// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Preprocessing
// Imports: public import Init public meta import Init public meta import Mathlib.Control.Basic public meta import Mathlib.Lean.Meta.Tactic.Rewrite public meta import Mathlib.Tactic.Linarith.Datatypes public meta import Mathlib.Util.AtomM public import Mathlib.Tactic.CancelDenoms.Core public import Mathlib.Tactic.Linarith.Datatypes public import Mathlib.Tactic.Zify
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_ineq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_numeral_x3f(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_rewriteType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_find_expr(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_io_mono_nanos_now();
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_List_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_ne_x3f_x27(lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(12, 252, 227, 83, 88, 185, 40, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "right"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__3_value),LEAN_SCALAR_PTR_LITERAL(18, 204, 165, 192, 253, 41, 237, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "splitConjunctions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__3_value),LEAN_SCALAR_PTR_LITERAL(60, 194, 114, 144, 224, 80, 8, 155)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "split conjunctions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "lt_of_not_ge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(172, 202, 18, 166, 1, 208, 224, 49)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "le_of_not_gt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(20, 53, 39, 207, 43, 74, 14, 72)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "filterComparisons"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__1_value),LEAN_SCALAR_PTR_LITERAL(221, 147, 44, 145, 10, 95, 36, 38)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "filter terms that are not proofs of comparisons"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "natCast_nonneg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 251, 248, 206, 215, 117, 62, 187)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Got exception when using cast "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mk__natCast__nonneg__prf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mk__natCast__nonneg__prf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__0 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__0_value;
static const lean_array_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__1 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__1_value;
static const lean_ctor_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__2 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__2_value;
static const lean_string_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "zifyProof failed on "};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__3 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__3_value;
static lean_once_cell_t lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "natToInt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(97, 182, 252, 97, 90, 197, 5, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "move nats to ints"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "add_one_le_iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(84, 114, 84, 144, 213, 130, 152, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(14, 81, 9, 215, 230, 198, 87, 3)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "strengthenStrictInt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__1_value),LEAN_SCALAR_PTR_LITERAL(185, 200, 101, 79, 229, 184, 234, 249)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "strengthen strict inequalities over int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "sub_eq_zero_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 36, 177, 113, 96, 233, 36, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "sub_nonpos_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(136, 91, 205, 28, 218, 191, 104, 34)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "sub_neg_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(217, 115, 118, 202, 222, 109, 96, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "compWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__1_value),LEAN_SCALAR_PTR_LITERAL(44, 130, 163, 125, 63, 56, 215, 40)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "make comparisons with zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Error in Linarith.normalizeDenominatorsLHS: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "without_one_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(179, 229, 226, 3, 208, 14, 47, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "OfScientific"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ofScientific"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(1, 219, 72, 84, 44, 38, 226, 47)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(101, 32, 126, 239, 82, 155, 222, 105)}};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Div"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(153, 247, 56, 19, 64, 245, 190, 87)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(25, 78, 24, 213, 240, 238, 239, 80)}};
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "cancelDenoms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__1_value),LEAN_SCALAR_PTR_LITERAL(148, 98, 11, 138, 68, 217, 76, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "cancel denominators"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = " finding squares"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "so we added proofs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__0_value;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__0_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__1_value;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3;
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__4 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__4_value;
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__5 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "mul_self_nonneg"};
static const lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__0 = (const lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__0_value;
static const lean_ctor_object lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(114, 153, 241, 129, 202, 48, 91, 1)}};
static const lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__1_value;
static const lean_string_object lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "sq_nonneg"};
static const lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__2_value;
static const lean_ctor_object lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 55, 193, 244, 121, 227, 18, 164)}};
static const lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__3 = (const lean_object*)&lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "found:"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mul_zero_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 130, 238, 45, 102, 47, 248, 21)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "zero_mul_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 18, 153, 232, 229, 90, 88, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "mul_nonneg_of_nonpos_of_nonpos"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(173, 143, 53, 142, 35, 79, 18, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_lt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(26, 46, 54, 245, 81, 108, 136, 63)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "mul_pos_of_neg_of_neg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 71, 118, 159, 118, 30, 168, 220)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " adding product terms"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "nlinarithExtras"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__1_value),LEAN_SCALAR_PTR_LITERAL(87, 131, 32, 61, 140, 197, 61, 122)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "nonlinear arithmetic extras"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__0 = (const lean_object*)&lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__1 = (const lean_object*)&lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__2(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "lt_or_gt_of_ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 252, 223, 6, 0, 180, 55, 7)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Or"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__2_value),LEAN_SCALAR_PTR_LITERAL(34, 237, 162, 225, 217, 98, 205, 196)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__3_value),LEAN_SCALAR_PTR_LITERAL(94, 178, 144, 142, 106, 224, 229, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe__aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe__aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "removeNe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 141, 101, 71, 34, 110, 254, 20)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 15, .m_data = "case split on ≠"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_removeNe___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToRealTransform;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nnrealToReal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_splitConjunctions___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 144, 255, 28, 87, 153, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "move nnreals to reals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Running preprocessors"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_Linarith_preprocess_spec__1(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_preprocess___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_preprocess___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_e_30_, v___y_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___boxed(lean_object* v_e_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0(v_e_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux(lean_object* v_proof_53_, lean_object* v_a_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_63_; 
lean_inc(v_a_57_);
lean_inc_ref(v_a_56_);
lean_inc(v_a_55_);
lean_inc_ref(v_a_54_);
lean_inc_ref(v_proof_53_);
v___x_63_ = lean_infer_type(v_proof_53_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v_a_64_; lean_object* v___x_65_; 
v_a_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_a_64_);
lean_dec_ref_known(v___x_63_, 1);
v___x_65_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_a_64_, v_a_55_);
if (lean_obj_tag(v___x_65_) == 0)
{
lean_object* v_a_66_; lean_object* v___x_67_; lean_object* v_fst_68_; 
v_a_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_a_66_);
lean_dec_ref_known(v___x_65_, 1);
v___x_67_ = l_Lean_Expr_getAppFnArgs(v_a_66_);
v_fst_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc(v_fst_68_);
if (lean_obj_tag(v_fst_68_) == 1)
{
lean_object* v_pre_69_; 
v_pre_69_ = lean_ctor_get(v_fst_68_, 0);
if (lean_obj_tag(v_pre_69_) == 0)
{
lean_object* v_snd_70_; lean_object* v_str_71_; lean_object* v___x_72_; uint8_t v___x_73_; 
v_snd_70_ = lean_ctor_get(v___x_67_, 1);
lean_inc(v_snd_70_);
lean_dec_ref(v___x_67_);
v_str_71_ = lean_ctor_get(v_fst_68_, 1);
lean_inc_ref(v_str_71_);
lean_dec_ref_known(v_fst_68_, 2);
v___x_72_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__0));
v___x_73_ = lean_string_dec_eq(v_str_71_, v___x_72_);
lean_dec_ref(v_str_71_);
if (v___x_73_ == 0)
{
lean_dec(v_snd_70_);
goto v___jp_59_;
}
else
{
lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_74_ = lean_array_get_size(v_snd_70_);
lean_dec(v_snd_70_);
v___x_75_ = lean_unsigned_to_nat(2u);
v___x_76_ = lean_nat_dec_eq(v___x_74_, v___x_75_);
if (v___x_76_ == 0)
{
goto v___jp_59_;
}
else
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_77_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__2));
v___x_78_ = lean_unsigned_to_nat(1u);
v___x_79_ = lean_mk_empty_array_with_capacity(v___x_78_);
v___x_80_ = lean_array_push(v___x_79_, v_proof_53_);
lean_inc_ref(v___x_80_);
v___x_81_ = l_Lean_Meta_mkAppM(v___x_77_, v___x_80_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
if (lean_obj_tag(v___x_81_) == 0)
{
lean_object* v_a_82_; lean_object* v___x_83_; 
v_a_82_ = lean_ctor_get(v___x_81_, 0);
lean_inc(v_a_82_);
lean_dec_ref_known(v___x_81_, 1);
v___x_83_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux(v_a_82_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
if (lean_obj_tag(v___x_83_) == 0)
{
lean_object* v_a_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v_a_84_ = lean_ctor_get(v___x_83_, 0);
lean_inc(v_a_84_);
lean_dec_ref_known(v___x_83_, 1);
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___closed__4));
v___x_86_ = l_Lean_Meta_mkAppM(v___x_85_, v___x_80_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
if (lean_obj_tag(v___x_86_) == 0)
{
lean_object* v_a_87_; lean_object* v___x_88_; 
v_a_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc(v_a_87_);
lean_dec_ref_known(v___x_86_, 1);
v___x_88_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux(v_a_87_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_97_; 
v_a_89_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_97_ == 0)
{
v___x_91_ = v___x_88_;
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_88_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = l_List_appendTR___redArg(v_a_84_, v_a_89_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v___x_93_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
else
{
lean_dec(v_a_84_);
return v___x_88_;
}
}
else
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
lean_dec(v_a_84_);
v_a_98_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_86_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_86_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
else
{
lean_dec_ref(v___x_80_);
return v___x_83_;
}
}
else
{
lean_object* v_a_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_113_; 
lean_dec_ref(v___x_80_);
v_a_106_ = lean_ctor_get(v___x_81_, 0);
v_isSharedCheck_113_ = !lean_is_exclusive(v___x_81_);
if (v_isSharedCheck_113_ == 0)
{
v___x_108_ = v___x_81_;
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_a_106_);
lean_dec(v___x_81_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_111_; 
if (v_isShared_109_ == 0)
{
v___x_111_ = v___x_108_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v_a_106_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_fst_68_, 2);
lean_dec_ref(v___x_67_);
goto v___jp_59_;
}
}
else
{
lean_dec(v_fst_68_);
lean_dec_ref(v___x_67_);
goto v___jp_59_;
}
}
else
{
lean_object* v_a_114_; lean_object* v___x_116_; uint8_t v_isShared_117_; uint8_t v_isSharedCheck_121_; 
lean_dec_ref(v_proof_53_);
v_a_114_ = lean_ctor_get(v___x_65_, 0);
v_isSharedCheck_121_ = !lean_is_exclusive(v___x_65_);
if (v_isSharedCheck_121_ == 0)
{
v___x_116_ = v___x_65_;
v_isShared_117_ = v_isSharedCheck_121_;
goto v_resetjp_115_;
}
else
{
lean_inc(v_a_114_);
lean_dec(v___x_65_);
v___x_116_ = lean_box(0);
v_isShared_117_ = v_isSharedCheck_121_;
goto v_resetjp_115_;
}
v_resetjp_115_:
{
lean_object* v___x_119_; 
if (v_isShared_117_ == 0)
{
v___x_119_ = v___x_116_;
goto v_reusejp_118_;
}
else
{
lean_object* v_reuseFailAlloc_120_; 
v_reuseFailAlloc_120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_120_, 0, v_a_114_);
v___x_119_ = v_reuseFailAlloc_120_;
goto v_reusejp_118_;
}
v_reusejp_118_:
{
return v___x_119_;
}
}
}
}
else
{
lean_object* v_a_122_; lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_129_; 
lean_dec_ref(v_proof_53_);
v_a_122_ = lean_ctor_get(v___x_63_, 0);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_129_ == 0)
{
v___x_124_ = v___x_63_;
v_isShared_125_ = v_isSharedCheck_129_;
goto v_resetjp_123_;
}
else
{
lean_inc(v_a_122_);
lean_dec(v___x_63_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_129_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
lean_object* v___x_127_; 
if (v_isShared_125_ == 0)
{
v___x_127_ = v___x_124_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v_a_122_);
v___x_127_ = v_reuseFailAlloc_128_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
return v___x_127_;
}
}
}
v___jp_59_:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_box(0);
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v_proof_53_);
lean_ctor_set(v___x_61_, 1, v___x_60_);
v___x_62_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux___boxed(lean_object* v_proof_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux(v_proof_130_, v_a_131_, v_a_132_, v_a_133_, v_a_134_);
lean_dec(v_a_134_);
lean_dec_ref(v_a_133_);
lean_dec(v_a_132_);
lean_dec_ref(v_a_131_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0(lean_object* v_h_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___y_171_; uint8_t v___y_172_; lean_object* v_a_177_; lean_object* v___x_180_; 
lean_inc(v___y_168_);
lean_inc_ref(v___y_167_);
lean_inc(v___y_166_);
lean_inc_ref(v___y_165_);
lean_inc_ref(v_h_164_);
v___x_180_ = lean_infer_type(v_h_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_180_) == 0)
{
lean_object* v_a_181_; lean_object* v___x_182_; lean_object* v_a_183_; lean_object* v___x_184_; lean_object* v___x_185_; uint8_t v___x_186_; 
v_a_181_ = lean_ctor_get(v___x_180_, 0);
lean_inc(v_a_181_);
lean_dec_ref_known(v___x_180_, 1);
v___x_182_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_a_181_, v___y_166_);
v_a_183_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_a_183_);
lean_dec_ref(v___x_182_);
v___x_184_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__1));
v___x_185_ = lean_unsigned_to_nat(1u);
v___x_186_ = l_Lean_Expr_isAppOfArity(v_a_183_, v___x_184_, v___x_185_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_183_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_196_; 
v_isSharedCheck_196_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_196_ == 0)
{
lean_object* v_unused_197_; 
v_unused_197_ = lean_ctor_get(v___x_187_, 0);
lean_dec(v_unused_197_);
v___x_189_ = v___x_187_;
v_isShared_190_ = v_isSharedCheck_196_;
goto v_resetjp_188_;
}
else
{
lean_dec(v___x_187_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_196_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_194_; 
v___x_191_ = lean_box(0);
v___x_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_192_, 0, v_h_164_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
if (v_isShared_190_ == 0)
{
lean_ctor_set(v___x_189_, 0, v___x_192_);
v___x_194_ = v___x_189_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v___x_192_);
v___x_194_ = v_reuseFailAlloc_195_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
return v___x_194_;
}
}
}
else
{
lean_object* v_a_198_; 
lean_dec_ref(v_h_164_);
v_a_198_ = lean_ctor_get(v___x_187_, 0);
lean_inc(v_a_198_);
lean_dec_ref_known(v___x_187_, 1);
v_a_177_ = v_a_198_;
goto v___jp_176_;
}
}
else
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = l_Lean_Expr_appArg_x21(v_a_183_);
lean_dec(v_a_183_);
v___x_200_ = lp_mathlib_Lean_Expr_ineq_x3f(v___x_199_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_250_; 
v_a_201_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_250_ == 0)
{
v___x_203_ = v___x_200_;
v_isShared_204_ = v_isSharedCheck_250_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_200_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_250_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v_fst_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_248_; 
v_fst_205_ = lean_ctor_get(v_a_201_, 0);
v_isSharedCheck_248_ = !lean_is_exclusive(v_a_201_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; 
v_unused_249_ = lean_ctor_get(v_a_201_, 1);
lean_dec(v_unused_249_);
v___x_207_ = v_a_201_;
v_isShared_208_ = v_isSharedCheck_248_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_fst_205_);
lean_dec(v_a_201_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_248_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
uint8_t v___x_209_; 
v___x_209_ = lean_unbox(v_fst_205_);
lean_dec(v_fst_205_);
switch(v___x_209_)
{
case 0:
{
lean_object* v___x_210_; lean_object* v___x_212_; 
lean_del_object(v___x_207_);
lean_dec_ref(v_h_164_);
v___x_210_ = lean_box(0);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 0, v___x_210_);
v___x_212_ = v___x_203_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v___x_210_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
case 1:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
lean_del_object(v___x_203_);
v___x_214_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__3));
v___x_215_ = lean_mk_empty_array_with_capacity(v___x_185_);
v___x_216_ = lean_array_push(v___x_215_, v_h_164_);
v___x_217_ = l_Lean_Meta_mkAppM(v___x_214_, v___x_216_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_217_) == 0)
{
lean_object* v_a_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_229_; 
v_a_218_ = lean_ctor_get(v___x_217_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_229_ == 0)
{
v___x_220_ = v___x_217_;
v_isShared_221_ = v_isSharedCheck_229_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_a_218_);
lean_dec(v___x_217_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_229_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v___x_222_; lean_object* v___x_224_; 
v___x_222_ = lean_box(0);
if (v_isShared_208_ == 0)
{
lean_ctor_set_tag(v___x_207_, 1);
lean_ctor_set(v___x_207_, 1, v___x_222_);
lean_ctor_set(v___x_207_, 0, v_a_218_);
v___x_224_ = v___x_207_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_218_);
lean_ctor_set(v_reuseFailAlloc_228_, 1, v___x_222_);
v___x_224_ = v_reuseFailAlloc_228_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
lean_object* v___x_226_; 
if (v_isShared_221_ == 0)
{
lean_ctor_set(v___x_220_, 0, v___x_224_);
v___x_226_ = v___x_220_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v___x_224_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
else
{
lean_object* v_a_230_; 
lean_del_object(v___x_207_);
v_a_230_ = lean_ctor_get(v___x_217_, 0);
lean_inc(v_a_230_);
lean_dec_ref_known(v___x_217_, 1);
v_a_177_ = v_a_230_;
goto v___jp_176_;
}
}
default: 
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
lean_del_object(v___x_203_);
v___x_231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__5));
v___x_232_ = lean_mk_empty_array_with_capacity(v___x_185_);
v___x_233_ = lean_array_push(v___x_232_, v_h_164_);
v___x_234_ = l_Lean_Meta_mkAppM(v___x_231_, v___x_233_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_234_) == 0)
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_246_; 
v_a_235_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_246_ == 0)
{
v___x_237_ = v___x_234_;
v_isShared_238_ = v_isSharedCheck_246_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_234_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_246_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_239_; lean_object* v___x_241_; 
v___x_239_ = lean_box(0);
if (v_isShared_208_ == 0)
{
lean_ctor_set_tag(v___x_207_, 1);
lean_ctor_set(v___x_207_, 1, v___x_239_);
lean_ctor_set(v___x_207_, 0, v_a_235_);
v___x_241_ = v___x_207_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_a_235_);
lean_ctor_set(v_reuseFailAlloc_245_, 1, v___x_239_);
v___x_241_ = v_reuseFailAlloc_245_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
lean_object* v___x_243_; 
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_241_);
v___x_243_ = v___x_237_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_241_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
else
{
lean_object* v_a_247_; 
lean_del_object(v___x_207_);
v_a_247_ = lean_ctor_get(v___x_234_, 0);
lean_inc(v_a_247_);
lean_dec_ref_known(v___x_234_, 1);
v_a_177_ = v_a_247_;
goto v___jp_176_;
}
}
}
}
}
}
else
{
lean_object* v_a_251_; 
lean_dec_ref(v_h_164_);
v_a_251_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_a_251_);
lean_dec_ref_known(v___x_200_, 1);
v_a_177_ = v_a_251_;
goto v___jp_176_;
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec_ref(v_h_164_);
v_a_252_ = lean_ctor_get(v___x_180_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_180_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_180_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_180_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
v___jp_170_:
{
if (v___y_172_ == 0)
{
lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec_ref(v___y_171_);
v___x_173_ = lean_box(0);
v___x_174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
return v___x_174_;
}
else
{
lean_object* v___x_175_; 
v___x_175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_175_, 0, v___y_171_);
return v___x_175_;
}
}
v___jp_176_:
{
uint8_t v___x_178_; 
v___x_178_ = l_Lean_Exception_isInterrupt(v_a_177_);
if (v___x_178_ == 0)
{
uint8_t v___x_179_; 
lean_inc_ref(v_a_177_);
v___x_179_ = l_Lean_Exception_isRuntime(v_a_177_);
v___y_171_ = v_a_177_;
v___y_172_ = v___x_179_;
goto v___jp_170_;
}
else
{
v___y_171_ = v_a_177_;
v___y_172_ = v___x_178_;
goto v___jp_170_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___boxed(lean_object* v_h_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0(v_h_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(lean_object* v_x_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = l_Lean_Meta_saveState___redArg(v___y_284_, v___y_286_);
if (lean_obj_tag(v___x_288_) == 0)
{
lean_object* v_a_289_; lean_object* v___x_290_; 
v_a_289_ = lean_ctor_get(v___x_288_, 0);
lean_inc(v_a_289_);
lean_dec_ref_known(v___x_288_, 1);
lean_inc(v___y_286_);
lean_inc_ref(v___y_285_);
lean_inc(v___y_284_);
lean_inc_ref(v___y_283_);
v___x_290_ = lean_apply_5(v_x_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, lean_box(0));
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_299_; 
lean_dec(v_a_289_);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_299_ == 0)
{
lean_object* v_unused_300_; 
v_unused_300_ = lean_ctor_get(v___x_290_, 0);
lean_dec(v_unused_300_);
v___x_292_ = v___x_290_;
v_isShared_293_ = v_isSharedCheck_299_;
goto v_resetjp_291_;
}
else
{
lean_dec(v___x_290_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_299_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
uint8_t v___x_294_; lean_object* v___x_295_; lean_object* v___x_297_; 
v___x_294_ = 1;
v___x_295_ = lean_box(v___x_294_);
if (v_isShared_293_ == 0)
{
lean_ctor_set(v___x_292_, 0, v___x_295_);
v___x_297_ = v___x_292_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v___x_295_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
else
{
lean_object* v_a_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_330_; 
v_a_301_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_330_ == 0)
{
v___x_303_ = v___x_290_;
v_isShared_304_ = v_isSharedCheck_330_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_a_301_);
lean_dec(v___x_290_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_330_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v___x_306_; 
lean_inc(v_a_301_);
if (v_isShared_304_ == 0)
{
v___x_306_ = v___x_303_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_301_);
v___x_306_ = v_reuseFailAlloc_329_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
uint8_t v___y_308_; uint8_t v___x_327_; 
v___x_327_ = l_Lean_Exception_isInterrupt(v_a_301_);
if (v___x_327_ == 0)
{
uint8_t v___x_328_; 
v___x_328_ = l_Lean_Exception_isRuntime(v_a_301_);
v___y_308_ = v___x_328_;
goto v___jp_307_;
}
else
{
lean_dec(v_a_301_);
v___y_308_ = v___x_327_;
goto v___jp_307_;
}
v___jp_307_:
{
if (v___y_308_ == 0)
{
lean_object* v___x_309_; 
lean_dec_ref(v___x_306_);
v___x_309_ = l_Lean_Meta_SavedState_restore___redArg(v_a_289_, v___y_284_, v___y_286_);
lean_dec(v_a_289_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_317_; 
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_317_ == 0)
{
lean_object* v_unused_318_; 
v_unused_318_ = lean_ctor_get(v___x_309_, 0);
lean_dec(v_unused_318_);
v___x_311_ = v___x_309_;
v_isShared_312_ = v_isSharedCheck_317_;
goto v_resetjp_310_;
}
else
{
lean_dec(v___x_309_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_317_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; lean_object* v___x_315_; 
v___x_313_ = lean_box(v___y_308_);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 0, v___x_313_);
v___x_315_ = v___x_311_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_313_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
else
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_326_; 
v_a_319_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___x_309_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_309_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_324_; 
if (v_isShared_322_ == 0)
{
v___x_324_ = v___x_321_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_a_319_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
else
{
lean_dec(v_a_289_);
return v___x_306_;
}
}
}
}
}
}
else
{
lean_object* v_a_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_338_; 
lean_dec_ref(v_x_282_);
v_a_331_ = lean_ctor_get(v___x_288_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v___x_288_);
if (v_isSharedCheck_338_ == 0)
{
v___x_333_ = v___x_288_;
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_a_331_);
lean_dec(v___x_288_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_336_; 
if (v_isShared_334_ == 0)
{
v___x_336_ = v___x_333_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v_a_331_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg___boxed(lean_object* v_x_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(v_x_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1(lean_object* v_00_u03b1_346_, lean_object* v_x_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(v_x_347_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___boxed(lean_object* v_00_u03b1_354_, lean_object* v_x_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1(v_00_u03b1_354_, v_x_355_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
lean_dec(v___y_357_);
lean_dec_ref(v___y_356_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(lean_object* v_msgData_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_368_; lean_object* v_env_369_; lean_object* v___x_370_; lean_object* v_mctx_371_; lean_object* v_lctx_372_; lean_object* v_options_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_368_ = lean_st_ref_get(v___y_366_);
v_env_369_ = lean_ctor_get(v___x_368_, 0);
lean_inc_ref(v_env_369_);
lean_dec(v___x_368_);
v___x_370_ = lean_st_ref_get(v___y_364_);
v_mctx_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc_ref(v_mctx_371_);
lean_dec(v___x_370_);
v_lctx_372_ = lean_ctor_get(v___y_363_, 2);
v_options_373_ = lean_ctor_get(v___y_365_, 2);
lean_inc_ref(v_options_373_);
lean_inc_ref(v_lctx_372_);
v___x_374_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_374_, 0, v_env_369_);
lean_ctor_set(v___x_374_, 1, v_mctx_371_);
lean_ctor_set(v___x_374_, 2, v_lctx_372_);
lean_ctor_set(v___x_374_, 3, v_options_373_);
v___x_375_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_msgData_362_);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0___boxed(lean_object* v_msgData_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(v_msgData_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(lean_object* v_msg_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_ref_390_; lean_object* v___x_391_; lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_400_; 
v_ref_390_ = lean_ctor_get(v___y_387_, 5);
v___x_391_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(v_msg_384_, v___y_385_, v___y_386_, v___y_387_, v___y_388_);
v_a_392_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_400_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_400_ == 0)
{
v___x_394_ = v___x_391_;
v_isShared_395_ = v_isSharedCheck_400_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_391_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_400_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_396_; lean_object* v___x_398_; 
lean_inc(v_ref_390_);
v___x_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_396_, 0, v_ref_390_);
lean_ctor_set(v___x_396_, 1, v_a_392_);
if (v_isShared_395_ == 0)
{
lean_ctor_set_tag(v___x_394_, 1);
lean_ctor_set(v___x_394_, 0, v___x_396_);
v___x_398_ = v___x_394_;
goto v_reusejp_397_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v___x_396_);
v___x_398_ = v_reuseFailAlloc_399_;
goto v_reusejp_397_;
}
v_reusejp_397_:
{
return v___x_398_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg___boxed(lean_object* v_msg_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v_msg_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
return v_res_407_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__0));
v___x_410_ = l_Lean_stringToMessageData(v___x_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0(lean_object* v_e_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v___y_419_; lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v___y_422_; lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(v_e_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
if (lean_obj_tag(v___x_425_) == 0)
{
lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_443_; 
v_a_426_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_443_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_443_ == 0)
{
v___x_428_ = v___x_425_;
v_isShared_429_ = v_isSharedCheck_443_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_425_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_443_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v_snd_430_; lean_object* v_snd_431_; lean_object* v_fst_432_; 
v_snd_430_ = lean_ctor_get(v_a_426_, 1);
lean_inc(v_snd_430_);
lean_dec(v_a_426_);
v_snd_431_ = lean_ctor_get(v_snd_430_, 1);
lean_inc(v_snd_431_);
lean_dec(v_snd_430_);
v_fst_432_ = lean_ctor_get(v_snd_431_, 0);
lean_inc(v_fst_432_);
lean_dec(v_snd_431_);
if (lean_obj_tag(v_fst_432_) == 4)
{
lean_object* v_declName_433_; 
v_declName_433_ = lean_ctor_get(v_fst_432_, 0);
lean_inc(v_declName_433_);
if (lean_obj_tag(v_declName_433_) == 1)
{
lean_object* v_pre_434_; 
v_pre_434_ = lean_ctor_get(v_declName_433_, 0);
if (lean_obj_tag(v_pre_434_) == 0)
{
lean_object* v_us_435_; lean_object* v_str_436_; lean_object* v___x_437_; uint8_t v___x_438_; 
v_us_435_ = lean_ctor_get(v_fst_432_, 1);
lean_inc(v_us_435_);
lean_dec_ref_known(v_fst_432_, 2);
v_str_436_ = lean_ctor_get(v_declName_433_, 1);
lean_inc_ref(v_str_436_);
lean_dec_ref_known(v_declName_433_, 2);
v___x_437_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__2));
v___x_438_ = lean_string_dec_eq(v_str_436_, v___x_437_);
lean_dec_ref(v_str_436_);
if (v___x_438_ == 0)
{
lean_dec(v_us_435_);
lean_del_object(v___x_428_);
v___y_419_ = v___y_413_;
v___y_420_ = v___y_414_;
v___y_421_ = v___y_415_;
v___y_422_ = v___y_416_;
goto v___jp_418_;
}
else
{
if (lean_obj_tag(v_us_435_) == 0)
{
lean_object* v___x_439_; lean_object* v___x_441_; 
v___x_439_ = lean_box(0);
if (v_isShared_429_ == 0)
{
lean_ctor_set(v___x_428_, 0, v___x_439_);
v___x_441_ = v___x_428_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_442_; 
v_reuseFailAlloc_442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_442_, 0, v___x_439_);
v___x_441_ = v_reuseFailAlloc_442_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
return v___x_441_;
}
}
else
{
lean_dec(v_us_435_);
lean_del_object(v___x_428_);
v___y_419_ = v___y_413_;
v___y_420_ = v___y_414_;
v___y_421_ = v___y_415_;
v___y_422_ = v___y_416_;
goto v___jp_418_;
}
}
}
else
{
lean_dec_ref_known(v_declName_433_, 2);
lean_dec_ref_known(v_fst_432_, 2);
lean_del_object(v___x_428_);
v___y_419_ = v___y_413_;
v___y_420_ = v___y_414_;
v___y_421_ = v___y_415_;
v___y_422_ = v___y_416_;
goto v___jp_418_;
}
}
else
{
lean_dec(v_declName_433_);
lean_dec_ref_known(v_fst_432_, 2);
lean_del_object(v___x_428_);
v___y_419_ = v___y_413_;
v___y_420_ = v___y_414_;
v___y_421_ = v___y_415_;
v___y_422_ = v___y_416_;
goto v___jp_418_;
}
}
else
{
lean_dec(v_fst_432_);
lean_del_object(v___x_428_);
v___y_419_ = v___y_413_;
v___y_420_ = v___y_414_;
v___y_421_ = v___y_415_;
v___y_422_ = v___y_416_;
goto v___jp_418_;
}
}
}
else
{
lean_object* v_a_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_451_; 
v_a_444_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_451_ == 0)
{
v___x_446_ = v___x_425_;
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_a_444_);
lean_dec(v___x_425_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_449_; 
if (v_isShared_447_ == 0)
{
v___x_449_ = v___x_446_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_a_444_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
v___jp_418_:
{
lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_423_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1);
v___x_424_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v___x_423_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
return v___x_424_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___boxed(lean_object* v_e_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0(v_e_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_);
lean_dec(v___y_456_);
lean_dec_ref(v___y_455_);
lean_dec(v___y_454_);
lean_dec_ref(v___y_453_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp(lean_object* v_e_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_){
_start:
{
lean_object* v___f_465_; lean_object* v___x_466_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___boxed), 6, 1);
lean_closure_set(v___f_465_, 0, v_e_459_);
v___x_466_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(v___f_465_, v_a_460_, v_a_461_, v_a_462_, v_a_463_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___boxed(lean_object* v_e_467_, lean_object* v_a_468_, lean_object* v_a_469_, lean_object* v_a_470_, lean_object* v_a_471_, lean_object* v_a_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_Mathlib_Tactic_Linarith_isNatProp(v_e_467_, v_a_468_, v_a_469_, v_a_470_, v_a_471_);
lean_dec(v_a_471_);
lean_dec_ref(v_a_470_);
lean_dec(v_a_469_);
lean_dec_ref(v_a_468_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0(lean_object* v_00_u03b1_474_, lean_object* v_msg_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v_msg_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___boxed(lean_object* v_00_u03b1_482_, lean_object* v_msg_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0(v_00_u03b1_482_, v_msg_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec(v___y_485_);
lean_dec_ref(v___y_484_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe(lean_object* v_e_491_){
_start:
{
lean_object* v___x_492_; lean_object* v_fst_493_; 
v___x_492_ = l_Lean_Expr_getAppFnArgs(v_e_491_);
v_fst_493_ = lean_ctor_get(v___x_492_, 0);
lean_inc(v_fst_493_);
if (lean_obj_tag(v_fst_493_) == 1)
{
lean_object* v_pre_494_; 
v_pre_494_ = lean_ctor_get(v_fst_493_, 0);
lean_inc(v_pre_494_);
if (lean_obj_tag(v_pre_494_) == 1)
{
lean_object* v_pre_495_; 
v_pre_495_ = lean_ctor_get(v_pre_494_, 0);
if (lean_obj_tag(v_pre_495_) == 0)
{
lean_object* v_snd_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_520_; 
v_snd_496_ = lean_ctor_get(v___x_492_, 1);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_520_ == 0)
{
lean_object* v_unused_521_; 
v_unused_521_ = lean_ctor_get(v___x_492_, 0);
lean_dec(v_unused_521_);
v___x_498_ = v___x_492_;
v_isShared_499_ = v_isSharedCheck_520_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_snd_496_);
lean_dec(v___x_492_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_520_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v_str_500_; lean_object* v_str_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v_str_500_ = lean_ctor_get(v_fst_493_, 1);
lean_inc_ref(v_str_500_);
lean_dec_ref_known(v_fst_493_, 2);
v_str_501_ = lean_ctor_get(v_pre_494_, 1);
lean_inc_ref(v_str_501_);
lean_dec_ref_known(v_pre_494_, 2);
v___x_502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__2));
v___x_503_ = lean_string_dec_eq(v_str_501_, v___x_502_);
lean_dec_ref(v_str_501_);
if (v___x_503_ == 0)
{
lean_object* v___x_504_; 
lean_dec_ref(v_str_500_);
lean_del_object(v___x_498_);
lean_dec(v_snd_496_);
v___x_504_ = lean_box(0);
return v___x_504_;
}
else
{
lean_object* v___x_505_; uint8_t v___x_506_; 
v___x_505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe___closed__0));
v___x_506_ = lean_string_dec_eq(v_str_500_, v___x_505_);
lean_dec_ref(v_str_500_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; 
lean_del_object(v___x_498_);
lean_dec(v_snd_496_);
v___x_507_ = lean_box(0);
return v___x_507_;
}
else
{
lean_object* v___x_508_; lean_object* v___x_509_; uint8_t v___x_510_; 
v___x_508_ = lean_array_get_size(v_snd_496_);
v___x_509_ = lean_unsigned_to_nat(3u);
v___x_510_ = lean_nat_dec_eq(v___x_508_, v___x_509_);
if (v___x_510_ == 0)
{
lean_object* v___x_511_; 
lean_del_object(v___x_498_);
lean_dec(v_snd_496_);
v___x_511_ = lean_box(0);
return v___x_511_;
}
else
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_517_; 
v___x_512_ = lean_unsigned_to_nat(0u);
v___x_513_ = lean_array_fget(v_snd_496_, v___x_512_);
v___x_514_ = lean_unsigned_to_nat(2u);
v___x_515_ = lean_array_fget(v_snd_496_, v___x_514_);
lean_dec(v_snd_496_);
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 1, v___x_513_);
lean_ctor_set(v___x_498_, 0, v___x_515_);
v___x_517_ = v___x_498_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v___x_515_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v___x_513_);
v___x_517_ = v_reuseFailAlloc_519_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
lean_object* v___x_518_; 
v___x_518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_518_, 0, v___x_517_);
return v___x_518_;
}
}
}
}
}
}
else
{
lean_object* v___x_522_; 
lean_dec_ref_known(v_pre_494_, 2);
lean_dec_ref_known(v_fst_493_, 2);
lean_dec_ref(v___x_492_);
v___x_522_ = lean_box(0);
return v___x_522_;
}
}
else
{
lean_object* v___x_523_; 
lean_dec(v_pre_494_);
lean_dec_ref_known(v_fst_493_, 2);
lean_dec_ref(v___x_492_);
v___x_523_ = lean_box(0);
return v___x_523_;
}
}
else
{
lean_object* v___x_524_; 
lean_dec(v_fst_493_);
lean_dec_ref(v___x_492_);
v___x_524_ = lean_box(0);
return v___x_524_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(lean_object* v_e_533_){
_start:
{
lean_object* v_a_535_; lean_object* v_b_536_; lean_object* v___x_540_; 
lean_inc_ref(v_e_533_);
v___x_540_ = lp_mathlib_Mathlib_Tactic_Linarith_isNatCoe(v_e_533_);
if (lean_obj_tag(v___x_540_) == 0)
{
lean_object* v___x_541_; lean_object* v_fst_542_; 
v___x_541_ = l_Lean_Expr_getAppFnArgs(v_e_533_);
v_fst_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc(v_fst_542_);
if (lean_obj_tag(v_fst_542_) == 1)
{
lean_object* v_pre_543_; 
v_pre_543_ = lean_ctor_get(v_fst_542_, 0);
lean_inc(v_pre_543_);
if (lean_obj_tag(v_pre_543_) == 1)
{
lean_object* v_pre_544_; 
v_pre_544_ = lean_ctor_get(v_pre_543_, 0);
if (lean_obj_tag(v_pre_544_) == 0)
{
lean_object* v_snd_545_; lean_object* v_str_546_; lean_object* v_str_547_; lean_object* v___x_548_; uint8_t v___x_549_; 
v_snd_545_ = lean_ctor_get(v___x_541_, 1);
lean_inc(v_snd_545_);
lean_dec_ref(v___x_541_);
v_str_546_ = lean_ctor_get(v_fst_542_, 1);
lean_inc_ref(v_str_546_);
lean_dec_ref_known(v_fst_542_, 2);
v_str_547_ = lean_ctor_get(v_pre_543_, 1);
lean_inc_ref(v_str_547_);
lean_dec_ref_known(v_pre_543_, 2);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__0));
v___x_549_ = lean_string_dec_eq(v_str_547_, v___x_548_);
if (v___x_549_ == 0)
{
lean_object* v___x_550_; uint8_t v___x_551_; 
v___x_550_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__1));
v___x_551_ = lean_string_dec_eq(v_str_547_, v___x_550_);
if (v___x_551_ == 0)
{
lean_object* v___x_552_; uint8_t v___x_553_; 
v___x_552_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__2));
v___x_553_ = lean_string_dec_eq(v_str_547_, v___x_552_);
if (v___x_553_ == 0)
{
lean_object* v___x_554_; uint8_t v___x_555_; 
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__3));
v___x_555_ = lean_string_dec_eq(v_str_547_, v___x_554_);
lean_dec_ref(v_str_547_);
if (v___x_555_ == 0)
{
lean_object* v___x_556_; 
lean_dec_ref(v_str_546_);
lean_dec(v_snd_545_);
v___x_556_ = lean_box(0);
return v___x_556_;
}
else
{
lean_object* v___x_557_; uint8_t v___x_558_; 
v___x_557_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__4));
v___x_558_ = lean_string_dec_eq(v_str_546_, v___x_557_);
lean_dec_ref(v_str_546_);
if (v___x_558_ == 0)
{
lean_object* v___x_559_; 
lean_dec(v_snd_545_);
v___x_559_ = lean_box(0);
return v___x_559_;
}
else
{
lean_object* v___x_560_; lean_object* v___x_561_; uint8_t v___x_562_; 
v___x_560_ = lean_array_get_size(v_snd_545_);
v___x_561_ = lean_unsigned_to_nat(3u);
v___x_562_ = lean_nat_dec_eq(v___x_560_, v___x_561_);
if (v___x_562_ == 0)
{
lean_object* v___x_563_; 
lean_dec(v_snd_545_);
v___x_563_ = lean_box(0);
return v___x_563_;
}
else
{
lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_564_ = lean_unsigned_to_nat(2u);
v___x_565_ = lean_array_fget(v_snd_545_, v___x_564_);
lean_dec(v_snd_545_);
v_e_533_ = v___x_565_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_567_; uint8_t v___x_568_; 
lean_dec_ref(v_str_547_);
v___x_567_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__5));
v___x_568_ = lean_string_dec_eq(v_str_546_, v___x_567_);
lean_dec_ref(v_str_546_);
if (v___x_568_ == 0)
{
lean_object* v___x_569_; 
lean_dec(v_snd_545_);
v___x_569_ = lean_box(0);
return v___x_569_;
}
else
{
lean_object* v___x_570_; lean_object* v___x_571_; uint8_t v___x_572_; 
v___x_570_ = lean_array_get_size(v_snd_545_);
v___x_571_ = lean_unsigned_to_nat(6u);
v___x_572_ = lean_nat_dec_eq(v___x_570_, v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; 
lean_dec(v_snd_545_);
v___x_573_ = lean_box(0);
return v___x_573_;
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_574_ = lean_unsigned_to_nat(4u);
v___x_575_ = lean_array_fget(v_snd_545_, v___x_574_);
v___x_576_ = lean_unsigned_to_nat(5u);
v___x_577_ = lean_array_fget(v_snd_545_, v___x_576_);
lean_dec(v_snd_545_);
v_a_535_ = v___x_575_;
v_b_536_ = v___x_577_;
goto v___jp_534_;
}
}
}
}
else
{
lean_object* v___x_578_; uint8_t v___x_579_; 
lean_dec_ref(v_str_547_);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__6));
v___x_579_ = lean_string_dec_eq(v_str_546_, v___x_578_);
lean_dec_ref(v_str_546_);
if (v___x_579_ == 0)
{
lean_object* v___x_580_; 
lean_dec(v_snd_545_);
v___x_580_ = lean_box(0);
return v___x_580_;
}
else
{
lean_object* v___x_581_; lean_object* v___x_582_; uint8_t v___x_583_; 
v___x_581_ = lean_array_get_size(v_snd_545_);
v___x_582_ = lean_unsigned_to_nat(6u);
v___x_583_ = lean_nat_dec_eq(v___x_581_, v___x_582_);
if (v___x_583_ == 0)
{
lean_object* v___x_584_; 
lean_dec(v_snd_545_);
v___x_584_ = lean_box(0);
return v___x_584_;
}
else
{
lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_585_ = lean_unsigned_to_nat(4u);
v___x_586_ = lean_array_fget(v_snd_545_, v___x_585_);
v___x_587_ = lean_unsigned_to_nat(5u);
v___x_588_ = lean_array_fget(v_snd_545_, v___x_587_);
lean_dec(v_snd_545_);
v_a_535_ = v___x_586_;
v_b_536_ = v___x_588_;
goto v___jp_534_;
}
}
}
}
else
{
lean_object* v___x_589_; uint8_t v___x_590_; 
lean_dec_ref(v_str_547_);
v___x_589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__7));
v___x_590_ = lean_string_dec_eq(v_str_546_, v___x_589_);
lean_dec_ref(v_str_546_);
if (v___x_590_ == 0)
{
lean_object* v___x_591_; 
lean_dec(v_snd_545_);
v___x_591_ = lean_box(0);
return v___x_591_;
}
else
{
lean_object* v___x_592_; lean_object* v___x_593_; uint8_t v___x_594_; 
v___x_592_ = lean_array_get_size(v_snd_545_);
v___x_593_ = lean_unsigned_to_nat(6u);
v___x_594_ = lean_nat_dec_eq(v___x_592_, v___x_593_);
if (v___x_594_ == 0)
{
lean_object* v___x_595_; 
lean_dec(v_snd_545_);
v___x_595_ = lean_box(0);
return v___x_595_;
}
else
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; 
v___x_596_ = lean_unsigned_to_nat(4u);
v___x_597_ = lean_array_fget(v_snd_545_, v___x_596_);
v___x_598_ = lean_unsigned_to_nat(5u);
v___x_599_ = lean_array_fget(v_snd_545_, v___x_598_);
lean_dec(v_snd_545_);
v_a_535_ = v___x_597_;
v_b_536_ = v___x_599_;
goto v___jp_534_;
}
}
}
}
else
{
lean_object* v___x_600_; 
lean_dec_ref_known(v_pre_543_, 2);
lean_dec_ref_known(v_fst_542_, 2);
lean_dec_ref(v___x_541_);
v___x_600_ = lean_box(0);
return v___x_600_;
}
}
else
{
lean_object* v___x_601_; 
lean_dec_ref_known(v_fst_542_, 2);
lean_dec(v_pre_543_);
lean_dec_ref(v___x_541_);
v___x_601_ = lean_box(0);
return v___x_601_;
}
}
else
{
lean_object* v___x_602_; 
lean_dec(v_fst_542_);
lean_dec_ref(v___x_541_);
v___x_602_ = lean_box(0);
return v___x_602_;
}
}
else
{
lean_object* v_val_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
lean_dec_ref(v_e_533_);
v_val_603_ = lean_ctor_get(v___x_540_, 0);
lean_inc(v_val_603_);
lean_dec_ref_known(v___x_540_, 1);
v___x_604_ = lean_box(0);
v___x_605_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_605_, 0, v_val_603_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
return v___x_605_;
}
v___jp_534_:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_537_ = lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(v_a_535_);
v___x_538_ = lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(v_b_536_);
v___x_539_ = l_List_appendTR___redArg(v___x_537_, v___x_538_);
return v___x_539_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg(lean_object* v_x_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = l_Lean_Meta_saveState___redArg(v___y_608_, v___y_610_);
if (lean_obj_tag(v___x_612_) == 0)
{
lean_object* v_a_613_; lean_object* v___x_614_; 
v_a_613_ = lean_ctor_get(v___x_612_, 0);
lean_inc(v_a_613_);
lean_dec_ref_known(v___x_612_, 1);
lean_inc(v___y_610_);
lean_inc_ref(v___y_609_);
lean_inc(v___y_608_);
lean_inc_ref(v___y_607_);
v___x_614_ = lean_apply_5(v_x_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, lean_box(0));
if (lean_obj_tag(v___x_614_) == 0)
{
lean_dec(v_a_613_);
return v___x_614_;
}
else
{
lean_object* v_a_615_; uint8_t v___y_617_; uint8_t v___x_635_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
lean_inc(v_a_615_);
v___x_635_ = l_Lean_Exception_isInterrupt(v_a_615_);
if (v___x_635_ == 0)
{
uint8_t v___x_636_; 
lean_inc(v_a_615_);
v___x_636_ = l_Lean_Exception_isRuntime(v_a_615_);
v___y_617_ = v___x_636_;
goto v___jp_616_;
}
else
{
v___y_617_ = v___x_635_;
goto v___jp_616_;
}
v___jp_616_:
{
if (v___y_617_ == 0)
{
lean_object* v___x_618_; 
lean_dec_ref_known(v___x_614_, 1);
v___x_618_ = l_Lean_Meta_SavedState_restore___redArg(v_a_613_, v___y_608_, v___y_610_);
lean_dec(v_a_613_);
if (lean_obj_tag(v___x_618_) == 0)
{
lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
v_isSharedCheck_625_ = !lean_is_exclusive(v___x_618_);
if (v_isSharedCheck_625_ == 0)
{
lean_object* v_unused_626_; 
v_unused_626_ = lean_ctor_get(v___x_618_, 0);
lean_dec(v_unused_626_);
v___x_620_ = v___x_618_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_dec(v___x_618_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_623_; 
if (v_isShared_621_ == 0)
{
lean_ctor_set_tag(v___x_620_, 1);
lean_ctor_set(v___x_620_, 0, v_a_615_);
v___x_623_ = v___x_620_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_a_615_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
else
{
lean_object* v_a_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_634_; 
lean_dec(v_a_615_);
v_a_627_ = lean_ctor_get(v___x_618_, 0);
v_isSharedCheck_634_ = !lean_is_exclusive(v___x_618_);
if (v_isSharedCheck_634_ == 0)
{
v___x_629_ = v___x_618_;
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_a_627_);
lean_dec(v___x_618_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_632_; 
if (v_isShared_630_ == 0)
{
v___x_632_ = v___x_629_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_633_; 
v_reuseFailAlloc_633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_633_, 0, v_a_627_);
v___x_632_ = v_reuseFailAlloc_633_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
return v___x_632_;
}
}
}
}
else
{
lean_dec(v_a_615_);
lean_dec(v_a_613_);
return v___x_614_;
}
}
}
}
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
lean_dec_ref(v_x_606_);
v_a_637_ = lean_ctor_get(v___x_612_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_612_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_612_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_612_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_642_; 
if (v_isShared_640_ == 0)
{
v___x_642_ = v___x_639_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v_a_637_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg___boxed(lean_object* v_x_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg(v_x_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0(lean_object* v_00_u03b1_652_, lean_object* v_x_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg(v_x_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___boxed(lean_object* v_00_u03b1_660_, lean_object* v_x_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_){
_start:
{
lean_object* v_res_667_; 
v_res_667_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0(v_00_u03b1_660_, v_x_661_, v___y_662_, v___y_663_, v___y_664_, v___y_665_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0(lean_object* v___x_668_, lean_object* v___x_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = l_Lean_Meta_mkAppM(v___x_668_, v___x_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v_a_676_; lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_684_; 
v_a_676_ = lean_ctor_get(v___x_675_, 0);
v_isSharedCheck_684_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_684_ == 0)
{
v___x_678_ = v___x_675_;
v_isShared_679_ = v_isSharedCheck_684_;
goto v_resetjp_677_;
}
else
{
lean_inc(v_a_676_);
lean_dec(v___x_675_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_684_;
goto v_resetjp_677_;
}
v_resetjp_677_:
{
lean_object* v___x_680_; lean_object* v___x_682_; 
v___x_680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_680_, 0, v_a_676_);
if (v_isShared_679_ == 0)
{
lean_ctor_set(v___x_678_, 0, v___x_680_);
v___x_682_ = v___x_678_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v___x_680_);
v___x_682_ = v_reuseFailAlloc_683_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
return v___x_682_;
}
}
}
else
{
lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_692_; 
v_a_685_ = lean_ctor_get(v___x_675_, 0);
v_isSharedCheck_692_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_692_ == 0)
{
v___x_687_ = v___x_675_;
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_675_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_690_; 
if (v_isShared_688_ == 0)
{
v___x_690_ = v___x_687_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_a_685_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0___boxed(lean_object* v___x_693_, lean_object* v___x_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0(v___x_693_, v___x_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
return v_res_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1(lean_object* v_____r_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_709_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___closed__0));
v___x_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1___boxed(lean_object* v_____r_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_){
_start:
{
lean_object* v_res_717_; 
v_res_717_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1(v_____r_711_, v___y_712_, v___y_713_, v___y_714_, v___y_715_);
lean_dec(v___y_715_);
lean_dec_ref(v___y_714_);
lean_dec(v___y_713_);
lean_dec_ref(v___y_712_);
return v_res_717_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0(void){
_start:
{
lean_object* v___x_718_; double v___x_719_; 
v___x_718_ = lean_unsigned_to_nat(0u);
v___x_719_ = lean_float_of_nat(v___x_718_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(lean_object* v_cls_723_, lean_object* v_msg_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_){
_start:
{
lean_object* v_ref_730_; lean_object* v___x_731_; lean_object* v_a_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_776_; 
v_ref_730_ = lean_ctor_get(v___y_727_, 5);
v___x_731_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(v_msg_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_);
v_a_732_ = lean_ctor_get(v___x_731_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_731_);
if (v_isSharedCheck_776_ == 0)
{
v___x_734_ = v___x_731_;
v_isShared_735_ = v_isSharedCheck_776_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_a_732_);
lean_dec(v___x_731_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_776_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_736_; lean_object* v_traceState_737_; lean_object* v_env_738_; lean_object* v_nextMacroScope_739_; lean_object* v_ngen_740_; lean_object* v_auxDeclNGen_741_; lean_object* v_cache_742_; lean_object* v_messages_743_; lean_object* v_infoState_744_; lean_object* v_snapshotTasks_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_775_; 
v___x_736_ = lean_st_ref_take(v___y_728_);
v_traceState_737_ = lean_ctor_get(v___x_736_, 4);
v_env_738_ = lean_ctor_get(v___x_736_, 0);
v_nextMacroScope_739_ = lean_ctor_get(v___x_736_, 1);
v_ngen_740_ = lean_ctor_get(v___x_736_, 2);
v_auxDeclNGen_741_ = lean_ctor_get(v___x_736_, 3);
v_cache_742_ = lean_ctor_get(v___x_736_, 5);
v_messages_743_ = lean_ctor_get(v___x_736_, 6);
v_infoState_744_ = lean_ctor_get(v___x_736_, 7);
v_snapshotTasks_745_ = lean_ctor_get(v___x_736_, 8);
v_isSharedCheck_775_ = !lean_is_exclusive(v___x_736_);
if (v_isSharedCheck_775_ == 0)
{
v___x_747_ = v___x_736_;
v_isShared_748_ = v_isSharedCheck_775_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_snapshotTasks_745_);
lean_inc(v_infoState_744_);
lean_inc(v_messages_743_);
lean_inc(v_cache_742_);
lean_inc(v_traceState_737_);
lean_inc(v_auxDeclNGen_741_);
lean_inc(v_ngen_740_);
lean_inc(v_nextMacroScope_739_);
lean_inc(v_env_738_);
lean_dec(v___x_736_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_775_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
uint64_t v_tid_749_; lean_object* v_traces_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_774_; 
v_tid_749_ = lean_ctor_get_uint64(v_traceState_737_, sizeof(void*)*1);
v_traces_750_ = lean_ctor_get(v_traceState_737_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v_traceState_737_);
if (v_isSharedCheck_774_ == 0)
{
v___x_752_ = v_traceState_737_;
v_isShared_753_ = v_isSharedCheck_774_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_traces_750_);
lean_dec(v_traceState_737_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_774_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_754_; double v___x_755_; uint8_t v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_764_; 
v___x_754_ = lean_box(0);
v___x_755_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0);
v___x_756_ = 0;
v___x_757_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1));
v___x_758_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_758_, 0, v_cls_723_);
lean_ctor_set(v___x_758_, 1, v___x_754_);
lean_ctor_set(v___x_758_, 2, v___x_757_);
lean_ctor_set_float(v___x_758_, sizeof(void*)*3, v___x_755_);
lean_ctor_set_float(v___x_758_, sizeof(void*)*3 + 8, v___x_755_);
lean_ctor_set_uint8(v___x_758_, sizeof(void*)*3 + 16, v___x_756_);
v___x_759_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__2));
v___x_760_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_760_, 0, v___x_758_);
lean_ctor_set(v___x_760_, 1, v_a_732_);
lean_ctor_set(v___x_760_, 2, v___x_759_);
lean_inc(v_ref_730_);
v___x_761_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_761_, 0, v_ref_730_);
lean_ctor_set(v___x_761_, 1, v___x_760_);
v___x_762_ = l_Lean_PersistentArray_push___redArg(v_traces_750_, v___x_761_);
if (v_isShared_753_ == 0)
{
lean_ctor_set(v___x_752_, 0, v___x_762_);
v___x_764_ = v___x_752_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v___x_762_);
lean_ctor_set_uint64(v_reuseFailAlloc_773_, sizeof(void*)*1, v_tid_749_);
v___x_764_ = v_reuseFailAlloc_773_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
lean_object* v___x_766_; 
if (v_isShared_748_ == 0)
{
lean_ctor_set(v___x_747_, 4, v___x_764_);
v___x_766_ = v___x_747_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v_env_738_);
lean_ctor_set(v_reuseFailAlloc_772_, 1, v_nextMacroScope_739_);
lean_ctor_set(v_reuseFailAlloc_772_, 2, v_ngen_740_);
lean_ctor_set(v_reuseFailAlloc_772_, 3, v_auxDeclNGen_741_);
lean_ctor_set(v_reuseFailAlloc_772_, 4, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_772_, 5, v_cache_742_);
lean_ctor_set(v_reuseFailAlloc_772_, 6, v_messages_743_);
lean_ctor_set(v_reuseFailAlloc_772_, 7, v_infoState_744_);
lean_ctor_set(v_reuseFailAlloc_772_, 8, v_snapshotTasks_745_);
v___x_766_ = v_reuseFailAlloc_772_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_770_; 
v___x_767_ = lean_st_ref_set(v___y_728_, v___x_766_);
v___x_768_ = lean_box(0);
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 0, v___x_768_);
v___x_770_ = v___x_734_;
goto v_reusejp_769_;
}
else
{
lean_object* v_reuseFailAlloc_771_; 
v_reuseFailAlloc_771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_771_, 0, v___x_768_);
v___x_770_ = v_reuseFailAlloc_771_;
goto v_reusejp_769_;
}
v_reusejp_769_:
{
return v___x_770_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___boxed(lean_object* v_cls_777_, lean_object* v_msg_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(v_cls_777_, v_msg_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
lean_dec(v___y_782_);
lean_dec_ref(v___y_781_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
return v_res_784_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6(void){
_start:
{
lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; 
v___x_797_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__5));
v___x_799_ = l_Lean_Name_append(v___x_798_, v___x_797_);
return v___x_799_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8(void){
_start:
{
lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_801_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__7));
v___x_802_ = l_Lean_stringToMessageData(v___x_801_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f(lean_object* v_p_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_){
_start:
{
lean_object* v___y_810_; lean_object* v_fst_820_; lean_object* v_snd_821_; lean_object* v___x_823_; uint8_t v_isShared_824_; uint8_t v_isSharedCheck_862_; 
v_fst_820_ = lean_ctor_get(v_p_803_, 0);
v_snd_821_ = lean_ctor_get(v_p_803_, 1);
v_isSharedCheck_862_ = !lean_is_exclusive(v_p_803_);
if (v_isSharedCheck_862_ == 0)
{
v___x_823_ = v_p_803_;
v_isShared_824_ = v_isSharedCheck_862_;
goto v_resetjp_822_;
}
else
{
lean_inc(v_snd_821_);
lean_inc(v_fst_820_);
lean_dec(v_p_803_);
v___x_823_ = lean_box(0);
v_isShared_824_ = v_isSharedCheck_862_;
goto v_resetjp_822_;
}
v___jp_809_:
{
lean_object* v_a_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_819_; 
v_a_811_ = lean_ctor_get(v___y_810_, 0);
v_isSharedCheck_819_ = !lean_is_exclusive(v___y_810_);
if (v_isSharedCheck_819_ == 0)
{
v___x_813_ = v___y_810_;
v_isShared_814_ = v_isSharedCheck_819_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_a_811_);
lean_dec(v___y_810_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_819_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v_a_815_; lean_object* v___x_817_; 
v_a_815_ = lean_ctor_get(v_a_811_, 0);
lean_inc(v_a_815_);
lean_dec(v_a_811_);
if (v_isShared_814_ == 0)
{
lean_ctor_set(v___x_813_, 0, v_a_815_);
v___x_817_ = v___x_813_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v_a_815_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
}
}
}
v_resetjp_822_:
{
lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___f_830_; lean_object* v___x_831_; 
v___x_825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__1));
v___x_826_ = lean_unsigned_to_nat(2u);
v___x_827_ = lean_mk_empty_array_with_capacity(v___x_826_);
v___x_828_ = lean_array_push(v___x_827_, v_snd_821_);
v___x_829_ = lean_array_push(v___x_828_, v_fst_820_);
v___f_830_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__0___boxed), 7, 2);
lean_closure_set(v___f_830_, 0, v___x_825_);
lean_closure_set(v___f_830_, 1, v___x_829_);
v___x_831_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__0___redArg(v___f_830_, v_a_804_, v_a_805_, v_a_806_, v_a_807_);
if (lean_obj_tag(v___x_831_) == 0)
{
lean_del_object(v___x_823_);
return v___x_831_;
}
else
{
lean_object* v_a_832_; uint8_t v___y_837_; uint8_t v___x_860_; 
v_a_832_ = lean_ctor_get(v___x_831_, 0);
lean_inc(v_a_832_);
v___x_860_ = l_Lean_Exception_isInterrupt(v_a_832_);
if (v___x_860_ == 0)
{
uint8_t v___x_861_; 
lean_inc(v_a_832_);
v___x_861_ = l_Lean_Exception_isRuntime(v_a_832_);
v___y_837_ = v___x_861_;
goto v___jp_836_;
}
else
{
v___y_837_ = v___x_860_;
goto v___jp_836_;
}
v___jp_833_:
{
lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_834_ = lean_box(0);
v___x_835_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1(v___x_834_, v_a_804_, v_a_805_, v_a_806_, v_a_807_);
v___y_810_ = v___x_835_;
goto v___jp_809_;
}
v___jp_836_:
{
if (v___y_837_ == 0)
{
lean_object* v_options_838_; uint8_t v_hasTrace_839_; 
lean_dec_ref_known(v___x_831_, 1);
v_options_838_ = lean_ctor_get(v_a_806_, 2);
v_hasTrace_839_ = lean_ctor_get_uint8(v_options_838_, sizeof(void*)*1);
if (v_hasTrace_839_ == 0)
{
lean_dec(v_a_832_);
lean_del_object(v___x_823_);
goto v___jp_833_;
}
else
{
lean_object* v_inheritedTraceOptions_840_; lean_object* v___x_841_; lean_object* v___x_842_; uint8_t v___x_843_; 
v_inheritedTraceOptions_840_ = lean_ctor_get(v_a_806_, 13);
v___x_841_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_842_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6);
v___x_843_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_840_, v_options_838_, v___x_842_);
if (v___x_843_ == 0)
{
lean_dec(v_a_832_);
lean_del_object(v___x_823_);
goto v___jp_833_;
}
else
{
lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_844_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__8);
v___x_845_ = l_Lean_Exception_toMessageData(v_a_832_);
if (v_isShared_824_ == 0)
{
lean_ctor_set_tag(v___x_823_, 7);
lean_ctor_set(v___x_823_, 1, v___x_845_);
lean_ctor_set(v___x_823_, 0, v___x_844_);
v___x_847_ = v___x_823_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v___x_844_);
lean_ctor_set(v_reuseFailAlloc_859_, 1, v___x_845_);
v___x_847_ = v_reuseFailAlloc_859_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
lean_object* v___x_848_; 
v___x_848_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(v___x_841_, v___x_847_, v_a_804_, v_a_805_, v_a_806_, v_a_807_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v_a_849_; lean_object* v___x_850_; 
v_a_849_ = lean_ctor_get(v___x_848_, 0);
lean_inc(v_a_849_);
lean_dec_ref_known(v___x_848_, 1);
v___x_850_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___lam__1(v_a_849_, v_a_804_, v_a_805_, v_a_806_, v_a_807_);
v___y_810_ = v___x_850_;
goto v___jp_809_;
}
else
{
lean_object* v_a_851_; lean_object* v___x_853_; uint8_t v_isShared_854_; uint8_t v_isSharedCheck_858_; 
v_a_851_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_858_ == 0)
{
v___x_853_ = v___x_848_;
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
else
{
lean_inc(v_a_851_);
lean_dec(v___x_848_);
v___x_853_ = lean_box(0);
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
v_resetjp_852_:
{
lean_object* v___x_856_; 
if (v_isShared_854_ == 0)
{
v___x_856_ = v___x_853_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_a_851_);
v___x_856_ = v_reuseFailAlloc_857_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
return v___x_856_;
}
}
}
}
}
}
}
else
{
lean_dec(v_a_832_);
lean_del_object(v___x_823_);
return v___x_831_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___boxed(lean_object* v_p_863_, lean_object* v_a_864_, lean_object* v_a_865_, lean_object* v_a_866_, lean_object* v_a_867_, lean_object* v_a_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f(v_p_863_, v_a_864_, v_a_865_, v_a_866_, v_a_867_);
lean_dec(v_a_867_);
lean_dec_ref(v_a_866_);
lean_dec(v_a_865_);
lean_dec_ref(v_a_864_);
return v_res_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mk__natCast__nonneg__prf(lean_object* v_p_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f(v_p_870_, v_a_871_, v_a_872_, v_a_873_, v_a_874_);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mk__natCast__nonneg__prf___boxed(lean_object* v_p_877_, lean_object* v_a_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_, lean_object* v_a_882_){
_start:
{
lean_object* v_res_883_; 
v_res_883_ = lp_mathlib_Mathlib_Tactic_Linarith_mk__natCast__nonneg__prf(v_p_877_, v_a_878_, v_a_879_, v_a_880_, v_a_881_);
lean_dec(v_a_881_);
lean_dec_ref(v_a_880_);
lean_dec(v_a_879_);
lean_dec_ref(v_a_878_);
return v_res_883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg(lean_object* v_k_884_, uint8_t v_allowLevelAssignments_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_){
_start:
{
lean_object* v___x_891_; 
v___x_891_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_885_, v_k_884_, v___y_886_, v___y_887_, v___y_888_, v___y_889_);
if (lean_obj_tag(v___x_891_) == 0)
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_899_; 
v_a_892_ = lean_ctor_get(v___x_891_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_891_);
if (v_isSharedCheck_899_ == 0)
{
v___x_894_ = v___x_891_;
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_891_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v_a_892_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
v_a_900_ = lean_ctor_get(v___x_891_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_891_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_891_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_891_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg___boxed(lean_object* v_k_908_, lean_object* v_allowLevelAssignments_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_915_; lean_object* v_res_916_; 
v_allowLevelAssignments_boxed_915_ = lean_unbox(v_allowLevelAssignments_909_);
v_res_916_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg(v_k_908_, v_allowLevelAssignments_boxed_915_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8(lean_object* v_00_u03b1_917_, lean_object* v_k_918_, uint8_t v_allowLevelAssignments_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
lean_object* v___x_925_; 
v___x_925_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg(v_k_918_, v_allowLevelAssignments_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_);
return v___x_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___boxed(lean_object* v_00_u03b1_926_, lean_object* v_k_927_, lean_object* v_allowLevelAssignments_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_934_; lean_object* v_res_935_; 
v_allowLevelAssignments_boxed_934_ = lean_unbox(v_allowLevelAssignments_928_);
v_res_935_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8(v_00_u03b1_926_, v_k_927_, v_allowLevelAssignments_boxed_934_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec(v___y_930_);
lean_dec_ref(v___y_929_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0(lean_object* v_e_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_){
_start:
{
lean_object* v___x_942_; uint8_t v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_942_ = lean_box(0);
v___x_943_ = 1;
v___x_944_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_944_, 0, v_e_936_);
lean_ctor_set(v___x_944_, 1, v___x_942_);
lean_ctor_set_uint8(v___x_944_, sizeof(void*)*2, v___x_943_);
v___x_945_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0___boxed(lean_object* v_e_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__0(v_e_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6(lean_object* v_init_953_, lean_object* v_x_954_){
_start:
{
if (lean_obj_tag(v_x_954_) == 0)
{
lean_object* v_k_955_; lean_object* v_l_956_; lean_object* v_r_957_; lean_object* v___x_958_; lean_object* v___x_959_; 
v_k_955_ = lean_ctor_get(v_x_954_, 1);
v_l_956_ = lean_ctor_get(v_x_954_, 3);
v_r_957_ = lean_ctor_get(v_x_954_, 4);
v___x_958_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6(v_init_953_, v_r_957_);
lean_inc(v_k_955_);
v___x_959_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_959_, 0, v_k_955_);
lean_ctor_set(v___x_959_, 1, v___x_958_);
v_init_953_ = v___x_959_;
v_x_954_ = v_l_956_;
goto _start;
}
else
{
return v_init_953_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6___boxed(lean_object* v_init_961_, lean_object* v_x_962_){
_start:
{
lean_object* v_res_963_; 
v_res_963_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6(v_init_961_, v_x_962_);
lean_dec(v_x_962_);
return v_res_963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0(lean_object* v_x_964_, lean_object* v_x_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_){
_start:
{
if (lean_obj_tag(v_x_964_) == 0)
{
lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_973_ = l_List_reverse___redArg(v_x_965_);
v___x_974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
return v___x_974_;
}
else
{
lean_object* v_head_975_; lean_object* v_tail_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_1016_; 
v_head_975_ = lean_ctor_get(v_x_964_, 0);
v_tail_976_ = lean_ctor_get(v_x_964_, 1);
v_isSharedCheck_1016_ = !lean_is_exclusive(v_x_964_);
if (v_isSharedCheck_1016_ == 0)
{
v___x_978_ = v_x_964_;
v_isShared_979_ = v_isSharedCheck_1016_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_tail_976_);
lean_inc(v_head_975_);
lean_dec(v_x_964_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_1016_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v_fst_980_; lean_object* v_snd_981_; lean_object* v___x_982_; 
v_fst_980_ = lean_ctor_get(v_head_975_, 0);
lean_inc(v_fst_980_);
v_snd_981_ = lean_ctor_get(v_head_975_, 1);
lean_inc(v_snd_981_);
lean_dec(v_head_975_);
v___x_982_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v_fst_980_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_);
if (lean_obj_tag(v___x_982_) == 0)
{
lean_object* v_a_983_; lean_object* v___x_984_; 
v_a_983_ = lean_ctor_get(v___x_982_, 0);
lean_inc(v_a_983_);
lean_dec_ref_known(v___x_982_, 1);
v___x_984_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v_snd_981_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_);
if (lean_obj_tag(v___x_984_) == 0)
{
lean_object* v_a_985_; lean_object* v_fst_986_; lean_object* v_fst_987_; lean_object* v___x_989_; uint8_t v_isShared_990_; uint8_t v_isSharedCheck_998_; 
v_a_985_ = lean_ctor_get(v___x_984_, 0);
lean_inc(v_a_985_);
lean_dec_ref_known(v___x_984_, 1);
v_fst_986_ = lean_ctor_get(v_a_983_, 0);
lean_inc(v_fst_986_);
lean_dec(v_a_983_);
v_fst_987_ = lean_ctor_get(v_a_985_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v_a_985_);
if (v_isSharedCheck_998_ == 0)
{
lean_object* v_unused_999_; 
v_unused_999_ = lean_ctor_get(v_a_985_, 1);
lean_dec(v_unused_999_);
v___x_989_ = v_a_985_;
v_isShared_990_ = v_isSharedCheck_998_;
goto v_resetjp_988_;
}
else
{
lean_inc(v_fst_987_);
lean_dec(v_a_985_);
v___x_989_ = lean_box(0);
v_isShared_990_ = v_isSharedCheck_998_;
goto v_resetjp_988_;
}
v_resetjp_988_:
{
lean_object* v___x_992_; 
if (v_isShared_990_ == 0)
{
lean_ctor_set(v___x_989_, 1, v_fst_987_);
lean_ctor_set(v___x_989_, 0, v_fst_986_);
v___x_992_ = v___x_989_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v_fst_986_);
lean_ctor_set(v_reuseFailAlloc_997_, 1, v_fst_987_);
v___x_992_ = v_reuseFailAlloc_997_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
lean_object* v___x_994_; 
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 1, v_x_965_);
lean_ctor_set(v___x_978_, 0, v___x_992_);
v___x_994_ = v___x_978_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_996_; 
v_reuseFailAlloc_996_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_996_, 0, v___x_992_);
lean_ctor_set(v_reuseFailAlloc_996_, 1, v_x_965_);
v___x_994_ = v_reuseFailAlloc_996_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
v_x_964_ = v_tail_976_;
v_x_965_ = v___x_994_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1007_; 
lean_dec(v_a_983_);
lean_del_object(v___x_978_);
lean_dec(v_tail_976_);
lean_dec(v_x_965_);
v_a_1000_ = lean_ctor_get(v___x_984_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_984_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_1002_ = v___x_984_;
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_a_1000_);
lean_dec(v___x_984_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v___x_1005_; 
if (v_isShared_1003_ == 0)
{
v___x_1005_ = v___x_1002_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_1000_);
v___x_1005_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
return v___x_1005_;
}
}
}
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
lean_dec(v_snd_981_);
lean_del_object(v___x_978_);
lean_dec(v_tail_976_);
lean_dec(v_x_965_);
v_a_1008_ = lean_ctor_get(v___x_982_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_982_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_982_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_982_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1008_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0___boxed(lean_object* v_x_1017_, lean_object* v_x_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_){
_start:
{
lean_object* v_res_1026_; 
v_res_1026_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0(v_x_1017_, v_x_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_, v___y_1024_);
lean_dec(v___y_1024_);
lean_dec_ref(v___y_1023_);
lean_dec(v___y_1022_);
lean_dec_ref(v___y_1021_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
return v_res_1026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(lean_object* v_k_1027_, lean_object* v_v_1028_, lean_object* v_t_1029_){
_start:
{
lean_object* v___y_1031_; lean_object* v___y_1032_; lean_object* v___y_1033_; lean_object* v___y_1034_; lean_object* v___y_1035_; lean_object* v___y_1036_; lean_object* v___y_1037_; lean_object* v___y_1038_; lean_object* v___y_1039_; lean_object* v___y_1040_; 
if (lean_obj_tag(v_t_1029_) == 0)
{
lean_object* v_size_1044_; lean_object* v_k_1045_; lean_object* v_v_1046_; lean_object* v_l_1047_; lean_object* v_r_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1310_; 
v_size_1044_ = lean_ctor_get(v_t_1029_, 0);
v_k_1045_ = lean_ctor_get(v_t_1029_, 1);
v_v_1046_ = lean_ctor_get(v_t_1029_, 2);
v_l_1047_ = lean_ctor_get(v_t_1029_, 3);
v_r_1048_ = lean_ctor_get(v_t_1029_, 4);
v_isSharedCheck_1310_ = !lean_is_exclusive(v_t_1029_);
if (v_isSharedCheck_1310_ == 0)
{
v___x_1050_ = v_t_1029_;
v_isShared_1051_ = v_isSharedCheck_1310_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_r_1048_);
lean_inc(v_l_1047_);
lean_inc(v_v_1046_);
lean_inc(v_k_1045_);
lean_inc(v_size_1044_);
lean_dec(v_t_1029_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1310_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v___y_1056_; lean_object* v___y_1057_; lean_object* v___y_1058_; lean_object* v___y_1059_; lean_object* v___y_1060_; lean_object* v___y_1061_; lean_object* v___y_1062_; lean_object* v___y_1063_; lean_object* v___y_1064_; lean_object* v___y_1172_; lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v___y_1178_; lean_object* v___y_1183_; lean_object* v___y_1184_; lean_object* v___y_1185_; lean_object* v___y_1186_; lean_object* v___y_1187_; lean_object* v___y_1188_; lean_object* v___y_1189_; lean_object* v___y_1190_; lean_object* v___y_1191_; lean_object* v___y_1192_; lean_object* v___y_1193_; lean_object* v___y_1194_; lean_object* v_fst_1301_; lean_object* v_snd_1302_; lean_object* v_fst_1303_; lean_object* v_snd_1304_; uint8_t v___x_1305_; 
v_fst_1301_ = lean_ctor_get(v_k_1027_, 0);
v_snd_1302_ = lean_ctor_get(v_k_1027_, 1);
v_fst_1303_ = lean_ctor_get(v_k_1045_, 0);
v_snd_1304_ = lean_ctor_get(v_k_1045_, 1);
v___x_1305_ = lean_nat_dec_lt(v_fst_1301_, v_fst_1303_);
if (v___x_1305_ == 0)
{
uint8_t v___x_1306_; 
v___x_1306_ = lean_nat_dec_eq(v_fst_1301_, v_fst_1303_);
if (v___x_1306_ == 0)
{
lean_dec(v_size_1044_);
goto v___jp_1072_;
}
else
{
uint8_t v___x_1307_; 
v___x_1307_ = lean_nat_dec_lt(v_snd_1302_, v_snd_1304_);
if (v___x_1307_ == 0)
{
uint8_t v___x_1308_; 
v___x_1308_ = lean_nat_dec_eq(v_snd_1302_, v_snd_1304_);
if (v___x_1308_ == 0)
{
lean_dec(v_size_1044_);
goto v___jp_1072_;
}
else
{
lean_object* v___x_1309_; 
lean_del_object(v___x_1050_);
lean_dec(v_v_1046_);
lean_dec(v_k_1045_);
v___x_1309_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1309_, 0, v_size_1044_);
lean_ctor_set(v___x_1309_, 1, v_k_1027_);
lean_ctor_set(v___x_1309_, 2, v_v_1028_);
lean_ctor_set(v___x_1309_, 3, v_l_1047_);
lean_ctor_set(v___x_1309_, 4, v_r_1048_);
return v___x_1309_;
}
}
else
{
lean_del_object(v___x_1050_);
lean_dec(v_size_1044_);
goto v___jp_1200_;
}
}
}
else
{
lean_del_object(v___x_1050_);
lean_dec(v_size_1044_);
goto v___jp_1200_;
}
v___jp_1052_:
{
lean_object* v___x_1065_; lean_object* v___x_1067_; 
v___x_1065_ = lean_nat_add(v___y_1063_, v___y_1064_);
lean_dec(v___y_1064_);
lean_dec(v___y_1063_);
if (v_isShared_1051_ == 0)
{
lean_ctor_set(v___x_1050_, 4, v___y_1062_);
lean_ctor_set(v___x_1050_, 0, v___x_1065_);
v___x_1067_ = v___x_1050_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v___x_1065_);
lean_ctor_set(v_reuseFailAlloc_1071_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1071_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1071_, 3, v_l_1047_);
lean_ctor_set(v_reuseFailAlloc_1071_, 4, v___y_1062_);
v___x_1067_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
lean_object* v___x_1068_; 
v___x_1068_ = lean_nat_add(v___y_1061_, v___y_1055_);
lean_dec(v___y_1055_);
if (lean_obj_tag(v___y_1058_) == 0)
{
lean_object* v_size_1069_; 
v_size_1069_ = lean_ctor_get(v___y_1058_, 0);
lean_inc(v_size_1069_);
v___y_1031_ = v___y_1054_;
v___y_1032_ = v___y_1053_;
v___y_1033_ = v___x_1067_;
v___y_1034_ = v___y_1057_;
v___y_1035_ = v___y_1056_;
v___y_1036_ = v___y_1059_;
v___y_1037_ = v___y_1058_;
v___y_1038_ = v___y_1060_;
v___y_1039_ = v___x_1068_;
v___y_1040_ = v_size_1069_;
goto v___jp_1030_;
}
else
{
lean_object* v___x_1070_; 
v___x_1070_ = lean_unsigned_to_nat(0u);
v___y_1031_ = v___y_1054_;
v___y_1032_ = v___y_1053_;
v___y_1033_ = v___x_1067_;
v___y_1034_ = v___y_1057_;
v___y_1035_ = v___y_1056_;
v___y_1036_ = v___y_1059_;
v___y_1037_ = v___y_1058_;
v___y_1038_ = v___y_1060_;
v___y_1039_ = v___x_1068_;
v___y_1040_ = v___x_1070_;
goto v___jp_1030_;
}
}
}
v___jp_1072_:
{
lean_object* v_impl_1073_; lean_object* v___x_1074_; 
v_impl_1073_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(v_k_1027_, v_v_1028_, v_r_1048_);
v___x_1074_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_1047_) == 0)
{
lean_object* v_size_1075_; lean_object* v_size_1076_; lean_object* v_k_1077_; lean_object* v_v_1078_; lean_object* v_l_1079_; lean_object* v_r_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; uint8_t v___x_1083_; 
v_size_1075_ = lean_ctor_get(v_l_1047_, 0);
v_size_1076_ = lean_ctor_get(v_impl_1073_, 0);
lean_inc(v_size_1076_);
v_k_1077_ = lean_ctor_get(v_impl_1073_, 1);
lean_inc(v_k_1077_);
v_v_1078_ = lean_ctor_get(v_impl_1073_, 2);
lean_inc(v_v_1078_);
v_l_1079_ = lean_ctor_get(v_impl_1073_, 3);
lean_inc(v_l_1079_);
v_r_1080_ = lean_ctor_get(v_impl_1073_, 4);
lean_inc(v_r_1080_);
v___x_1081_ = lean_unsigned_to_nat(3u);
v___x_1082_ = lean_nat_mul(v___x_1081_, v_size_1075_);
v___x_1083_ = lean_nat_dec_lt(v___x_1082_, v_size_1076_);
lean_dec(v___x_1082_);
if (v___x_1083_ == 0)
{
lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; 
lean_dec(v_r_1080_);
lean_dec(v_l_1079_);
lean_dec(v_v_1078_);
lean_dec(v_k_1077_);
lean_del_object(v___x_1050_);
v___x_1084_ = lean_nat_add(v___x_1074_, v_size_1075_);
v___x_1085_ = lean_nat_add(v___x_1084_, v_size_1076_);
lean_dec(v_size_1076_);
lean_dec(v___x_1084_);
v___x_1086_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1086_, 0, v___x_1085_);
lean_ctor_set(v___x_1086_, 1, v_k_1045_);
lean_ctor_set(v___x_1086_, 2, v_v_1046_);
lean_ctor_set(v___x_1086_, 3, v_l_1047_);
lean_ctor_set(v___x_1086_, 4, v_impl_1073_);
return v___x_1086_;
}
else
{
lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1121_; 
v_isSharedCheck_1121_ = !lean_is_exclusive(v_impl_1073_);
if (v_isSharedCheck_1121_ == 0)
{
lean_object* v_unused_1122_; lean_object* v_unused_1123_; lean_object* v_unused_1124_; lean_object* v_unused_1125_; lean_object* v_unused_1126_; 
v_unused_1122_ = lean_ctor_get(v_impl_1073_, 4);
lean_dec(v_unused_1122_);
v_unused_1123_ = lean_ctor_get(v_impl_1073_, 3);
lean_dec(v_unused_1123_);
v_unused_1124_ = lean_ctor_get(v_impl_1073_, 2);
lean_dec(v_unused_1124_);
v_unused_1125_ = lean_ctor_get(v_impl_1073_, 1);
lean_dec(v_unused_1125_);
v_unused_1126_ = lean_ctor_get(v_impl_1073_, 0);
lean_dec(v_unused_1126_);
v___x_1088_ = v_impl_1073_;
v_isShared_1089_ = v_isSharedCheck_1121_;
goto v_resetjp_1087_;
}
else
{
lean_dec(v_impl_1073_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1121_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v_size_1090_; lean_object* v_k_1091_; lean_object* v_v_1092_; lean_object* v_l_1093_; lean_object* v_r_1094_; lean_object* v_size_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; uint8_t v___x_1098_; 
v_size_1090_ = lean_ctor_get(v_l_1079_, 0);
v_k_1091_ = lean_ctor_get(v_l_1079_, 1);
v_v_1092_ = lean_ctor_get(v_l_1079_, 2);
v_l_1093_ = lean_ctor_get(v_l_1079_, 3);
v_r_1094_ = lean_ctor_get(v_l_1079_, 4);
v_size_1095_ = lean_ctor_get(v_r_1080_, 0);
v___x_1096_ = lean_unsigned_to_nat(2u);
v___x_1097_ = lean_nat_mul(v___x_1096_, v_size_1095_);
v___x_1098_ = lean_nat_dec_lt(v_size_1090_, v___x_1097_);
lean_dec(v___x_1097_);
if (v___x_1098_ == 0)
{
lean_object* v___x_1099_; lean_object* v___x_1100_; 
lean_inc(v_size_1095_);
lean_inc(v_r_1094_);
lean_inc(v_l_1093_);
lean_inc(v_v_1092_);
lean_inc(v_k_1091_);
lean_del_object(v___x_1088_);
lean_dec(v_l_1079_);
v___x_1099_ = lean_nat_add(v___x_1074_, v_size_1075_);
v___x_1100_ = lean_nat_add(v___x_1099_, v_size_1076_);
lean_dec(v_size_1076_);
if (lean_obj_tag(v_l_1093_) == 0)
{
lean_object* v_size_1101_; 
v_size_1101_ = lean_ctor_get(v_l_1093_, 0);
lean_inc(v_size_1101_);
v___y_1053_ = v_k_1077_;
v___y_1054_ = v___x_1100_;
v___y_1055_ = v_size_1095_;
v___y_1056_ = v_v_1078_;
v___y_1057_ = v_v_1092_;
v___y_1058_ = v_r_1094_;
v___y_1059_ = v_r_1080_;
v___y_1060_ = v_k_1091_;
v___y_1061_ = v___x_1074_;
v___y_1062_ = v_l_1093_;
v___y_1063_ = v___x_1099_;
v___y_1064_ = v_size_1101_;
goto v___jp_1052_;
}
else
{
lean_object* v___x_1102_; 
v___x_1102_ = lean_unsigned_to_nat(0u);
v___y_1053_ = v_k_1077_;
v___y_1054_ = v___x_1100_;
v___y_1055_ = v_size_1095_;
v___y_1056_ = v_v_1078_;
v___y_1057_ = v_v_1092_;
v___y_1058_ = v_r_1094_;
v___y_1059_ = v_r_1080_;
v___y_1060_ = v_k_1091_;
v___y_1061_ = v___x_1074_;
v___y_1062_ = v_l_1093_;
v___y_1063_ = v___x_1099_;
v___y_1064_ = v___x_1102_;
goto v___jp_1052_;
}
}
else
{
lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1107_; 
lean_del_object(v___x_1050_);
v___x_1103_ = lean_nat_add(v___x_1074_, v_size_1075_);
v___x_1104_ = lean_nat_add(v___x_1103_, v_size_1076_);
lean_dec(v_size_1076_);
v___x_1105_ = lean_nat_add(v___x_1103_, v_size_1090_);
lean_dec(v___x_1103_);
lean_inc_ref(v_l_1047_);
if (v_isShared_1089_ == 0)
{
lean_ctor_set(v___x_1088_, 4, v_l_1079_);
lean_ctor_set(v___x_1088_, 3, v_l_1047_);
lean_ctor_set(v___x_1088_, 2, v_v_1046_);
lean_ctor_set(v___x_1088_, 1, v_k_1045_);
lean_ctor_set(v___x_1088_, 0, v___x_1105_);
v___x_1107_ = v___x_1088_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v___x_1105_);
lean_ctor_set(v_reuseFailAlloc_1120_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1120_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1120_, 3, v_l_1047_);
lean_ctor_set(v_reuseFailAlloc_1120_, 4, v_l_1079_);
v___x_1107_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1114_; 
v_isSharedCheck_1114_ = !lean_is_exclusive(v_l_1047_);
if (v_isSharedCheck_1114_ == 0)
{
lean_object* v_unused_1115_; lean_object* v_unused_1116_; lean_object* v_unused_1117_; lean_object* v_unused_1118_; lean_object* v_unused_1119_; 
v_unused_1115_ = lean_ctor_get(v_l_1047_, 4);
lean_dec(v_unused_1115_);
v_unused_1116_ = lean_ctor_get(v_l_1047_, 3);
lean_dec(v_unused_1116_);
v_unused_1117_ = lean_ctor_get(v_l_1047_, 2);
lean_dec(v_unused_1117_);
v_unused_1118_ = lean_ctor_get(v_l_1047_, 1);
lean_dec(v_unused_1118_);
v_unused_1119_ = lean_ctor_get(v_l_1047_, 0);
lean_dec(v_unused_1119_);
v___x_1109_ = v_l_1047_;
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
else
{
lean_dec(v_l_1047_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v___x_1112_; 
if (v_isShared_1110_ == 0)
{
lean_ctor_set(v___x_1109_, 4, v_r_1080_);
lean_ctor_set(v___x_1109_, 3, v___x_1107_);
lean_ctor_set(v___x_1109_, 2, v_v_1078_);
lean_ctor_set(v___x_1109_, 1, v_k_1077_);
lean_ctor_set(v___x_1109_, 0, v___x_1104_);
v___x_1112_ = v___x_1109_;
goto v_reusejp_1111_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v___x_1104_);
lean_ctor_set(v_reuseFailAlloc_1113_, 1, v_k_1077_);
lean_ctor_set(v_reuseFailAlloc_1113_, 2, v_v_1078_);
lean_ctor_set(v_reuseFailAlloc_1113_, 3, v___x_1107_);
lean_ctor_set(v_reuseFailAlloc_1113_, 4, v_r_1080_);
v___x_1112_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1111_;
}
v_reusejp_1111_:
{
return v___x_1112_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_1127_; 
lean_del_object(v___x_1050_);
v_l_1127_ = lean_ctor_get(v_impl_1073_, 3);
lean_inc(v_l_1127_);
if (lean_obj_tag(v_l_1127_) == 0)
{
lean_object* v_r_1128_; lean_object* v_k_1129_; lean_object* v_v_1130_; lean_object* v___x_1132_; uint8_t v_isShared_1133_; uint8_t v_isSharedCheck_1151_; 
v_r_1128_ = lean_ctor_get(v_impl_1073_, 4);
v_k_1129_ = lean_ctor_get(v_impl_1073_, 1);
v_v_1130_ = lean_ctor_get(v_impl_1073_, 2);
v_isSharedCheck_1151_ = !lean_is_exclusive(v_impl_1073_);
if (v_isSharedCheck_1151_ == 0)
{
lean_object* v_unused_1152_; lean_object* v_unused_1153_; 
v_unused_1152_ = lean_ctor_get(v_impl_1073_, 3);
lean_dec(v_unused_1152_);
v_unused_1153_ = lean_ctor_get(v_impl_1073_, 0);
lean_dec(v_unused_1153_);
v___x_1132_ = v_impl_1073_;
v_isShared_1133_ = v_isSharedCheck_1151_;
goto v_resetjp_1131_;
}
else
{
lean_inc(v_r_1128_);
lean_inc(v_v_1130_);
lean_inc(v_k_1129_);
lean_dec(v_impl_1073_);
v___x_1132_ = lean_box(0);
v_isShared_1133_ = v_isSharedCheck_1151_;
goto v_resetjp_1131_;
}
v_resetjp_1131_:
{
lean_object* v_k_1134_; lean_object* v_v_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1147_; 
v_k_1134_ = lean_ctor_get(v_l_1127_, 1);
v_v_1135_ = lean_ctor_get(v_l_1127_, 2);
v_isSharedCheck_1147_ = !lean_is_exclusive(v_l_1127_);
if (v_isSharedCheck_1147_ == 0)
{
lean_object* v_unused_1148_; lean_object* v_unused_1149_; lean_object* v_unused_1150_; 
v_unused_1148_ = lean_ctor_get(v_l_1127_, 4);
lean_dec(v_unused_1148_);
v_unused_1149_ = lean_ctor_get(v_l_1127_, 3);
lean_dec(v_unused_1149_);
v_unused_1150_ = lean_ctor_get(v_l_1127_, 0);
lean_dec(v_unused_1150_);
v___x_1137_ = v_l_1127_;
v_isShared_1138_ = v_isSharedCheck_1147_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_v_1135_);
lean_inc(v_k_1134_);
lean_dec(v_l_1127_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1147_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1139_; lean_object* v___x_1141_; 
v___x_1139_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_1128_, 2);
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 4, v_r_1128_);
lean_ctor_set(v___x_1137_, 3, v_r_1128_);
lean_ctor_set(v___x_1137_, 2, v_v_1046_);
lean_ctor_set(v___x_1137_, 1, v_k_1045_);
lean_ctor_set(v___x_1137_, 0, v___x_1074_);
v___x_1141_ = v___x_1137_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v___x_1074_);
lean_ctor_set(v_reuseFailAlloc_1146_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1146_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1146_, 3, v_r_1128_);
lean_ctor_set(v_reuseFailAlloc_1146_, 4, v_r_1128_);
v___x_1141_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
lean_object* v___x_1143_; 
lean_inc(v_r_1128_);
if (v_isShared_1133_ == 0)
{
lean_ctor_set(v___x_1132_, 3, v_r_1128_);
lean_ctor_set(v___x_1132_, 0, v___x_1074_);
v___x_1143_ = v___x_1132_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v___x_1074_);
lean_ctor_set(v_reuseFailAlloc_1145_, 1, v_k_1129_);
lean_ctor_set(v_reuseFailAlloc_1145_, 2, v_v_1130_);
lean_ctor_set(v_reuseFailAlloc_1145_, 3, v_r_1128_);
lean_ctor_set(v_reuseFailAlloc_1145_, 4, v_r_1128_);
v___x_1143_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
lean_object* v___x_1144_; 
v___x_1144_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1144_, 0, v___x_1139_);
lean_ctor_set(v___x_1144_, 1, v_k_1134_);
lean_ctor_set(v___x_1144_, 2, v_v_1135_);
lean_ctor_set(v___x_1144_, 3, v___x_1141_);
lean_ctor_set(v___x_1144_, 4, v___x_1143_);
return v___x_1144_;
}
}
}
}
}
else
{
lean_object* v_r_1154_; 
v_r_1154_ = lean_ctor_get(v_impl_1073_, 4);
lean_inc(v_r_1154_);
if (lean_obj_tag(v_r_1154_) == 0)
{
lean_object* v_k_1155_; lean_object* v_v_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1165_; 
v_k_1155_ = lean_ctor_get(v_impl_1073_, 1);
v_v_1156_ = lean_ctor_get(v_impl_1073_, 2);
v_isSharedCheck_1165_ = !lean_is_exclusive(v_impl_1073_);
if (v_isSharedCheck_1165_ == 0)
{
lean_object* v_unused_1166_; lean_object* v_unused_1167_; lean_object* v_unused_1168_; 
v_unused_1166_ = lean_ctor_get(v_impl_1073_, 4);
lean_dec(v_unused_1166_);
v_unused_1167_ = lean_ctor_get(v_impl_1073_, 3);
lean_dec(v_unused_1167_);
v_unused_1168_ = lean_ctor_get(v_impl_1073_, 0);
lean_dec(v_unused_1168_);
v___x_1158_ = v_impl_1073_;
v_isShared_1159_ = v_isSharedCheck_1165_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_v_1156_);
lean_inc(v_k_1155_);
lean_dec(v_impl_1073_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1165_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1160_; lean_object* v___x_1162_; 
v___x_1160_ = lean_unsigned_to_nat(3u);
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 4, v_l_1127_);
lean_ctor_set(v___x_1158_, 2, v_v_1046_);
lean_ctor_set(v___x_1158_, 1, v_k_1045_);
lean_ctor_set(v___x_1158_, 0, v___x_1074_);
v___x_1162_ = v___x_1158_;
goto v_reusejp_1161_;
}
else
{
lean_object* v_reuseFailAlloc_1164_; 
v_reuseFailAlloc_1164_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1164_, 0, v___x_1074_);
lean_ctor_set(v_reuseFailAlloc_1164_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1164_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1164_, 3, v_l_1127_);
lean_ctor_set(v_reuseFailAlloc_1164_, 4, v_l_1127_);
v___x_1162_ = v_reuseFailAlloc_1164_;
goto v_reusejp_1161_;
}
v_reusejp_1161_:
{
lean_object* v___x_1163_; 
v___x_1163_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1160_);
lean_ctor_set(v___x_1163_, 1, v_k_1155_);
lean_ctor_set(v___x_1163_, 2, v_v_1156_);
lean_ctor_set(v___x_1163_, 3, v___x_1162_);
lean_ctor_set(v___x_1163_, 4, v_r_1154_);
return v___x_1163_;
}
}
}
else
{
lean_object* v___x_1169_; lean_object* v___x_1170_; 
v___x_1169_ = lean_unsigned_to_nat(2u);
v___x_1170_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1169_);
lean_ctor_set(v___x_1170_, 1, v_k_1045_);
lean_ctor_set(v___x_1170_, 2, v_v_1046_);
lean_ctor_set(v___x_1170_, 3, v_r_1154_);
lean_ctor_set(v___x_1170_, 4, v_impl_1073_);
return v___x_1170_;
}
}
}
}
v___jp_1171_:
{
lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; 
v___x_1179_ = lean_nat_add(v___y_1175_, v___y_1178_);
lean_dec(v___y_1178_);
lean_dec(v___y_1175_);
v___x_1180_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v_k_1045_);
lean_ctor_set(v___x_1180_, 2, v_v_1046_);
lean_ctor_set(v___x_1180_, 3, v___y_1174_);
lean_ctor_set(v___x_1180_, 4, v_r_1048_);
v___x_1181_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1181_, 0, v___y_1177_);
lean_ctor_set(v___x_1181_, 1, v___y_1176_);
lean_ctor_set(v___x_1181_, 2, v___y_1173_);
lean_ctor_set(v___x_1181_, 3, v___y_1172_);
lean_ctor_set(v___x_1181_, 4, v___x_1180_);
return v___x_1181_;
}
v___jp_1182_:
{
lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; 
v___x_1195_ = lean_nat_add(v___y_1189_, v___y_1194_);
lean_dec(v___y_1194_);
lean_dec(v___y_1189_);
v___x_1196_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1196_, 0, v___x_1195_);
lean_ctor_set(v___x_1196_, 1, v___y_1193_);
lean_ctor_set(v___x_1196_, 2, v___y_1185_);
lean_ctor_set(v___x_1196_, 3, v___y_1192_);
lean_ctor_set(v___x_1196_, 4, v___y_1186_);
v___x_1197_ = lean_nat_add(v___y_1191_, v___y_1190_);
lean_dec(v___y_1190_);
if (lean_obj_tag(v___y_1184_) == 0)
{
lean_object* v_size_1198_; 
v_size_1198_ = lean_ctor_get(v___y_1184_, 0);
lean_inc(v_size_1198_);
v___y_1172_ = v___x_1196_;
v___y_1173_ = v___y_1183_;
v___y_1174_ = v___y_1184_;
v___y_1175_ = v___x_1197_;
v___y_1176_ = v___y_1188_;
v___y_1177_ = v___y_1187_;
v___y_1178_ = v_size_1198_;
goto v___jp_1171_;
}
else
{
lean_object* v___x_1199_; 
v___x_1199_ = lean_unsigned_to_nat(0u);
v___y_1172_ = v___x_1196_;
v___y_1173_ = v___y_1183_;
v___y_1174_ = v___y_1184_;
v___y_1175_ = v___x_1197_;
v___y_1176_ = v___y_1188_;
v___y_1177_ = v___y_1187_;
v___y_1178_ = v___x_1199_;
goto v___jp_1171_;
}
}
v___jp_1200_:
{
lean_object* v_impl_1201_; lean_object* v___x_1202_; 
v_impl_1201_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(v_k_1027_, v_v_1028_, v_l_1047_);
v___x_1202_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_1048_) == 0)
{
lean_object* v_size_1203_; lean_object* v_size_1204_; lean_object* v_k_1205_; lean_object* v_v_1206_; lean_object* v_l_1207_; lean_object* v_r_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; uint8_t v___x_1211_; 
v_size_1203_ = lean_ctor_get(v_r_1048_, 0);
v_size_1204_ = lean_ctor_get(v_impl_1201_, 0);
lean_inc(v_size_1204_);
v_k_1205_ = lean_ctor_get(v_impl_1201_, 1);
lean_inc(v_k_1205_);
v_v_1206_ = lean_ctor_get(v_impl_1201_, 2);
lean_inc(v_v_1206_);
v_l_1207_ = lean_ctor_get(v_impl_1201_, 3);
lean_inc(v_l_1207_);
v_r_1208_ = lean_ctor_get(v_impl_1201_, 4);
lean_inc(v_r_1208_);
v___x_1209_ = lean_unsigned_to_nat(3u);
v___x_1210_ = lean_nat_mul(v___x_1209_, v_size_1203_);
v___x_1211_ = lean_nat_dec_lt(v___x_1210_, v_size_1204_);
lean_dec(v___x_1210_);
if (v___x_1211_ == 0)
{
lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; 
lean_dec(v_r_1208_);
lean_dec(v_l_1207_);
lean_dec(v_v_1206_);
lean_dec(v_k_1205_);
v___x_1212_ = lean_nat_add(v___x_1202_, v_size_1204_);
lean_dec(v_size_1204_);
v___x_1213_ = lean_nat_add(v___x_1212_, v_size_1203_);
lean_dec(v___x_1212_);
v___x_1214_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1214_, 0, v___x_1213_);
lean_ctor_set(v___x_1214_, 1, v_k_1045_);
lean_ctor_set(v___x_1214_, 2, v_v_1046_);
lean_ctor_set(v___x_1214_, 3, v_impl_1201_);
lean_ctor_set(v___x_1214_, 4, v_r_1048_);
return v___x_1214_;
}
else
{
lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1251_; 
v_isSharedCheck_1251_ = !lean_is_exclusive(v_impl_1201_);
if (v_isSharedCheck_1251_ == 0)
{
lean_object* v_unused_1252_; lean_object* v_unused_1253_; lean_object* v_unused_1254_; lean_object* v_unused_1255_; lean_object* v_unused_1256_; 
v_unused_1252_ = lean_ctor_get(v_impl_1201_, 4);
lean_dec(v_unused_1252_);
v_unused_1253_ = lean_ctor_get(v_impl_1201_, 3);
lean_dec(v_unused_1253_);
v_unused_1254_ = lean_ctor_get(v_impl_1201_, 2);
lean_dec(v_unused_1254_);
v_unused_1255_ = lean_ctor_get(v_impl_1201_, 1);
lean_dec(v_unused_1255_);
v_unused_1256_ = lean_ctor_get(v_impl_1201_, 0);
lean_dec(v_unused_1256_);
v___x_1216_ = v_impl_1201_;
v_isShared_1217_ = v_isSharedCheck_1251_;
goto v_resetjp_1215_;
}
else
{
lean_dec(v_impl_1201_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1251_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
lean_object* v_size_1218_; lean_object* v_size_1219_; lean_object* v_k_1220_; lean_object* v_v_1221_; lean_object* v_l_1222_; lean_object* v_r_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; uint8_t v___x_1226_; 
v_size_1218_ = lean_ctor_get(v_l_1207_, 0);
v_size_1219_ = lean_ctor_get(v_r_1208_, 0);
v_k_1220_ = lean_ctor_get(v_r_1208_, 1);
v_v_1221_ = lean_ctor_get(v_r_1208_, 2);
v_l_1222_ = lean_ctor_get(v_r_1208_, 3);
v_r_1223_ = lean_ctor_get(v_r_1208_, 4);
v___x_1224_ = lean_unsigned_to_nat(2u);
v___x_1225_ = lean_nat_mul(v___x_1224_, v_size_1218_);
v___x_1226_ = lean_nat_dec_lt(v_size_1219_, v___x_1225_);
lean_dec(v___x_1225_);
if (v___x_1226_ == 0)
{
lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; 
lean_inc(v_r_1223_);
lean_inc(v_l_1222_);
lean_inc(v_v_1221_);
lean_inc(v_k_1220_);
lean_del_object(v___x_1216_);
lean_dec(v_r_1208_);
v___x_1227_ = lean_nat_add(v___x_1202_, v_size_1204_);
lean_dec(v_size_1204_);
v___x_1228_ = lean_nat_add(v___x_1227_, v_size_1203_);
lean_dec(v___x_1227_);
v___x_1229_ = lean_nat_add(v___x_1202_, v_size_1218_);
if (lean_obj_tag(v_l_1222_) == 0)
{
lean_object* v_size_1230_; 
v_size_1230_ = lean_ctor_get(v_l_1222_, 0);
lean_inc(v_size_1230_);
lean_inc(v_size_1203_);
v___y_1183_ = v_v_1221_;
v___y_1184_ = v_r_1223_;
v___y_1185_ = v_v_1206_;
v___y_1186_ = v_l_1222_;
v___y_1187_ = v___x_1228_;
v___y_1188_ = v_k_1220_;
v___y_1189_ = v___x_1229_;
v___y_1190_ = v_size_1203_;
v___y_1191_ = v___x_1202_;
v___y_1192_ = v_l_1207_;
v___y_1193_ = v_k_1205_;
v___y_1194_ = v_size_1230_;
goto v___jp_1182_;
}
else
{
lean_object* v___x_1231_; 
v___x_1231_ = lean_unsigned_to_nat(0u);
lean_inc(v_size_1203_);
v___y_1183_ = v_v_1221_;
v___y_1184_ = v_r_1223_;
v___y_1185_ = v_v_1206_;
v___y_1186_ = v_l_1222_;
v___y_1187_ = v___x_1228_;
v___y_1188_ = v_k_1220_;
v___y_1189_ = v___x_1229_;
v___y_1190_ = v_size_1203_;
v___y_1191_ = v___x_1202_;
v___y_1192_ = v_l_1207_;
v___y_1193_ = v_k_1205_;
v___y_1194_ = v___x_1231_;
goto v___jp_1182_;
}
}
else
{
lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1237_; 
v___x_1232_ = lean_nat_add(v___x_1202_, v_size_1204_);
lean_dec(v_size_1204_);
v___x_1233_ = lean_nat_add(v___x_1232_, v_size_1203_);
lean_dec(v___x_1232_);
v___x_1234_ = lean_nat_add(v___x_1202_, v_size_1203_);
v___x_1235_ = lean_nat_add(v___x_1234_, v_size_1219_);
lean_dec(v___x_1234_);
lean_inc_ref(v_r_1048_);
if (v_isShared_1217_ == 0)
{
lean_ctor_set(v___x_1216_, 4, v_r_1048_);
lean_ctor_set(v___x_1216_, 3, v_r_1208_);
lean_ctor_set(v___x_1216_, 2, v_v_1046_);
lean_ctor_set(v___x_1216_, 1, v_k_1045_);
lean_ctor_set(v___x_1216_, 0, v___x_1235_);
v___x_1237_ = v___x_1216_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1250_; 
v_reuseFailAlloc_1250_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1250_, 0, v___x_1235_);
lean_ctor_set(v_reuseFailAlloc_1250_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1250_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1250_, 3, v_r_1208_);
lean_ctor_set(v_reuseFailAlloc_1250_, 4, v_r_1048_);
v___x_1237_ = v_reuseFailAlloc_1250_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1244_; 
v_isSharedCheck_1244_ = !lean_is_exclusive(v_r_1048_);
if (v_isSharedCheck_1244_ == 0)
{
lean_object* v_unused_1245_; lean_object* v_unused_1246_; lean_object* v_unused_1247_; lean_object* v_unused_1248_; lean_object* v_unused_1249_; 
v_unused_1245_ = lean_ctor_get(v_r_1048_, 4);
lean_dec(v_unused_1245_);
v_unused_1246_ = lean_ctor_get(v_r_1048_, 3);
lean_dec(v_unused_1246_);
v_unused_1247_ = lean_ctor_get(v_r_1048_, 2);
lean_dec(v_unused_1247_);
v_unused_1248_ = lean_ctor_get(v_r_1048_, 1);
lean_dec(v_unused_1248_);
v_unused_1249_ = lean_ctor_get(v_r_1048_, 0);
lean_dec(v_unused_1249_);
v___x_1239_ = v_r_1048_;
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
else
{
lean_dec(v_r_1048_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1242_; 
if (v_isShared_1240_ == 0)
{
lean_ctor_set(v___x_1239_, 4, v___x_1237_);
lean_ctor_set(v___x_1239_, 3, v_l_1207_);
lean_ctor_set(v___x_1239_, 2, v_v_1206_);
lean_ctor_set(v___x_1239_, 1, v_k_1205_);
lean_ctor_set(v___x_1239_, 0, v___x_1233_);
v___x_1242_ = v___x_1239_;
goto v_reusejp_1241_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v___x_1233_);
lean_ctor_set(v_reuseFailAlloc_1243_, 1, v_k_1205_);
lean_ctor_set(v_reuseFailAlloc_1243_, 2, v_v_1206_);
lean_ctor_set(v_reuseFailAlloc_1243_, 3, v_l_1207_);
lean_ctor_set(v_reuseFailAlloc_1243_, 4, v___x_1237_);
v___x_1242_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1241_;
}
v_reusejp_1241_:
{
return v___x_1242_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_1257_; 
v_l_1257_ = lean_ctor_get(v_impl_1201_, 3);
lean_inc(v_l_1257_);
if (lean_obj_tag(v_l_1257_) == 0)
{
lean_object* v_r_1258_; lean_object* v_k_1259_; lean_object* v_v_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1269_; 
v_r_1258_ = lean_ctor_get(v_impl_1201_, 4);
v_k_1259_ = lean_ctor_get(v_impl_1201_, 1);
v_v_1260_ = lean_ctor_get(v_impl_1201_, 2);
v_isSharedCheck_1269_ = !lean_is_exclusive(v_impl_1201_);
if (v_isSharedCheck_1269_ == 0)
{
lean_object* v_unused_1270_; lean_object* v_unused_1271_; 
v_unused_1270_ = lean_ctor_get(v_impl_1201_, 3);
lean_dec(v_unused_1270_);
v_unused_1271_ = lean_ctor_get(v_impl_1201_, 0);
lean_dec(v_unused_1271_);
v___x_1262_ = v_impl_1201_;
v_isShared_1263_ = v_isSharedCheck_1269_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_r_1258_);
lean_inc(v_v_1260_);
lean_inc(v_k_1259_);
lean_dec(v_impl_1201_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1269_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v___x_1264_; lean_object* v___x_1266_; 
v___x_1264_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_1258_);
if (v_isShared_1263_ == 0)
{
lean_ctor_set(v___x_1262_, 3, v_r_1258_);
lean_ctor_set(v___x_1262_, 2, v_v_1046_);
lean_ctor_set(v___x_1262_, 1, v_k_1045_);
lean_ctor_set(v___x_1262_, 0, v___x_1202_);
v___x_1266_ = v___x_1262_;
goto v_reusejp_1265_;
}
else
{
lean_object* v_reuseFailAlloc_1268_; 
v_reuseFailAlloc_1268_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1268_, 0, v___x_1202_);
lean_ctor_set(v_reuseFailAlloc_1268_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1268_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1268_, 3, v_r_1258_);
lean_ctor_set(v_reuseFailAlloc_1268_, 4, v_r_1258_);
v___x_1266_ = v_reuseFailAlloc_1268_;
goto v_reusejp_1265_;
}
v_reusejp_1265_:
{
lean_object* v___x_1267_; 
v___x_1267_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1264_);
lean_ctor_set(v___x_1267_, 1, v_k_1259_);
lean_ctor_set(v___x_1267_, 2, v_v_1260_);
lean_ctor_set(v___x_1267_, 3, v_l_1257_);
lean_ctor_set(v___x_1267_, 4, v___x_1266_);
return v___x_1267_;
}
}
}
else
{
lean_object* v_r_1272_; 
v_r_1272_ = lean_ctor_get(v_impl_1201_, 4);
lean_inc(v_r_1272_);
if (lean_obj_tag(v_r_1272_) == 0)
{
lean_object* v_k_1273_; lean_object* v_v_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1295_; 
v_k_1273_ = lean_ctor_get(v_impl_1201_, 1);
v_v_1274_ = lean_ctor_get(v_impl_1201_, 2);
v_isSharedCheck_1295_ = !lean_is_exclusive(v_impl_1201_);
if (v_isSharedCheck_1295_ == 0)
{
lean_object* v_unused_1296_; lean_object* v_unused_1297_; lean_object* v_unused_1298_; 
v_unused_1296_ = lean_ctor_get(v_impl_1201_, 4);
lean_dec(v_unused_1296_);
v_unused_1297_ = lean_ctor_get(v_impl_1201_, 3);
lean_dec(v_unused_1297_);
v_unused_1298_ = lean_ctor_get(v_impl_1201_, 0);
lean_dec(v_unused_1298_);
v___x_1276_ = v_impl_1201_;
v_isShared_1277_ = v_isSharedCheck_1295_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_v_1274_);
lean_inc(v_k_1273_);
lean_dec(v_impl_1201_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1295_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
lean_object* v_k_1278_; lean_object* v_v_1279_; lean_object* v___x_1281_; uint8_t v_isShared_1282_; uint8_t v_isSharedCheck_1291_; 
v_k_1278_ = lean_ctor_get(v_r_1272_, 1);
v_v_1279_ = lean_ctor_get(v_r_1272_, 2);
v_isSharedCheck_1291_ = !lean_is_exclusive(v_r_1272_);
if (v_isSharedCheck_1291_ == 0)
{
lean_object* v_unused_1292_; lean_object* v_unused_1293_; lean_object* v_unused_1294_; 
v_unused_1292_ = lean_ctor_get(v_r_1272_, 4);
lean_dec(v_unused_1292_);
v_unused_1293_ = lean_ctor_get(v_r_1272_, 3);
lean_dec(v_unused_1293_);
v_unused_1294_ = lean_ctor_get(v_r_1272_, 0);
lean_dec(v_unused_1294_);
v___x_1281_ = v_r_1272_;
v_isShared_1282_ = v_isSharedCheck_1291_;
goto v_resetjp_1280_;
}
else
{
lean_inc(v_v_1279_);
lean_inc(v_k_1278_);
lean_dec(v_r_1272_);
v___x_1281_ = lean_box(0);
v_isShared_1282_ = v_isSharedCheck_1291_;
goto v_resetjp_1280_;
}
v_resetjp_1280_:
{
lean_object* v___x_1283_; lean_object* v___x_1285_; 
v___x_1283_ = lean_unsigned_to_nat(3u);
if (v_isShared_1282_ == 0)
{
lean_ctor_set(v___x_1281_, 4, v_l_1257_);
lean_ctor_set(v___x_1281_, 3, v_l_1257_);
lean_ctor_set(v___x_1281_, 2, v_v_1274_);
lean_ctor_set(v___x_1281_, 1, v_k_1273_);
lean_ctor_set(v___x_1281_, 0, v___x_1202_);
v___x_1285_ = v___x_1281_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1202_);
lean_ctor_set(v_reuseFailAlloc_1290_, 1, v_k_1273_);
lean_ctor_set(v_reuseFailAlloc_1290_, 2, v_v_1274_);
lean_ctor_set(v_reuseFailAlloc_1290_, 3, v_l_1257_);
lean_ctor_set(v_reuseFailAlloc_1290_, 4, v_l_1257_);
v___x_1285_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
lean_object* v___x_1287_; 
if (v_isShared_1277_ == 0)
{
lean_ctor_set(v___x_1276_, 4, v_l_1257_);
lean_ctor_set(v___x_1276_, 2, v_v_1046_);
lean_ctor_set(v___x_1276_, 1, v_k_1045_);
lean_ctor_set(v___x_1276_, 0, v___x_1202_);
v___x_1287_ = v___x_1276_;
goto v_reusejp_1286_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v___x_1202_);
lean_ctor_set(v_reuseFailAlloc_1289_, 1, v_k_1045_);
lean_ctor_set(v_reuseFailAlloc_1289_, 2, v_v_1046_);
lean_ctor_set(v_reuseFailAlloc_1289_, 3, v_l_1257_);
lean_ctor_set(v_reuseFailAlloc_1289_, 4, v_l_1257_);
v___x_1287_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1286_;
}
v_reusejp_1286_:
{
lean_object* v___x_1288_; 
v___x_1288_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1283_);
lean_ctor_set(v___x_1288_, 1, v_k_1278_);
lean_ctor_set(v___x_1288_, 2, v_v_1279_);
lean_ctor_set(v___x_1288_, 3, v___x_1285_);
lean_ctor_set(v___x_1288_, 4, v___x_1287_);
return v___x_1288_;
}
}
}
}
}
else
{
lean_object* v___x_1299_; lean_object* v___x_1300_; 
v___x_1299_ = lean_unsigned_to_nat(2u);
v___x_1300_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1300_, 0, v___x_1299_);
lean_ctor_set(v___x_1300_, 1, v_k_1045_);
lean_ctor_set(v___x_1300_, 2, v_v_1046_);
lean_ctor_set(v___x_1300_, 3, v_impl_1201_);
lean_ctor_set(v___x_1300_, 4, v_r_1272_);
return v___x_1300_;
}
}
}
}
}
}
else
{
lean_object* v___x_1311_; lean_object* v___x_1312_; 
v___x_1311_ = lean_unsigned_to_nat(1u);
v___x_1312_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1312_, 0, v___x_1311_);
lean_ctor_set(v___x_1312_, 1, v_k_1027_);
lean_ctor_set(v___x_1312_, 2, v_v_1028_);
lean_ctor_set(v___x_1312_, 3, v_t_1029_);
lean_ctor_set(v___x_1312_, 4, v_t_1029_);
return v___x_1312_;
}
v___jp_1030_:
{
lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
v___x_1041_ = lean_nat_add(v___y_1039_, v___y_1040_);
lean_dec(v___y_1040_);
lean_dec(v___y_1039_);
v___x_1042_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1041_);
lean_ctor_set(v___x_1042_, 1, v___y_1032_);
lean_ctor_set(v___x_1042_, 2, v___y_1035_);
lean_ctor_set(v___x_1042_, 3, v___y_1037_);
lean_ctor_set(v___x_1042_, 4, v___y_1036_);
v___x_1043_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1043_, 0, v___y_1031_);
lean_ctor_set(v___x_1043_, 1, v___y_1038_);
lean_ctor_set(v___x_1043_, 2, v___y_1034_);
lean_ctor_set(v___x_1043_, 3, v___y_1033_);
lean_ctor_set(v___x_1043_, 4, v___x_1042_);
return v___x_1043_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg(lean_object* v_k_1313_, lean_object* v_t_1314_){
_start:
{
if (lean_obj_tag(v_t_1314_) == 0)
{
lean_object* v_k_1315_; lean_object* v_l_1316_; lean_object* v_r_1317_; lean_object* v_fst_1318_; lean_object* v_snd_1319_; lean_object* v_fst_1320_; lean_object* v_snd_1321_; uint8_t v___x_1322_; 
v_k_1315_ = lean_ctor_get(v_t_1314_, 1);
v_l_1316_ = lean_ctor_get(v_t_1314_, 3);
v_r_1317_ = lean_ctor_get(v_t_1314_, 4);
v_fst_1318_ = lean_ctor_get(v_k_1313_, 0);
v_snd_1319_ = lean_ctor_get(v_k_1313_, 1);
v_fst_1320_ = lean_ctor_get(v_k_1315_, 0);
v_snd_1321_ = lean_ctor_get(v_k_1315_, 1);
v___x_1322_ = lean_nat_dec_lt(v_fst_1318_, v_fst_1320_);
if (v___x_1322_ == 0)
{
uint8_t v___x_1323_; 
v___x_1323_ = lean_nat_dec_eq(v_fst_1318_, v_fst_1320_);
if (v___x_1323_ == 0)
{
v_t_1314_ = v_r_1317_;
goto _start;
}
else
{
uint8_t v___x_1325_; 
v___x_1325_ = lean_nat_dec_lt(v_snd_1319_, v_snd_1321_);
if (v___x_1325_ == 0)
{
uint8_t v___x_1326_; 
v___x_1326_ = lean_nat_dec_eq(v_snd_1319_, v_snd_1321_);
if (v___x_1326_ == 0)
{
v_t_1314_ = v_r_1317_;
goto _start;
}
else
{
return v___x_1326_;
}
}
else
{
v_t_1314_ = v_l_1316_;
goto _start;
}
}
}
else
{
v_t_1314_ = v_l_1316_;
goto _start;
}
}
else
{
uint8_t v___x_1330_; 
v___x_1330_ = 0;
return v___x_1330_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg___boxed(lean_object* v_k_1331_, lean_object* v_t_1332_){
_start:
{
uint8_t v_res_1333_; lean_object* v_r_1334_; 
v_res_1333_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg(v_k_1331_, v_t_1332_);
lean_dec(v_t_1332_);
lean_dec_ref(v_k_1331_);
v_r_1334_ = lean_box(v_res_1333_);
return v_r_1334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(lean_object* v_as_x27_1335_, lean_object* v_b_1336_){
_start:
{
if (lean_obj_tag(v_as_x27_1335_) == 0)
{
return v_b_1336_;
}
else
{
lean_object* v_head_1337_; lean_object* v_tail_1338_; uint8_t v___x_1339_; 
v_head_1337_ = lean_ctor_get(v_as_x27_1335_, 0);
v_tail_1338_ = lean_ctor_get(v_as_x27_1335_, 1);
v___x_1339_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg(v_head_1337_, v_b_1336_);
if (v___x_1339_ == 0)
{
lean_object* v___x_1340_; lean_object* v___x_1341_; 
v___x_1340_ = lean_box(0);
lean_inc(v_head_1337_);
v___x_1341_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(v_head_1337_, v___x_1340_, v_b_1336_);
v_as_x27_1335_ = v_tail_1338_;
v_b_1336_ = v___x_1341_;
goto _start;
}
else
{
v_as_x27_1335_ = v_tail_1338_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg___boxed(lean_object* v_as_x27_1344_, lean_object* v_b_1345_){
_start:
{
lean_object* v_res_1346_; 
v_res_1346_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(v_as_x27_1344_, v_b_1345_);
lean_dec(v_as_x27_1344_);
return v_res_1346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5(lean_object* v_x_1347_, lean_object* v_x_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_){
_start:
{
if (lean_obj_tag(v_x_1348_) == 0)
{
lean_object* v___x_1356_; 
v___x_1356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1356_, 0, v_x_1347_);
return v___x_1356_;
}
else
{
lean_object* v_head_1357_; lean_object* v_tail_1358_; lean_object* v___y_1360_; uint8_t v___y_1361_; lean_object* v___y_1364_; lean_object* v_a_1365_; lean_object* v___x_1368_; 
v_head_1357_ = lean_ctor_get(v_x_1348_, 0);
lean_inc(v_head_1357_);
v_tail_1358_ = lean_ctor_get(v_x_1348_, 1);
lean_inc(v_tail_1358_);
lean_dec_ref_known(v_x_1348_, 2);
lean_inc(v___y_1354_);
lean_inc_ref(v___y_1353_);
lean_inc(v___y_1352_);
lean_inc_ref(v___y_1351_);
v___x_1368_ = lean_infer_type(v_head_1357_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1368_) == 0)
{
lean_object* v_a_1369_; lean_object* v___x_1370_; 
v_a_1369_ = lean_ctor_get(v___x_1368_, 0);
lean_inc(v_a_1369_);
lean_dec_ref_known(v___x_1368_, 1);
v___x_1370_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_1369_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1370_) == 0)
{
lean_object* v_a_1371_; lean_object* v_snd_1372_; lean_object* v_snd_1373_; lean_object* v_fst_1374_; lean_object* v_snd_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
v_a_1371_ = lean_ctor_get(v___x_1370_, 0);
lean_inc(v_a_1371_);
lean_dec_ref_known(v___x_1370_, 1);
v_snd_1372_ = lean_ctor_get(v_a_1371_, 1);
lean_inc(v_snd_1372_);
lean_dec(v_a_1371_);
v_snd_1373_ = lean_ctor_get(v_snd_1372_, 1);
lean_inc(v_snd_1373_);
lean_dec(v_snd_1372_);
v_fst_1374_ = lean_ctor_get(v_snd_1373_, 0);
lean_inc(v_fst_1374_);
v_snd_1375_ = lean_ctor_get(v_snd_1373_, 1);
lean_inc(v_snd_1375_);
lean_dec(v_snd_1373_);
v___x_1376_ = lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(v_fst_1374_);
v___x_1377_ = lean_box(0);
v___x_1378_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0(v___x_1376_, v___x_1377_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1378_) == 0)
{
lean_object* v_a_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc(v_a_1379_);
lean_dec_ref_known(v___x_1378_, 1);
v___x_1380_ = lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons(v_snd_1375_);
v___x_1381_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__0(v___x_1380_, v___x_1377_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1381_) == 0)
{
lean_object* v_a_1382_; lean_object* v_r_1383_; lean_object* v___x_1384_; 
v_a_1382_ = lean_ctor_get(v___x_1381_, 0);
lean_inc(v_a_1382_);
lean_dec_ref_known(v___x_1381_, 1);
v_r_1383_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(v_a_1379_, v_x_1347_);
lean_dec(v_a_1379_);
v___x_1384_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(v_a_1382_, v_r_1383_);
lean_dec(v_a_1382_);
v_x_1347_ = v___x_1384_;
v_x_1348_ = v_tail_1358_;
goto _start;
}
else
{
lean_object* v_a_1386_; lean_object* v___x_1388_; uint8_t v_isShared_1389_; uint8_t v_isSharedCheck_1393_; 
lean_dec(v_a_1379_);
v_a_1386_ = lean_ctor_get(v___x_1381_, 0);
v_isSharedCheck_1393_ = !lean_is_exclusive(v___x_1381_);
if (v_isSharedCheck_1393_ == 0)
{
v___x_1388_ = v___x_1381_;
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
else
{
lean_inc(v_a_1386_);
lean_dec(v___x_1381_);
v___x_1388_ = lean_box(0);
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
v_resetjp_1387_:
{
lean_object* v___x_1391_; 
lean_inc(v_a_1386_);
if (v_isShared_1389_ == 0)
{
v___x_1391_ = v___x_1388_;
goto v_reusejp_1390_;
}
else
{
lean_object* v_reuseFailAlloc_1392_; 
v_reuseFailAlloc_1392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1392_, 0, v_a_1386_);
v___x_1391_ = v_reuseFailAlloc_1392_;
goto v_reusejp_1390_;
}
v_reusejp_1390_:
{
v___y_1364_ = v___x_1391_;
v_a_1365_ = v_a_1386_;
goto v___jp_1363_;
}
}
}
}
else
{
lean_object* v_a_1394_; lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1401_; 
lean_dec(v_snd_1375_);
v_a_1394_ = lean_ctor_get(v___x_1378_, 0);
v_isSharedCheck_1401_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1401_ == 0)
{
v___x_1396_ = v___x_1378_;
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
else
{
lean_inc(v_a_1394_);
lean_dec(v___x_1378_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v___x_1399_; 
lean_inc(v_a_1394_);
if (v_isShared_1397_ == 0)
{
v___x_1399_ = v___x_1396_;
goto v_reusejp_1398_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v_a_1394_);
v___x_1399_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1398_;
}
v_reusejp_1398_:
{
v___y_1364_ = v___x_1399_;
v_a_1365_ = v_a_1394_;
goto v___jp_1363_;
}
}
}
}
else
{
lean_object* v_a_1402_; lean_object* v___x_1404_; uint8_t v_isShared_1405_; uint8_t v_isSharedCheck_1409_; 
v_a_1402_ = lean_ctor_get(v___x_1370_, 0);
v_isSharedCheck_1409_ = !lean_is_exclusive(v___x_1370_);
if (v_isSharedCheck_1409_ == 0)
{
v___x_1404_ = v___x_1370_;
v_isShared_1405_ = v_isSharedCheck_1409_;
goto v_resetjp_1403_;
}
else
{
lean_inc(v_a_1402_);
lean_dec(v___x_1370_);
v___x_1404_ = lean_box(0);
v_isShared_1405_ = v_isSharedCheck_1409_;
goto v_resetjp_1403_;
}
v_resetjp_1403_:
{
lean_object* v___x_1407_; 
lean_inc(v_a_1402_);
if (v_isShared_1405_ == 0)
{
v___x_1407_ = v___x_1404_;
goto v_reusejp_1406_;
}
else
{
lean_object* v_reuseFailAlloc_1408_; 
v_reuseFailAlloc_1408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1408_, 0, v_a_1402_);
v___x_1407_ = v_reuseFailAlloc_1408_;
goto v_reusejp_1406_;
}
v_reusejp_1406_:
{
v___y_1364_ = v___x_1407_;
v_a_1365_ = v_a_1402_;
goto v___jp_1363_;
}
}
}
}
else
{
lean_object* v_a_1410_; lean_object* v___x_1412_; uint8_t v_isShared_1413_; uint8_t v_isSharedCheck_1417_; 
v_a_1410_ = lean_ctor_get(v___x_1368_, 0);
v_isSharedCheck_1417_ = !lean_is_exclusive(v___x_1368_);
if (v_isSharedCheck_1417_ == 0)
{
v___x_1412_ = v___x_1368_;
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
else
{
lean_inc(v_a_1410_);
lean_dec(v___x_1368_);
v___x_1412_ = lean_box(0);
v_isShared_1413_ = v_isSharedCheck_1417_;
goto v_resetjp_1411_;
}
v_resetjp_1411_:
{
lean_object* v___x_1415_; 
lean_inc(v_a_1410_);
if (v_isShared_1413_ == 0)
{
v___x_1415_ = v___x_1412_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1416_; 
v_reuseFailAlloc_1416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1416_, 0, v_a_1410_);
v___x_1415_ = v_reuseFailAlloc_1416_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
v___y_1364_ = v___x_1415_;
v_a_1365_ = v_a_1410_;
goto v___jp_1363_;
}
}
}
v___jp_1359_:
{
if (v___y_1361_ == 0)
{
lean_dec_ref(v___y_1360_);
v_x_1348_ = v_tail_1358_;
goto _start;
}
else
{
lean_dec(v_tail_1358_);
lean_dec(v_x_1347_);
return v___y_1360_;
}
}
v___jp_1363_:
{
uint8_t v___x_1366_; 
v___x_1366_ = l_Lean_Exception_isInterrupt(v_a_1365_);
if (v___x_1366_ == 0)
{
uint8_t v___x_1367_; 
v___x_1367_ = l_Lean_Exception_isRuntime(v_a_1365_);
v___y_1360_ = v___y_1364_;
v___y_1361_ = v___x_1367_;
goto v___jp_1359_;
}
else
{
lean_dec_ref(v_a_1365_);
v___y_1360_ = v___y_1364_;
v___y_1361_ = v___x_1366_;
goto v___jp_1359_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5___boxed(lean_object* v_x_1418_, lean_object* v_x_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_){
_start:
{
lean_object* v_res_1427_; 
v_res_1427_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5(v_x_1418_, v_x_1419_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_, v___y_1424_, v___y_1425_);
lean_dec(v___y_1425_);
lean_dec_ref(v___y_1424_);
lean_dec(v___y_1423_);
lean_dec_ref(v___y_1422_);
lean_dec(v___y_1421_);
lean_dec_ref(v___y_1420_);
return v_res_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg(lean_object* v___x_1428_, lean_object* v_x_1429_, lean_object* v_x_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_){
_start:
{
if (lean_obj_tag(v_x_1429_) == 0)
{
lean_object* v___x_1436_; lean_object* v___x_1437_; 
v___x_1436_ = l_List_reverse___redArg(v_x_1430_);
v___x_1437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1437_, 0, v___x_1436_);
return v___x_1437_;
}
else
{
lean_object* v_head_1438_; lean_object* v_tail_1439_; lean_object* v___x_1441_; uint8_t v_isShared_1442_; uint8_t v_isSharedCheck_1471_; 
v_head_1438_ = lean_ctor_get(v_x_1429_, 0);
v_tail_1439_ = lean_ctor_get(v_x_1429_, 1);
v_isSharedCheck_1471_ = !lean_is_exclusive(v_x_1429_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1441_ = v_x_1429_;
v_isShared_1442_ = v_isSharedCheck_1471_;
goto v_resetjp_1440_;
}
else
{
lean_inc(v_tail_1439_);
lean_inc(v_head_1438_);
lean_dec(v_x_1429_);
v___x_1441_ = lean_box(0);
v_isShared_1442_ = v_isSharedCheck_1471_;
goto v_resetjp_1440_;
}
v_resetjp_1440_:
{
lean_object* v_fst_1443_; lean_object* v_snd_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1470_; 
v_fst_1443_ = lean_ctor_get(v_head_1438_, 0);
v_snd_1444_ = lean_ctor_get(v_head_1438_, 1);
v_isSharedCheck_1470_ = !lean_is_exclusive(v_head_1438_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1446_ = v_head_1438_;
v_isShared_1447_ = v_isSharedCheck_1470_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_snd_1444_);
lean_inc(v_fst_1443_);
lean_dec(v_head_1438_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1470_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1452_; 
v___x_1448_ = l_Lean_instInhabitedExpr;
v___x_1449_ = lean_array_get_borrowed(v___x_1448_, v___x_1428_, v_fst_1443_);
lean_dec(v_fst_1443_);
v___x_1450_ = lean_array_get_borrowed(v___x_1448_, v___x_1428_, v_snd_1444_);
lean_dec(v_snd_1444_);
lean_inc(v___x_1450_);
lean_inc(v___x_1449_);
if (v_isShared_1447_ == 0)
{
lean_ctor_set(v___x_1446_, 1, v___x_1450_);
lean_ctor_set(v___x_1446_, 0, v___x_1449_);
v___x_1452_ = v___x_1446_;
goto v_reusejp_1451_;
}
else
{
lean_object* v_reuseFailAlloc_1469_; 
v_reuseFailAlloc_1469_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1469_, 0, v___x_1449_);
lean_ctor_set(v_reuseFailAlloc_1469_, 1, v___x_1450_);
v___x_1452_ = v_reuseFailAlloc_1469_;
goto v_reusejp_1451_;
}
v_reusejp_1451_:
{
lean_object* v___x_1453_; 
v___x_1453_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f(v___x_1452_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_);
if (lean_obj_tag(v___x_1453_) == 0)
{
lean_object* v_a_1454_; 
v_a_1454_ = lean_ctor_get(v___x_1453_, 0);
lean_inc(v_a_1454_);
lean_dec_ref_known(v___x_1453_, 1);
if (lean_obj_tag(v_a_1454_) == 0)
{
lean_del_object(v___x_1441_);
v_x_1429_ = v_tail_1439_;
goto _start;
}
else
{
lean_object* v_val_1456_; lean_object* v___x_1458_; 
v_val_1456_ = lean_ctor_get(v_a_1454_, 0);
lean_inc(v_val_1456_);
lean_dec_ref_known(v_a_1454_, 1);
if (v_isShared_1442_ == 0)
{
lean_ctor_set(v___x_1441_, 1, v_x_1430_);
lean_ctor_set(v___x_1441_, 0, v_val_1456_);
v___x_1458_ = v___x_1441_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v_val_1456_);
lean_ctor_set(v_reuseFailAlloc_1460_, 1, v_x_1430_);
v___x_1458_ = v_reuseFailAlloc_1460_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
v_x_1429_ = v_tail_1439_;
v_x_1430_ = v___x_1458_;
goto _start;
}
}
}
else
{
lean_object* v_a_1461_; lean_object* v___x_1463_; uint8_t v_isShared_1464_; uint8_t v_isSharedCheck_1468_; 
lean_del_object(v___x_1441_);
lean_dec(v_tail_1439_);
lean_dec(v_x_1430_);
v_a_1461_ = lean_ctor_get(v___x_1453_, 0);
v_isSharedCheck_1468_ = !lean_is_exclusive(v___x_1453_);
if (v_isSharedCheck_1468_ == 0)
{
v___x_1463_ = v___x_1453_;
v_isShared_1464_ = v_isSharedCheck_1468_;
goto v_resetjp_1462_;
}
else
{
lean_inc(v_a_1461_);
lean_dec(v___x_1453_);
v___x_1463_ = lean_box(0);
v_isShared_1464_ = v_isSharedCheck_1468_;
goto v_resetjp_1462_;
}
v_resetjp_1462_:
{
lean_object* v___x_1466_; 
if (v_isShared_1464_ == 0)
{
v___x_1466_ = v___x_1463_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v_a_1461_);
v___x_1466_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
return v___x_1466_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg___boxed(lean_object* v___x_1472_, lean_object* v_x_1473_, lean_object* v_x_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_){
_start:
{
lean_object* v_res_1480_; 
v_res_1480_ = lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg(v___x_1472_, v_x_1473_, v_x_1474_, v___y_1475_, v___y_1476_, v___y_1477_, v___y_1478_);
lean_dec(v___y_1478_);
lean_dec_ref(v___y_1477_);
lean_dec(v___y_1476_);
lean_dec_ref(v___y_1475_);
lean_dec_ref(v___x_1472_);
return v_res_1480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1(lean_object* v___x_1481_, lean_object* v_a_1482_, lean_object* v___x_1483_, lean_object* v_g_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_){
_start:
{
lean_object* v___x_1492_; 
lean_inc(v_a_1482_);
v___x_1492_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_natToInt_spec__5(v___x_1481_, v_a_1482_, v___y_1485_, v___y_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
if (lean_obj_tag(v___x_1492_) == 0)
{
lean_object* v_a_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; 
v_a_1493_ = lean_ctor_get(v___x_1492_, 0);
lean_inc(v_a_1493_);
lean_dec_ref_known(v___x_1492_, 1);
v___x_1494_ = lean_st_ref_get(v___y_1486_);
v___x_1495_ = lean_box(0);
v___x_1496_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_natToInt_spec__6(v___x_1495_, v_a_1493_);
lean_dec(v_a_1493_);
v___x_1497_ = lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg(v___x_1494_, v___x_1496_, v___x_1483_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec(v___x_1494_);
if (lean_obj_tag(v___x_1497_) == 0)
{
lean_object* v_a_1498_; lean_object* v___x_1500_; uint8_t v_isShared_1501_; uint8_t v_isSharedCheck_1508_; 
v_a_1498_ = lean_ctor_get(v___x_1497_, 0);
v_isSharedCheck_1508_ = !lean_is_exclusive(v___x_1497_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1500_ = v___x_1497_;
v_isShared_1501_ = v_isSharedCheck_1508_;
goto v_resetjp_1499_;
}
else
{
lean_inc(v_a_1498_);
lean_dec(v___x_1497_);
v___x_1500_ = lean_box(0);
v_isShared_1501_ = v_isSharedCheck_1508_;
goto v_resetjp_1499_;
}
v_resetjp_1499_:
{
lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1506_; 
v___x_1502_ = l_List_appendTR___redArg(v_a_1498_, v_a_1482_);
v___x_1503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1503_, 0, v_g_1484_);
lean_ctor_set(v___x_1503_, 1, v___x_1502_);
v___x_1504_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1504_, 0, v___x_1503_);
lean_ctor_set(v___x_1504_, 1, v___x_1495_);
if (v_isShared_1501_ == 0)
{
lean_ctor_set(v___x_1500_, 0, v___x_1504_);
v___x_1506_ = v___x_1500_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v___x_1504_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
return v___x_1506_;
}
}
}
else
{
lean_object* v_a_1509_; lean_object* v___x_1511_; uint8_t v_isShared_1512_; uint8_t v_isSharedCheck_1516_; 
lean_dec(v_g_1484_);
lean_dec(v_a_1482_);
v_a_1509_ = lean_ctor_get(v___x_1497_, 0);
v_isSharedCheck_1516_ = !lean_is_exclusive(v___x_1497_);
if (v_isSharedCheck_1516_ == 0)
{
v___x_1511_ = v___x_1497_;
v_isShared_1512_ = v_isSharedCheck_1516_;
goto v_resetjp_1510_;
}
else
{
lean_inc(v_a_1509_);
lean_dec(v___x_1497_);
v___x_1511_ = lean_box(0);
v_isShared_1512_ = v_isSharedCheck_1516_;
goto v_resetjp_1510_;
}
v_resetjp_1510_:
{
lean_object* v___x_1514_; 
if (v_isShared_1512_ == 0)
{
v___x_1514_ = v___x_1511_;
goto v_reusejp_1513_;
}
else
{
lean_object* v_reuseFailAlloc_1515_; 
v_reuseFailAlloc_1515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1515_, 0, v_a_1509_);
v___x_1514_ = v_reuseFailAlloc_1515_;
goto v_reusejp_1513_;
}
v_reusejp_1513_:
{
return v___x_1514_;
}
}
}
}
else
{
lean_object* v_a_1517_; lean_object* v___x_1519_; uint8_t v_isShared_1520_; uint8_t v_isSharedCheck_1524_; 
lean_dec(v_g_1484_);
lean_dec(v___x_1483_);
lean_dec(v_a_1482_);
v_a_1517_ = lean_ctor_get(v___x_1492_, 0);
v_isSharedCheck_1524_ = !lean_is_exclusive(v___x_1492_);
if (v_isSharedCheck_1524_ == 0)
{
v___x_1519_ = v___x_1492_;
v_isShared_1520_ = v_isSharedCheck_1524_;
goto v_resetjp_1518_;
}
else
{
lean_inc(v_a_1517_);
lean_dec(v___x_1492_);
v___x_1519_ = lean_box(0);
v_isShared_1520_ = v_isSharedCheck_1524_;
goto v_resetjp_1518_;
}
v_resetjp_1518_:
{
lean_object* v___x_1522_; 
if (v_isShared_1520_ == 0)
{
v___x_1522_ = v___x_1519_;
goto v_reusejp_1521_;
}
else
{
lean_object* v_reuseFailAlloc_1523_; 
v_reuseFailAlloc_1523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1523_, 0, v_a_1517_);
v___x_1522_ = v_reuseFailAlloc_1523_;
goto v_reusejp_1521_;
}
v_reusejp_1521_:
{
return v___x_1522_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1___boxed(lean_object* v___x_1525_, lean_object* v_a_1526_, lean_object* v___x_1527_, lean_object* v_g_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_){
_start:
{
lean_object* v_res_1536_; 
v_res_1536_ = lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1(v___x_1525_, v_a_1526_, v___x_1527_, v_g_1528_, v___y_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_, v___y_1534_);
lean_dec(v___y_1534_);
lean_dec_ref(v___y_1533_);
lean_dec(v___y_1532_);
lean_dec_ref(v___y_1531_);
lean_dec(v___y_1530_);
lean_dec_ref(v___y_1529_);
return v_res_1536_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0(lean_object* v_x_1537_){
_start:
{
uint8_t v___x_1538_; 
v___x_1538_ = 0;
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0___boxed(lean_object* v_x_1539_){
_start:
{
uint8_t v_res_1540_; lean_object* v_r_1541_; 
v_res_1540_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___lam__0(v_x_1539_);
lean_dec(v_x_1539_);
v_r_1541_ = lean_box(v_res_1540_);
return v_r_1541_;
}
}
static lean_object* _init_lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4(void){
_start:
{
lean_object* v___x_1549_; lean_object* v___x_1550_; 
v___x_1549_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__3));
v___x_1550_ = l_Lean_stringToMessageData(v___x_1549_);
return v___x_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4(lean_object* v_g_1551_, lean_object* v_x_1552_, lean_object* v_x_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_){
_start:
{
if (lean_obj_tag(v_x_1552_) == 0)
{
lean_object* v___x_1559_; lean_object* v___x_1560_; 
lean_dec(v_g_1551_);
v___x_1559_ = l_List_reverse___redArg(v_x_1553_);
v___x_1560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1560_, 0, v___x_1559_);
return v___x_1560_;
}
else
{
lean_object* v_head_1561_; lean_object* v_tail_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1654_; 
v_head_1561_ = lean_ctor_get(v_x_1552_, 0);
v_tail_1562_ = lean_ctor_get(v_x_1552_, 1);
v_isSharedCheck_1654_ = !lean_is_exclusive(v_x_1552_);
if (v_isSharedCheck_1654_ == 0)
{
v___x_1564_ = v_x_1552_;
v_isShared_1565_ = v_isSharedCheck_1654_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_tail_1562_);
lean_inc(v_head_1561_);
lean_dec(v_x_1552_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1654_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v_a_1567_; lean_object* v___y_1573_; lean_object* v___x_1583_; 
lean_inc(v___y_1557_);
lean_inc_ref(v___y_1556_);
lean_inc(v___y_1555_);
lean_inc_ref(v___y_1554_);
lean_inc(v_head_1561_);
v___x_1583_ = lean_infer_type(v_head_1561_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
if (lean_obj_tag(v___x_1583_) == 0)
{
lean_object* v_a_1584_; lean_object* v___x_1585_; lean_object* v_a_1586_; lean_object* v___x_1587_; 
v_a_1584_ = lean_ctor_get(v___x_1583_, 0);
lean_inc(v_a_1584_);
lean_dec_ref_known(v___x_1583_, 1);
v___x_1585_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_a_1584_, v___y_1555_);
v_a_1586_ = lean_ctor_get(v___x_1585_, 0);
lean_inc(v_a_1586_);
lean_dec_ref(v___x_1585_);
v___x_1587_ = l_Lean_Meta_whnfR(v_a_1586_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
if (lean_obj_tag(v___x_1587_) == 0)
{
lean_object* v_a_1588_; lean_object* v___x_1589_; 
v_a_1588_ = lean_ctor_get(v___x_1587_, 0);
lean_inc_n(v_a_1588_, 2);
lean_dec_ref_known(v___x_1587_, 1);
v___x_1589_ = lp_mathlib_Mathlib_Tactic_Linarith_isNatProp(v_a_1588_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
if (lean_obj_tag(v___x_1589_) == 0)
{
lean_object* v_a_1590_; uint8_t v___x_1591_; 
v_a_1590_ = lean_ctor_get(v___x_1589_, 0);
lean_inc(v_a_1590_);
lean_dec_ref_known(v___x_1589_, 1);
v___x_1591_ = lean_unbox(v_a_1590_);
if (v___x_1591_ == 0)
{
lean_dec(v_a_1590_);
lean_dec(v_a_1588_);
v_a_1567_ = v_head_1561_;
goto v___jp_1566_;
}
else
{
lean_object* v___f_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; uint8_t v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; uint8_t v___x_1601_; uint8_t v___x_1602_; uint8_t v___x_1603_; uint8_t v___x_1604_; uint8_t v___x_1605_; uint8_t v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; 
v___f_1592_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__0));
v___x_1593_ = lean_box(0);
lean_inc(v_head_1561_);
v___x_1594_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed), 12, 3);
lean_closure_set(v___x_1594_, 0, v___x_1593_);
lean_closure_set(v___x_1594_, 1, v_head_1561_);
lean_closure_set(v___x_1594_, 2, v_a_1588_);
lean_inc(v_g_1551_);
v___x_1595_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_run__for___boxed), 10, 3);
lean_closure_set(v___x_1595_, 0, lean_box(0));
lean_closure_set(v___x_1595_, 1, v_g_1551_);
lean_closure_set(v___x_1595_, 2, v___x_1594_);
v___x_1596_ = lean_box(0);
v___x_1597_ = lean_box(1);
v___x_1598_ = 0;
v___x_1599_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__1));
v___x_1600_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1600_, 0, v___x_1593_);
lean_ctor_set(v___x_1600_, 1, v___x_1596_);
lean_ctor_set(v___x_1600_, 2, v___x_1593_);
lean_ctor_set(v___x_1600_, 3, v___f_1592_);
lean_ctor_set(v___x_1600_, 4, v___x_1597_);
lean_ctor_set(v___x_1600_, 5, v___x_1597_);
lean_ctor_set(v___x_1600_, 6, v___x_1593_);
lean_ctor_set(v___x_1600_, 7, v___x_1599_);
v___x_1601_ = lean_unbox(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8, v___x_1601_);
v___x_1602_ = lean_unbox(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 1, v___x_1602_);
v___x_1603_ = lean_unbox(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 2, v___x_1603_);
v___x_1604_ = lean_unbox(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 3, v___x_1604_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 4, v___x_1598_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 5, v___x_1598_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 6, v___x_1598_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 7, v___x_1598_);
v___x_1605_ = lean_unbox(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 8, v___x_1605_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 9, v___x_1598_);
v___x_1606_ = lean_unbox(v_a_1590_);
lean_dec(v_a_1590_);
lean_ctor_set_uint8(v___x_1600_, sizeof(void*)*8 + 10, v___x_1606_);
v___x_1607_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__2));
v___x_1608_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_1595_, v___x_1600_, v___x_1607_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v_a_1609_; lean_object* v_fst_1610_; lean_object* v_fst_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1636_; 
v_a_1609_ = lean_ctor_get(v___x_1608_, 0);
lean_inc(v_a_1609_);
lean_dec_ref_known(v___x_1608_, 1);
v_fst_1610_ = lean_ctor_get(v_a_1609_, 0);
lean_inc(v_fst_1610_);
lean_dec(v_a_1609_);
v_fst_1611_ = lean_ctor_get(v_fst_1610_, 0);
v_isSharedCheck_1636_ = !lean_is_exclusive(v_fst_1610_);
if (v_isSharedCheck_1636_ == 0)
{
lean_object* v_unused_1637_; 
v_unused_1637_ = lean_ctor_get(v_fst_1610_, 1);
lean_dec(v_unused_1637_);
v___x_1613_ = v_fst_1610_;
v_isShared_1614_ = v_isSharedCheck_1636_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_fst_1611_);
lean_dec(v_fst_1610_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1636_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
if (lean_obj_tag(v_fst_1611_) == 1)
{
lean_object* v_val_1615_; lean_object* v_fst_1616_; lean_object* v_snd_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; 
lean_del_object(v___x_1613_);
v_val_1615_ = lean_ctor_get(v_fst_1611_, 0);
lean_inc(v_val_1615_);
lean_dec_ref_known(v_fst_1611_, 1);
v_fst_1616_ = lean_ctor_get(v_val_1615_, 0);
lean_inc(v_fst_1616_);
v_snd_1617_ = lean_ctor_get(v_val_1615_, 1);
lean_inc(v_snd_1617_);
lean_dec(v_val_1615_);
v___x_1618_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___boxed), 6, 1);
lean_closure_set(v___x_1618_, 0, v_snd_1617_);
v___x_1619_ = lp_mathlib_succeeds___at___00Mathlib_Tactic_Linarith_isNatProp_spec__1___redArg(v___x_1618_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
if (lean_obj_tag(v___x_1619_) == 0)
{
lean_object* v_a_1620_; uint8_t v___x_1621_; 
v_a_1620_ = lean_ctor_get(v___x_1619_, 0);
lean_inc(v_a_1620_);
lean_dec_ref_known(v___x_1619_, 1);
v___x_1621_ = lean_unbox(v_a_1620_);
lean_dec(v_a_1620_);
if (v___x_1621_ == 0)
{
lean_dec(v_fst_1616_);
v_a_1567_ = v_head_1561_;
goto v___jp_1566_;
}
else
{
lean_dec(v_head_1561_);
v_a_1567_ = v_fst_1616_;
goto v___jp_1566_;
}
}
else
{
lean_object* v_a_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1629_; 
lean_dec(v_fst_1616_);
lean_del_object(v___x_1564_);
lean_dec(v_tail_1562_);
lean_dec(v_head_1561_);
lean_dec(v_x_1553_);
lean_dec(v_g_1551_);
v_a_1622_ = lean_ctor_get(v___x_1619_, 0);
v_isSharedCheck_1629_ = !lean_is_exclusive(v___x_1619_);
if (v_isSharedCheck_1629_ == 0)
{
v___x_1624_ = v___x_1619_;
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_a_1622_);
lean_dec(v___x_1619_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1627_; 
if (v_isShared_1625_ == 0)
{
v___x_1627_ = v___x_1624_;
goto v_reusejp_1626_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v_a_1622_);
v___x_1627_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1626_;
}
v_reusejp_1626_:
{
return v___x_1627_;
}
}
}
}
else
{
lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1633_; 
lean_dec(v_fst_1611_);
v___x_1630_ = lean_obj_once(&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4, &lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4_once, _init_lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__4);
v___x_1631_ = l_Lean_MessageData_ofExpr(v_head_1561_);
if (v_isShared_1614_ == 0)
{
lean_ctor_set_tag(v___x_1613_, 7);
lean_ctor_set(v___x_1613_, 1, v___x_1631_);
lean_ctor_set(v___x_1613_, 0, v___x_1630_);
v___x_1633_ = v___x_1613_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1635_; 
v_reuseFailAlloc_1635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1635_, 0, v___x_1630_);
lean_ctor_set(v_reuseFailAlloc_1635_, 1, v___x_1631_);
v___x_1633_ = v_reuseFailAlloc_1635_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
lean_object* v___x_1634_; 
v___x_1634_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v___x_1633_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_);
v___y_1573_ = v___x_1634_;
goto v___jp_1572_;
}
}
}
}
else
{
lean_object* v_a_1638_; lean_object* v___x_1640_; uint8_t v_isShared_1641_; uint8_t v_isSharedCheck_1645_; 
lean_del_object(v___x_1564_);
lean_dec(v_tail_1562_);
lean_dec(v_head_1561_);
lean_dec(v_x_1553_);
lean_dec(v_g_1551_);
v_a_1638_ = lean_ctor_get(v___x_1608_, 0);
v_isSharedCheck_1645_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1645_ == 0)
{
v___x_1640_ = v___x_1608_;
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_a_1638_);
lean_dec(v___x_1608_);
v___x_1640_ = lean_box(0);
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
v_resetjp_1639_:
{
lean_object* v___x_1643_; 
if (v_isShared_1641_ == 0)
{
v___x_1643_ = v___x_1640_;
goto v_reusejp_1642_;
}
else
{
lean_object* v_reuseFailAlloc_1644_; 
v_reuseFailAlloc_1644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1644_, 0, v_a_1638_);
v___x_1643_ = v_reuseFailAlloc_1644_;
goto v_reusejp_1642_;
}
v_reusejp_1642_:
{
return v___x_1643_;
}
}
}
}
}
else
{
lean_object* v_a_1646_; lean_object* v___x_1648_; uint8_t v_isShared_1649_; uint8_t v_isSharedCheck_1653_; 
lean_dec(v_a_1588_);
lean_del_object(v___x_1564_);
lean_dec(v_tail_1562_);
lean_dec(v_head_1561_);
lean_dec(v_x_1553_);
lean_dec(v_g_1551_);
v_a_1646_ = lean_ctor_get(v___x_1589_, 0);
v_isSharedCheck_1653_ = !lean_is_exclusive(v___x_1589_);
if (v_isSharedCheck_1653_ == 0)
{
v___x_1648_ = v___x_1589_;
v_isShared_1649_ = v_isSharedCheck_1653_;
goto v_resetjp_1647_;
}
else
{
lean_inc(v_a_1646_);
lean_dec(v___x_1589_);
v___x_1648_ = lean_box(0);
v_isShared_1649_ = v_isSharedCheck_1653_;
goto v_resetjp_1647_;
}
v_resetjp_1647_:
{
lean_object* v___x_1651_; 
if (v_isShared_1649_ == 0)
{
v___x_1651_ = v___x_1648_;
goto v_reusejp_1650_;
}
else
{
lean_object* v_reuseFailAlloc_1652_; 
v_reuseFailAlloc_1652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1652_, 0, v_a_1646_);
v___x_1651_ = v_reuseFailAlloc_1652_;
goto v_reusejp_1650_;
}
v_reusejp_1650_:
{
return v___x_1651_;
}
}
}
}
else
{
lean_dec(v_head_1561_);
v___y_1573_ = v___x_1587_;
goto v___jp_1572_;
}
}
else
{
lean_dec(v_head_1561_);
v___y_1573_ = v___x_1583_;
goto v___jp_1572_;
}
v___jp_1566_:
{
lean_object* v___x_1569_; 
if (v_isShared_1565_ == 0)
{
lean_ctor_set(v___x_1564_, 1, v_x_1553_);
lean_ctor_set(v___x_1564_, 0, v_a_1567_);
v___x_1569_ = v___x_1564_;
goto v_reusejp_1568_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v_a_1567_);
lean_ctor_set(v_reuseFailAlloc_1571_, 1, v_x_1553_);
v___x_1569_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1568_;
}
v_reusejp_1568_:
{
v_x_1552_ = v_tail_1562_;
v_x_1553_ = v___x_1569_;
goto _start;
}
}
v___jp_1572_:
{
if (lean_obj_tag(v___y_1573_) == 0)
{
lean_object* v_a_1574_; 
v_a_1574_ = lean_ctor_get(v___y_1573_, 0);
lean_inc(v_a_1574_);
lean_dec_ref_known(v___y_1573_, 1);
v_a_1567_ = v_a_1574_;
goto v___jp_1566_;
}
else
{
lean_object* v_a_1575_; lean_object* v___x_1577_; uint8_t v_isShared_1578_; uint8_t v_isSharedCheck_1582_; 
lean_del_object(v___x_1564_);
lean_dec(v_tail_1562_);
lean_dec(v_x_1553_);
lean_dec(v_g_1551_);
v_a_1575_ = lean_ctor_get(v___y_1573_, 0);
v_isSharedCheck_1582_ = !lean_is_exclusive(v___y_1573_);
if (v_isSharedCheck_1582_ == 0)
{
v___x_1577_ = v___y_1573_;
v_isShared_1578_ = v_isSharedCheck_1582_;
goto v_resetjp_1576_;
}
else
{
lean_inc(v_a_1575_);
lean_dec(v___y_1573_);
v___x_1577_ = lean_box(0);
v_isShared_1578_ = v_isSharedCheck_1582_;
goto v_resetjp_1576_;
}
v_resetjp_1576_:
{
lean_object* v___x_1580_; 
if (v_isShared_1578_ == 0)
{
v___x_1580_ = v___x_1577_;
goto v_reusejp_1579_;
}
else
{
lean_object* v_reuseFailAlloc_1581_; 
v_reuseFailAlloc_1581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1581_, 0, v_a_1575_);
v___x_1580_ = v_reuseFailAlloc_1581_;
goto v_reusejp_1579_;
}
v_reusejp_1579_:
{
return v___x_1580_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___boxed(lean_object* v_g_1655_, lean_object* v_x_1656_, lean_object* v_x_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_){
_start:
{
lean_object* v_res_1663_; 
v_res_1663_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4(v_g_1655_, v_x_1656_, v_x_1657_, v___y_1658_, v___y_1659_, v___y_1660_, v___y_1661_);
lean_dec(v___y_1661_);
lean_dec_ref(v___y_1660_);
lean_dec(v___y_1659_);
lean_dec_ref(v___y_1658_);
return v_res_1663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2(lean_object* v___f_1664_, lean_object* v_g_1665_, lean_object* v_l_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_){
_start:
{
lean_object* v___x_1672_; lean_object* v___x_1673_; 
v___x_1672_ = lean_box(0);
lean_inc(v_g_1665_);
v___x_1673_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4(v_g_1665_, v_l_1666_, v___x_1672_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_);
if (lean_obj_tag(v___x_1673_) == 0)
{
lean_object* v_a_1674_; uint8_t v___x_1675_; lean_object* v___x_1676_; lean_object* v___f_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; uint8_t v___x_1680_; lean_object* v___x_1681_; 
v_a_1674_ = lean_ctor_get(v___x_1673_, 0);
lean_inc(v_a_1674_);
lean_dec_ref_known(v___x_1673_, 1);
v___x_1675_ = 2;
v___x_1676_ = lean_box(1);
v___f_1677_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__1___boxed), 11, 4);
lean_closure_set(v___f_1677_, 0, v___x_1676_);
lean_closure_set(v___f_1677_, 1, v_a_1674_);
lean_closure_set(v___f_1677_, 2, v___x_1672_);
lean_closure_set(v___f_1677_, 3, v_g_1665_);
v___x_1678_ = lean_box(v___x_1675_);
v___x_1679_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_AtomM_run___boxed), 9, 4);
lean_closure_set(v___x_1679_, 0, lean_box(0));
lean_closure_set(v___x_1679_, 1, v___x_1678_);
lean_closure_set(v___x_1679_, 2, v___f_1677_);
lean_closure_set(v___x_1679_, 3, v___f_1664_);
v___x_1680_ = 0;
v___x_1681_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Linarith_natToInt_spec__8___redArg(v___x_1679_, v___x_1680_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_);
return v___x_1681_;
}
else
{
lean_object* v_a_1682_; lean_object* v___x_1684_; uint8_t v_isShared_1685_; uint8_t v_isSharedCheck_1689_; 
lean_dec(v_g_1665_);
lean_dec_ref(v___f_1664_);
v_a_1682_ = lean_ctor_get(v___x_1673_, 0);
v_isSharedCheck_1689_ = !lean_is_exclusive(v___x_1673_);
if (v_isSharedCheck_1689_ == 0)
{
v___x_1684_ = v___x_1673_;
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
else
{
lean_inc(v_a_1682_);
lean_dec(v___x_1673_);
v___x_1684_ = lean_box(0);
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
v_resetjp_1683_:
{
lean_object* v___x_1687_; 
if (v_isShared_1685_ == 0)
{
v___x_1687_ = v___x_1684_;
goto v_reusejp_1686_;
}
else
{
lean_object* v_reuseFailAlloc_1688_; 
v_reuseFailAlloc_1688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1688_, 0, v_a_1682_);
v___x_1687_ = v_reuseFailAlloc_1688_;
goto v_reusejp_1686_;
}
v_reusejp_1686_:
{
return v___x_1687_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2___boxed(lean_object* v___f_1690_, lean_object* v_g_1691_, lean_object* v_l_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_){
_start:
{
lean_object* v_res_1698_; 
v_res_1698_ = lp_mathlib_Mathlib_Tactic_Linarith_natToInt___lam__2(v___f_1690_, v_g_1691_, v_l_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_);
lean_dec(v___y_1696_);
lean_dec_ref(v___y_1695_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
return v_res_1698_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1(lean_object* v_00_u03b2_1716_, lean_object* v_k_1717_, lean_object* v_t_1718_){
_start:
{
uint8_t v___x_1719_; 
v___x_1719_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___redArg(v_k_1717_, v_t_1718_);
return v___x_1719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1___boxed(lean_object* v_00_u03b2_1720_, lean_object* v_k_1721_, lean_object* v_t_1722_){
_start:
{
uint8_t v_res_1723_; lean_object* v_r_1724_; 
v_res_1723_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_natToInt_spec__1(v_00_u03b2_1720_, v_k_1721_, v_t_1722_);
lean_dec(v_t_1722_);
lean_dec_ref(v_k_1721_);
v_r_1724_ = lean_box(v_res_1723_);
return v_r_1724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2(lean_object* v_00_u03b2_1725_, lean_object* v_k_1726_, lean_object* v_v_1727_, lean_object* v_t_1728_, lean_object* v_hl_1729_){
_start:
{
lean_object* v___x_1730_; 
v___x_1730_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_natToInt_spec__2___redArg(v_k_1726_, v_v_1727_, v_t_1728_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3(lean_object* v_as_1731_, lean_object* v_as_x27_1732_, lean_object* v_b_1733_, lean_object* v_a_1734_){
_start:
{
lean_object* v___x_1735_; 
v___x_1735_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___redArg(v_as_x27_1732_, v_b_1733_);
return v___x_1735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3___boxed(lean_object* v_as_1736_, lean_object* v_as_x27_1737_, lean_object* v_b_1738_, lean_object* v_a_1739_){
_start:
{
lean_object* v_res_1740_; 
v_res_1740_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__3(v_as_1736_, v_as_x27_1737_, v_b_1738_, v_a_1739_);
lean_dec(v_as_x27_1737_);
lean_dec(v_as_1736_);
return v_res_1740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7(lean_object* v___x_1741_, lean_object* v_x_1742_, lean_object* v_x_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_){
_start:
{
lean_object* v___x_1751_; 
v___x_1751_ = lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___redArg(v___x_1741_, v_x_1742_, v_x_1743_, v___y_1746_, v___y_1747_, v___y_1748_, v___y_1749_);
return v___x_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7___boxed(lean_object* v___x_1752_, lean_object* v_x_1753_, lean_object* v_x_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_){
_start:
{
lean_object* v_res_1762_; 
v_res_1762_ = lp_mathlib_List_filterMapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__7(v___x_1752_, v_x_1753_, v_x_1754_, v___y_1755_, v___y_1756_, v___y_1757_, v___y_1758_, v___y_1759_, v___y_1760_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
lean_dec(v___y_1758_);
lean_dec_ref(v___y_1757_);
lean_dec(v___y_1756_);
lean_dec_ref(v___y_1755_);
lean_dec_ref(v___x_1752_);
return v_res_1762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f(lean_object* v_pf_1773_, lean_object* v_a_1774_, lean_object* v_a_1775_, lean_object* v_a_1776_, lean_object* v_a_1777_){
_start:
{
lean_object* v___x_1782_; 
lean_inc(v_a_1777_);
lean_inc_ref(v_a_1776_);
lean_inc(v_a_1775_);
lean_inc_ref(v_a_1774_);
lean_inc_ref(v_pf_1773_);
v___x_1782_ = lean_infer_type(v_pf_1773_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1782_) == 0)
{
lean_object* v_a_1783_; lean_object* v___x_1784_; 
v_a_1783_ = lean_ctor_get(v___x_1782_, 0);
lean_inc(v_a_1783_);
lean_dec_ref_known(v___x_1782_, 1);
v___x_1784_ = lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(v_a_1783_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1784_) == 0)
{
lean_object* v_a_1785_; lean_object* v_fst_1786_; uint8_t v___x_1787_; 
v_a_1785_ = lean_ctor_get(v___x_1784_, 0);
lean_inc(v_a_1785_);
lean_dec_ref_known(v___x_1784_, 1);
v_fst_1786_ = lean_ctor_get(v_a_1785_, 0);
v___x_1787_ = lean_unbox(v_fst_1786_);
if (v___x_1787_ == 0)
{
lean_object* v_snd_1788_; lean_object* v_fst_1789_; uint8_t v___x_1790_; 
v_snd_1788_ = lean_ctor_get(v_a_1785_, 1);
lean_inc(v_snd_1788_);
lean_dec(v_a_1785_);
v_fst_1789_ = lean_ctor_get(v_snd_1788_, 0);
v___x_1790_ = lean_unbox(v_fst_1789_);
if (v___x_1790_ == 1)
{
lean_object* v_snd_1791_; lean_object* v_fst_1792_; 
v_snd_1791_ = lean_ctor_get(v_snd_1788_, 1);
lean_inc(v_snd_1791_);
lean_dec(v_snd_1788_);
v_fst_1792_ = lean_ctor_get(v_snd_1791_, 0);
lean_inc(v_fst_1792_);
if (lean_obj_tag(v_fst_1792_) == 4)
{
lean_object* v_declName_1793_; 
v_declName_1793_ = lean_ctor_get(v_fst_1792_, 0);
lean_inc(v_declName_1793_);
if (lean_obj_tag(v_declName_1793_) == 1)
{
lean_object* v_pre_1794_; 
v_pre_1794_ = lean_ctor_get(v_declName_1793_, 0);
if (lean_obj_tag(v_pre_1794_) == 0)
{
lean_object* v_snd_1795_; lean_object* v_us_1796_; lean_object* v_str_1797_; lean_object* v___x_1798_; uint8_t v___x_1799_; 
v_snd_1795_ = lean_ctor_get(v_snd_1791_, 1);
lean_inc(v_snd_1795_);
lean_dec(v_snd_1791_);
v_us_1796_ = lean_ctor_get(v_fst_1792_, 1);
lean_inc(v_us_1796_);
lean_dec_ref_known(v_fst_1792_, 2);
v_str_1797_ = lean_ctor_get(v_declName_1793_, 1);
lean_inc_ref(v_str_1797_);
lean_dec_ref_known(v_declName_1793_, 2);
v___x_1798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0));
v___x_1799_ = lean_string_dec_eq(v_str_1797_, v___x_1798_);
lean_dec_ref(v_str_1797_);
if (v___x_1799_ == 0)
{
lean_dec(v_us_1796_);
lean_dec(v_snd_1795_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
else
{
if (lean_obj_tag(v_us_1796_) == 0)
{
lean_object* v_fst_1800_; lean_object* v_snd_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; 
v_fst_1800_ = lean_ctor_get(v_snd_1795_, 0);
lean_inc(v_fst_1800_);
v_snd_1801_ = lean_ctor_get(v_snd_1795_, 1);
lean_inc(v_snd_1801_);
lean_dec(v_snd_1795_);
v___x_1802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2));
v___x_1803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1803_, 0, v_snd_1801_);
v___x_1804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1804_, 0, v_fst_1800_);
v___x_1805_ = lean_unsigned_to_nat(2u);
v___x_1806_ = lean_mk_empty_array_with_capacity(v___x_1805_);
v___x_1807_ = lean_array_push(v___x_1806_, v___x_1803_);
v___x_1808_ = lean_array_push(v___x_1807_, v___x_1804_);
v___x_1809_ = l_Lean_Meta_mkAppOptM(v___x_1802_, v___x_1808_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1809_) == 0)
{
lean_object* v_a_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; 
v_a_1810_ = lean_ctor_get(v___x_1809_, 0);
lean_inc(v_a_1810_);
lean_dec_ref_known(v___x_1809_, 1);
v___x_1811_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5));
v___x_1812_ = lean_unsigned_to_nat(1u);
v___x_1813_ = lean_mk_empty_array_with_capacity(v___x_1812_);
lean_inc_ref(v___x_1813_);
v___x_1814_ = lean_array_push(v___x_1813_, v_a_1810_);
v___x_1815_ = l_Lean_Meta_mkAppM(v___x_1811_, v___x_1814_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1815_) == 0)
{
lean_object* v_a_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; 
v_a_1816_ = lean_ctor_get(v___x_1815_, 0);
lean_inc(v_a_1816_);
lean_dec_ref_known(v___x_1815_, 1);
v___x_1817_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons___lam__0___closed__3));
v___x_1818_ = lean_array_push(v___x_1813_, v_pf_1773_);
v___x_1819_ = l_Lean_Meta_mkAppM(v___x_1817_, v___x_1818_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1819_) == 0)
{
lean_object* v_a_1820_; lean_object* v___x_1822_; uint8_t v_isShared_1823_; uint8_t v_isSharedCheck_1829_; 
v_a_1820_ = lean_ctor_get(v___x_1819_, 0);
v_isSharedCheck_1829_ = !lean_is_exclusive(v___x_1819_);
if (v_isSharedCheck_1829_ == 0)
{
v___x_1822_ = v___x_1819_;
v_isShared_1823_ = v_isSharedCheck_1829_;
goto v_resetjp_1821_;
}
else
{
lean_inc(v_a_1820_);
lean_dec(v___x_1819_);
v___x_1822_ = lean_box(0);
v_isShared_1823_ = v_isSharedCheck_1829_;
goto v_resetjp_1821_;
}
v_resetjp_1821_:
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1827_; 
v___x_1824_ = l_Lean_Expr_app___override(v_a_1816_, v_a_1820_);
v___x_1825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1825_, 0, v___x_1824_);
if (v_isShared_1823_ == 0)
{
lean_ctor_set(v___x_1822_, 0, v___x_1825_);
v___x_1827_ = v___x_1822_;
goto v_reusejp_1826_;
}
else
{
lean_object* v_reuseFailAlloc_1828_; 
v_reuseFailAlloc_1828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1828_, 0, v___x_1825_);
v___x_1827_ = v_reuseFailAlloc_1828_;
goto v_reusejp_1826_;
}
v_reusejp_1826_:
{
return v___x_1827_;
}
}
}
else
{
lean_object* v_a_1830_; lean_object* v___x_1832_; uint8_t v_isShared_1833_; uint8_t v_isSharedCheck_1837_; 
lean_dec(v_a_1816_);
v_a_1830_ = lean_ctor_get(v___x_1819_, 0);
v_isSharedCheck_1837_ = !lean_is_exclusive(v___x_1819_);
if (v_isSharedCheck_1837_ == 0)
{
v___x_1832_ = v___x_1819_;
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
else
{
lean_inc(v_a_1830_);
lean_dec(v___x_1819_);
v___x_1832_ = lean_box(0);
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
v_resetjp_1831_:
{
lean_object* v___x_1835_; 
if (v_isShared_1833_ == 0)
{
v___x_1835_ = v___x_1832_;
goto v_reusejp_1834_;
}
else
{
lean_object* v_reuseFailAlloc_1836_; 
v_reuseFailAlloc_1836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1836_, 0, v_a_1830_);
v___x_1835_ = v_reuseFailAlloc_1836_;
goto v_reusejp_1834_;
}
v_reusejp_1834_:
{
return v___x_1835_;
}
}
}
}
else
{
lean_object* v_a_1838_; lean_object* v___x_1840_; uint8_t v_isShared_1841_; uint8_t v_isSharedCheck_1845_; 
lean_dec_ref(v___x_1813_);
lean_dec_ref(v_pf_1773_);
v_a_1838_ = lean_ctor_get(v___x_1815_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1815_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1840_ = v___x_1815_;
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
else
{
lean_inc(v_a_1838_);
lean_dec(v___x_1815_);
v___x_1840_ = lean_box(0);
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
v_resetjp_1839_:
{
lean_object* v___x_1843_; 
if (v_isShared_1841_ == 0)
{
v___x_1843_ = v___x_1840_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v_a_1838_);
v___x_1843_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
return v___x_1843_;
}
}
}
}
else
{
lean_object* v_a_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1853_; 
lean_dec_ref(v_pf_1773_);
v_a_1846_ = lean_ctor_get(v___x_1809_, 0);
v_isSharedCheck_1853_ = !lean_is_exclusive(v___x_1809_);
if (v_isSharedCheck_1853_ == 0)
{
v___x_1848_ = v___x_1809_;
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_a_1846_);
lean_dec(v___x_1809_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v___x_1851_; 
if (v_isShared_1849_ == 0)
{
v___x_1851_ = v___x_1848_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v_a_1846_);
v___x_1851_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
return v___x_1851_;
}
}
}
}
else
{
lean_dec(v_us_1796_);
lean_dec(v_snd_1795_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
}
else
{
lean_dec_ref_known(v_declName_1793_, 2);
lean_dec_ref_known(v_fst_1792_, 2);
lean_dec(v_snd_1791_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec_ref_known(v_fst_1792_, 2);
lean_dec(v_declName_1793_);
lean_dec(v_snd_1791_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec(v_fst_1792_);
lean_dec(v_snd_1791_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec(v_snd_1788_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_object* v_snd_1854_; lean_object* v_fst_1855_; uint8_t v___x_1856_; 
v_snd_1854_ = lean_ctor_get(v_a_1785_, 1);
lean_inc(v_snd_1854_);
lean_dec(v_a_1785_);
v_fst_1855_ = lean_ctor_get(v_snd_1854_, 0);
v___x_1856_ = lean_unbox(v_fst_1855_);
if (v___x_1856_ == 2)
{
lean_object* v_snd_1857_; lean_object* v_fst_1858_; 
v_snd_1857_ = lean_ctor_get(v_snd_1854_, 1);
lean_inc(v_snd_1857_);
lean_dec(v_snd_1854_);
v_fst_1858_ = lean_ctor_get(v_snd_1857_, 0);
lean_inc(v_fst_1858_);
if (lean_obj_tag(v_fst_1858_) == 4)
{
lean_object* v_declName_1859_; 
v_declName_1859_ = lean_ctor_get(v_fst_1858_, 0);
lean_inc(v_declName_1859_);
if (lean_obj_tag(v_declName_1859_) == 1)
{
lean_object* v_pre_1860_; 
v_pre_1860_ = lean_ctor_get(v_declName_1859_, 0);
if (lean_obj_tag(v_pre_1860_) == 0)
{
lean_object* v_snd_1861_; lean_object* v_us_1862_; lean_object* v_str_1863_; lean_object* v___x_1864_; uint8_t v___x_1865_; 
v_snd_1861_ = lean_ctor_get(v_snd_1857_, 1);
lean_inc(v_snd_1861_);
lean_dec(v_snd_1857_);
v_us_1862_ = lean_ctor_get(v_fst_1858_, 1);
lean_inc(v_us_1862_);
lean_dec_ref_known(v_fst_1858_, 2);
v_str_1863_ = lean_ctor_get(v_declName_1859_, 1);
lean_inc_ref(v_str_1863_);
lean_dec_ref_known(v_declName_1859_, 2);
v___x_1864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__0));
v___x_1865_ = lean_string_dec_eq(v_str_1863_, v___x_1864_);
lean_dec_ref(v_str_1863_);
if (v___x_1865_ == 0)
{
lean_dec(v_us_1862_);
lean_dec(v_snd_1861_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
else
{
if (lean_obj_tag(v_us_1862_) == 0)
{
lean_object* v_fst_1866_; lean_object* v_snd_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; 
v_fst_1866_ = lean_ctor_get(v_snd_1861_, 0);
lean_inc(v_fst_1866_);
v_snd_1867_ = lean_ctor_get(v_snd_1861_, 1);
lean_inc(v_snd_1867_);
lean_dec(v_snd_1861_);
v___x_1868_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__2));
v___x_1869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1869_, 0, v_fst_1866_);
v___x_1870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1870_, 0, v_snd_1867_);
v___x_1871_ = lean_unsigned_to_nat(2u);
v___x_1872_ = lean_mk_empty_array_with_capacity(v___x_1871_);
v___x_1873_ = lean_array_push(v___x_1872_, v___x_1869_);
v___x_1874_ = lean_array_push(v___x_1873_, v___x_1870_);
v___x_1875_ = l_Lean_Meta_mkAppOptM(v___x_1868_, v___x_1874_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1875_) == 0)
{
lean_object* v_a_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v_a_1876_ = lean_ctor_get(v___x_1875_, 0);
lean_inc(v_a_1876_);
lean_dec_ref_known(v___x_1875_, 1);
v___x_1877_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___closed__5));
v___x_1878_ = lean_unsigned_to_nat(1u);
v___x_1879_ = lean_mk_empty_array_with_capacity(v___x_1878_);
v___x_1880_ = lean_array_push(v___x_1879_, v_a_1876_);
v___x_1881_ = l_Lean_Meta_mkAppM(v___x_1877_, v___x_1880_, v_a_1774_, v_a_1775_, v_a_1776_, v_a_1777_);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_object* v_a_1882_; lean_object* v___x_1884_; uint8_t v_isShared_1885_; uint8_t v_isSharedCheck_1891_; 
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1884_ = v___x_1881_;
v_isShared_1885_ = v_isSharedCheck_1891_;
goto v_resetjp_1883_;
}
else
{
lean_inc(v_a_1882_);
lean_dec(v___x_1881_);
v___x_1884_ = lean_box(0);
v_isShared_1885_ = v_isSharedCheck_1891_;
goto v_resetjp_1883_;
}
v_resetjp_1883_:
{
lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1889_; 
v___x_1886_ = l_Lean_Expr_app___override(v_a_1882_, v_pf_1773_);
v___x_1887_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1887_, 0, v___x_1886_);
if (v_isShared_1885_ == 0)
{
lean_ctor_set(v___x_1884_, 0, v___x_1887_);
v___x_1889_ = v___x_1884_;
goto v_reusejp_1888_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1890_, 0, v___x_1887_);
v___x_1889_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1888_;
}
v_reusejp_1888_:
{
return v___x_1889_;
}
}
}
else
{
lean_object* v_a_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_1899_; 
lean_dec_ref(v_pf_1773_);
v_a_1892_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1899_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1899_ == 0)
{
v___x_1894_ = v___x_1881_;
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_a_1892_);
lean_dec(v___x_1881_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v___x_1897_; 
if (v_isShared_1895_ == 0)
{
v___x_1897_ = v___x_1894_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v_a_1892_);
v___x_1897_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
return v___x_1897_;
}
}
}
}
else
{
lean_object* v_a_1900_; lean_object* v___x_1902_; uint8_t v_isShared_1903_; uint8_t v_isSharedCheck_1907_; 
lean_dec_ref(v_pf_1773_);
v_a_1900_ = lean_ctor_get(v___x_1875_, 0);
v_isSharedCheck_1907_ = !lean_is_exclusive(v___x_1875_);
if (v_isSharedCheck_1907_ == 0)
{
v___x_1902_ = v___x_1875_;
v_isShared_1903_ = v_isSharedCheck_1907_;
goto v_resetjp_1901_;
}
else
{
lean_inc(v_a_1900_);
lean_dec(v___x_1875_);
v___x_1902_ = lean_box(0);
v_isShared_1903_ = v_isSharedCheck_1907_;
goto v_resetjp_1901_;
}
v_resetjp_1901_:
{
lean_object* v___x_1905_; 
if (v_isShared_1903_ == 0)
{
v___x_1905_ = v___x_1902_;
goto v_reusejp_1904_;
}
else
{
lean_object* v_reuseFailAlloc_1906_; 
v_reuseFailAlloc_1906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1906_, 0, v_a_1900_);
v___x_1905_ = v_reuseFailAlloc_1906_;
goto v_reusejp_1904_;
}
v_reusejp_1904_:
{
return v___x_1905_;
}
}
}
}
else
{
lean_dec(v_us_1862_);
lean_dec(v_snd_1861_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
}
else
{
lean_dec_ref_known(v_declName_1859_, 2);
lean_dec_ref_known(v_fst_1858_, 2);
lean_dec(v_snd_1857_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec(v_declName_1859_);
lean_dec_ref_known(v_fst_1858_, 2);
lean_dec(v_snd_1857_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec(v_fst_1858_);
lean_dec(v_snd_1857_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
else
{
lean_dec(v_snd_1854_);
lean_dec_ref(v_pf_1773_);
goto v___jp_1779_;
}
}
}
else
{
lean_object* v_a_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1915_; 
lean_dec_ref(v_pf_1773_);
v_a_1908_ = lean_ctor_get(v___x_1784_, 0);
v_isSharedCheck_1915_ = !lean_is_exclusive(v___x_1784_);
if (v_isSharedCheck_1915_ == 0)
{
v___x_1910_ = v___x_1784_;
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_a_1908_);
lean_dec(v___x_1784_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1913_; 
if (v_isShared_1911_ == 0)
{
v___x_1913_ = v___x_1910_;
goto v_reusejp_1912_;
}
else
{
lean_object* v_reuseFailAlloc_1914_; 
v_reuseFailAlloc_1914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1914_, 0, v_a_1908_);
v___x_1913_ = v_reuseFailAlloc_1914_;
goto v_reusejp_1912_;
}
v_reusejp_1912_:
{
return v___x_1913_;
}
}
}
}
else
{
lean_object* v_a_1916_; lean_object* v___x_1918_; uint8_t v_isShared_1919_; uint8_t v_isSharedCheck_1923_; 
lean_dec_ref(v_pf_1773_);
v_a_1916_ = lean_ctor_get(v___x_1782_, 0);
v_isSharedCheck_1923_ = !lean_is_exclusive(v___x_1782_);
if (v_isSharedCheck_1923_ == 0)
{
v___x_1918_ = v___x_1782_;
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
else
{
lean_inc(v_a_1916_);
lean_dec(v___x_1782_);
v___x_1918_ = lean_box(0);
v_isShared_1919_ = v_isSharedCheck_1923_;
goto v_resetjp_1917_;
}
v_resetjp_1917_:
{
lean_object* v___x_1921_; 
if (v_isShared_1919_ == 0)
{
v___x_1921_ = v___x_1918_;
goto v_reusejp_1920_;
}
else
{
lean_object* v_reuseFailAlloc_1922_; 
v_reuseFailAlloc_1922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1922_, 0, v_a_1916_);
v___x_1921_ = v_reuseFailAlloc_1922_;
goto v_reusejp_1920_;
}
v_reusejp_1920_:
{
return v___x_1921_;
}
}
}
v___jp_1779_:
{
lean_object* v___x_1780_; lean_object* v___x_1781_; 
v___x_1780_ = lean_box(0);
v___x_1781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1781_, 0, v___x_1780_);
return v___x_1781_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f___boxed(lean_object* v_pf_1924_, lean_object* v_a_1925_, lean_object* v_a_1926_, lean_object* v_a_1927_, lean_object* v_a_1928_, lean_object* v_a_1929_){
_start:
{
lean_object* v_res_1930_; 
v_res_1930_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f(v_pf_1924_, v_a_1925_, v_a_1926_, v_a_1927_, v_a_1928_);
lean_dec(v_a_1928_);
lean_dec_ref(v_a_1927_);
lean_dec(v_a_1926_);
lean_dec_ref(v_a_1925_);
return v_res_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof(lean_object* v_pf_1931_, lean_object* v_a_1932_, lean_object* v_a_1933_, lean_object* v_a_1934_, lean_object* v_a_1935_){
_start:
{
lean_object* v___x_1937_; 
v___x_1937_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f(v_pf_1931_, v_a_1932_, v_a_1933_, v_a_1934_, v_a_1935_);
return v___x_1937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof___boxed(lean_object* v_pf_1938_, lean_object* v_a_1939_, lean_object* v_a_1940_, lean_object* v_a_1941_, lean_object* v_a_1942_, lean_object* v_a_1943_){
_start:
{
lean_object* v_res_1944_; 
v_res_1944_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof(v_pf_1938_, v_a_1939_, v_a_1940_, v_a_1941_, v_a_1942_);
lean_dec(v_a_1942_);
lean_dec_ref(v_a_1941_);
lean_dec(v_a_1940_);
lean_dec_ref(v_a_1939_);
return v_res_1944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0(lean_object* v_h_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v___x_1951_; 
lean_inc_ref(v_h_1945_);
v___x_1951_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNonstrictIntProof_x3f(v_h_1945_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_);
if (lean_obj_tag(v___x_1951_) == 0)
{
lean_object* v_a_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_1964_; 
v_a_1952_ = lean_ctor_get(v___x_1951_, 0);
v_isSharedCheck_1964_ = !lean_is_exclusive(v___x_1951_);
if (v_isSharedCheck_1964_ == 0)
{
v___x_1954_ = v___x_1951_;
v_isShared_1955_ = v_isSharedCheck_1964_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_a_1952_);
lean_dec(v___x_1951_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_1964_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v___y_1957_; 
if (lean_obj_tag(v_a_1952_) == 0)
{
v___y_1957_ = v_h_1945_;
goto v___jp_1956_;
}
else
{
lean_object* v_val_1963_; 
lean_dec_ref(v_h_1945_);
v_val_1963_ = lean_ctor_get(v_a_1952_, 0);
lean_inc(v_val_1963_);
lean_dec_ref_known(v_a_1952_, 1);
v___y_1957_ = v_val_1963_;
goto v___jp_1956_;
}
v___jp_1956_:
{
lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1961_; 
v___x_1958_ = lean_box(0);
v___x_1959_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1959_, 0, v___y_1957_);
lean_ctor_set(v___x_1959_, 1, v___x_1958_);
if (v_isShared_1955_ == 0)
{
lean_ctor_set(v___x_1954_, 0, v___x_1959_);
v___x_1961_ = v___x_1954_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_1962_; 
v_reuseFailAlloc_1962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1962_, 0, v___x_1959_);
v___x_1961_ = v_reuseFailAlloc_1962_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
return v___x_1961_;
}
}
}
}
else
{
lean_object* v_a_1965_; lean_object* v___x_1967_; uint8_t v_isShared_1968_; uint8_t v_isSharedCheck_1972_; 
lean_dec_ref(v_h_1945_);
v_a_1965_ = lean_ctor_get(v___x_1951_, 0);
v_isSharedCheck_1972_ = !lean_is_exclusive(v___x_1951_);
if (v_isSharedCheck_1972_ == 0)
{
v___x_1967_ = v___x_1951_;
v_isShared_1968_ = v_isSharedCheck_1972_;
goto v_resetjp_1966_;
}
else
{
lean_inc(v_a_1965_);
lean_dec(v___x_1951_);
v___x_1967_ = lean_box(0);
v_isShared_1968_ = v_isSharedCheck_1972_;
goto v_resetjp_1966_;
}
v_resetjp_1966_:
{
lean_object* v___x_1970_; 
if (v_isShared_1968_ == 0)
{
v___x_1970_ = v___x_1967_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1971_; 
v_reuseFailAlloc_1971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1971_, 0, v_a_1965_);
v___x_1970_ = v_reuseFailAlloc_1971_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
return v___x_1970_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0___boxed(lean_object* v_h_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v_res_1979_; 
v_res_1979_ = lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt___lam__0(v_h_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_);
lean_dec(v___y_1977_);
lean_dec_ref(v___y_1976_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(lean_object* v_x_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_){
_start:
{
lean_object* v___x_2001_; 
v___x_2001_ = l_Lean_Meta_saveState___redArg(v___y_1997_, v___y_1999_);
if (lean_obj_tag(v___x_2001_) == 0)
{
lean_object* v_a_2002_; lean_object* v___x_2003_; 
v_a_2002_ = lean_ctor_get(v___x_2001_, 0);
lean_inc(v_a_2002_);
lean_dec_ref_known(v___x_2001_, 1);
lean_inc(v___y_1999_);
lean_inc_ref(v___y_1998_);
lean_inc(v___y_1997_);
lean_inc_ref(v___y_1996_);
v___x_2003_ = lean_apply_5(v_x_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, lean_box(0));
if (lean_obj_tag(v___x_2003_) == 0)
{
lean_object* v_a_2004_; lean_object* v___x_2006_; uint8_t v_isShared_2007_; uint8_t v_isSharedCheck_2012_; 
lean_dec(v_a_2002_);
v_a_2004_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2012_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2012_ == 0)
{
v___x_2006_ = v___x_2003_;
v_isShared_2007_ = v_isSharedCheck_2012_;
goto v_resetjp_2005_;
}
else
{
lean_inc(v_a_2004_);
lean_dec(v___x_2003_);
v___x_2006_ = lean_box(0);
v_isShared_2007_ = v_isSharedCheck_2012_;
goto v_resetjp_2005_;
}
v_resetjp_2005_:
{
lean_object* v___x_2008_; lean_object* v___x_2010_; 
v___x_2008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2008_, 0, v_a_2004_);
if (v_isShared_2007_ == 0)
{
lean_ctor_set(v___x_2006_, 0, v___x_2008_);
v___x_2010_ = v___x_2006_;
goto v_reusejp_2009_;
}
else
{
lean_object* v_reuseFailAlloc_2011_; 
v_reuseFailAlloc_2011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2011_, 0, v___x_2008_);
v___x_2010_ = v_reuseFailAlloc_2011_;
goto v_reusejp_2009_;
}
v_reusejp_2009_:
{
return v___x_2010_;
}
}
}
else
{
lean_object* v_a_2013_; lean_object* v___x_2015_; uint8_t v_isShared_2016_; uint8_t v_isSharedCheck_2042_; 
v_a_2013_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2042_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2042_ == 0)
{
v___x_2015_ = v___x_2003_;
v_isShared_2016_ = v_isSharedCheck_2042_;
goto v_resetjp_2014_;
}
else
{
lean_inc(v_a_2013_);
lean_dec(v___x_2003_);
v___x_2015_ = lean_box(0);
v_isShared_2016_ = v_isSharedCheck_2042_;
goto v_resetjp_2014_;
}
v_resetjp_2014_:
{
lean_object* v___x_2018_; 
lean_inc(v_a_2013_);
if (v_isShared_2016_ == 0)
{
v___x_2018_ = v___x_2015_;
goto v_reusejp_2017_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v_a_2013_);
v___x_2018_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2017_;
}
v_reusejp_2017_:
{
uint8_t v___y_2020_; uint8_t v___x_2039_; 
v___x_2039_ = l_Lean_Exception_isInterrupt(v_a_2013_);
if (v___x_2039_ == 0)
{
uint8_t v___x_2040_; 
v___x_2040_ = l_Lean_Exception_isRuntime(v_a_2013_);
v___y_2020_ = v___x_2040_;
goto v___jp_2019_;
}
else
{
lean_dec(v_a_2013_);
v___y_2020_ = v___x_2039_;
goto v___jp_2019_;
}
v___jp_2019_:
{
if (v___y_2020_ == 0)
{
lean_object* v___x_2021_; 
lean_dec_ref(v___x_2018_);
v___x_2021_ = l_Lean_Meta_SavedState_restore___redArg(v_a_2002_, v___y_1997_, v___y_1999_);
lean_dec(v_a_2002_);
if (lean_obj_tag(v___x_2021_) == 0)
{
lean_object* v___x_2023_; uint8_t v_isShared_2024_; uint8_t v_isSharedCheck_2029_; 
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_2021_);
if (v_isSharedCheck_2029_ == 0)
{
lean_object* v_unused_2030_; 
v_unused_2030_ = lean_ctor_get(v___x_2021_, 0);
lean_dec(v_unused_2030_);
v___x_2023_ = v___x_2021_;
v_isShared_2024_ = v_isSharedCheck_2029_;
goto v_resetjp_2022_;
}
else
{
lean_dec(v___x_2021_);
v___x_2023_ = lean_box(0);
v_isShared_2024_ = v_isSharedCheck_2029_;
goto v_resetjp_2022_;
}
v_resetjp_2022_:
{
lean_object* v___x_2025_; lean_object* v___x_2027_; 
v___x_2025_ = lean_box(0);
if (v_isShared_2024_ == 0)
{
lean_ctor_set(v___x_2023_, 0, v___x_2025_);
v___x_2027_ = v___x_2023_;
goto v_reusejp_2026_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v___x_2025_);
v___x_2027_ = v_reuseFailAlloc_2028_;
goto v_reusejp_2026_;
}
v_reusejp_2026_:
{
return v___x_2027_;
}
}
}
else
{
lean_object* v_a_2031_; lean_object* v___x_2033_; uint8_t v_isShared_2034_; uint8_t v_isSharedCheck_2038_; 
v_a_2031_ = lean_ctor_get(v___x_2021_, 0);
v_isSharedCheck_2038_ = !lean_is_exclusive(v___x_2021_);
if (v_isSharedCheck_2038_ == 0)
{
v___x_2033_ = v___x_2021_;
v_isShared_2034_ = v_isSharedCheck_2038_;
goto v_resetjp_2032_;
}
else
{
lean_inc(v_a_2031_);
lean_dec(v___x_2021_);
v___x_2033_ = lean_box(0);
v_isShared_2034_ = v_isSharedCheck_2038_;
goto v_resetjp_2032_;
}
v_resetjp_2032_:
{
lean_object* v___x_2036_; 
if (v_isShared_2034_ == 0)
{
v___x_2036_ = v___x_2033_;
goto v_reusejp_2035_;
}
else
{
lean_object* v_reuseFailAlloc_2037_; 
v_reuseFailAlloc_2037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2037_, 0, v_a_2031_);
v___x_2036_ = v_reuseFailAlloc_2037_;
goto v_reusejp_2035_;
}
v_reusejp_2035_:
{
return v___x_2036_;
}
}
}
}
else
{
lean_dec(v_a_2002_);
return v___x_2018_;
}
}
}
}
}
}
else
{
lean_object* v_a_2043_; lean_object* v___x_2045_; uint8_t v_isShared_2046_; uint8_t v_isSharedCheck_2050_; 
lean_dec_ref(v_x_1995_);
v_a_2043_ = lean_ctor_get(v___x_2001_, 0);
v_isSharedCheck_2050_ = !lean_is_exclusive(v___x_2001_);
if (v_isSharedCheck_2050_ == 0)
{
v___x_2045_ = v___x_2001_;
v_isShared_2046_ = v_isSharedCheck_2050_;
goto v_resetjp_2044_;
}
else
{
lean_inc(v_a_2043_);
lean_dec(v___x_2001_);
v___x_2045_ = lean_box(0);
v_isShared_2046_ = v_isSharedCheck_2050_;
goto v_resetjp_2044_;
}
v_resetjp_2044_:
{
lean_object* v___x_2048_; 
if (v_isShared_2046_ == 0)
{
v___x_2048_ = v___x_2045_;
goto v_reusejp_2047_;
}
else
{
lean_object* v_reuseFailAlloc_2049_; 
v_reuseFailAlloc_2049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2049_, 0, v_a_2043_);
v___x_2048_ = v_reuseFailAlloc_2049_;
goto v_reusejp_2047_;
}
v_reusejp_2047_:
{
return v___x_2048_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg___boxed(lean_object* v_x_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_){
_start:
{
lean_object* v_res_2057_; 
v_res_2057_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(v_x_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_);
lean_dec(v___y_2055_);
lean_dec_ref(v___y_2054_);
lean_dec(v___y_2053_);
lean_dec_ref(v___y_2052_);
return v_res_2057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0(lean_object* v_00_u03b1_2058_, lean_object* v_x_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_){
_start:
{
lean_object* v___x_2065_; 
v___x_2065_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(v_x_2059_, v___y_2060_, v___y_2061_, v___y_2062_, v___y_2063_);
return v___x_2065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___boxed(lean_object* v_00_u03b1_2066_, lean_object* v_x_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_){
_start:
{
lean_object* v_res_2073_; 
v_res_2073_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0(v_00_u03b1_2066_, v_x_2067_, v___y_2068_, v___y_2069_, v___y_2070_, v___y_2071_);
lean_dec(v___y_2071_);
lean_dec_ref(v___y_2070_);
lean_dec(v___y_2069_);
lean_dec_ref(v___y_2068_);
return v_res_2073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f(lean_object* v_e_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_, lean_object* v_a_2093_){
_start:
{
lean_object* v___x_2095_; 
lean_inc(v_a_2093_);
lean_inc_ref(v_a_2092_);
lean_inc(v_a_2091_);
lean_inc_ref(v_a_2090_);
lean_inc_ref(v_e_2089_);
v___x_2095_ = lean_infer_type(v_e_2089_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
if (lean_obj_tag(v___x_2095_) == 0)
{
lean_object* v_a_2096_; lean_object* v___x_2097_; 
v_a_2096_ = lean_ctor_get(v___x_2095_, 0);
lean_inc(v_a_2096_);
lean_dec_ref_known(v___x_2095_, 1);
v___x_2097_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_2096_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
if (lean_obj_tag(v___x_2097_) == 0)
{
lean_object* v_a_2098_; lean_object* v_fst_2099_; uint8_t v___x_2100_; 
v_a_2098_ = lean_ctor_get(v___x_2097_, 0);
lean_inc(v_a_2098_);
lean_dec_ref_known(v___x_2097_, 1);
v_fst_2099_ = lean_ctor_get(v_a_2098_, 0);
lean_inc(v_fst_2099_);
lean_dec(v_a_2098_);
v___x_2100_ = lean_unbox(v_fst_2099_);
lean_dec(v_fst_2099_);
switch(v___x_2100_)
{
case 0:
{
lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; 
v___x_2101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__1));
v___x_2102_ = lean_unsigned_to_nat(1u);
v___x_2103_ = lean_mk_empty_array_with_capacity(v___x_2102_);
v___x_2104_ = lean_array_push(v___x_2103_, v_e_2089_);
v___x_2105_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_2105_, 0, v___x_2101_);
lean_closure_set(v___x_2105_, 1, v___x_2104_);
v___x_2106_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(v___x_2105_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
return v___x_2106_;
}
case 1:
{
lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; 
v___x_2107_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__3));
v___x_2108_ = lean_unsigned_to_nat(1u);
v___x_2109_ = lean_mk_empty_array_with_capacity(v___x_2108_);
v___x_2110_ = lean_array_push(v___x_2109_, v_e_2089_);
v___x_2111_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_2111_, 0, v___x_2107_);
lean_closure_set(v___x_2111_, 1, v___x_2110_);
v___x_2112_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(v___x_2111_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
return v___x_2112_;
}
default: 
{
lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; 
v___x_2113_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___closed__5));
v___x_2114_ = lean_unsigned_to_nat(1u);
v___x_2115_ = lean_mk_empty_array_with_capacity(v___x_2114_);
v___x_2116_ = lean_array_push(v___x_2115_, v_e_2089_);
v___x_2117_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_2117_, 0, v___x_2113_);
lean_closure_set(v___x_2117_, 1, v___x_2116_);
v___x_2118_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_Linarith_rearrangeComparison_x3f_spec__0___redArg(v___x_2117_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
return v___x_2118_;
}
}
}
else
{
lean_object* v_a_2119_; lean_object* v___x_2121_; uint8_t v_isShared_2122_; uint8_t v_isSharedCheck_2126_; 
lean_dec_ref(v_e_2089_);
v_a_2119_ = lean_ctor_get(v___x_2097_, 0);
v_isSharedCheck_2126_ = !lean_is_exclusive(v___x_2097_);
if (v_isSharedCheck_2126_ == 0)
{
v___x_2121_ = v___x_2097_;
v_isShared_2122_ = v_isSharedCheck_2126_;
goto v_resetjp_2120_;
}
else
{
lean_inc(v_a_2119_);
lean_dec(v___x_2097_);
v___x_2121_ = lean_box(0);
v_isShared_2122_ = v_isSharedCheck_2126_;
goto v_resetjp_2120_;
}
v_resetjp_2120_:
{
lean_object* v___x_2124_; 
if (v_isShared_2122_ == 0)
{
v___x_2124_ = v___x_2121_;
goto v_reusejp_2123_;
}
else
{
lean_object* v_reuseFailAlloc_2125_; 
v_reuseFailAlloc_2125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2125_, 0, v_a_2119_);
v___x_2124_ = v_reuseFailAlloc_2125_;
goto v_reusejp_2123_;
}
v_reusejp_2123_:
{
return v___x_2124_;
}
}
}
}
else
{
lean_object* v_a_2127_; lean_object* v___x_2129_; uint8_t v_isShared_2130_; uint8_t v_isSharedCheck_2134_; 
lean_dec_ref(v_e_2089_);
v_a_2127_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2134_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2134_ == 0)
{
v___x_2129_ = v___x_2095_;
v_isShared_2130_ = v_isSharedCheck_2134_;
goto v_resetjp_2128_;
}
else
{
lean_inc(v_a_2127_);
lean_dec(v___x_2095_);
v___x_2129_ = lean_box(0);
v_isShared_2130_ = v_isSharedCheck_2134_;
goto v_resetjp_2128_;
}
v_resetjp_2128_:
{
lean_object* v___x_2132_; 
if (v_isShared_2130_ == 0)
{
v___x_2132_ = v___x_2129_;
goto v_reusejp_2131_;
}
else
{
lean_object* v_reuseFailAlloc_2133_; 
v_reuseFailAlloc_2133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2133_, 0, v_a_2127_);
v___x_2132_ = v_reuseFailAlloc_2133_;
goto v_reusejp_2131_;
}
v_reusejp_2131_:
{
return v___x_2132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f___boxed(lean_object* v_e_2135_, lean_object* v_a_2136_, lean_object* v_a_2137_, lean_object* v_a_2138_, lean_object* v_a_2139_, lean_object* v_a_2140_){
_start:
{
lean_object* v_res_2141_; 
v_res_2141_ = lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f(v_e_2135_, v_a_2136_, v_a_2137_, v_a_2138_, v_a_2139_);
lean_dec(v_a_2139_);
lean_dec_ref(v_a_2138_);
lean_dec(v_a_2137_);
lean_dec_ref(v_a_2136_);
return v_res_2141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison(lean_object* v_e_2142_, lean_object* v_a_2143_, lean_object* v_a_2144_, lean_object* v_a_2145_, lean_object* v_a_2146_){
_start:
{
lean_object* v___x_2148_; 
v___x_2148_ = lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f(v_e_2142_, v_a_2143_, v_a_2144_, v_a_2145_, v_a_2146_);
return v___x_2148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison___boxed(lean_object* v_e_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_, lean_object* v_a_2152_, lean_object* v_a_2153_, lean_object* v_a_2154_){
_start:
{
lean_object* v_res_2155_; 
v_res_2155_ = lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison(v_e_2149_, v_a_2150_, v_a_2151_, v_a_2152_, v_a_2153_);
lean_dec(v_a_2153_);
lean_dec_ref(v_a_2152_);
lean_dec(v_a_2151_);
lean_dec_ref(v_a_2150_);
return v_res_2155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0(lean_object* v_e_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_){
_start:
{
lean_object* v___x_2162_; 
v___x_2162_ = lp_mathlib_Mathlib_Tactic_Linarith_rearrangeComparison_x3f(v_e_2156_, v___y_2157_, v___y_2158_, v___y_2159_, v___y_2160_);
if (lean_obj_tag(v___x_2162_) == 0)
{
lean_object* v_a_2163_; lean_object* v___x_2165_; uint8_t v_isShared_2166_; uint8_t v_isSharedCheck_2177_; 
v_a_2163_ = lean_ctor_get(v___x_2162_, 0);
v_isSharedCheck_2177_ = !lean_is_exclusive(v___x_2162_);
if (v_isSharedCheck_2177_ == 0)
{
v___x_2165_ = v___x_2162_;
v_isShared_2166_ = v_isSharedCheck_2177_;
goto v_resetjp_2164_;
}
else
{
lean_inc(v_a_2163_);
lean_dec(v___x_2162_);
v___x_2165_ = lean_box(0);
v_isShared_2166_ = v_isSharedCheck_2177_;
goto v_resetjp_2164_;
}
v_resetjp_2164_:
{
if (lean_obj_tag(v_a_2163_) == 0)
{
lean_object* v___x_2167_; lean_object* v___x_2169_; 
v___x_2167_ = lean_box(0);
if (v_isShared_2166_ == 0)
{
lean_ctor_set(v___x_2165_, 0, v___x_2167_);
v___x_2169_ = v___x_2165_;
goto v_reusejp_2168_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v___x_2167_);
v___x_2169_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2168_;
}
v_reusejp_2168_:
{
return v___x_2169_;
}
}
else
{
lean_object* v_val_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2175_; 
v_val_2171_ = lean_ctor_get(v_a_2163_, 0);
lean_inc(v_val_2171_);
lean_dec_ref_known(v_a_2163_, 1);
v___x_2172_ = lean_box(0);
v___x_2173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2173_, 0, v_val_2171_);
lean_ctor_set(v___x_2173_, 1, v___x_2172_);
if (v_isShared_2166_ == 0)
{
lean_ctor_set(v___x_2165_, 0, v___x_2173_);
v___x_2175_ = v___x_2165_;
goto v_reusejp_2174_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v___x_2173_);
v___x_2175_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2174_;
}
v_reusejp_2174_:
{
return v___x_2175_;
}
}
}
}
else
{
lean_object* v_a_2178_; lean_object* v___x_2180_; uint8_t v_isShared_2181_; uint8_t v_isSharedCheck_2185_; 
v_a_2178_ = lean_ctor_get(v___x_2162_, 0);
v_isSharedCheck_2185_ = !lean_is_exclusive(v___x_2162_);
if (v_isSharedCheck_2185_ == 0)
{
v___x_2180_ = v___x_2162_;
v_isShared_2181_ = v_isSharedCheck_2185_;
goto v_resetjp_2179_;
}
else
{
lean_inc(v_a_2178_);
lean_dec(v___x_2162_);
v___x_2180_ = lean_box(0);
v_isShared_2181_ = v_isSharedCheck_2185_;
goto v_resetjp_2179_;
}
v_resetjp_2179_:
{
lean_object* v___x_2183_; 
if (v_isShared_2181_ == 0)
{
v___x_2183_ = v___x_2180_;
goto v_reusejp_2182_;
}
else
{
lean_object* v_reuseFailAlloc_2184_; 
v_reuseFailAlloc_2184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2184_, 0, v_a_2178_);
v___x_2183_ = v_reuseFailAlloc_2184_;
goto v_reusejp_2182_;
}
v_reusejp_2182_:
{
return v___x_2183_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0___boxed(lean_object* v_e_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_){
_start:
{
lean_object* v_res_2192_; 
v_res_2192_ = lp_mathlib_Mathlib_Tactic_Linarith_compWithZero___lam__0(v_e_2186_, v___y_2187_, v___y_2188_, v___y_2189_, v___y_2190_);
lean_dec(v___y_2190_);
lean_dec_ref(v___y_2189_);
lean_dec(v___y_2188_);
lean_dec_ref(v___y_2187_);
return v_res_2192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0(lean_object* v_a_2208_, lean_object* v_x_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_){
_start:
{
lean_object* v___x_2215_; 
v___x_2215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2215_, 0, v_a_2208_);
return v___x_2215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0___boxed(lean_object* v_a_2216_, lean_object* v_x_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_){
_start:
{
lean_object* v_res_2223_; 
v_res_2223_ = lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0(v_a_2216_, v_x_2217_, v___y_2218_, v___y_2219_, v___y_2220_, v___y_2221_);
lean_dec(v___y_2221_);
lean_dec_ref(v___y_2220_);
lean_dec(v___y_2219_);
lean_dec_ref(v___y_2218_);
return v_res_2223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS(lean_object* v_h_2231_, lean_object* v_lhs_2232_, lean_object* v_a_2233_, lean_object* v_a_2234_, lean_object* v_a_2235_, lean_object* v_a_2236_){
_start:
{
lean_object* v___y_2239_; lean_object* v___y_2240_; lean_object* v___y_2241_; lean_object* v___y_2242_; lean_object* v___y_2243_; lean_object* v___y_2244_; lean_object* v___y_2245_; uint8_t v___y_2246_; lean_object* v___x_2253_; 
v___x_2253_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(v_lhs_2232_, v_a_2233_, v_a_2234_, v_a_2235_, v_a_2236_);
if (lean_obj_tag(v___x_2253_) == 0)
{
lean_object* v_a_2254_; lean_object* v_fst_2255_; lean_object* v_snd_2256_; lean_object* v_lhs_x27_2258_; lean_object* v___y_2259_; lean_object* v___y_2260_; lean_object* v___y_2261_; lean_object* v___y_2262_; lean_object* v___x_2279_; uint8_t v___x_2280_; 
v_a_2254_ = lean_ctor_get(v___x_2253_, 0);
lean_inc(v_a_2254_);
lean_dec_ref_known(v___x_2253_, 1);
v_fst_2255_ = lean_ctor_get(v_a_2254_, 0);
lean_inc(v_fst_2255_);
v_snd_2256_ = lean_ctor_get(v_a_2254_, 1);
lean_inc(v_snd_2256_);
lean_dec(v_a_2254_);
v___x_2279_ = lean_unsigned_to_nat(1u);
v___x_2280_ = lean_nat_dec_eq(v_fst_2255_, v___x_2279_);
if (v___x_2280_ == 0)
{
v_lhs_x27_2258_ = v_snd_2256_;
v___y_2259_ = v_a_2233_;
v___y_2260_ = v_a_2234_;
v___y_2261_ = v_a_2235_;
v___y_2262_ = v_a_2236_;
goto v___jp_2257_;
}
else
{
lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; 
v___x_2281_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__2));
v___x_2282_ = lean_mk_empty_array_with_capacity(v___x_2279_);
v___x_2283_ = lean_array_push(v___x_2282_, v_snd_2256_);
v___x_2284_ = l_Lean_Meta_mkAppM(v___x_2281_, v___x_2283_, v_a_2233_, v_a_2234_, v_a_2235_, v_a_2236_);
if (lean_obj_tag(v___x_2284_) == 0)
{
lean_object* v_a_2285_; 
v_a_2285_ = lean_ctor_get(v___x_2284_, 0);
lean_inc(v_a_2285_);
lean_dec_ref_known(v___x_2284_, 1);
v_lhs_x27_2258_ = v_a_2285_;
v___y_2259_ = v_a_2233_;
v___y_2260_ = v_a_2234_;
v___y_2261_ = v_a_2235_;
v___y_2262_ = v_a_2236_;
goto v___jp_2257_;
}
else
{
lean_dec(v_fst_2255_);
lean_dec_ref(v_h_2231_);
return v___x_2284_;
}
}
v___jp_2257_:
{
lean_object* v___x_2263_; 
v___x_2263_ = lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(v_fst_2255_, v_h_2231_, v___y_2259_, v___y_2260_, v___y_2261_, v___y_2262_);
if (lean_obj_tag(v___x_2263_) == 0)
{
lean_object* v_a_2264_; lean_object* v_snd_2265_; lean_object* v___x_2266_; 
v_a_2264_ = lean_ctor_get(v___x_2263_, 0);
lean_inc(v_a_2264_);
lean_dec_ref_known(v___x_2263_, 1);
v_snd_2265_ = lean_ctor_get(v_a_2264_, 1);
lean_inc(v_snd_2265_);
lean_dec(v_a_2264_);
v___x_2266_ = lp_mathlib_Lean_Expr_rewriteType(v_snd_2265_, v_lhs_x27_2258_, v___y_2259_, v___y_2260_, v___y_2261_, v___y_2262_);
if (lean_obj_tag(v___x_2266_) == 0)
{
return v___x_2266_;
}
else
{
lean_object* v_a_2267_; lean_object* v___f_2268_; uint8_t v___x_2269_; 
v_a_2267_ = lean_ctor_get(v___x_2266_, 0);
lean_inc_n(v_a_2267_, 2);
v___f_2268_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___lam__0___boxed), 7, 1);
lean_closure_set(v___f_2268_, 0, v_a_2267_);
v___x_2269_ = l_Lean_Exception_isInterrupt(v_a_2267_);
if (v___x_2269_ == 0)
{
uint8_t v___x_2270_; 
lean_inc(v_a_2267_);
v___x_2270_ = l_Lean_Exception_isRuntime(v_a_2267_);
v___y_2239_ = v___y_2261_;
v___y_2240_ = v___x_2266_;
v___y_2241_ = v_a_2267_;
v___y_2242_ = v___y_2262_;
v___y_2243_ = v___y_2259_;
v___y_2244_ = v___y_2260_;
v___y_2245_ = v___f_2268_;
v___y_2246_ = v___x_2270_;
goto v___jp_2238_;
}
else
{
v___y_2239_ = v___y_2261_;
v___y_2240_ = v___x_2266_;
v___y_2241_ = v_a_2267_;
v___y_2242_ = v___y_2262_;
v___y_2243_ = v___y_2259_;
v___y_2244_ = v___y_2260_;
v___y_2245_ = v___f_2268_;
v___y_2246_ = v___x_2269_;
goto v___jp_2238_;
}
}
}
else
{
lean_object* v_a_2271_; lean_object* v___x_2273_; uint8_t v_isShared_2274_; uint8_t v_isSharedCheck_2278_; 
lean_dec_ref(v_lhs_x27_2258_);
v_a_2271_ = lean_ctor_get(v___x_2263_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2273_ = v___x_2263_;
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
else
{
lean_inc(v_a_2271_);
lean_dec(v___x_2263_);
v___x_2273_ = lean_box(0);
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
v_resetjp_2272_:
{
lean_object* v___x_2276_; 
if (v_isShared_2274_ == 0)
{
v___x_2276_ = v___x_2273_;
goto v_reusejp_2275_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v_a_2271_);
v___x_2276_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2275_;
}
v_reusejp_2275_:
{
return v___x_2276_;
}
}
}
}
}
else
{
lean_object* v_a_2286_; lean_object* v___x_2288_; uint8_t v_isShared_2289_; uint8_t v_isSharedCheck_2293_; 
lean_dec_ref(v_h_2231_);
v_a_2286_ = lean_ctor_get(v___x_2253_, 0);
v_isSharedCheck_2293_ = !lean_is_exclusive(v___x_2253_);
if (v_isSharedCheck_2293_ == 0)
{
v___x_2288_ = v___x_2253_;
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
else
{
lean_inc(v_a_2286_);
lean_dec(v___x_2253_);
v___x_2288_ = lean_box(0);
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
v_resetjp_2287_:
{
lean_object* v___x_2291_; 
if (v_isShared_2289_ == 0)
{
v___x_2291_ = v___x_2288_;
goto v_reusejp_2290_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v_a_2286_);
v___x_2291_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2290_;
}
v_reusejp_2290_:
{
return v___x_2291_;
}
}
}
v___jp_2238_:
{
if (v___y_2246_ == 0)
{
lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_987__overap_2251_; lean_object* v___x_2252_; 
lean_dec_ref(v___y_2240_);
v___x_2247_ = l_Lean_Exception_toMessageData(v___y_2241_);
v___x_2248_ = l_Lean_MessageData_toString(v___x_2247_);
v___x_2249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___closed__0));
v___x_2250_ = lean_string_append(v___x_2249_, v___x_2248_);
lean_dec_ref(v___x_2248_);
v___x_987__overap_2251_ = lean_dbg_trace(v___x_2250_, v___y_2245_);
lean_inc(v___y_2242_);
lean_inc_ref(v___y_2239_);
lean_inc(v___y_2244_);
lean_inc_ref(v___y_2243_);
v___x_2252_ = lean_apply_5(v___x_987__overap_2251_, v___y_2243_, v___y_2244_, v___y_2239_, v___y_2242_, lean_box(0));
return v___x_2252_;
}
else
{
lean_dec_ref(v___y_2245_);
lean_dec_ref(v___y_2241_);
return v___y_2240_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS___boxed(lean_object* v_h_2294_, lean_object* v_lhs_2295_, lean_object* v_a_2296_, lean_object* v_a_2297_, lean_object* v_a_2298_, lean_object* v_a_2299_, lean_object* v_a_2300_){
_start:
{
lean_object* v_res_2301_; 
v_res_2301_ = lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS(v_h_2294_, v_lhs_2295_, v_a_2296_, v_a_2297_, v_a_2298_, v_a_2299_);
lean_dec(v_a_2299_);
lean_dec_ref(v_a_2298_);
lean_dec(v_a_2297_);
lean_dec_ref(v_a_2296_);
return v_res_2301_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0(lean_object* v_x_2322_){
_start:
{
if (lean_obj_tag(v_x_2322_) == 4)
{
lean_object* v_declName_2323_; uint8_t v___y_2325_; lean_object* v___x_2330_; uint8_t v___x_2331_; 
v_declName_2323_ = lean_ctor_get(v_x_2322_, 0);
v___x_2330_ = ((lean_object*)(lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__8));
v___x_2331_ = lean_name_eq(v_declName_2323_, v___x_2330_);
if (v___x_2331_ == 0)
{
lean_object* v___x_2332_; uint8_t v___x_2333_; 
v___x_2332_ = ((lean_object*)(lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__11));
v___x_2333_ = lean_name_eq(v_declName_2323_, v___x_2332_);
v___y_2325_ = v___x_2333_;
goto v___jp_2324_;
}
else
{
v___y_2325_ = v___x_2331_;
goto v___jp_2324_;
}
v___jp_2324_:
{
if (v___y_2325_ == 0)
{
lean_object* v___x_2326_; uint8_t v___x_2327_; 
v___x_2326_ = ((lean_object*)(lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__2));
v___x_2327_ = lean_name_eq(v_declName_2323_, v___x_2326_);
if (v___x_2327_ == 0)
{
lean_object* v___x_2328_; uint8_t v___x_2329_; 
v___x_2328_ = ((lean_object*)(lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___closed__5));
v___x_2329_ = lean_name_eq(v_declName_2323_, v___x_2328_);
return v___x_2329_;
}
else
{
return v___x_2327_;
}
}
else
{
return v___y_2325_;
}
}
}
else
{
uint8_t v___x_2334_; 
v___x_2334_ = 0;
return v___x_2334_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0___boxed(lean_object* v_x_2335_){
_start:
{
uint8_t v_res_2336_; lean_object* v_r_2337_; 
v_res_2336_ = lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___lam__0(v_x_2335_);
lean_dec_ref(v_x_2335_);
v_r_2337_ = lean_box(v_res_2336_);
return v_r_2337_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0(lean_object* v_e_2339_){
_start:
{
lean_object* v___f_2340_; lean_object* v___x_2341_; 
v___f_2340_ = ((lean_object*)(lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___closed__0));
v___x_2341_ = lean_find_expr(v___f_2340_, v_e_2339_);
if (lean_obj_tag(v___x_2341_) == 0)
{
uint8_t v___x_2342_; 
v___x_2342_ = 0;
return v___x_2342_;
}
else
{
uint8_t v___x_2343_; 
lean_dec_ref_known(v___x_2341_, 1);
v___x_2343_ = 1;
return v___x_2343_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0___boxed(lean_object* v_e_2344_){
_start:
{
uint8_t v_res_2345_; lean_object* v_r_2346_; 
v_res_2345_ = lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0(v_e_2344_);
lean_dec_ref(v_e_2344_);
v_r_2346_ = lean_box(v_res_2345_);
return v_r_2346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0(lean_object* v_pf_2347_, lean_object* v___y_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_){
_start:
{
lean_object* v___x_2353_; 
v___x_2353_ = l_Lean_Meta_saveState___redArg(v___y_2349_, v___y_2351_);
if (lean_obj_tag(v___x_2353_) == 0)
{
lean_object* v_a_2354_; lean_object* v___y_2356_; uint8_t v___y_2357_; lean_object* v___y_2378_; lean_object* v_a_2379_; lean_object* v___x_2382_; 
v_a_2354_ = lean_ctor_get(v___x_2353_, 0);
lean_inc(v_a_2354_);
lean_dec_ref_known(v___x_2353_, 1);
lean_inc(v___y_2351_);
lean_inc_ref(v___y_2350_);
lean_inc(v___y_2349_);
lean_inc_ref(v___y_2348_);
lean_inc_ref(v_pf_2347_);
v___x_2382_ = lean_infer_type(v_pf_2347_, v___y_2348_, v___y_2349_, v___y_2350_, v___y_2351_);
if (lean_obj_tag(v___x_2382_) == 0)
{
lean_object* v_a_2383_; lean_object* v___x_2384_; 
v_a_2383_ = lean_ctor_get(v___x_2382_, 0);
lean_inc(v_a_2383_);
lean_dec_ref_known(v___x_2382_, 1);
v___x_2384_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_a_2383_, v___y_2348_, v___y_2349_, v___y_2350_, v___y_2351_);
if (lean_obj_tag(v___x_2384_) == 0)
{
lean_object* v_a_2385_; lean_object* v_snd_2386_; lean_object* v___x_2388_; uint8_t v_isShared_2389_; uint8_t v_isSharedCheck_2423_; 
v_a_2385_ = lean_ctor_get(v___x_2384_, 0);
lean_inc(v_a_2385_);
lean_dec_ref_known(v___x_2384_, 1);
v_snd_2386_ = lean_ctor_get(v_a_2385_, 1);
v_isSharedCheck_2423_ = !lean_is_exclusive(v_a_2385_);
if (v_isSharedCheck_2423_ == 0)
{
lean_object* v_unused_2424_; 
v_unused_2424_ = lean_ctor_get(v_a_2385_, 0);
lean_dec(v_unused_2424_);
v___x_2388_ = v_a_2385_;
v_isShared_2389_ = v_isSharedCheck_2423_;
goto v_resetjp_2387_;
}
else
{
lean_inc(v_snd_2386_);
lean_dec(v_a_2385_);
v___x_2388_ = lean_box(0);
v_isShared_2389_ = v_isSharedCheck_2423_;
goto v_resetjp_2387_;
}
v_resetjp_2387_:
{
uint8_t v___x_2412_; 
v___x_2412_ = lp_mathlib_Lean_Expr_containsConst___at___00Mathlib_Tactic_Linarith_cancelDenoms_spec__0(v_snd_2386_);
if (v___x_2412_ == 0)
{
lean_object* v___x_2413_; lean_object* v___x_2414_; lean_object* v_a_2415_; lean_object* v___x_2417_; uint8_t v_isShared_2418_; uint8_t v_isSharedCheck_2422_; 
lean_del_object(v___x_2388_);
lean_dec(v_snd_2386_);
v___x_2413_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1);
v___x_2414_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v___x_2413_, v___y_2348_, v___y_2349_, v___y_2350_, v___y_2351_);
v_a_2415_ = lean_ctor_get(v___x_2414_, 0);
v_isSharedCheck_2422_ = !lean_is_exclusive(v___x_2414_);
if (v_isSharedCheck_2422_ == 0)
{
v___x_2417_ = v___x_2414_;
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
else
{
lean_inc(v_a_2415_);
lean_dec(v___x_2414_);
v___x_2417_ = lean_box(0);
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
v_resetjp_2416_:
{
lean_object* v___x_2420_; 
lean_inc(v_a_2415_);
if (v_isShared_2418_ == 0)
{
v___x_2420_ = v___x_2417_;
goto v_reusejp_2419_;
}
else
{
lean_object* v_reuseFailAlloc_2421_; 
v_reuseFailAlloc_2421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2421_, 0, v_a_2415_);
v___x_2420_ = v_reuseFailAlloc_2421_;
goto v_reusejp_2419_;
}
v_reusejp_2419_:
{
v___y_2378_ = v___x_2420_;
v_a_2379_ = v_a_2415_;
goto v___jp_2377_;
}
}
}
else
{
goto v___jp_2390_;
}
v___jp_2390_:
{
lean_object* v___x_2391_; 
lean_inc_ref(v_pf_2347_);
v___x_2391_ = lp_mathlib_Mathlib_Tactic_Linarith_normalizeDenominatorsLHS(v_pf_2347_, v_snd_2386_, v___y_2348_, v___y_2349_, v___y_2350_, v___y_2351_);
if (lean_obj_tag(v___x_2391_) == 0)
{
lean_object* v_a_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2403_; 
lean_dec(v_a_2354_);
lean_dec_ref(v_pf_2347_);
v_a_2392_ = lean_ctor_get(v___x_2391_, 0);
v_isSharedCheck_2403_ = !lean_is_exclusive(v___x_2391_);
if (v_isSharedCheck_2403_ == 0)
{
v___x_2394_ = v___x_2391_;
v_isShared_2395_ = v_isSharedCheck_2403_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_a_2392_);
lean_dec(v___x_2391_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2403_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v___x_2396_; lean_object* v___x_2398_; 
v___x_2396_ = lean_box(0);
if (v_isShared_2389_ == 0)
{
lean_ctor_set_tag(v___x_2388_, 1);
lean_ctor_set(v___x_2388_, 1, v___x_2396_);
lean_ctor_set(v___x_2388_, 0, v_a_2392_);
v___x_2398_ = v___x_2388_;
goto v_reusejp_2397_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v_a_2392_);
lean_ctor_set(v_reuseFailAlloc_2402_, 1, v___x_2396_);
v___x_2398_ = v_reuseFailAlloc_2402_;
goto v_reusejp_2397_;
}
v_reusejp_2397_:
{
lean_object* v___x_2400_; 
if (v_isShared_2395_ == 0)
{
lean_ctor_set(v___x_2394_, 0, v___x_2398_);
v___x_2400_ = v___x_2394_;
goto v_reusejp_2399_;
}
else
{
lean_object* v_reuseFailAlloc_2401_; 
v_reuseFailAlloc_2401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2401_, 0, v___x_2398_);
v___x_2400_ = v_reuseFailAlloc_2401_;
goto v_reusejp_2399_;
}
v_reusejp_2399_:
{
return v___x_2400_;
}
}
}
}
else
{
lean_object* v_a_2404_; lean_object* v___x_2406_; uint8_t v_isShared_2407_; uint8_t v_isSharedCheck_2411_; 
lean_del_object(v___x_2388_);
v_a_2404_ = lean_ctor_get(v___x_2391_, 0);
v_isSharedCheck_2411_ = !lean_is_exclusive(v___x_2391_);
if (v_isSharedCheck_2411_ == 0)
{
v___x_2406_ = v___x_2391_;
v_isShared_2407_ = v_isSharedCheck_2411_;
goto v_resetjp_2405_;
}
else
{
lean_inc(v_a_2404_);
lean_dec(v___x_2391_);
v___x_2406_ = lean_box(0);
v_isShared_2407_ = v_isSharedCheck_2411_;
goto v_resetjp_2405_;
}
v_resetjp_2405_:
{
lean_object* v___x_2409_; 
lean_inc(v_a_2404_);
if (v_isShared_2407_ == 0)
{
v___x_2409_ = v___x_2406_;
goto v_reusejp_2408_;
}
else
{
lean_object* v_reuseFailAlloc_2410_; 
v_reuseFailAlloc_2410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2410_, 0, v_a_2404_);
v___x_2409_ = v_reuseFailAlloc_2410_;
goto v_reusejp_2408_;
}
v_reusejp_2408_:
{
v___y_2378_ = v___x_2409_;
v_a_2379_ = v_a_2404_;
goto v___jp_2377_;
}
}
}
}
}
}
else
{
lean_object* v_a_2425_; lean_object* v___x_2427_; uint8_t v_isShared_2428_; uint8_t v_isSharedCheck_2432_; 
v_a_2425_ = lean_ctor_get(v___x_2384_, 0);
v_isSharedCheck_2432_ = !lean_is_exclusive(v___x_2384_);
if (v_isSharedCheck_2432_ == 0)
{
v___x_2427_ = v___x_2384_;
v_isShared_2428_ = v_isSharedCheck_2432_;
goto v_resetjp_2426_;
}
else
{
lean_inc(v_a_2425_);
lean_dec(v___x_2384_);
v___x_2427_ = lean_box(0);
v_isShared_2428_ = v_isSharedCheck_2432_;
goto v_resetjp_2426_;
}
v_resetjp_2426_:
{
lean_object* v___x_2430_; 
lean_inc(v_a_2425_);
if (v_isShared_2428_ == 0)
{
v___x_2430_ = v___x_2427_;
goto v_reusejp_2429_;
}
else
{
lean_object* v_reuseFailAlloc_2431_; 
v_reuseFailAlloc_2431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2431_, 0, v_a_2425_);
v___x_2430_ = v_reuseFailAlloc_2431_;
goto v_reusejp_2429_;
}
v_reusejp_2429_:
{
v___y_2378_ = v___x_2430_;
v_a_2379_ = v_a_2425_;
goto v___jp_2377_;
}
}
}
}
else
{
lean_object* v_a_2433_; lean_object* v___x_2435_; uint8_t v_isShared_2436_; uint8_t v_isSharedCheck_2440_; 
v_a_2433_ = lean_ctor_get(v___x_2382_, 0);
v_isSharedCheck_2440_ = !lean_is_exclusive(v___x_2382_);
if (v_isSharedCheck_2440_ == 0)
{
v___x_2435_ = v___x_2382_;
v_isShared_2436_ = v_isSharedCheck_2440_;
goto v_resetjp_2434_;
}
else
{
lean_inc(v_a_2433_);
lean_dec(v___x_2382_);
v___x_2435_ = lean_box(0);
v_isShared_2436_ = v_isSharedCheck_2440_;
goto v_resetjp_2434_;
}
v_resetjp_2434_:
{
lean_object* v___x_2438_; 
lean_inc(v_a_2433_);
if (v_isShared_2436_ == 0)
{
v___x_2438_ = v___x_2435_;
goto v_reusejp_2437_;
}
else
{
lean_object* v_reuseFailAlloc_2439_; 
v_reuseFailAlloc_2439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2439_, 0, v_a_2433_);
v___x_2438_ = v_reuseFailAlloc_2439_;
goto v_reusejp_2437_;
}
v_reusejp_2437_:
{
v___y_2378_ = v___x_2438_;
v_a_2379_ = v_a_2433_;
goto v___jp_2377_;
}
}
}
v___jp_2355_:
{
if (v___y_2357_ == 0)
{
lean_object* v___x_2358_; 
lean_dec_ref(v___y_2356_);
v___x_2358_ = l_Lean_Meta_SavedState_restore___redArg(v_a_2354_, v___y_2349_, v___y_2351_);
lean_dec(v_a_2354_);
if (lean_obj_tag(v___x_2358_) == 0)
{
lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2367_; 
v_isSharedCheck_2367_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2367_ == 0)
{
lean_object* v_unused_2368_; 
v_unused_2368_ = lean_ctor_get(v___x_2358_, 0);
lean_dec(v_unused_2368_);
v___x_2360_ = v___x_2358_;
v_isShared_2361_ = v_isSharedCheck_2367_;
goto v_resetjp_2359_;
}
else
{
lean_dec(v___x_2358_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2367_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2365_; 
v___x_2362_ = lean_box(0);
v___x_2363_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2363_, 0, v_pf_2347_);
lean_ctor_set(v___x_2363_, 1, v___x_2362_);
if (v_isShared_2361_ == 0)
{
lean_ctor_set(v___x_2360_, 0, v___x_2363_);
v___x_2365_ = v___x_2360_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v___x_2363_);
v___x_2365_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
return v___x_2365_;
}
}
}
else
{
lean_object* v_a_2369_; lean_object* v___x_2371_; uint8_t v_isShared_2372_; uint8_t v_isSharedCheck_2376_; 
lean_dec_ref(v_pf_2347_);
v_a_2369_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2376_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2376_ == 0)
{
v___x_2371_ = v___x_2358_;
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
else
{
lean_inc(v_a_2369_);
lean_dec(v___x_2358_);
v___x_2371_ = lean_box(0);
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
v_resetjp_2370_:
{
lean_object* v___x_2374_; 
if (v_isShared_2372_ == 0)
{
v___x_2374_ = v___x_2371_;
goto v_reusejp_2373_;
}
else
{
lean_object* v_reuseFailAlloc_2375_; 
v_reuseFailAlloc_2375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2375_, 0, v_a_2369_);
v___x_2374_ = v_reuseFailAlloc_2375_;
goto v_reusejp_2373_;
}
v_reusejp_2373_:
{
return v___x_2374_;
}
}
}
}
else
{
lean_dec(v_a_2354_);
lean_dec_ref(v_pf_2347_);
return v___y_2356_;
}
}
v___jp_2377_:
{
uint8_t v___x_2380_; 
v___x_2380_ = l_Lean_Exception_isInterrupt(v_a_2379_);
if (v___x_2380_ == 0)
{
uint8_t v___x_2381_; 
v___x_2381_ = l_Lean_Exception_isRuntime(v_a_2379_);
v___y_2356_ = v___y_2378_;
v___y_2357_ = v___x_2381_;
goto v___jp_2355_;
}
else
{
lean_dec_ref(v_a_2379_);
v___y_2356_ = v___y_2378_;
v___y_2357_ = v___x_2380_;
goto v___jp_2355_;
}
}
}
else
{
lean_object* v_a_2441_; lean_object* v___x_2443_; uint8_t v_isShared_2444_; uint8_t v_isSharedCheck_2448_; 
lean_dec_ref(v_pf_2347_);
v_a_2441_ = lean_ctor_get(v___x_2353_, 0);
v_isSharedCheck_2448_ = !lean_is_exclusive(v___x_2353_);
if (v_isSharedCheck_2448_ == 0)
{
v___x_2443_ = v___x_2353_;
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
else
{
lean_inc(v_a_2441_);
lean_dec(v___x_2353_);
v___x_2443_ = lean_box(0);
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
v_resetjp_2442_:
{
lean_object* v___x_2446_; 
if (v_isShared_2444_ == 0)
{
v___x_2446_ = v___x_2443_;
goto v_reusejp_2445_;
}
else
{
lean_object* v_reuseFailAlloc_2447_; 
v_reuseFailAlloc_2447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2447_, 0, v_a_2441_);
v___x_2446_ = v_reuseFailAlloc_2447_;
goto v_reusejp_2445_;
}
v_reusejp_2445_:
{
return v___x_2446_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0___boxed(lean_object* v_pf_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_){
_start:
{
lean_object* v_res_2455_; 
v_res_2455_ = lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms___lam__0(v_pf_2449_, v___y_2450_, v___y_2451_, v___y_2452_, v___y_2453_);
lean_dec(v___y_2453_);
lean_dec_ref(v___y_2452_);
lean_dec(v___y_2451_);
lean_dec_ref(v___y_2450_);
return v_res_2455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(lean_object* v_k_2471_, lean_object* v_v_2472_, lean_object* v_t_2473_){
_start:
{
lean_object* v___y_2475_; lean_object* v___y_2476_; lean_object* v___y_2477_; lean_object* v___y_2478_; lean_object* v___y_2479_; lean_object* v___y_2480_; lean_object* v___y_2481_; lean_object* v___y_2482_; lean_object* v___y_2483_; lean_object* v___y_2484_; 
if (lean_obj_tag(v_t_2473_) == 0)
{
lean_object* v_size_2488_; lean_object* v_k_2489_; lean_object* v_v_2490_; lean_object* v_l_2491_; lean_object* v_r_2492_; lean_object* v___x_2494_; uint8_t v_isShared_2495_; uint8_t v_isSharedCheck_2756_; 
v_size_2488_ = lean_ctor_get(v_t_2473_, 0);
v_k_2489_ = lean_ctor_get(v_t_2473_, 1);
v_v_2490_ = lean_ctor_get(v_t_2473_, 2);
v_l_2491_ = lean_ctor_get(v_t_2473_, 3);
v_r_2492_ = lean_ctor_get(v_t_2473_, 4);
v_isSharedCheck_2756_ = !lean_is_exclusive(v_t_2473_);
if (v_isSharedCheck_2756_ == 0)
{
v___x_2494_ = v_t_2473_;
v_isShared_2495_ = v_isSharedCheck_2756_;
goto v_resetjp_2493_;
}
else
{
lean_inc(v_r_2492_);
lean_inc(v_l_2491_);
lean_inc(v_v_2490_);
lean_inc(v_k_2489_);
lean_inc(v_size_2488_);
lean_dec(v_t_2473_);
v___x_2494_ = lean_box(0);
v_isShared_2495_ = v_isSharedCheck_2756_;
goto v_resetjp_2493_;
}
v_resetjp_2493_:
{
lean_object* v___y_2497_; lean_object* v___y_2498_; lean_object* v___y_2499_; lean_object* v___y_2500_; lean_object* v___y_2501_; lean_object* v___y_2502_; lean_object* v___y_2503_; lean_object* v___y_2504_; lean_object* v___y_2505_; lean_object* v___y_2506_; lean_object* v___y_2507_; lean_object* v___y_2508_; lean_object* v___y_2616_; lean_object* v___y_2617_; lean_object* v___y_2618_; lean_object* v___y_2619_; lean_object* v___y_2620_; lean_object* v___y_2621_; lean_object* v___y_2622_; lean_object* v___y_2627_; lean_object* v___y_2628_; lean_object* v___y_2629_; lean_object* v___y_2630_; lean_object* v___y_2631_; lean_object* v___y_2632_; lean_object* v___y_2633_; lean_object* v___y_2634_; lean_object* v___y_2635_; lean_object* v___y_2636_; lean_object* v___y_2637_; lean_object* v___y_2638_; lean_object* v_fst_2745_; lean_object* v_snd_2746_; lean_object* v_fst_2747_; lean_object* v_snd_2748_; uint8_t v___x_2749_; 
v_fst_2745_ = lean_ctor_get(v_k_2471_, 0);
v_snd_2746_ = lean_ctor_get(v_k_2471_, 1);
v_fst_2747_ = lean_ctor_get(v_k_2489_, 0);
v_snd_2748_ = lean_ctor_get(v_k_2489_, 1);
v___x_2749_ = lean_nat_dec_lt(v_fst_2745_, v_fst_2747_);
if (v___x_2749_ == 0)
{
uint8_t v___x_2750_; 
v___x_2750_ = lean_nat_dec_eq(v_fst_2745_, v_fst_2747_);
if (v___x_2750_ == 0)
{
lean_dec(v_size_2488_);
goto v___jp_2516_;
}
else
{
uint8_t v___x_2751_; 
v___x_2751_ = lean_unbox(v_snd_2746_);
if (v___x_2751_ == 0)
{
uint8_t v___x_2752_; 
lean_del_object(v___x_2494_);
v___x_2752_ = lean_unbox(v_snd_2748_);
if (v___x_2752_ == 1)
{
lean_dec(v_size_2488_);
goto v___jp_2644_;
}
else
{
lean_object* v___x_2753_; 
lean_dec(v_v_2490_);
lean_dec(v_k_2489_);
v___x_2753_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2753_, 0, v_size_2488_);
lean_ctor_set(v___x_2753_, 1, v_k_2471_);
lean_ctor_set(v___x_2753_, 2, v_v_2472_);
lean_ctor_set(v___x_2753_, 3, v_l_2491_);
lean_ctor_set(v___x_2753_, 4, v_r_2492_);
return v___x_2753_;
}
}
else
{
uint8_t v___x_2754_; 
v___x_2754_ = lean_unbox(v_snd_2748_);
if (v___x_2754_ == 0)
{
lean_dec(v_size_2488_);
goto v___jp_2516_;
}
else
{
lean_object* v___x_2755_; 
lean_del_object(v___x_2494_);
lean_dec(v_v_2490_);
lean_dec(v_k_2489_);
v___x_2755_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2755_, 0, v_size_2488_);
lean_ctor_set(v___x_2755_, 1, v_k_2471_);
lean_ctor_set(v___x_2755_, 2, v_v_2472_);
lean_ctor_set(v___x_2755_, 3, v_l_2491_);
lean_ctor_set(v___x_2755_, 4, v_r_2492_);
return v___x_2755_;
}
}
}
}
else
{
lean_del_object(v___x_2494_);
lean_dec(v_size_2488_);
goto v___jp_2644_;
}
v___jp_2496_:
{
lean_object* v___x_2509_; lean_object* v___x_2511_; 
v___x_2509_ = lean_nat_add(v___y_2499_, v___y_2508_);
lean_dec(v___y_2508_);
lean_dec(v___y_2499_);
if (v_isShared_2495_ == 0)
{
lean_ctor_set(v___x_2494_, 4, v___y_2504_);
lean_ctor_set(v___x_2494_, 0, v___x_2509_);
v___x_2511_ = v___x_2494_;
goto v_reusejp_2510_;
}
else
{
lean_object* v_reuseFailAlloc_2515_; 
v_reuseFailAlloc_2515_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2515_, 0, v___x_2509_);
lean_ctor_set(v_reuseFailAlloc_2515_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2515_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2515_, 3, v_l_2491_);
lean_ctor_set(v_reuseFailAlloc_2515_, 4, v___y_2504_);
v___x_2511_ = v_reuseFailAlloc_2515_;
goto v_reusejp_2510_;
}
v_reusejp_2510_:
{
lean_object* v___x_2512_; 
v___x_2512_ = lean_nat_add(v___y_2505_, v___y_2503_);
lean_dec(v___y_2503_);
if (lean_obj_tag(v___y_2498_) == 0)
{
lean_object* v_size_2513_; 
v_size_2513_ = lean_ctor_get(v___y_2498_, 0);
lean_inc(v_size_2513_);
v___y_2475_ = v___y_2497_;
v___y_2476_ = v___x_2511_;
v___y_2477_ = v___y_2498_;
v___y_2478_ = v___y_2501_;
v___y_2479_ = v___y_2500_;
v___y_2480_ = v___y_2502_;
v___y_2481_ = v___x_2512_;
v___y_2482_ = v___y_2506_;
v___y_2483_ = v___y_2507_;
v___y_2484_ = v_size_2513_;
goto v___jp_2474_;
}
else
{
lean_object* v___x_2514_; 
v___x_2514_ = lean_unsigned_to_nat(0u);
v___y_2475_ = v___y_2497_;
v___y_2476_ = v___x_2511_;
v___y_2477_ = v___y_2498_;
v___y_2478_ = v___y_2501_;
v___y_2479_ = v___y_2500_;
v___y_2480_ = v___y_2502_;
v___y_2481_ = v___x_2512_;
v___y_2482_ = v___y_2506_;
v___y_2483_ = v___y_2507_;
v___y_2484_ = v___x_2514_;
goto v___jp_2474_;
}
}
}
v___jp_2516_:
{
lean_object* v_impl_2517_; lean_object* v___x_2518_; 
v_impl_2517_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(v_k_2471_, v_v_2472_, v_r_2492_);
v___x_2518_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_2491_) == 0)
{
lean_object* v_size_2519_; lean_object* v_size_2520_; lean_object* v_k_2521_; lean_object* v_v_2522_; lean_object* v_l_2523_; lean_object* v_r_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; uint8_t v___x_2527_; 
v_size_2519_ = lean_ctor_get(v_l_2491_, 0);
v_size_2520_ = lean_ctor_get(v_impl_2517_, 0);
lean_inc(v_size_2520_);
v_k_2521_ = lean_ctor_get(v_impl_2517_, 1);
lean_inc(v_k_2521_);
v_v_2522_ = lean_ctor_get(v_impl_2517_, 2);
lean_inc(v_v_2522_);
v_l_2523_ = lean_ctor_get(v_impl_2517_, 3);
lean_inc(v_l_2523_);
v_r_2524_ = lean_ctor_get(v_impl_2517_, 4);
lean_inc(v_r_2524_);
v___x_2525_ = lean_unsigned_to_nat(3u);
v___x_2526_ = lean_nat_mul(v___x_2525_, v_size_2519_);
v___x_2527_ = lean_nat_dec_lt(v___x_2526_, v_size_2520_);
lean_dec(v___x_2526_);
if (v___x_2527_ == 0)
{
lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; 
lean_dec(v_r_2524_);
lean_dec(v_l_2523_);
lean_dec(v_v_2522_);
lean_dec(v_k_2521_);
lean_del_object(v___x_2494_);
v___x_2528_ = lean_nat_add(v___x_2518_, v_size_2519_);
v___x_2529_ = lean_nat_add(v___x_2528_, v_size_2520_);
lean_dec(v_size_2520_);
lean_dec(v___x_2528_);
v___x_2530_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2530_, 0, v___x_2529_);
lean_ctor_set(v___x_2530_, 1, v_k_2489_);
lean_ctor_set(v___x_2530_, 2, v_v_2490_);
lean_ctor_set(v___x_2530_, 3, v_l_2491_);
lean_ctor_set(v___x_2530_, 4, v_impl_2517_);
return v___x_2530_;
}
else
{
lean_object* v___x_2532_; uint8_t v_isShared_2533_; uint8_t v_isSharedCheck_2565_; 
v_isSharedCheck_2565_ = !lean_is_exclusive(v_impl_2517_);
if (v_isSharedCheck_2565_ == 0)
{
lean_object* v_unused_2566_; lean_object* v_unused_2567_; lean_object* v_unused_2568_; lean_object* v_unused_2569_; lean_object* v_unused_2570_; 
v_unused_2566_ = lean_ctor_get(v_impl_2517_, 4);
lean_dec(v_unused_2566_);
v_unused_2567_ = lean_ctor_get(v_impl_2517_, 3);
lean_dec(v_unused_2567_);
v_unused_2568_ = lean_ctor_get(v_impl_2517_, 2);
lean_dec(v_unused_2568_);
v_unused_2569_ = lean_ctor_get(v_impl_2517_, 1);
lean_dec(v_unused_2569_);
v_unused_2570_ = lean_ctor_get(v_impl_2517_, 0);
lean_dec(v_unused_2570_);
v___x_2532_ = v_impl_2517_;
v_isShared_2533_ = v_isSharedCheck_2565_;
goto v_resetjp_2531_;
}
else
{
lean_dec(v_impl_2517_);
v___x_2532_ = lean_box(0);
v_isShared_2533_ = v_isSharedCheck_2565_;
goto v_resetjp_2531_;
}
v_resetjp_2531_:
{
lean_object* v_size_2534_; lean_object* v_k_2535_; lean_object* v_v_2536_; lean_object* v_l_2537_; lean_object* v_r_2538_; lean_object* v_size_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; uint8_t v___x_2542_; 
v_size_2534_ = lean_ctor_get(v_l_2523_, 0);
v_k_2535_ = lean_ctor_get(v_l_2523_, 1);
v_v_2536_ = lean_ctor_get(v_l_2523_, 2);
v_l_2537_ = lean_ctor_get(v_l_2523_, 3);
v_r_2538_ = lean_ctor_get(v_l_2523_, 4);
v_size_2539_ = lean_ctor_get(v_r_2524_, 0);
v___x_2540_ = lean_unsigned_to_nat(2u);
v___x_2541_ = lean_nat_mul(v___x_2540_, v_size_2539_);
v___x_2542_ = lean_nat_dec_lt(v_size_2534_, v___x_2541_);
lean_dec(v___x_2541_);
if (v___x_2542_ == 0)
{
lean_object* v___x_2543_; lean_object* v___x_2544_; 
lean_inc(v_size_2539_);
lean_inc(v_r_2538_);
lean_inc(v_l_2537_);
lean_inc(v_v_2536_);
lean_inc(v_k_2535_);
lean_del_object(v___x_2532_);
lean_dec(v_l_2523_);
v___x_2543_ = lean_nat_add(v___x_2518_, v_size_2519_);
v___x_2544_ = lean_nat_add(v___x_2543_, v_size_2520_);
lean_dec(v_size_2520_);
if (lean_obj_tag(v_l_2537_) == 0)
{
lean_object* v_size_2545_; 
v_size_2545_ = lean_ctor_get(v_l_2537_, 0);
lean_inc(v_size_2545_);
v___y_2497_ = v_v_2536_;
v___y_2498_ = v_r_2538_;
v___y_2499_ = v___x_2543_;
v___y_2500_ = v_v_2522_;
v___y_2501_ = v_r_2524_;
v___y_2502_ = v_k_2521_;
v___y_2503_ = v_size_2539_;
v___y_2504_ = v_l_2537_;
v___y_2505_ = v___x_2518_;
v___y_2506_ = v___x_2544_;
v___y_2507_ = v_k_2535_;
v___y_2508_ = v_size_2545_;
goto v___jp_2496_;
}
else
{
lean_object* v___x_2546_; 
v___x_2546_ = lean_unsigned_to_nat(0u);
v___y_2497_ = v_v_2536_;
v___y_2498_ = v_r_2538_;
v___y_2499_ = v___x_2543_;
v___y_2500_ = v_v_2522_;
v___y_2501_ = v_r_2524_;
v___y_2502_ = v_k_2521_;
v___y_2503_ = v_size_2539_;
v___y_2504_ = v_l_2537_;
v___y_2505_ = v___x_2518_;
v___y_2506_ = v___x_2544_;
v___y_2507_ = v_k_2535_;
v___y_2508_ = v___x_2546_;
goto v___jp_2496_;
}
}
else
{
lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2551_; 
lean_del_object(v___x_2494_);
v___x_2547_ = lean_nat_add(v___x_2518_, v_size_2519_);
v___x_2548_ = lean_nat_add(v___x_2547_, v_size_2520_);
lean_dec(v_size_2520_);
v___x_2549_ = lean_nat_add(v___x_2547_, v_size_2534_);
lean_dec(v___x_2547_);
lean_inc_ref(v_l_2491_);
if (v_isShared_2533_ == 0)
{
lean_ctor_set(v___x_2532_, 4, v_l_2523_);
lean_ctor_set(v___x_2532_, 3, v_l_2491_);
lean_ctor_set(v___x_2532_, 2, v_v_2490_);
lean_ctor_set(v___x_2532_, 1, v_k_2489_);
lean_ctor_set(v___x_2532_, 0, v___x_2549_);
v___x_2551_ = v___x_2532_;
goto v_reusejp_2550_;
}
else
{
lean_object* v_reuseFailAlloc_2564_; 
v_reuseFailAlloc_2564_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2564_, 0, v___x_2549_);
lean_ctor_set(v_reuseFailAlloc_2564_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2564_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2564_, 3, v_l_2491_);
lean_ctor_set(v_reuseFailAlloc_2564_, 4, v_l_2523_);
v___x_2551_ = v_reuseFailAlloc_2564_;
goto v_reusejp_2550_;
}
v_reusejp_2550_:
{
lean_object* v___x_2553_; uint8_t v_isShared_2554_; uint8_t v_isSharedCheck_2558_; 
v_isSharedCheck_2558_ = !lean_is_exclusive(v_l_2491_);
if (v_isSharedCheck_2558_ == 0)
{
lean_object* v_unused_2559_; lean_object* v_unused_2560_; lean_object* v_unused_2561_; lean_object* v_unused_2562_; lean_object* v_unused_2563_; 
v_unused_2559_ = lean_ctor_get(v_l_2491_, 4);
lean_dec(v_unused_2559_);
v_unused_2560_ = lean_ctor_get(v_l_2491_, 3);
lean_dec(v_unused_2560_);
v_unused_2561_ = lean_ctor_get(v_l_2491_, 2);
lean_dec(v_unused_2561_);
v_unused_2562_ = lean_ctor_get(v_l_2491_, 1);
lean_dec(v_unused_2562_);
v_unused_2563_ = lean_ctor_get(v_l_2491_, 0);
lean_dec(v_unused_2563_);
v___x_2553_ = v_l_2491_;
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
else
{
lean_dec(v_l_2491_);
v___x_2553_ = lean_box(0);
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
v_resetjp_2552_:
{
lean_object* v___x_2556_; 
if (v_isShared_2554_ == 0)
{
lean_ctor_set(v___x_2553_, 4, v_r_2524_);
lean_ctor_set(v___x_2553_, 3, v___x_2551_);
lean_ctor_set(v___x_2553_, 2, v_v_2522_);
lean_ctor_set(v___x_2553_, 1, v_k_2521_);
lean_ctor_set(v___x_2553_, 0, v___x_2548_);
v___x_2556_ = v___x_2553_;
goto v_reusejp_2555_;
}
else
{
lean_object* v_reuseFailAlloc_2557_; 
v_reuseFailAlloc_2557_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2557_, 0, v___x_2548_);
lean_ctor_set(v_reuseFailAlloc_2557_, 1, v_k_2521_);
lean_ctor_set(v_reuseFailAlloc_2557_, 2, v_v_2522_);
lean_ctor_set(v_reuseFailAlloc_2557_, 3, v___x_2551_);
lean_ctor_set(v_reuseFailAlloc_2557_, 4, v_r_2524_);
v___x_2556_ = v_reuseFailAlloc_2557_;
goto v_reusejp_2555_;
}
v_reusejp_2555_:
{
return v___x_2556_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_2571_; 
lean_del_object(v___x_2494_);
v_l_2571_ = lean_ctor_get(v_impl_2517_, 3);
lean_inc(v_l_2571_);
if (lean_obj_tag(v_l_2571_) == 0)
{
lean_object* v_r_2572_; lean_object* v_k_2573_; lean_object* v_v_2574_; lean_object* v___x_2576_; uint8_t v_isShared_2577_; uint8_t v_isSharedCheck_2595_; 
v_r_2572_ = lean_ctor_get(v_impl_2517_, 4);
v_k_2573_ = lean_ctor_get(v_impl_2517_, 1);
v_v_2574_ = lean_ctor_get(v_impl_2517_, 2);
v_isSharedCheck_2595_ = !lean_is_exclusive(v_impl_2517_);
if (v_isSharedCheck_2595_ == 0)
{
lean_object* v_unused_2596_; lean_object* v_unused_2597_; 
v_unused_2596_ = lean_ctor_get(v_impl_2517_, 3);
lean_dec(v_unused_2596_);
v_unused_2597_ = lean_ctor_get(v_impl_2517_, 0);
lean_dec(v_unused_2597_);
v___x_2576_ = v_impl_2517_;
v_isShared_2577_ = v_isSharedCheck_2595_;
goto v_resetjp_2575_;
}
else
{
lean_inc(v_r_2572_);
lean_inc(v_v_2574_);
lean_inc(v_k_2573_);
lean_dec(v_impl_2517_);
v___x_2576_ = lean_box(0);
v_isShared_2577_ = v_isSharedCheck_2595_;
goto v_resetjp_2575_;
}
v_resetjp_2575_:
{
lean_object* v_k_2578_; lean_object* v_v_2579_; lean_object* v___x_2581_; uint8_t v_isShared_2582_; uint8_t v_isSharedCheck_2591_; 
v_k_2578_ = lean_ctor_get(v_l_2571_, 1);
v_v_2579_ = lean_ctor_get(v_l_2571_, 2);
v_isSharedCheck_2591_ = !lean_is_exclusive(v_l_2571_);
if (v_isSharedCheck_2591_ == 0)
{
lean_object* v_unused_2592_; lean_object* v_unused_2593_; lean_object* v_unused_2594_; 
v_unused_2592_ = lean_ctor_get(v_l_2571_, 4);
lean_dec(v_unused_2592_);
v_unused_2593_ = lean_ctor_get(v_l_2571_, 3);
lean_dec(v_unused_2593_);
v_unused_2594_ = lean_ctor_get(v_l_2571_, 0);
lean_dec(v_unused_2594_);
v___x_2581_ = v_l_2571_;
v_isShared_2582_ = v_isSharedCheck_2591_;
goto v_resetjp_2580_;
}
else
{
lean_inc(v_v_2579_);
lean_inc(v_k_2578_);
lean_dec(v_l_2571_);
v___x_2581_ = lean_box(0);
v_isShared_2582_ = v_isSharedCheck_2591_;
goto v_resetjp_2580_;
}
v_resetjp_2580_:
{
lean_object* v___x_2583_; lean_object* v___x_2585_; 
v___x_2583_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_2572_, 2);
if (v_isShared_2582_ == 0)
{
lean_ctor_set(v___x_2581_, 4, v_r_2572_);
lean_ctor_set(v___x_2581_, 3, v_r_2572_);
lean_ctor_set(v___x_2581_, 2, v_v_2490_);
lean_ctor_set(v___x_2581_, 1, v_k_2489_);
lean_ctor_set(v___x_2581_, 0, v___x_2518_);
v___x_2585_ = v___x_2581_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2590_; 
v_reuseFailAlloc_2590_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2590_, 0, v___x_2518_);
lean_ctor_set(v_reuseFailAlloc_2590_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2590_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2590_, 3, v_r_2572_);
lean_ctor_set(v_reuseFailAlloc_2590_, 4, v_r_2572_);
v___x_2585_ = v_reuseFailAlloc_2590_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
lean_object* v___x_2587_; 
lean_inc(v_r_2572_);
if (v_isShared_2577_ == 0)
{
lean_ctor_set(v___x_2576_, 3, v_r_2572_);
lean_ctor_set(v___x_2576_, 0, v___x_2518_);
v___x_2587_ = v___x_2576_;
goto v_reusejp_2586_;
}
else
{
lean_object* v_reuseFailAlloc_2589_; 
v_reuseFailAlloc_2589_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2589_, 0, v___x_2518_);
lean_ctor_set(v_reuseFailAlloc_2589_, 1, v_k_2573_);
lean_ctor_set(v_reuseFailAlloc_2589_, 2, v_v_2574_);
lean_ctor_set(v_reuseFailAlloc_2589_, 3, v_r_2572_);
lean_ctor_set(v_reuseFailAlloc_2589_, 4, v_r_2572_);
v___x_2587_ = v_reuseFailAlloc_2589_;
goto v_reusejp_2586_;
}
v_reusejp_2586_:
{
lean_object* v___x_2588_; 
v___x_2588_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2588_, 0, v___x_2583_);
lean_ctor_set(v___x_2588_, 1, v_k_2578_);
lean_ctor_set(v___x_2588_, 2, v_v_2579_);
lean_ctor_set(v___x_2588_, 3, v___x_2585_);
lean_ctor_set(v___x_2588_, 4, v___x_2587_);
return v___x_2588_;
}
}
}
}
}
else
{
lean_object* v_r_2598_; 
v_r_2598_ = lean_ctor_get(v_impl_2517_, 4);
lean_inc(v_r_2598_);
if (lean_obj_tag(v_r_2598_) == 0)
{
lean_object* v_k_2599_; lean_object* v_v_2600_; lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2609_; 
v_k_2599_ = lean_ctor_get(v_impl_2517_, 1);
v_v_2600_ = lean_ctor_get(v_impl_2517_, 2);
v_isSharedCheck_2609_ = !lean_is_exclusive(v_impl_2517_);
if (v_isSharedCheck_2609_ == 0)
{
lean_object* v_unused_2610_; lean_object* v_unused_2611_; lean_object* v_unused_2612_; 
v_unused_2610_ = lean_ctor_get(v_impl_2517_, 4);
lean_dec(v_unused_2610_);
v_unused_2611_ = lean_ctor_get(v_impl_2517_, 3);
lean_dec(v_unused_2611_);
v_unused_2612_ = lean_ctor_get(v_impl_2517_, 0);
lean_dec(v_unused_2612_);
v___x_2602_ = v_impl_2517_;
v_isShared_2603_ = v_isSharedCheck_2609_;
goto v_resetjp_2601_;
}
else
{
lean_inc(v_v_2600_);
lean_inc(v_k_2599_);
lean_dec(v_impl_2517_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2609_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v___x_2604_; lean_object* v___x_2606_; 
v___x_2604_ = lean_unsigned_to_nat(3u);
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 4, v_l_2571_);
lean_ctor_set(v___x_2602_, 2, v_v_2490_);
lean_ctor_set(v___x_2602_, 1, v_k_2489_);
lean_ctor_set(v___x_2602_, 0, v___x_2518_);
v___x_2606_ = v___x_2602_;
goto v_reusejp_2605_;
}
else
{
lean_object* v_reuseFailAlloc_2608_; 
v_reuseFailAlloc_2608_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2608_, 0, v___x_2518_);
lean_ctor_set(v_reuseFailAlloc_2608_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2608_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2608_, 3, v_l_2571_);
lean_ctor_set(v_reuseFailAlloc_2608_, 4, v_l_2571_);
v___x_2606_ = v_reuseFailAlloc_2608_;
goto v_reusejp_2605_;
}
v_reusejp_2605_:
{
lean_object* v___x_2607_; 
v___x_2607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2607_, 0, v___x_2604_);
lean_ctor_set(v___x_2607_, 1, v_k_2599_);
lean_ctor_set(v___x_2607_, 2, v_v_2600_);
lean_ctor_set(v___x_2607_, 3, v___x_2606_);
lean_ctor_set(v___x_2607_, 4, v_r_2598_);
return v___x_2607_;
}
}
}
else
{
lean_object* v___x_2613_; lean_object* v___x_2614_; 
v___x_2613_ = lean_unsigned_to_nat(2u);
v___x_2614_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2614_, 0, v___x_2613_);
lean_ctor_set(v___x_2614_, 1, v_k_2489_);
lean_ctor_set(v___x_2614_, 2, v_v_2490_);
lean_ctor_set(v___x_2614_, 3, v_r_2598_);
lean_ctor_set(v___x_2614_, 4, v_impl_2517_);
return v___x_2614_;
}
}
}
}
v___jp_2615_:
{
lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; 
v___x_2623_ = lean_nat_add(v___y_2617_, v___y_2622_);
lean_dec(v___y_2622_);
lean_dec(v___y_2617_);
v___x_2624_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2624_, 0, v___x_2623_);
lean_ctor_set(v___x_2624_, 1, v_k_2489_);
lean_ctor_set(v___x_2624_, 2, v_v_2490_);
lean_ctor_set(v___x_2624_, 3, v___y_2621_);
lean_ctor_set(v___x_2624_, 4, v_r_2492_);
v___x_2625_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2625_, 0, v___y_2618_);
lean_ctor_set(v___x_2625_, 1, v___y_2620_);
lean_ctor_set(v___x_2625_, 2, v___y_2619_);
lean_ctor_set(v___x_2625_, 3, v___y_2616_);
lean_ctor_set(v___x_2625_, 4, v___x_2624_);
return v___x_2625_;
}
v___jp_2626_:
{
lean_object* v___x_2639_; lean_object* v___x_2640_; lean_object* v___x_2641_; 
v___x_2639_ = lean_nat_add(v___y_2628_, v___y_2638_);
lean_dec(v___y_2638_);
lean_dec(v___y_2628_);
v___x_2640_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2640_, 0, v___x_2639_);
lean_ctor_set(v___x_2640_, 1, v___y_2630_);
lean_ctor_set(v___x_2640_, 2, v___y_2627_);
lean_ctor_set(v___x_2640_, 3, v___y_2629_);
lean_ctor_set(v___x_2640_, 4, v___y_2634_);
v___x_2641_ = lean_nat_add(v___y_2637_, v___y_2633_);
lean_dec(v___y_2633_);
if (lean_obj_tag(v___y_2636_) == 0)
{
lean_object* v_size_2642_; 
v_size_2642_ = lean_ctor_get(v___y_2636_, 0);
lean_inc(v_size_2642_);
v___y_2616_ = v___x_2640_;
v___y_2617_ = v___x_2641_;
v___y_2618_ = v___y_2631_;
v___y_2619_ = v___y_2632_;
v___y_2620_ = v___y_2635_;
v___y_2621_ = v___y_2636_;
v___y_2622_ = v_size_2642_;
goto v___jp_2615_;
}
else
{
lean_object* v___x_2643_; 
v___x_2643_ = lean_unsigned_to_nat(0u);
v___y_2616_ = v___x_2640_;
v___y_2617_ = v___x_2641_;
v___y_2618_ = v___y_2631_;
v___y_2619_ = v___y_2632_;
v___y_2620_ = v___y_2635_;
v___y_2621_ = v___y_2636_;
v___y_2622_ = v___x_2643_;
goto v___jp_2615_;
}
}
v___jp_2644_:
{
lean_object* v_impl_2645_; lean_object* v___x_2646_; 
v_impl_2645_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(v_k_2471_, v_v_2472_, v_l_2491_);
v___x_2646_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_2492_) == 0)
{
lean_object* v_size_2647_; lean_object* v_size_2648_; lean_object* v_k_2649_; lean_object* v_v_2650_; lean_object* v_l_2651_; lean_object* v_r_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; uint8_t v___x_2655_; 
v_size_2647_ = lean_ctor_get(v_r_2492_, 0);
v_size_2648_ = lean_ctor_get(v_impl_2645_, 0);
lean_inc(v_size_2648_);
v_k_2649_ = lean_ctor_get(v_impl_2645_, 1);
lean_inc(v_k_2649_);
v_v_2650_ = lean_ctor_get(v_impl_2645_, 2);
lean_inc(v_v_2650_);
v_l_2651_ = lean_ctor_get(v_impl_2645_, 3);
lean_inc(v_l_2651_);
v_r_2652_ = lean_ctor_get(v_impl_2645_, 4);
lean_inc(v_r_2652_);
v___x_2653_ = lean_unsigned_to_nat(3u);
v___x_2654_ = lean_nat_mul(v___x_2653_, v_size_2647_);
v___x_2655_ = lean_nat_dec_lt(v___x_2654_, v_size_2648_);
lean_dec(v___x_2654_);
if (v___x_2655_ == 0)
{
lean_object* v___x_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; 
lean_dec(v_r_2652_);
lean_dec(v_l_2651_);
lean_dec(v_v_2650_);
lean_dec(v_k_2649_);
v___x_2656_ = lean_nat_add(v___x_2646_, v_size_2648_);
lean_dec(v_size_2648_);
v___x_2657_ = lean_nat_add(v___x_2656_, v_size_2647_);
lean_dec(v___x_2656_);
v___x_2658_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2658_, 0, v___x_2657_);
lean_ctor_set(v___x_2658_, 1, v_k_2489_);
lean_ctor_set(v___x_2658_, 2, v_v_2490_);
lean_ctor_set(v___x_2658_, 3, v_impl_2645_);
lean_ctor_set(v___x_2658_, 4, v_r_2492_);
return v___x_2658_;
}
else
{
lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2695_; 
v_isSharedCheck_2695_ = !lean_is_exclusive(v_impl_2645_);
if (v_isSharedCheck_2695_ == 0)
{
lean_object* v_unused_2696_; lean_object* v_unused_2697_; lean_object* v_unused_2698_; lean_object* v_unused_2699_; lean_object* v_unused_2700_; 
v_unused_2696_ = lean_ctor_get(v_impl_2645_, 4);
lean_dec(v_unused_2696_);
v_unused_2697_ = lean_ctor_get(v_impl_2645_, 3);
lean_dec(v_unused_2697_);
v_unused_2698_ = lean_ctor_get(v_impl_2645_, 2);
lean_dec(v_unused_2698_);
v_unused_2699_ = lean_ctor_get(v_impl_2645_, 1);
lean_dec(v_unused_2699_);
v_unused_2700_ = lean_ctor_get(v_impl_2645_, 0);
lean_dec(v_unused_2700_);
v___x_2660_ = v_impl_2645_;
v_isShared_2661_ = v_isSharedCheck_2695_;
goto v_resetjp_2659_;
}
else
{
lean_dec(v_impl_2645_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2695_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v_size_2662_; lean_object* v_size_2663_; lean_object* v_k_2664_; lean_object* v_v_2665_; lean_object* v_l_2666_; lean_object* v_r_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; uint8_t v___x_2670_; 
v_size_2662_ = lean_ctor_get(v_l_2651_, 0);
v_size_2663_ = lean_ctor_get(v_r_2652_, 0);
v_k_2664_ = lean_ctor_get(v_r_2652_, 1);
v_v_2665_ = lean_ctor_get(v_r_2652_, 2);
v_l_2666_ = lean_ctor_get(v_r_2652_, 3);
v_r_2667_ = lean_ctor_get(v_r_2652_, 4);
v___x_2668_ = lean_unsigned_to_nat(2u);
v___x_2669_ = lean_nat_mul(v___x_2668_, v_size_2662_);
v___x_2670_ = lean_nat_dec_lt(v_size_2663_, v___x_2669_);
lean_dec(v___x_2669_);
if (v___x_2670_ == 0)
{
lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; 
lean_inc(v_r_2667_);
lean_inc(v_l_2666_);
lean_inc(v_v_2665_);
lean_inc(v_k_2664_);
lean_del_object(v___x_2660_);
lean_dec(v_r_2652_);
v___x_2671_ = lean_nat_add(v___x_2646_, v_size_2648_);
lean_dec(v_size_2648_);
v___x_2672_ = lean_nat_add(v___x_2671_, v_size_2647_);
lean_dec(v___x_2671_);
v___x_2673_ = lean_nat_add(v___x_2646_, v_size_2662_);
if (lean_obj_tag(v_l_2666_) == 0)
{
lean_object* v_size_2674_; 
v_size_2674_ = lean_ctor_get(v_l_2666_, 0);
lean_inc(v_size_2674_);
lean_inc(v_size_2647_);
v___y_2627_ = v_v_2650_;
v___y_2628_ = v___x_2673_;
v___y_2629_ = v_l_2651_;
v___y_2630_ = v_k_2649_;
v___y_2631_ = v___x_2672_;
v___y_2632_ = v_v_2665_;
v___y_2633_ = v_size_2647_;
v___y_2634_ = v_l_2666_;
v___y_2635_ = v_k_2664_;
v___y_2636_ = v_r_2667_;
v___y_2637_ = v___x_2646_;
v___y_2638_ = v_size_2674_;
goto v___jp_2626_;
}
else
{
lean_object* v___x_2675_; 
v___x_2675_ = lean_unsigned_to_nat(0u);
lean_inc(v_size_2647_);
v___y_2627_ = v_v_2650_;
v___y_2628_ = v___x_2673_;
v___y_2629_ = v_l_2651_;
v___y_2630_ = v_k_2649_;
v___y_2631_ = v___x_2672_;
v___y_2632_ = v_v_2665_;
v___y_2633_ = v_size_2647_;
v___y_2634_ = v_l_2666_;
v___y_2635_ = v_k_2664_;
v___y_2636_ = v_r_2667_;
v___y_2637_ = v___x_2646_;
v___y_2638_ = v___x_2675_;
goto v___jp_2626_;
}
}
else
{
lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v___x_2679_; lean_object* v___x_2681_; 
v___x_2676_ = lean_nat_add(v___x_2646_, v_size_2648_);
lean_dec(v_size_2648_);
v___x_2677_ = lean_nat_add(v___x_2676_, v_size_2647_);
lean_dec(v___x_2676_);
v___x_2678_ = lean_nat_add(v___x_2646_, v_size_2647_);
v___x_2679_ = lean_nat_add(v___x_2678_, v_size_2663_);
lean_dec(v___x_2678_);
lean_inc_ref(v_r_2492_);
if (v_isShared_2661_ == 0)
{
lean_ctor_set(v___x_2660_, 4, v_r_2492_);
lean_ctor_set(v___x_2660_, 3, v_r_2652_);
lean_ctor_set(v___x_2660_, 2, v_v_2490_);
lean_ctor_set(v___x_2660_, 1, v_k_2489_);
lean_ctor_set(v___x_2660_, 0, v___x_2679_);
v___x_2681_ = v___x_2660_;
goto v_reusejp_2680_;
}
else
{
lean_object* v_reuseFailAlloc_2694_; 
v_reuseFailAlloc_2694_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2694_, 0, v___x_2679_);
lean_ctor_set(v_reuseFailAlloc_2694_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2694_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2694_, 3, v_r_2652_);
lean_ctor_set(v_reuseFailAlloc_2694_, 4, v_r_2492_);
v___x_2681_ = v_reuseFailAlloc_2694_;
goto v_reusejp_2680_;
}
v_reusejp_2680_:
{
lean_object* v___x_2683_; uint8_t v_isShared_2684_; uint8_t v_isSharedCheck_2688_; 
v_isSharedCheck_2688_ = !lean_is_exclusive(v_r_2492_);
if (v_isSharedCheck_2688_ == 0)
{
lean_object* v_unused_2689_; lean_object* v_unused_2690_; lean_object* v_unused_2691_; lean_object* v_unused_2692_; lean_object* v_unused_2693_; 
v_unused_2689_ = lean_ctor_get(v_r_2492_, 4);
lean_dec(v_unused_2689_);
v_unused_2690_ = lean_ctor_get(v_r_2492_, 3);
lean_dec(v_unused_2690_);
v_unused_2691_ = lean_ctor_get(v_r_2492_, 2);
lean_dec(v_unused_2691_);
v_unused_2692_ = lean_ctor_get(v_r_2492_, 1);
lean_dec(v_unused_2692_);
v_unused_2693_ = lean_ctor_get(v_r_2492_, 0);
lean_dec(v_unused_2693_);
v___x_2683_ = v_r_2492_;
v_isShared_2684_ = v_isSharedCheck_2688_;
goto v_resetjp_2682_;
}
else
{
lean_dec(v_r_2492_);
v___x_2683_ = lean_box(0);
v_isShared_2684_ = v_isSharedCheck_2688_;
goto v_resetjp_2682_;
}
v_resetjp_2682_:
{
lean_object* v___x_2686_; 
if (v_isShared_2684_ == 0)
{
lean_ctor_set(v___x_2683_, 4, v___x_2681_);
lean_ctor_set(v___x_2683_, 3, v_l_2651_);
lean_ctor_set(v___x_2683_, 2, v_v_2650_);
lean_ctor_set(v___x_2683_, 1, v_k_2649_);
lean_ctor_set(v___x_2683_, 0, v___x_2677_);
v___x_2686_ = v___x_2683_;
goto v_reusejp_2685_;
}
else
{
lean_object* v_reuseFailAlloc_2687_; 
v_reuseFailAlloc_2687_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2687_, 0, v___x_2677_);
lean_ctor_set(v_reuseFailAlloc_2687_, 1, v_k_2649_);
lean_ctor_set(v_reuseFailAlloc_2687_, 2, v_v_2650_);
lean_ctor_set(v_reuseFailAlloc_2687_, 3, v_l_2651_);
lean_ctor_set(v_reuseFailAlloc_2687_, 4, v___x_2681_);
v___x_2686_ = v_reuseFailAlloc_2687_;
goto v_reusejp_2685_;
}
v_reusejp_2685_:
{
return v___x_2686_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_2701_; 
v_l_2701_ = lean_ctor_get(v_impl_2645_, 3);
lean_inc(v_l_2701_);
if (lean_obj_tag(v_l_2701_) == 0)
{
lean_object* v_r_2702_; lean_object* v_k_2703_; lean_object* v_v_2704_; lean_object* v___x_2706_; uint8_t v_isShared_2707_; uint8_t v_isSharedCheck_2713_; 
v_r_2702_ = lean_ctor_get(v_impl_2645_, 4);
v_k_2703_ = lean_ctor_get(v_impl_2645_, 1);
v_v_2704_ = lean_ctor_get(v_impl_2645_, 2);
v_isSharedCheck_2713_ = !lean_is_exclusive(v_impl_2645_);
if (v_isSharedCheck_2713_ == 0)
{
lean_object* v_unused_2714_; lean_object* v_unused_2715_; 
v_unused_2714_ = lean_ctor_get(v_impl_2645_, 3);
lean_dec(v_unused_2714_);
v_unused_2715_ = lean_ctor_get(v_impl_2645_, 0);
lean_dec(v_unused_2715_);
v___x_2706_ = v_impl_2645_;
v_isShared_2707_ = v_isSharedCheck_2713_;
goto v_resetjp_2705_;
}
else
{
lean_inc(v_r_2702_);
lean_inc(v_v_2704_);
lean_inc(v_k_2703_);
lean_dec(v_impl_2645_);
v___x_2706_ = lean_box(0);
v_isShared_2707_ = v_isSharedCheck_2713_;
goto v_resetjp_2705_;
}
v_resetjp_2705_:
{
lean_object* v___x_2708_; lean_object* v___x_2710_; 
v___x_2708_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_2702_);
if (v_isShared_2707_ == 0)
{
lean_ctor_set(v___x_2706_, 3, v_r_2702_);
lean_ctor_set(v___x_2706_, 2, v_v_2490_);
lean_ctor_set(v___x_2706_, 1, v_k_2489_);
lean_ctor_set(v___x_2706_, 0, v___x_2646_);
v___x_2710_ = v___x_2706_;
goto v_reusejp_2709_;
}
else
{
lean_object* v_reuseFailAlloc_2712_; 
v_reuseFailAlloc_2712_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2712_, 0, v___x_2646_);
lean_ctor_set(v_reuseFailAlloc_2712_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2712_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2712_, 3, v_r_2702_);
lean_ctor_set(v_reuseFailAlloc_2712_, 4, v_r_2702_);
v___x_2710_ = v_reuseFailAlloc_2712_;
goto v_reusejp_2709_;
}
v_reusejp_2709_:
{
lean_object* v___x_2711_; 
v___x_2711_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2711_, 0, v___x_2708_);
lean_ctor_set(v___x_2711_, 1, v_k_2703_);
lean_ctor_set(v___x_2711_, 2, v_v_2704_);
lean_ctor_set(v___x_2711_, 3, v_l_2701_);
lean_ctor_set(v___x_2711_, 4, v___x_2710_);
return v___x_2711_;
}
}
}
else
{
lean_object* v_r_2716_; 
v_r_2716_ = lean_ctor_get(v_impl_2645_, 4);
lean_inc(v_r_2716_);
if (lean_obj_tag(v_r_2716_) == 0)
{
lean_object* v_k_2717_; lean_object* v_v_2718_; lean_object* v___x_2720_; uint8_t v_isShared_2721_; uint8_t v_isSharedCheck_2739_; 
v_k_2717_ = lean_ctor_get(v_impl_2645_, 1);
v_v_2718_ = lean_ctor_get(v_impl_2645_, 2);
v_isSharedCheck_2739_ = !lean_is_exclusive(v_impl_2645_);
if (v_isSharedCheck_2739_ == 0)
{
lean_object* v_unused_2740_; lean_object* v_unused_2741_; lean_object* v_unused_2742_; 
v_unused_2740_ = lean_ctor_get(v_impl_2645_, 4);
lean_dec(v_unused_2740_);
v_unused_2741_ = lean_ctor_get(v_impl_2645_, 3);
lean_dec(v_unused_2741_);
v_unused_2742_ = lean_ctor_get(v_impl_2645_, 0);
lean_dec(v_unused_2742_);
v___x_2720_ = v_impl_2645_;
v_isShared_2721_ = v_isSharedCheck_2739_;
goto v_resetjp_2719_;
}
else
{
lean_inc(v_v_2718_);
lean_inc(v_k_2717_);
lean_dec(v_impl_2645_);
v___x_2720_ = lean_box(0);
v_isShared_2721_ = v_isSharedCheck_2739_;
goto v_resetjp_2719_;
}
v_resetjp_2719_:
{
lean_object* v_k_2722_; lean_object* v_v_2723_; lean_object* v___x_2725_; uint8_t v_isShared_2726_; uint8_t v_isSharedCheck_2735_; 
v_k_2722_ = lean_ctor_get(v_r_2716_, 1);
v_v_2723_ = lean_ctor_get(v_r_2716_, 2);
v_isSharedCheck_2735_ = !lean_is_exclusive(v_r_2716_);
if (v_isSharedCheck_2735_ == 0)
{
lean_object* v_unused_2736_; lean_object* v_unused_2737_; lean_object* v_unused_2738_; 
v_unused_2736_ = lean_ctor_get(v_r_2716_, 4);
lean_dec(v_unused_2736_);
v_unused_2737_ = lean_ctor_get(v_r_2716_, 3);
lean_dec(v_unused_2737_);
v_unused_2738_ = lean_ctor_get(v_r_2716_, 0);
lean_dec(v_unused_2738_);
v___x_2725_ = v_r_2716_;
v_isShared_2726_ = v_isSharedCheck_2735_;
goto v_resetjp_2724_;
}
else
{
lean_inc(v_v_2723_);
lean_inc(v_k_2722_);
lean_dec(v_r_2716_);
v___x_2725_ = lean_box(0);
v_isShared_2726_ = v_isSharedCheck_2735_;
goto v_resetjp_2724_;
}
v_resetjp_2724_:
{
lean_object* v___x_2727_; lean_object* v___x_2729_; 
v___x_2727_ = lean_unsigned_to_nat(3u);
if (v_isShared_2726_ == 0)
{
lean_ctor_set(v___x_2725_, 4, v_l_2701_);
lean_ctor_set(v___x_2725_, 3, v_l_2701_);
lean_ctor_set(v___x_2725_, 2, v_v_2718_);
lean_ctor_set(v___x_2725_, 1, v_k_2717_);
lean_ctor_set(v___x_2725_, 0, v___x_2646_);
v___x_2729_ = v___x_2725_;
goto v_reusejp_2728_;
}
else
{
lean_object* v_reuseFailAlloc_2734_; 
v_reuseFailAlloc_2734_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2734_, 0, v___x_2646_);
lean_ctor_set(v_reuseFailAlloc_2734_, 1, v_k_2717_);
lean_ctor_set(v_reuseFailAlloc_2734_, 2, v_v_2718_);
lean_ctor_set(v_reuseFailAlloc_2734_, 3, v_l_2701_);
lean_ctor_set(v_reuseFailAlloc_2734_, 4, v_l_2701_);
v___x_2729_ = v_reuseFailAlloc_2734_;
goto v_reusejp_2728_;
}
v_reusejp_2728_:
{
lean_object* v___x_2731_; 
if (v_isShared_2721_ == 0)
{
lean_ctor_set(v___x_2720_, 4, v_l_2701_);
lean_ctor_set(v___x_2720_, 2, v_v_2490_);
lean_ctor_set(v___x_2720_, 1, v_k_2489_);
lean_ctor_set(v___x_2720_, 0, v___x_2646_);
v___x_2731_ = v___x_2720_;
goto v_reusejp_2730_;
}
else
{
lean_object* v_reuseFailAlloc_2733_; 
v_reuseFailAlloc_2733_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2733_, 0, v___x_2646_);
lean_ctor_set(v_reuseFailAlloc_2733_, 1, v_k_2489_);
lean_ctor_set(v_reuseFailAlloc_2733_, 2, v_v_2490_);
lean_ctor_set(v_reuseFailAlloc_2733_, 3, v_l_2701_);
lean_ctor_set(v_reuseFailAlloc_2733_, 4, v_l_2701_);
v___x_2731_ = v_reuseFailAlloc_2733_;
goto v_reusejp_2730_;
}
v_reusejp_2730_:
{
lean_object* v___x_2732_; 
v___x_2732_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2732_, 0, v___x_2727_);
lean_ctor_set(v___x_2732_, 1, v_k_2722_);
lean_ctor_set(v___x_2732_, 2, v_v_2723_);
lean_ctor_set(v___x_2732_, 3, v___x_2729_);
lean_ctor_set(v___x_2732_, 4, v___x_2731_);
return v___x_2732_;
}
}
}
}
}
else
{
lean_object* v___x_2743_; lean_object* v___x_2744_; 
v___x_2743_ = lean_unsigned_to_nat(2u);
v___x_2744_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2744_, 0, v___x_2743_);
lean_ctor_set(v___x_2744_, 1, v_k_2489_);
lean_ctor_set(v___x_2744_, 2, v_v_2490_);
lean_ctor_set(v___x_2744_, 3, v_impl_2645_);
lean_ctor_set(v___x_2744_, 4, v_r_2716_);
return v___x_2744_;
}
}
}
}
}
}
else
{
lean_object* v___x_2757_; lean_object* v___x_2758_; 
v___x_2757_ = lean_unsigned_to_nat(1u);
v___x_2758_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2758_, 0, v___x_2757_);
lean_ctor_set(v___x_2758_, 1, v_k_2471_);
lean_ctor_set(v___x_2758_, 2, v_v_2472_);
lean_ctor_set(v___x_2758_, 3, v_t_2473_);
lean_ctor_set(v___x_2758_, 4, v_t_2473_);
return v___x_2758_;
}
v___jp_2474_:
{
lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; 
v___x_2485_ = lean_nat_add(v___y_2481_, v___y_2484_);
lean_dec(v___y_2484_);
lean_dec(v___y_2481_);
v___x_2486_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2486_, 0, v___x_2485_);
lean_ctor_set(v___x_2486_, 1, v___y_2480_);
lean_ctor_set(v___x_2486_, 2, v___y_2479_);
lean_ctor_set(v___x_2486_, 3, v___y_2477_);
lean_ctor_set(v___x_2486_, 4, v___y_2478_);
v___x_2487_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2487_, 0, v___y_2482_);
lean_ctor_set(v___x_2487_, 1, v___y_2483_);
lean_ctor_set(v___x_2487_, 2, v___y_2475_);
lean_ctor_set(v___x_2487_, 3, v___y_2476_);
lean_ctor_set(v___x_2487_, 4, v___x_2486_);
return v___x_2487_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(lean_object* v_k_2759_, lean_object* v_t_2760_){
_start:
{
if (lean_obj_tag(v_t_2760_) == 0)
{
lean_object* v_k_2761_; lean_object* v_l_2762_; lean_object* v_r_2763_; lean_object* v_fst_2764_; lean_object* v_snd_2765_; lean_object* v_fst_2766_; lean_object* v_snd_2767_; uint8_t v___x_2768_; 
v_k_2761_ = lean_ctor_get(v_t_2760_, 1);
v_l_2762_ = lean_ctor_get(v_t_2760_, 3);
v_r_2763_ = lean_ctor_get(v_t_2760_, 4);
v_fst_2764_ = lean_ctor_get(v_k_2759_, 0);
v_snd_2765_ = lean_ctor_get(v_k_2759_, 1);
v_fst_2766_ = lean_ctor_get(v_k_2761_, 0);
v_snd_2767_ = lean_ctor_get(v_k_2761_, 1);
v___x_2768_ = lean_nat_dec_lt(v_fst_2764_, v_fst_2766_);
if (v___x_2768_ == 0)
{
uint8_t v___x_2769_; 
v___x_2769_ = lean_nat_dec_eq(v_fst_2764_, v_fst_2766_);
if (v___x_2769_ == 0)
{
v_t_2760_ = v_r_2763_;
goto _start;
}
else
{
uint8_t v___x_2771_; 
v___x_2771_ = lean_unbox(v_snd_2765_);
if (v___x_2771_ == 0)
{
uint8_t v___x_2772_; 
v___x_2772_ = lean_unbox(v_snd_2767_);
if (v___x_2772_ == 1)
{
v_t_2760_ = v_l_2762_;
goto _start;
}
else
{
return v___x_2769_;
}
}
else
{
uint8_t v___x_2774_; 
v___x_2774_ = lean_unbox(v_snd_2767_);
if (v___x_2774_ == 0)
{
v_t_2760_ = v_r_2763_;
goto _start;
}
else
{
uint8_t v___x_2776_; 
v___x_2776_ = lean_unbox(v_snd_2765_);
return v___x_2776_;
}
}
}
}
else
{
v_t_2760_ = v_l_2762_;
goto _start;
}
}
else
{
uint8_t v___x_2778_; 
v___x_2778_ = 0;
return v___x_2778_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg___boxed(lean_object* v_k_2779_, lean_object* v_t_2780_){
_start:
{
uint8_t v_res_2781_; lean_object* v_r_2782_; 
v_res_2781_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(v_k_2779_, v_t_2780_);
lean_dec(v_t_2780_);
lean_dec_ref(v_k_2779_);
v_r_2782_ = lean_box(v_res_2781_);
return v_r_2782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(lean_object* v_f_2783_, lean_object* v_e_x27_2784_, lean_object* v_a_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_, lean_object* v___y_2791_){
_start:
{
lean_object* v___x_2793_; 
lean_inc(v___y_2791_);
lean_inc_ref(v___y_2790_);
lean_inc(v___y_2789_);
lean_inc_ref(v___y_2788_);
lean_inc(v___y_2787_);
lean_inc_ref(v___y_2786_);
lean_inc_ref(v_e_x27_2784_);
v___x_2793_ = lean_apply_9(v_f_2783_, v_a_2785_, v_e_x27_2784_, v___y_2786_, v___y_2787_, v___y_2788_, v___y_2789_, v___y_2790_, v___y_2791_, lean_box(0));
if (lean_obj_tag(v___x_2793_) == 0)
{
lean_object* v_a_2794_; lean_object* v___x_2796_; uint8_t v_isShared_2797_; uint8_t v_isSharedCheck_2802_; 
v_a_2794_ = lean_ctor_get(v___x_2793_, 0);
v_isSharedCheck_2802_ = !lean_is_exclusive(v___x_2793_);
if (v_isSharedCheck_2802_ == 0)
{
v___x_2796_ = v___x_2793_;
v_isShared_2797_ = v_isSharedCheck_2802_;
goto v_resetjp_2795_;
}
else
{
lean_inc(v_a_2794_);
lean_dec(v___x_2793_);
v___x_2796_ = lean_box(0);
v_isShared_2797_ = v_isSharedCheck_2802_;
goto v_resetjp_2795_;
}
v_resetjp_2795_:
{
lean_object* v___x_2798_; lean_object* v___x_2800_; 
v___x_2798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2798_, 0, v_e_x27_2784_);
lean_ctor_set(v___x_2798_, 1, v_a_2794_);
if (v_isShared_2797_ == 0)
{
lean_ctor_set(v___x_2796_, 0, v___x_2798_);
v___x_2800_ = v___x_2796_;
goto v_reusejp_2799_;
}
else
{
lean_object* v_reuseFailAlloc_2801_; 
v_reuseFailAlloc_2801_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2801_, 0, v___x_2798_);
v___x_2800_ = v_reuseFailAlloc_2801_;
goto v_reusejp_2799_;
}
v_reusejp_2799_:
{
return v___x_2800_;
}
}
}
else
{
lean_object* v_a_2803_; lean_object* v___x_2805_; uint8_t v_isShared_2806_; uint8_t v_isSharedCheck_2810_; 
lean_dec_ref(v_e_x27_2784_);
v_a_2803_ = lean_ctor_get(v___x_2793_, 0);
v_isSharedCheck_2810_ = !lean_is_exclusive(v___x_2793_);
if (v_isSharedCheck_2810_ == 0)
{
v___x_2805_ = v___x_2793_;
v_isShared_2806_ = v_isSharedCheck_2810_;
goto v_resetjp_2804_;
}
else
{
lean_inc(v_a_2803_);
lean_dec(v___x_2793_);
v___x_2805_ = lean_box(0);
v_isShared_2806_ = v_isSharedCheck_2810_;
goto v_resetjp_2804_;
}
v_resetjp_2804_:
{
lean_object* v___x_2808_; 
if (v_isShared_2806_ == 0)
{
v___x_2808_ = v___x_2805_;
goto v_reusejp_2807_;
}
else
{
lean_object* v_reuseFailAlloc_2809_; 
v_reuseFailAlloc_2809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2809_, 0, v_a_2803_);
v___x_2808_ = v_reuseFailAlloc_2809_;
goto v_reusejp_2807_;
}
v_reusejp_2807_:
{
return v___x_2808_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_f_2811_, lean_object* v_e_x27_2812_, lean_object* v_a_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_){
_start:
{
lean_object* v_res_2821_; 
v_res_2821_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2811_, v_e_x27_2812_, v_a_2813_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_);
lean_dec(v___y_2819_);
lean_dec_ref(v___y_2818_);
lean_dec(v___y_2817_);
lean_dec_ref(v___y_2816_);
lean_dec(v___y_2815_);
lean_dec_ref(v___y_2814_);
return v_res_2821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg(lean_object* v_f_2822_, lean_object* v_x_2823_, lean_object* v___y_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_, lean_object* v___y_2830_){
_start:
{
switch(lean_obj_tag(v_x_2823_))
{
case 7:
{
lean_object* v_binderName_2832_; lean_object* v_binderType_2833_; lean_object* v_body_2834_; uint8_t v_binderInfo_2835_; lean_object* v___x_2836_; 
v_binderName_2832_ = lean_ctor_get(v_x_2823_, 0);
v_binderType_2833_ = lean_ctor_get(v_x_2823_, 1);
v_body_2834_ = lean_ctor_get(v_x_2823_, 2);
v_binderInfo_2835_ = lean_ctor_get_uint8(v_x_2823_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_2833_);
lean_inc_ref(v_f_2822_);
v___x_2836_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_binderType_2833_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2836_) == 0)
{
lean_object* v_a_2837_; lean_object* v_fst_2838_; lean_object* v_snd_2839_; lean_object* v___x_2840_; 
v_a_2837_ = lean_ctor_get(v___x_2836_, 0);
lean_inc(v_a_2837_);
lean_dec_ref_known(v___x_2836_, 1);
v_fst_2838_ = lean_ctor_get(v_a_2837_, 0);
lean_inc(v_fst_2838_);
v_snd_2839_ = lean_ctor_get(v_a_2837_, 1);
lean_inc(v_snd_2839_);
lean_dec(v_a_2837_);
lean_inc_ref(v_body_2834_);
v___x_2840_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_body_2834_, v_snd_2839_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2840_) == 0)
{
lean_object* v_a_2841_; lean_object* v___x_2843_; uint8_t v_isShared_2844_; uint8_t v_isSharedCheck_2870_; 
v_a_2841_ = lean_ctor_get(v___x_2840_, 0);
v_isSharedCheck_2870_ = !lean_is_exclusive(v___x_2840_);
if (v_isSharedCheck_2870_ == 0)
{
v___x_2843_ = v___x_2840_;
v_isShared_2844_ = v_isSharedCheck_2870_;
goto v_resetjp_2842_;
}
else
{
lean_inc(v_a_2841_);
lean_dec(v___x_2840_);
v___x_2843_ = lean_box(0);
v_isShared_2844_ = v_isSharedCheck_2870_;
goto v_resetjp_2842_;
}
v_resetjp_2842_:
{
lean_object* v_fst_2845_; lean_object* v_snd_2846_; lean_object* v___x_2848_; uint8_t v_isShared_2849_; uint8_t v_isSharedCheck_2869_; 
v_fst_2845_ = lean_ctor_get(v_a_2841_, 0);
v_snd_2846_ = lean_ctor_get(v_a_2841_, 1);
v_isSharedCheck_2869_ = !lean_is_exclusive(v_a_2841_);
if (v_isSharedCheck_2869_ == 0)
{
v___x_2848_ = v_a_2841_;
v_isShared_2849_ = v_isSharedCheck_2869_;
goto v_resetjp_2847_;
}
else
{
lean_inc(v_snd_2846_);
lean_inc(v_fst_2845_);
lean_dec(v_a_2841_);
v___x_2848_ = lean_box(0);
v_isShared_2849_ = v_isSharedCheck_2869_;
goto v_resetjp_2847_;
}
v_resetjp_2847_:
{
lean_object* v___y_2851_; uint8_t v___y_2859_; size_t v___x_2863_; size_t v___x_2864_; uint8_t v___x_2865_; 
v___x_2863_ = lean_ptr_addr(v_binderType_2833_);
v___x_2864_ = lean_ptr_addr(v_fst_2838_);
v___x_2865_ = lean_usize_dec_eq(v___x_2863_, v___x_2864_);
if (v___x_2865_ == 0)
{
v___y_2859_ = v___x_2865_;
goto v___jp_2858_;
}
else
{
size_t v___x_2866_; size_t v___x_2867_; uint8_t v___x_2868_; 
v___x_2866_ = lean_ptr_addr(v_body_2834_);
v___x_2867_ = lean_ptr_addr(v_fst_2845_);
v___x_2868_ = lean_usize_dec_eq(v___x_2866_, v___x_2867_);
v___y_2859_ = v___x_2868_;
goto v___jp_2858_;
}
v___jp_2850_:
{
lean_object* v___x_2853_; 
if (v_isShared_2849_ == 0)
{
lean_ctor_set(v___x_2848_, 0, v___y_2851_);
v___x_2853_ = v___x_2848_;
goto v_reusejp_2852_;
}
else
{
lean_object* v_reuseFailAlloc_2857_; 
v_reuseFailAlloc_2857_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2857_, 0, v___y_2851_);
lean_ctor_set(v_reuseFailAlloc_2857_, 1, v_snd_2846_);
v___x_2853_ = v_reuseFailAlloc_2857_;
goto v_reusejp_2852_;
}
v_reusejp_2852_:
{
lean_object* v___x_2855_; 
if (v_isShared_2844_ == 0)
{
lean_ctor_set(v___x_2843_, 0, v___x_2853_);
v___x_2855_ = v___x_2843_;
goto v_reusejp_2854_;
}
else
{
lean_object* v_reuseFailAlloc_2856_; 
v_reuseFailAlloc_2856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2856_, 0, v___x_2853_);
v___x_2855_ = v_reuseFailAlloc_2856_;
goto v_reusejp_2854_;
}
v_reusejp_2854_:
{
return v___x_2855_;
}
}
}
v___jp_2858_:
{
if (v___y_2859_ == 0)
{
lean_object* v___x_2860_; 
lean_inc(v_binderName_2832_);
lean_dec_ref_known(v_x_2823_, 3);
v___x_2860_ = l_Lean_Expr_forallE___override(v_binderName_2832_, v_fst_2838_, v_fst_2845_, v_binderInfo_2835_);
v___y_2851_ = v___x_2860_;
goto v___jp_2850_;
}
else
{
uint8_t v___x_2861_; 
v___x_2861_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_2835_, v_binderInfo_2835_);
if (v___x_2861_ == 0)
{
lean_object* v___x_2862_; 
lean_inc(v_binderName_2832_);
lean_dec_ref_known(v_x_2823_, 3);
v___x_2862_ = l_Lean_Expr_forallE___override(v_binderName_2832_, v_fst_2838_, v_fst_2845_, v_binderInfo_2835_);
v___y_2851_ = v___x_2862_;
goto v___jp_2850_;
}
else
{
lean_dec(v_fst_2845_);
lean_dec(v_fst_2838_);
v___y_2851_ = v_x_2823_;
goto v___jp_2850_;
}
}
}
}
}
}
else
{
lean_dec(v_fst_2838_);
lean_dec_ref_known(v_x_2823_, 3);
return v___x_2840_;
}
}
else
{
lean_dec_ref_known(v_x_2823_, 3);
lean_dec_ref(v_f_2822_);
return v___x_2836_;
}
}
case 6:
{
lean_object* v_binderName_2871_; lean_object* v_binderType_2872_; lean_object* v_body_2873_; uint8_t v_binderInfo_2874_; lean_object* v___x_2875_; 
v_binderName_2871_ = lean_ctor_get(v_x_2823_, 0);
v_binderType_2872_ = lean_ctor_get(v_x_2823_, 1);
v_body_2873_ = lean_ctor_get(v_x_2823_, 2);
v_binderInfo_2874_ = lean_ctor_get_uint8(v_x_2823_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_2872_);
lean_inc_ref(v_f_2822_);
v___x_2875_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_binderType_2872_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2875_) == 0)
{
lean_object* v_a_2876_; lean_object* v_fst_2877_; lean_object* v_snd_2878_; lean_object* v___x_2879_; 
v_a_2876_ = lean_ctor_get(v___x_2875_, 0);
lean_inc(v_a_2876_);
lean_dec_ref_known(v___x_2875_, 1);
v_fst_2877_ = lean_ctor_get(v_a_2876_, 0);
lean_inc(v_fst_2877_);
v_snd_2878_ = lean_ctor_get(v_a_2876_, 1);
lean_inc(v_snd_2878_);
lean_dec(v_a_2876_);
lean_inc_ref(v_body_2873_);
v___x_2879_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_body_2873_, v_snd_2878_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2879_) == 0)
{
lean_object* v_a_2880_; lean_object* v___x_2882_; uint8_t v_isShared_2883_; uint8_t v_isSharedCheck_2909_; 
v_a_2880_ = lean_ctor_get(v___x_2879_, 0);
v_isSharedCheck_2909_ = !lean_is_exclusive(v___x_2879_);
if (v_isSharedCheck_2909_ == 0)
{
v___x_2882_ = v___x_2879_;
v_isShared_2883_ = v_isSharedCheck_2909_;
goto v_resetjp_2881_;
}
else
{
lean_inc(v_a_2880_);
lean_dec(v___x_2879_);
v___x_2882_ = lean_box(0);
v_isShared_2883_ = v_isSharedCheck_2909_;
goto v_resetjp_2881_;
}
v_resetjp_2881_:
{
lean_object* v_fst_2884_; lean_object* v_snd_2885_; lean_object* v___x_2887_; uint8_t v_isShared_2888_; uint8_t v_isSharedCheck_2908_; 
v_fst_2884_ = lean_ctor_get(v_a_2880_, 0);
v_snd_2885_ = lean_ctor_get(v_a_2880_, 1);
v_isSharedCheck_2908_ = !lean_is_exclusive(v_a_2880_);
if (v_isSharedCheck_2908_ == 0)
{
v___x_2887_ = v_a_2880_;
v_isShared_2888_ = v_isSharedCheck_2908_;
goto v_resetjp_2886_;
}
else
{
lean_inc(v_snd_2885_);
lean_inc(v_fst_2884_);
lean_dec(v_a_2880_);
v___x_2887_ = lean_box(0);
v_isShared_2888_ = v_isSharedCheck_2908_;
goto v_resetjp_2886_;
}
v_resetjp_2886_:
{
lean_object* v___y_2890_; uint8_t v___y_2898_; size_t v___x_2902_; size_t v___x_2903_; uint8_t v___x_2904_; 
v___x_2902_ = lean_ptr_addr(v_binderType_2872_);
v___x_2903_ = lean_ptr_addr(v_fst_2877_);
v___x_2904_ = lean_usize_dec_eq(v___x_2902_, v___x_2903_);
if (v___x_2904_ == 0)
{
v___y_2898_ = v___x_2904_;
goto v___jp_2897_;
}
else
{
size_t v___x_2905_; size_t v___x_2906_; uint8_t v___x_2907_; 
v___x_2905_ = lean_ptr_addr(v_body_2873_);
v___x_2906_ = lean_ptr_addr(v_fst_2884_);
v___x_2907_ = lean_usize_dec_eq(v___x_2905_, v___x_2906_);
v___y_2898_ = v___x_2907_;
goto v___jp_2897_;
}
v___jp_2889_:
{
lean_object* v___x_2892_; 
if (v_isShared_2888_ == 0)
{
lean_ctor_set(v___x_2887_, 0, v___y_2890_);
v___x_2892_ = v___x_2887_;
goto v_reusejp_2891_;
}
else
{
lean_object* v_reuseFailAlloc_2896_; 
v_reuseFailAlloc_2896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2896_, 0, v___y_2890_);
lean_ctor_set(v_reuseFailAlloc_2896_, 1, v_snd_2885_);
v___x_2892_ = v_reuseFailAlloc_2896_;
goto v_reusejp_2891_;
}
v_reusejp_2891_:
{
lean_object* v___x_2894_; 
if (v_isShared_2883_ == 0)
{
lean_ctor_set(v___x_2882_, 0, v___x_2892_);
v___x_2894_ = v___x_2882_;
goto v_reusejp_2893_;
}
else
{
lean_object* v_reuseFailAlloc_2895_; 
v_reuseFailAlloc_2895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2895_, 0, v___x_2892_);
v___x_2894_ = v_reuseFailAlloc_2895_;
goto v_reusejp_2893_;
}
v_reusejp_2893_:
{
return v___x_2894_;
}
}
}
v___jp_2897_:
{
if (v___y_2898_ == 0)
{
lean_object* v___x_2899_; 
lean_inc(v_binderName_2871_);
lean_dec_ref_known(v_x_2823_, 3);
v___x_2899_ = l_Lean_Expr_lam___override(v_binderName_2871_, v_fst_2877_, v_fst_2884_, v_binderInfo_2874_);
v___y_2890_ = v___x_2899_;
goto v___jp_2889_;
}
else
{
uint8_t v___x_2900_; 
v___x_2900_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_2874_, v_binderInfo_2874_);
if (v___x_2900_ == 0)
{
lean_object* v___x_2901_; 
lean_inc(v_binderName_2871_);
lean_dec_ref_known(v_x_2823_, 3);
v___x_2901_ = l_Lean_Expr_lam___override(v_binderName_2871_, v_fst_2877_, v_fst_2884_, v_binderInfo_2874_);
v___y_2890_ = v___x_2901_;
goto v___jp_2889_;
}
else
{
lean_dec(v_fst_2884_);
lean_dec(v_fst_2877_);
v___y_2890_ = v_x_2823_;
goto v___jp_2889_;
}
}
}
}
}
}
else
{
lean_dec(v_fst_2877_);
lean_dec_ref_known(v_x_2823_, 3);
return v___x_2879_;
}
}
else
{
lean_dec_ref_known(v_x_2823_, 3);
lean_dec_ref(v_f_2822_);
return v___x_2875_;
}
}
case 10:
{
lean_object* v_data_2910_; lean_object* v_expr_2911_; lean_object* v___x_2912_; 
v_data_2910_ = lean_ctor_get(v_x_2823_, 0);
v_expr_2911_ = lean_ctor_get(v_x_2823_, 1);
lean_inc_ref(v_expr_2911_);
v___x_2912_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_expr_2911_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2912_) == 0)
{
lean_object* v_a_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_2935_; 
v_a_2913_ = lean_ctor_get(v___x_2912_, 0);
v_isSharedCheck_2935_ = !lean_is_exclusive(v___x_2912_);
if (v_isSharedCheck_2935_ == 0)
{
v___x_2915_ = v___x_2912_;
v_isShared_2916_ = v_isSharedCheck_2935_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_a_2913_);
lean_dec(v___x_2912_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_2935_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v_fst_2917_; lean_object* v_snd_2918_; lean_object* v___x_2920_; uint8_t v_isShared_2921_; uint8_t v_isSharedCheck_2934_; 
v_fst_2917_ = lean_ctor_get(v_a_2913_, 0);
v_snd_2918_ = lean_ctor_get(v_a_2913_, 1);
v_isSharedCheck_2934_ = !lean_is_exclusive(v_a_2913_);
if (v_isSharedCheck_2934_ == 0)
{
v___x_2920_ = v_a_2913_;
v_isShared_2921_ = v_isSharedCheck_2934_;
goto v_resetjp_2919_;
}
else
{
lean_inc(v_snd_2918_);
lean_inc(v_fst_2917_);
lean_dec(v_a_2913_);
v___x_2920_ = lean_box(0);
v_isShared_2921_ = v_isSharedCheck_2934_;
goto v_resetjp_2919_;
}
v_resetjp_2919_:
{
lean_object* v___y_2923_; size_t v___x_2930_; size_t v___x_2931_; uint8_t v___x_2932_; 
v___x_2930_ = lean_ptr_addr(v_expr_2911_);
v___x_2931_ = lean_ptr_addr(v_fst_2917_);
v___x_2932_ = lean_usize_dec_eq(v___x_2930_, v___x_2931_);
if (v___x_2932_ == 0)
{
lean_object* v___x_2933_; 
lean_inc(v_data_2910_);
lean_dec_ref_known(v_x_2823_, 2);
v___x_2933_ = l_Lean_Expr_mdata___override(v_data_2910_, v_fst_2917_);
v___y_2923_ = v___x_2933_;
goto v___jp_2922_;
}
else
{
lean_dec(v_fst_2917_);
v___y_2923_ = v_x_2823_;
goto v___jp_2922_;
}
v___jp_2922_:
{
lean_object* v___x_2925_; 
if (v_isShared_2921_ == 0)
{
lean_ctor_set(v___x_2920_, 0, v___y_2923_);
v___x_2925_ = v___x_2920_;
goto v_reusejp_2924_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v___y_2923_);
lean_ctor_set(v_reuseFailAlloc_2929_, 1, v_snd_2918_);
v___x_2925_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2924_;
}
v_reusejp_2924_:
{
lean_object* v___x_2927_; 
if (v_isShared_2916_ == 0)
{
lean_ctor_set(v___x_2915_, 0, v___x_2925_);
v___x_2927_ = v___x_2915_;
goto v_reusejp_2926_;
}
else
{
lean_object* v_reuseFailAlloc_2928_; 
v_reuseFailAlloc_2928_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2928_, 0, v___x_2925_);
v___x_2927_ = v_reuseFailAlloc_2928_;
goto v_reusejp_2926_;
}
v_reusejp_2926_:
{
return v___x_2927_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_x_2823_, 2);
return v___x_2912_;
}
}
case 8:
{
lean_object* v_declName_2936_; lean_object* v_type_2937_; lean_object* v_value_2938_; lean_object* v_body_2939_; uint8_t v_nondep_2940_; lean_object* v___x_2941_; 
v_declName_2936_ = lean_ctor_get(v_x_2823_, 0);
v_type_2937_ = lean_ctor_get(v_x_2823_, 1);
v_value_2938_ = lean_ctor_get(v_x_2823_, 2);
v_body_2939_ = lean_ctor_get(v_x_2823_, 3);
v_nondep_2940_ = lean_ctor_get_uint8(v_x_2823_, sizeof(void*)*4 + 8);
lean_inc_ref(v_type_2937_);
lean_inc_ref(v_f_2822_);
v___x_2941_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_type_2937_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2941_) == 0)
{
lean_object* v_a_2942_; lean_object* v_fst_2943_; lean_object* v_snd_2944_; lean_object* v___x_2945_; 
v_a_2942_ = lean_ctor_get(v___x_2941_, 0);
lean_inc(v_a_2942_);
lean_dec_ref_known(v___x_2941_, 1);
v_fst_2943_ = lean_ctor_get(v_a_2942_, 0);
lean_inc(v_fst_2943_);
v_snd_2944_ = lean_ctor_get(v_a_2942_, 1);
lean_inc(v_snd_2944_);
lean_dec(v_a_2942_);
lean_inc_ref(v_value_2938_);
lean_inc_ref(v_f_2822_);
v___x_2945_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_value_2938_, v_snd_2944_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2945_) == 0)
{
lean_object* v_a_2946_; lean_object* v_fst_2947_; lean_object* v_snd_2948_; lean_object* v___x_2949_; 
v_a_2946_ = lean_ctor_get(v___x_2945_, 0);
lean_inc(v_a_2946_);
lean_dec_ref_known(v___x_2945_, 1);
v_fst_2947_ = lean_ctor_get(v_a_2946_, 0);
lean_inc(v_fst_2947_);
v_snd_2948_ = lean_ctor_get(v_a_2946_, 1);
lean_inc(v_snd_2948_);
lean_dec(v_a_2946_);
lean_inc_ref(v_body_2939_);
v___x_2949_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_body_2939_, v_snd_2948_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2949_) == 0)
{
lean_object* v_a_2950_; lean_object* v___x_2952_; uint8_t v_isShared_2953_; uint8_t v_isSharedCheck_2981_; 
v_a_2950_ = lean_ctor_get(v___x_2949_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2949_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2952_ = v___x_2949_;
v_isShared_2953_ = v_isSharedCheck_2981_;
goto v_resetjp_2951_;
}
else
{
lean_inc(v_a_2950_);
lean_dec(v___x_2949_);
v___x_2952_ = lean_box(0);
v_isShared_2953_ = v_isSharedCheck_2981_;
goto v_resetjp_2951_;
}
v_resetjp_2951_:
{
lean_object* v_fst_2954_; lean_object* v_snd_2955_; lean_object* v___x_2957_; uint8_t v_isShared_2958_; uint8_t v_isSharedCheck_2980_; 
v_fst_2954_ = lean_ctor_get(v_a_2950_, 0);
v_snd_2955_ = lean_ctor_get(v_a_2950_, 1);
v_isSharedCheck_2980_ = !lean_is_exclusive(v_a_2950_);
if (v_isSharedCheck_2980_ == 0)
{
v___x_2957_ = v_a_2950_;
v_isShared_2958_ = v_isSharedCheck_2980_;
goto v_resetjp_2956_;
}
else
{
lean_inc(v_snd_2955_);
lean_inc(v_fst_2954_);
lean_dec(v_a_2950_);
v___x_2957_ = lean_box(0);
v_isShared_2958_ = v_isSharedCheck_2980_;
goto v_resetjp_2956_;
}
v_resetjp_2956_:
{
lean_object* v___y_2960_; uint8_t v___y_2968_; size_t v___x_2974_; size_t v___x_2975_; uint8_t v___x_2976_; 
v___x_2974_ = lean_ptr_addr(v_type_2937_);
v___x_2975_ = lean_ptr_addr(v_fst_2943_);
v___x_2976_ = lean_usize_dec_eq(v___x_2974_, v___x_2975_);
if (v___x_2976_ == 0)
{
v___y_2968_ = v___x_2976_;
goto v___jp_2967_;
}
else
{
size_t v___x_2977_; size_t v___x_2978_; uint8_t v___x_2979_; 
v___x_2977_ = lean_ptr_addr(v_value_2938_);
v___x_2978_ = lean_ptr_addr(v_fst_2947_);
v___x_2979_ = lean_usize_dec_eq(v___x_2977_, v___x_2978_);
v___y_2968_ = v___x_2979_;
goto v___jp_2967_;
}
v___jp_2959_:
{
lean_object* v___x_2962_; 
if (v_isShared_2958_ == 0)
{
lean_ctor_set(v___x_2957_, 0, v___y_2960_);
v___x_2962_ = v___x_2957_;
goto v_reusejp_2961_;
}
else
{
lean_object* v_reuseFailAlloc_2966_; 
v_reuseFailAlloc_2966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2966_, 0, v___y_2960_);
lean_ctor_set(v_reuseFailAlloc_2966_, 1, v_snd_2955_);
v___x_2962_ = v_reuseFailAlloc_2966_;
goto v_reusejp_2961_;
}
v_reusejp_2961_:
{
lean_object* v___x_2964_; 
if (v_isShared_2953_ == 0)
{
lean_ctor_set(v___x_2952_, 0, v___x_2962_);
v___x_2964_ = v___x_2952_;
goto v_reusejp_2963_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v___x_2962_);
v___x_2964_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2963_;
}
v_reusejp_2963_:
{
return v___x_2964_;
}
}
}
v___jp_2967_:
{
if (v___y_2968_ == 0)
{
lean_object* v___x_2969_; 
lean_inc(v_declName_2936_);
lean_dec_ref_known(v_x_2823_, 4);
v___x_2969_ = l_Lean_Expr_letE___override(v_declName_2936_, v_fst_2943_, v_fst_2947_, v_fst_2954_, v_nondep_2940_);
v___y_2960_ = v___x_2969_;
goto v___jp_2959_;
}
else
{
size_t v___x_2970_; size_t v___x_2971_; uint8_t v___x_2972_; 
v___x_2970_ = lean_ptr_addr(v_body_2939_);
v___x_2971_ = lean_ptr_addr(v_fst_2954_);
v___x_2972_ = lean_usize_dec_eq(v___x_2970_, v___x_2971_);
if (v___x_2972_ == 0)
{
lean_object* v___x_2973_; 
lean_inc(v_declName_2936_);
lean_dec_ref_known(v_x_2823_, 4);
v___x_2973_ = l_Lean_Expr_letE___override(v_declName_2936_, v_fst_2943_, v_fst_2947_, v_fst_2954_, v_nondep_2940_);
v___y_2960_ = v___x_2973_;
goto v___jp_2959_;
}
else
{
lean_dec(v_fst_2954_);
lean_dec(v_fst_2947_);
lean_dec(v_fst_2943_);
v___y_2960_ = v_x_2823_;
goto v___jp_2959_;
}
}
}
}
}
}
else
{
lean_dec(v_fst_2947_);
lean_dec(v_fst_2943_);
lean_dec_ref_known(v_x_2823_, 4);
return v___x_2949_;
}
}
else
{
lean_dec(v_fst_2943_);
lean_dec_ref_known(v_x_2823_, 4);
lean_dec_ref(v_f_2822_);
return v___x_2945_;
}
}
else
{
lean_dec_ref_known(v_x_2823_, 4);
lean_dec_ref(v_f_2822_);
return v___x_2941_;
}
}
case 5:
{
lean_object* v_fn_2982_; lean_object* v_arg_2983_; lean_object* v___x_2984_; 
v_fn_2982_ = lean_ctor_get(v_x_2823_, 0);
v_arg_2983_ = lean_ctor_get(v_x_2823_, 1);
lean_inc_ref(v_fn_2982_);
lean_inc_ref(v_f_2822_);
v___x_2984_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_fn_2982_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2984_) == 0)
{
lean_object* v_a_2985_; lean_object* v_fst_2986_; lean_object* v_snd_2987_; lean_object* v___x_2988_; 
v_a_2985_ = lean_ctor_get(v___x_2984_, 0);
lean_inc(v_a_2985_);
lean_dec_ref_known(v___x_2984_, 1);
v_fst_2986_ = lean_ctor_get(v_a_2985_, 0);
lean_inc(v_fst_2986_);
v_snd_2987_ = lean_ctor_get(v_a_2985_, 1);
lean_inc(v_snd_2987_);
lean_dec(v_a_2985_);
lean_inc_ref(v_arg_2983_);
v___x_2988_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_arg_2983_, v_snd_2987_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_2988_) == 0)
{
lean_object* v_a_2989_; lean_object* v___x_2991_; uint8_t v_isShared_2992_; uint8_t v_isSharedCheck_3016_; 
v_a_2989_ = lean_ctor_get(v___x_2988_, 0);
v_isSharedCheck_3016_ = !lean_is_exclusive(v___x_2988_);
if (v_isSharedCheck_3016_ == 0)
{
v___x_2991_ = v___x_2988_;
v_isShared_2992_ = v_isSharedCheck_3016_;
goto v_resetjp_2990_;
}
else
{
lean_inc(v_a_2989_);
lean_dec(v___x_2988_);
v___x_2991_ = lean_box(0);
v_isShared_2992_ = v_isSharedCheck_3016_;
goto v_resetjp_2990_;
}
v_resetjp_2990_:
{
lean_object* v_fst_2993_; lean_object* v_snd_2994_; lean_object* v___x_2996_; uint8_t v_isShared_2997_; uint8_t v_isSharedCheck_3015_; 
v_fst_2993_ = lean_ctor_get(v_a_2989_, 0);
v_snd_2994_ = lean_ctor_get(v_a_2989_, 1);
v_isSharedCheck_3015_ = !lean_is_exclusive(v_a_2989_);
if (v_isSharedCheck_3015_ == 0)
{
v___x_2996_ = v_a_2989_;
v_isShared_2997_ = v_isSharedCheck_3015_;
goto v_resetjp_2995_;
}
else
{
lean_inc(v_snd_2994_);
lean_inc(v_fst_2993_);
lean_dec(v_a_2989_);
v___x_2996_ = lean_box(0);
v_isShared_2997_ = v_isSharedCheck_3015_;
goto v_resetjp_2995_;
}
v_resetjp_2995_:
{
lean_object* v___y_2999_; uint8_t v___y_3007_; size_t v___x_3009_; size_t v___x_3010_; uint8_t v___x_3011_; 
v___x_3009_ = lean_ptr_addr(v_fn_2982_);
v___x_3010_ = lean_ptr_addr(v_fst_2986_);
v___x_3011_ = lean_usize_dec_eq(v___x_3009_, v___x_3010_);
if (v___x_3011_ == 0)
{
v___y_3007_ = v___x_3011_;
goto v___jp_3006_;
}
else
{
size_t v___x_3012_; size_t v___x_3013_; uint8_t v___x_3014_; 
v___x_3012_ = lean_ptr_addr(v_arg_2983_);
v___x_3013_ = lean_ptr_addr(v_fst_2993_);
v___x_3014_ = lean_usize_dec_eq(v___x_3012_, v___x_3013_);
v___y_3007_ = v___x_3014_;
goto v___jp_3006_;
}
v___jp_2998_:
{
lean_object* v___x_3001_; 
if (v_isShared_2997_ == 0)
{
lean_ctor_set(v___x_2996_, 0, v___y_2999_);
v___x_3001_ = v___x_2996_;
goto v_reusejp_3000_;
}
else
{
lean_object* v_reuseFailAlloc_3005_; 
v_reuseFailAlloc_3005_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3005_, 0, v___y_2999_);
lean_ctor_set(v_reuseFailAlloc_3005_, 1, v_snd_2994_);
v___x_3001_ = v_reuseFailAlloc_3005_;
goto v_reusejp_3000_;
}
v_reusejp_3000_:
{
lean_object* v___x_3003_; 
if (v_isShared_2992_ == 0)
{
lean_ctor_set(v___x_2991_, 0, v___x_3001_);
v___x_3003_ = v___x_2991_;
goto v_reusejp_3002_;
}
else
{
lean_object* v_reuseFailAlloc_3004_; 
v_reuseFailAlloc_3004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3004_, 0, v___x_3001_);
v___x_3003_ = v_reuseFailAlloc_3004_;
goto v_reusejp_3002_;
}
v_reusejp_3002_:
{
return v___x_3003_;
}
}
}
v___jp_3006_:
{
if (v___y_3007_ == 0)
{
lean_object* v___x_3008_; 
lean_dec_ref_known(v_x_2823_, 2);
v___x_3008_ = l_Lean_Expr_app___override(v_fst_2986_, v_fst_2993_);
v___y_2999_ = v___x_3008_;
goto v___jp_2998_;
}
else
{
lean_dec(v_fst_2993_);
lean_dec(v_fst_2986_);
v___y_2999_ = v_x_2823_;
goto v___jp_2998_;
}
}
}
}
}
else
{
lean_dec(v_fst_2986_);
lean_dec_ref_known(v_x_2823_, 2);
return v___x_2988_;
}
}
else
{
lean_dec_ref_known(v_x_2823_, 2);
lean_dec_ref(v_f_2822_);
return v___x_2984_;
}
}
case 11:
{
lean_object* v_typeName_3017_; lean_object* v_idx_3018_; lean_object* v_struct_3019_; lean_object* v___x_3020_; 
v_typeName_3017_ = lean_ctor_get(v_x_2823_, 0);
v_idx_3018_ = lean_ctor_get(v_x_2823_, 1);
v_struct_3019_ = lean_ctor_get(v_x_2823_, 2);
lean_inc_ref(v_struct_3019_);
v___x_3020_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___lam__0(v_f_2822_, v_struct_3019_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_, v___y_2830_);
if (lean_obj_tag(v___x_3020_) == 0)
{
lean_object* v_a_3021_; lean_object* v___x_3023_; uint8_t v_isShared_3024_; uint8_t v_isSharedCheck_3043_; 
v_a_3021_ = lean_ctor_get(v___x_3020_, 0);
v_isSharedCheck_3043_ = !lean_is_exclusive(v___x_3020_);
if (v_isSharedCheck_3043_ == 0)
{
v___x_3023_ = v___x_3020_;
v_isShared_3024_ = v_isSharedCheck_3043_;
goto v_resetjp_3022_;
}
else
{
lean_inc(v_a_3021_);
lean_dec(v___x_3020_);
v___x_3023_ = lean_box(0);
v_isShared_3024_ = v_isSharedCheck_3043_;
goto v_resetjp_3022_;
}
v_resetjp_3022_:
{
lean_object* v_fst_3025_; lean_object* v_snd_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3042_; 
v_fst_3025_ = lean_ctor_get(v_a_3021_, 0);
v_snd_3026_ = lean_ctor_get(v_a_3021_, 1);
v_isSharedCheck_3042_ = !lean_is_exclusive(v_a_3021_);
if (v_isSharedCheck_3042_ == 0)
{
v___x_3028_ = v_a_3021_;
v_isShared_3029_ = v_isSharedCheck_3042_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_snd_3026_);
lean_inc(v_fst_3025_);
lean_dec(v_a_3021_);
v___x_3028_ = lean_box(0);
v_isShared_3029_ = v_isSharedCheck_3042_;
goto v_resetjp_3027_;
}
v_resetjp_3027_:
{
lean_object* v___y_3031_; size_t v___x_3038_; size_t v___x_3039_; uint8_t v___x_3040_; 
v___x_3038_ = lean_ptr_addr(v_struct_3019_);
v___x_3039_ = lean_ptr_addr(v_fst_3025_);
v___x_3040_ = lean_usize_dec_eq(v___x_3038_, v___x_3039_);
if (v___x_3040_ == 0)
{
lean_object* v___x_3041_; 
lean_inc(v_idx_3018_);
lean_inc(v_typeName_3017_);
lean_dec_ref_known(v_x_2823_, 3);
v___x_3041_ = l_Lean_Expr_proj___override(v_typeName_3017_, v_idx_3018_, v_fst_3025_);
v___y_3031_ = v___x_3041_;
goto v___jp_3030_;
}
else
{
lean_dec(v_fst_3025_);
v___y_3031_ = v_x_2823_;
goto v___jp_3030_;
}
v___jp_3030_:
{
lean_object* v___x_3033_; 
if (v_isShared_3029_ == 0)
{
lean_ctor_set(v___x_3028_, 0, v___y_3031_);
v___x_3033_ = v___x_3028_;
goto v_reusejp_3032_;
}
else
{
lean_object* v_reuseFailAlloc_3037_; 
v_reuseFailAlloc_3037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3037_, 0, v___y_3031_);
lean_ctor_set(v_reuseFailAlloc_3037_, 1, v_snd_3026_);
v___x_3033_ = v_reuseFailAlloc_3037_;
goto v_reusejp_3032_;
}
v_reusejp_3032_:
{
lean_object* v___x_3035_; 
if (v_isShared_3024_ == 0)
{
lean_ctor_set(v___x_3023_, 0, v___x_3033_);
v___x_3035_ = v___x_3023_;
goto v_reusejp_3034_;
}
else
{
lean_object* v_reuseFailAlloc_3036_; 
v_reuseFailAlloc_3036_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3036_, 0, v___x_3033_);
v___x_3035_ = v_reuseFailAlloc_3036_;
goto v_reusejp_3034_;
}
v_reusejp_3034_:
{
return v___x_3035_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_x_2823_, 3);
return v___x_3020_;
}
}
default: 
{
lean_object* v___x_3044_; lean_object* v___x_3045_; 
lean_dec_ref(v_f_2822_);
v___x_3044_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3044_, 0, v_x_2823_);
lean_ctor_set(v___x_3044_, 1, v___y_2824_);
v___x_3045_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3045_, 0, v___x_3044_);
return v___x_3045_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg___boxed(lean_object* v_f_3046_, lean_object* v_x_3047_, lean_object* v___y_3048_, lean_object* v___y_3049_, lean_object* v___y_3050_, lean_object* v___y_3051_, lean_object* v___y_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_){
_start:
{
lean_object* v_res_3056_; 
v_res_3056_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg(v_f_3046_, v_x_3047_, v___y_3048_, v___y_3049_, v___y_3050_, v___y_3051_, v___y_3052_, v___y_3053_, v___y_3054_);
lean_dec(v___y_3054_);
lean_dec_ref(v___y_3053_);
lean_dec(v___y_3052_);
lean_dec_ref(v___y_3051_);
lean_dec(v___y_3050_);
lean_dec_ref(v___y_3049_);
return v_res_3056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(lean_object* v_f_3057_, lean_object* v_init_3058_, lean_object* v_e_3059_, lean_object* v___y_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_){
_start:
{
lean_object* v___x_3067_; 
v___x_3067_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg(v_f_3057_, v_e_3059_, v_init_3058_, v___y_3060_, v___y_3061_, v___y_3062_, v___y_3063_, v___y_3064_, v___y_3065_);
if (lean_obj_tag(v___x_3067_) == 0)
{
lean_object* v_a_3068_; lean_object* v___x_3070_; uint8_t v_isShared_3071_; uint8_t v_isSharedCheck_3076_; 
v_a_3068_ = lean_ctor_get(v___x_3067_, 0);
v_isSharedCheck_3076_ = !lean_is_exclusive(v___x_3067_);
if (v_isSharedCheck_3076_ == 0)
{
v___x_3070_ = v___x_3067_;
v_isShared_3071_ = v_isSharedCheck_3076_;
goto v_resetjp_3069_;
}
else
{
lean_inc(v_a_3068_);
lean_dec(v___x_3067_);
v___x_3070_ = lean_box(0);
v_isShared_3071_ = v_isSharedCheck_3076_;
goto v_resetjp_3069_;
}
v_resetjp_3069_:
{
lean_object* v_snd_3072_; lean_object* v___x_3074_; 
v_snd_3072_ = lean_ctor_get(v_a_3068_, 1);
lean_inc(v_snd_3072_);
lean_dec(v_a_3068_);
if (v_isShared_3071_ == 0)
{
lean_ctor_set(v___x_3070_, 0, v_snd_3072_);
v___x_3074_ = v___x_3070_;
goto v_reusejp_3073_;
}
else
{
lean_object* v_reuseFailAlloc_3075_; 
v_reuseFailAlloc_3075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3075_, 0, v_snd_3072_);
v___x_3074_ = v_reuseFailAlloc_3075_;
goto v_reusejp_3073_;
}
v_reusejp_3073_:
{
return v___x_3074_;
}
}
}
else
{
lean_object* v_a_3077_; lean_object* v___x_3079_; uint8_t v_isShared_3080_; uint8_t v_isSharedCheck_3084_; 
v_a_3077_ = lean_ctor_get(v___x_3067_, 0);
v_isSharedCheck_3084_ = !lean_is_exclusive(v___x_3067_);
if (v_isSharedCheck_3084_ == 0)
{
v___x_3079_ = v___x_3067_;
v_isShared_3080_ = v_isSharedCheck_3084_;
goto v_resetjp_3078_;
}
else
{
lean_inc(v_a_3077_);
lean_dec(v___x_3067_);
v___x_3079_ = lean_box(0);
v_isShared_3080_ = v_isSharedCheck_3084_;
goto v_resetjp_3078_;
}
v_resetjp_3078_:
{
lean_object* v___x_3082_; 
if (v_isShared_3080_ == 0)
{
v___x_3082_ = v___x_3079_;
goto v_reusejp_3081_;
}
else
{
lean_object* v_reuseFailAlloc_3083_; 
v_reuseFailAlloc_3083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3083_, 0, v_a_3077_);
v___x_3082_ = v_reuseFailAlloc_3083_;
goto v_reusejp_3081_;
}
v_reusejp_3081_:
{
return v___x_3082_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg___boxed(lean_object* v_f_3085_, lean_object* v_init_3086_, lean_object* v_e_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_){
_start:
{
lean_object* v_res_3095_; 
v_res_3095_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(v_f_3085_, v_init_3086_, v_e_3087_, v___y_3088_, v___y_3089_, v___y_3090_, v___y_3091_, v___y_3092_, v___y_3093_);
lean_dec(v___y_3093_);
lean_dec_ref(v___y_3092_);
lean_dec(v___y_3091_);
lean_dec_ref(v___y_3090_);
lean_dec(v___y_3089_);
lean_dec_ref(v___y_3088_);
return v_res_3095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares___boxed(lean_object* v_s_3098_, lean_object* v_e_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_, lean_object* v_a_3102_, lean_object* v_a_3103_, lean_object* v_a_3104_, lean_object* v_a_3105_, lean_object* v_a_3106_){
_start:
{
lean_object* v_res_3107_; 
v_res_3107_ = lp_mathlib_Mathlib_Tactic_Linarith_findSquares(v_s_3098_, v_e_3099_, v_a_3100_, v_a_3101_, v_a_3102_, v_a_3103_, v_a_3104_, v_a_3105_);
lean_dec(v_a_3105_);
lean_dec_ref(v_a_3104_);
lean_dec(v_a_3103_);
lean_dec_ref(v_a_3102_);
lean_dec(v_a_3101_);
lean_dec_ref(v_a_3100_);
return v_res_3107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_findSquares(lean_object* v_s_3108_, lean_object* v_e_3109_, lean_object* v_a_3110_, lean_object* v_a_3111_, lean_object* v_a_3112_, lean_object* v_a_3113_, lean_object* v_a_3114_, lean_object* v_a_3115_){
_start:
{
lean_object* v___y_3118_; lean_object* v___y_3119_; lean_object* v___y_3120_; lean_object* v___y_3121_; lean_object* v___y_3122_; lean_object* v___y_3123_; lean_object* v___y_3127_; lean_object* v___y_3128_; lean_object* v___y_3129_; lean_object* v___y_3130_; lean_object* v___y_3131_; lean_object* v___y_3132_; uint8_t v___x_3135_; 
v___x_3135_ = l_Lean_Expr_hasLooseBVars(v_e_3109_);
if (v___x_3135_ == 0)
{
lean_object* v___x_3136_; lean_object* v_fst_3137_; 
lean_inc_ref(v_e_3109_);
v___x_3136_ = l_Lean_Expr_getAppFnArgs(v_e_3109_);
v_fst_3137_ = lean_ctor_get(v___x_3136_, 0);
lean_inc(v_fst_3137_);
if (lean_obj_tag(v_fst_3137_) == 1)
{
lean_object* v_pre_3138_; 
v_pre_3138_ = lean_ctor_get(v_fst_3137_, 0);
lean_inc(v_pre_3138_);
if (lean_obj_tag(v_pre_3138_) == 1)
{
lean_object* v_pre_3139_; 
v_pre_3139_ = lean_ctor_get(v_pre_3138_, 0);
if (lean_obj_tag(v_pre_3139_) == 0)
{
lean_object* v_snd_3140_; lean_object* v_str_3141_; lean_object* v_str_3142_; lean_object* v___x_3143_; uint8_t v___x_3144_; 
v_snd_3140_ = lean_ctor_get(v___x_3136_, 1);
lean_inc(v_snd_3140_);
lean_dec_ref(v___x_3136_);
v_str_3141_ = lean_ctor_get(v_fst_3137_, 1);
lean_inc_ref(v_str_3141_);
lean_dec_ref_known(v_fst_3137_, 2);
v_str_3142_ = lean_ctor_get(v_pre_3138_, 1);
lean_inc_ref(v_str_3142_);
lean_dec_ref_known(v_pre_3138_, 2);
v___x_3143_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__0));
v___x_3144_ = lean_string_dec_eq(v_str_3142_, v___x_3143_);
if (v___x_3144_ == 0)
{
lean_object* v___x_3145_; uint8_t v___x_3146_; 
v___x_3145_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__1));
v___x_3146_ = lean_string_dec_eq(v_str_3142_, v___x_3145_);
lean_dec_ref(v_str_3142_);
if (v___x_3146_ == 0)
{
lean_dec_ref(v_str_3141_);
lean_dec(v_snd_3140_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3147_; uint8_t v___x_3148_; 
v___x_3147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_getNatComparisons___closed__6));
v___x_3148_ = lean_string_dec_eq(v_str_3141_, v___x_3147_);
lean_dec_ref(v_str_3141_);
if (v___x_3148_ == 0)
{
lean_dec(v_snd_3140_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3149_; lean_object* v___x_3150_; uint8_t v___x_3151_; 
v___x_3149_ = lean_array_get_size(v_snd_3140_);
v___x_3150_ = lean_unsigned_to_nat(6u);
v___x_3151_ = lean_nat_dec_eq(v___x_3149_, v___x_3150_);
if (v___x_3151_ == 0)
{
lean_dec(v_snd_3140_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3152_; lean_object* v___x_3153_; lean_object* v___x_3154_; 
v___x_3152_ = lean_unsigned_to_nat(4u);
v___x_3153_ = lean_array_fget(v_snd_3140_, v___x_3152_);
lean_inc(v___x_3153_);
v___x_3154_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v___x_3153_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
if (lean_obj_tag(v___x_3154_) == 0)
{
lean_object* v_a_3155_; lean_object* v_fst_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v___x_3159_; 
v_a_3155_ = lean_ctor_get(v___x_3154_, 0);
lean_inc(v_a_3155_);
lean_dec_ref_known(v___x_3154_, 1);
v_fst_3156_ = lean_ctor_get(v_a_3155_, 0);
lean_inc(v_fst_3156_);
lean_dec(v_a_3155_);
v___x_3157_ = lean_unsigned_to_nat(5u);
v___x_3158_ = lean_array_fget(v_snd_3140_, v___x_3157_);
lean_dec(v_snd_3140_);
v___x_3159_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v___x_3158_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
if (lean_obj_tag(v___x_3159_) == 0)
{
lean_object* v_a_3160_; lean_object* v_fst_3161_; lean_object* v___x_3163_; uint8_t v_isShared_3164_; uint8_t v_isSharedCheck_3187_; 
v_a_3160_ = lean_ctor_get(v___x_3159_, 0);
lean_inc(v_a_3160_);
lean_dec_ref_known(v___x_3159_, 1);
v_fst_3161_ = lean_ctor_get(v_a_3160_, 0);
v_isSharedCheck_3187_ = !lean_is_exclusive(v_a_3160_);
if (v_isSharedCheck_3187_ == 0)
{
lean_object* v_unused_3188_; 
v_unused_3188_ = lean_ctor_get(v_a_3160_, 1);
lean_dec(v_unused_3188_);
v___x_3163_ = v_a_3160_;
v_isShared_3164_ = v_isSharedCheck_3187_;
goto v_resetjp_3162_;
}
else
{
lean_inc(v_fst_3161_);
lean_dec(v_a_3160_);
v___x_3163_ = lean_box(0);
v_isShared_3164_ = v_isSharedCheck_3187_;
goto v_resetjp_3162_;
}
v_resetjp_3162_:
{
uint8_t v___x_3165_; 
v___x_3165_ = lean_nat_dec_eq(v_fst_3156_, v_fst_3161_);
lean_dec(v_fst_3161_);
if (v___x_3165_ == 0)
{
lean_object* v___x_3166_; lean_object* v___x_3167_; 
lean_del_object(v___x_3163_);
lean_dec(v_fst_3156_);
lean_dec(v___x_3153_);
v___x_3166_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_findSquares___boxed), 9, 0);
v___x_3167_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(v___x_3166_, v_s_3108_, v_e_3109_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
return v___x_3167_;
}
else
{
lean_object* v___x_3168_; 
lean_dec_ref(v_e_3109_);
v___x_3168_ = lp_mathlib_Mathlib_Tactic_Linarith_findSquares(v_s_3108_, v___x_3153_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
if (lean_obj_tag(v___x_3168_) == 0)
{
lean_object* v_a_3169_; lean_object* v___x_3171_; uint8_t v_isShared_3172_; uint8_t v_isSharedCheck_3186_; 
v_a_3169_ = lean_ctor_get(v___x_3168_, 0);
v_isSharedCheck_3186_ = !lean_is_exclusive(v___x_3168_);
if (v_isSharedCheck_3186_ == 0)
{
v___x_3171_ = v___x_3168_;
v_isShared_3172_ = v_isSharedCheck_3186_;
goto v_resetjp_3170_;
}
else
{
lean_inc(v_a_3169_);
lean_dec(v___x_3168_);
v___x_3171_ = lean_box(0);
v_isShared_3172_ = v_isSharedCheck_3186_;
goto v_resetjp_3170_;
}
v_resetjp_3170_:
{
lean_object* v___x_3173_; lean_object* v___x_3175_; 
v___x_3173_ = lean_box(v___x_3135_);
if (v_isShared_3164_ == 0)
{
lean_ctor_set(v___x_3163_, 1, v___x_3173_);
lean_ctor_set(v___x_3163_, 0, v_fst_3156_);
v___x_3175_ = v___x_3163_;
goto v_reusejp_3174_;
}
else
{
lean_object* v_reuseFailAlloc_3185_; 
v_reuseFailAlloc_3185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3185_, 0, v_fst_3156_);
lean_ctor_set(v_reuseFailAlloc_3185_, 1, v___x_3173_);
v___x_3175_ = v_reuseFailAlloc_3185_;
goto v_reusejp_3174_;
}
v_reusejp_3174_:
{
uint8_t v___x_3176_; 
v___x_3176_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(v___x_3175_, v_a_3169_);
if (v___x_3176_ == 0)
{
lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3180_; 
v___x_3177_ = lean_box(0);
v___x_3178_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(v___x_3175_, v___x_3177_, v_a_3169_);
if (v_isShared_3172_ == 0)
{
lean_ctor_set(v___x_3171_, 0, v___x_3178_);
v___x_3180_ = v___x_3171_;
goto v_reusejp_3179_;
}
else
{
lean_object* v_reuseFailAlloc_3181_; 
v_reuseFailAlloc_3181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3181_, 0, v___x_3178_);
v___x_3180_ = v_reuseFailAlloc_3181_;
goto v_reusejp_3179_;
}
v_reusejp_3179_:
{
return v___x_3180_;
}
}
else
{
lean_object* v___x_3183_; 
lean_dec_ref(v___x_3175_);
if (v_isShared_3172_ == 0)
{
v___x_3183_ = v___x_3171_;
goto v_reusejp_3182_;
}
else
{
lean_object* v_reuseFailAlloc_3184_; 
v_reuseFailAlloc_3184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3184_, 0, v_a_3169_);
v___x_3183_ = v_reuseFailAlloc_3184_;
goto v_reusejp_3182_;
}
v_reusejp_3182_:
{
return v___x_3183_;
}
}
}
}
}
else
{
lean_del_object(v___x_3163_);
lean_dec(v_fst_3156_);
return v___x_3168_;
}
}
}
}
else
{
lean_object* v_a_3189_; lean_object* v___x_3191_; uint8_t v_isShared_3192_; uint8_t v_isSharedCheck_3196_; 
lean_dec(v_fst_3156_);
lean_dec(v___x_3153_);
lean_dec_ref(v_e_3109_);
lean_dec(v_s_3108_);
v_a_3189_ = lean_ctor_get(v___x_3159_, 0);
v_isSharedCheck_3196_ = !lean_is_exclusive(v___x_3159_);
if (v_isSharedCheck_3196_ == 0)
{
v___x_3191_ = v___x_3159_;
v_isShared_3192_ = v_isSharedCheck_3196_;
goto v_resetjp_3190_;
}
else
{
lean_inc(v_a_3189_);
lean_dec(v___x_3159_);
v___x_3191_ = lean_box(0);
v_isShared_3192_ = v_isSharedCheck_3196_;
goto v_resetjp_3190_;
}
v_resetjp_3190_:
{
lean_object* v___x_3194_; 
if (v_isShared_3192_ == 0)
{
v___x_3194_ = v___x_3191_;
goto v_reusejp_3193_;
}
else
{
lean_object* v_reuseFailAlloc_3195_; 
v_reuseFailAlloc_3195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3195_, 0, v_a_3189_);
v___x_3194_ = v_reuseFailAlloc_3195_;
goto v_reusejp_3193_;
}
v_reusejp_3193_:
{
return v___x_3194_;
}
}
}
}
else
{
lean_object* v_a_3197_; lean_object* v___x_3199_; uint8_t v_isShared_3200_; uint8_t v_isSharedCheck_3204_; 
lean_dec(v___x_3153_);
lean_dec(v_snd_3140_);
lean_dec_ref(v_e_3109_);
lean_dec(v_s_3108_);
v_a_3197_ = lean_ctor_get(v___x_3154_, 0);
v_isSharedCheck_3204_ = !lean_is_exclusive(v___x_3154_);
if (v_isSharedCheck_3204_ == 0)
{
v___x_3199_ = v___x_3154_;
v_isShared_3200_ = v_isSharedCheck_3204_;
goto v_resetjp_3198_;
}
else
{
lean_inc(v_a_3197_);
lean_dec(v___x_3154_);
v___x_3199_ = lean_box(0);
v_isShared_3200_ = v_isSharedCheck_3204_;
goto v_resetjp_3198_;
}
v_resetjp_3198_:
{
lean_object* v___x_3202_; 
if (v_isShared_3200_ == 0)
{
v___x_3202_ = v___x_3199_;
goto v_reusejp_3201_;
}
else
{
lean_object* v_reuseFailAlloc_3203_; 
v_reuseFailAlloc_3203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3203_, 0, v_a_3197_);
v___x_3202_ = v_reuseFailAlloc_3203_;
goto v_reusejp_3201_;
}
v_reusejp_3201_:
{
return v___x_3202_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3205_; uint8_t v___x_3206_; 
lean_dec_ref(v_str_3142_);
v___x_3205_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_findSquares___closed__1));
v___x_3206_ = lean_string_dec_eq(v_str_3141_, v___x_3205_);
lean_dec_ref(v_str_3141_);
if (v___x_3206_ == 0)
{
lean_dec(v_snd_3140_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3207_; lean_object* v___x_3208_; uint8_t v___x_3209_; 
v___x_3207_ = lean_array_get_size(v_snd_3140_);
v___x_3208_ = lean_unsigned_to_nat(6u);
v___x_3209_ = lean_nat_dec_eq(v___x_3207_, v___x_3208_);
if (v___x_3209_ == 0)
{
lean_dec(v_snd_3140_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
else
{
lean_object* v___x_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; 
v___x_3210_ = lean_unsigned_to_nat(5u);
v___x_3211_ = lean_array_fget_borrowed(v_snd_3140_, v___x_3210_);
lean_inc(v___x_3211_);
v___x_3212_ = lp_mathlib_Lean_Expr_numeral_x3f(v___x_3211_);
if (lean_obj_tag(v___x_3212_) == 1)
{
lean_object* v_val_3213_; lean_object* v___x_3214_; uint8_t v___x_3215_; 
v_val_3213_ = lean_ctor_get(v___x_3212_, 0);
lean_inc(v_val_3213_);
lean_dec_ref_known(v___x_3212_, 1);
v___x_3214_ = lean_unsigned_to_nat(2u);
v___x_3215_ = lean_nat_dec_eq(v_val_3213_, v___x_3214_);
lean_dec(v_val_3213_);
if (v___x_3215_ == 0)
{
lean_dec(v_snd_3140_);
v___y_3127_ = v_a_3110_;
v___y_3128_ = v_a_3111_;
v___y_3129_ = v_a_3112_;
v___y_3130_ = v_a_3113_;
v___y_3131_ = v_a_3114_;
v___y_3132_ = v_a_3115_;
goto v___jp_3126_;
}
else
{
lean_object* v___x_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; 
lean_dec_ref(v_e_3109_);
v___x_3216_ = lean_unsigned_to_nat(4u);
v___x_3217_ = lean_array_fget(v_snd_3140_, v___x_3216_);
lean_dec(v_snd_3140_);
lean_inc(v___x_3217_);
v___x_3218_ = lp_mathlib_Mathlib_Tactic_Linarith_findSquares(v_s_3108_, v___x_3217_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
if (lean_obj_tag(v___x_3218_) == 0)
{
lean_object* v_a_3219_; lean_object* v___x_3220_; 
v_a_3219_ = lean_ctor_get(v___x_3218_, 0);
lean_inc(v_a_3219_);
lean_dec_ref_known(v___x_3218_, 1);
v___x_3220_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v___x_3217_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_);
if (lean_obj_tag(v___x_3220_) == 0)
{
lean_object* v_a_3221_; lean_object* v___x_3223_; uint8_t v_isShared_3224_; uint8_t v_isSharedCheck_3244_; 
v_a_3221_ = lean_ctor_get(v___x_3220_, 0);
v_isSharedCheck_3244_ = !lean_is_exclusive(v___x_3220_);
if (v_isSharedCheck_3244_ == 0)
{
v___x_3223_ = v___x_3220_;
v_isShared_3224_ = v_isSharedCheck_3244_;
goto v_resetjp_3222_;
}
else
{
lean_inc(v_a_3221_);
lean_dec(v___x_3220_);
v___x_3223_ = lean_box(0);
v_isShared_3224_ = v_isSharedCheck_3244_;
goto v_resetjp_3222_;
}
v_resetjp_3222_:
{
lean_object* v_fst_3225_; lean_object* v___x_3227_; uint8_t v_isShared_3228_; uint8_t v_isSharedCheck_3242_; 
v_fst_3225_ = lean_ctor_get(v_a_3221_, 0);
v_isSharedCheck_3242_ = !lean_is_exclusive(v_a_3221_);
if (v_isSharedCheck_3242_ == 0)
{
lean_object* v_unused_3243_; 
v_unused_3243_ = lean_ctor_get(v_a_3221_, 1);
lean_dec(v_unused_3243_);
v___x_3227_ = v_a_3221_;
v_isShared_3228_ = v_isSharedCheck_3242_;
goto v_resetjp_3226_;
}
else
{
lean_inc(v_fst_3225_);
lean_dec(v_a_3221_);
v___x_3227_ = lean_box(0);
v_isShared_3228_ = v_isSharedCheck_3242_;
goto v_resetjp_3226_;
}
v_resetjp_3226_:
{
lean_object* v___x_3229_; lean_object* v___x_3231_; 
v___x_3229_ = lean_box(v___x_3215_);
if (v_isShared_3228_ == 0)
{
lean_ctor_set(v___x_3227_, 1, v___x_3229_);
v___x_3231_ = v___x_3227_;
goto v_reusejp_3230_;
}
else
{
lean_object* v_reuseFailAlloc_3241_; 
v_reuseFailAlloc_3241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3241_, 0, v_fst_3225_);
lean_ctor_set(v_reuseFailAlloc_3241_, 1, v___x_3229_);
v___x_3231_ = v_reuseFailAlloc_3241_;
goto v_reusejp_3230_;
}
v_reusejp_3230_:
{
uint8_t v___x_3232_; 
v___x_3232_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(v___x_3231_, v_a_3219_);
if (v___x_3232_ == 0)
{
lean_object* v___x_3233_; lean_object* v___x_3234_; lean_object* v___x_3236_; 
v___x_3233_ = lean_box(0);
v___x_3234_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(v___x_3231_, v___x_3233_, v_a_3219_);
if (v_isShared_3224_ == 0)
{
lean_ctor_set(v___x_3223_, 0, v___x_3234_);
v___x_3236_ = v___x_3223_;
goto v_reusejp_3235_;
}
else
{
lean_object* v_reuseFailAlloc_3237_; 
v_reuseFailAlloc_3237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3237_, 0, v___x_3234_);
v___x_3236_ = v_reuseFailAlloc_3237_;
goto v_reusejp_3235_;
}
v_reusejp_3235_:
{
return v___x_3236_;
}
}
else
{
lean_object* v___x_3239_; 
lean_dec_ref(v___x_3231_);
if (v_isShared_3224_ == 0)
{
lean_ctor_set(v___x_3223_, 0, v_a_3219_);
v___x_3239_ = v___x_3223_;
goto v_reusejp_3238_;
}
else
{
lean_object* v_reuseFailAlloc_3240_; 
v_reuseFailAlloc_3240_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3240_, 0, v_a_3219_);
v___x_3239_ = v_reuseFailAlloc_3240_;
goto v_reusejp_3238_;
}
v_reusejp_3238_:
{
return v___x_3239_;
}
}
}
}
}
}
else
{
lean_object* v_a_3245_; lean_object* v___x_3247_; uint8_t v_isShared_3248_; uint8_t v_isSharedCheck_3252_; 
lean_dec(v_a_3219_);
v_a_3245_ = lean_ctor_get(v___x_3220_, 0);
v_isSharedCheck_3252_ = !lean_is_exclusive(v___x_3220_);
if (v_isSharedCheck_3252_ == 0)
{
v___x_3247_ = v___x_3220_;
v_isShared_3248_ = v_isSharedCheck_3252_;
goto v_resetjp_3246_;
}
else
{
lean_inc(v_a_3245_);
lean_dec(v___x_3220_);
v___x_3247_ = lean_box(0);
v_isShared_3248_ = v_isSharedCheck_3252_;
goto v_resetjp_3246_;
}
v_resetjp_3246_:
{
lean_object* v___x_3250_; 
if (v_isShared_3248_ == 0)
{
v___x_3250_ = v___x_3247_;
goto v_reusejp_3249_;
}
else
{
lean_object* v_reuseFailAlloc_3251_; 
v_reuseFailAlloc_3251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3251_, 0, v_a_3245_);
v___x_3250_ = v_reuseFailAlloc_3251_;
goto v_reusejp_3249_;
}
v_reusejp_3249_:
{
return v___x_3250_;
}
}
}
}
else
{
lean_dec(v___x_3217_);
return v___x_3218_;
}
}
}
else
{
lean_dec(v___x_3212_);
lean_dec(v_snd_3140_);
v___y_3127_ = v_a_3110_;
v___y_3128_ = v_a_3111_;
v___y_3129_ = v_a_3112_;
v___y_3130_ = v_a_3113_;
v___y_3131_ = v_a_3114_;
v___y_3132_ = v_a_3115_;
goto v___jp_3126_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_3138_, 2);
lean_dec_ref_known(v_fst_3137_, 2);
lean_dec_ref(v___x_3136_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
}
else
{
lean_dec_ref_known(v_fst_3137_, 2);
lean_dec(v_pre_3138_);
lean_dec_ref(v___x_3136_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
}
else
{
lean_dec(v_fst_3137_);
lean_dec_ref(v___x_3136_);
v___y_3118_ = v_a_3110_;
v___y_3119_ = v_a_3111_;
v___y_3120_ = v_a_3112_;
v___y_3121_ = v_a_3113_;
v___y_3122_ = v_a_3114_;
v___y_3123_ = v_a_3115_;
goto v___jp_3117_;
}
}
else
{
lean_object* v___x_3253_; 
lean_dec_ref(v_e_3109_);
v___x_3253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3253_, 0, v_s_3108_);
return v___x_3253_;
}
v___jp_3117_:
{
lean_object* v___x_3124_; lean_object* v___x_3125_; 
v___x_3124_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_findSquares___boxed), 9, 0);
v___x_3125_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(v___x_3124_, v_s_3108_, v_e_3109_, v___y_3118_, v___y_3119_, v___y_3120_, v___y_3121_, v___y_3122_, v___y_3123_);
return v___x_3125_;
}
v___jp_3126_:
{
lean_object* v___x_3133_; lean_object* v___x_3134_; 
v___x_3133_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_findSquares___boxed), 9, 0);
v___x_3134_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(v___x_3133_, v_s_3108_, v_e_3109_, v___y_3127_, v___y_3128_, v___y_3129_, v___y_3130_, v___y_3131_, v___y_3132_);
return v___x_3134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0(lean_object* v_00_u03b1_3254_, lean_object* v_f_3255_, lean_object* v_x_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_, lean_object* v___y_3263_){
_start:
{
lean_object* v___x_3265_; 
v___x_3265_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___redArg(v_f_3255_, v_x_3256_, v___y_3257_, v___y_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_, v___y_3263_);
return v___x_3265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0___boxed(lean_object* v_00_u03b1_3266_, lean_object* v_f_3267_, lean_object* v_x_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_, lean_object* v___y_3272_, lean_object* v___y_3273_, lean_object* v___y_3274_, lean_object* v___y_3275_, lean_object* v___y_3276_){
_start:
{
lean_object* v_res_3277_; 
v_res_3277_ = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0_spec__0(v_00_u03b1_3266_, v_f_3267_, v_x_3268_, v___y_3269_, v___y_3270_, v___y_3271_, v___y_3272_, v___y_3273_, v___y_3274_, v___y_3275_);
lean_dec(v___y_3275_);
lean_dec_ref(v___y_3274_);
lean_dec(v___y_3273_);
lean_dec_ref(v___y_3272_);
lean_dec(v___y_3271_);
lean_dec_ref(v___y_3270_);
return v_res_3277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0(lean_object* v_00_u03b1_3278_, lean_object* v_f_3279_, lean_object* v_init_3280_, lean_object* v_e_3281_, lean_object* v___y_3282_, lean_object* v___y_3283_, lean_object* v___y_3284_, lean_object* v___y_3285_, lean_object* v___y_3286_, lean_object* v___y_3287_){
_start:
{
lean_object* v___x_3289_; 
v___x_3289_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___redArg(v_f_3279_, v_init_3280_, v_e_3281_, v___y_3282_, v___y_3283_, v___y_3284_, v___y_3285_, v___y_3286_, v___y_3287_);
return v___x_3289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0___boxed(lean_object* v_00_u03b1_3290_, lean_object* v_f_3291_, lean_object* v_init_3292_, lean_object* v_e_3293_, lean_object* v___y_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_){
_start:
{
lean_object* v_res_3301_; 
v_res_3301_ = lp_mathlib_Lean_Expr_foldlM___at___00Mathlib_Tactic_Linarith_findSquares_spec__0(v_00_u03b1_3290_, v_f_3291_, v_init_3292_, v_e_3293_, v___y_3294_, v___y_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_);
lean_dec(v___y_3299_);
lean_dec_ref(v___y_3298_);
lean_dec(v___y_3297_);
lean_dec_ref(v___y_3296_);
lean_dec(v___y_3295_);
lean_dec_ref(v___y_3294_);
return v_res_3301_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1(lean_object* v_00_u03b2_3302_, lean_object* v_k_3303_, lean_object* v_t_3304_){
_start:
{
uint8_t v___x_3305_; 
v___x_3305_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___redArg(v_k_3303_, v_t_3304_);
return v___x_3305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1___boxed(lean_object* v_00_u03b2_3306_, lean_object* v_k_3307_, lean_object* v_t_3308_){
_start:
{
uint8_t v_res_3309_; lean_object* v_r_3310_; 
v_res_3309_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Tactic_Linarith_findSquares_spec__1(v_00_u03b2_3306_, v_k_3307_, v_t_3308_);
lean_dec(v_t_3308_);
lean_dec_ref(v_k_3307_);
v_r_3310_ = lean_box(v_res_3309_);
return v_r_3310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2(lean_object* v_00_u03b2_3311_, lean_object* v_k_3312_, lean_object* v_v_3313_, lean_object* v_t_3314_, lean_object* v_hl_3315_){
_start:
{
lean_object* v___x_3316_; 
v___x_3316_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_findSquares_spec__2___redArg(v_k_3312_, v_v_3313_, v_t_3314_);
return v___x_3316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg(lean_object* v_e_3317_, lean_object* v___y_3318_){
_start:
{
uint8_t v___x_3320_; 
v___x_3320_ = l_Lean_Expr_hasMVar(v_e_3317_);
if (v___x_3320_ == 0)
{
lean_object* v___x_3321_; 
v___x_3321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3321_, 0, v_e_3317_);
return v___x_3321_;
}
else
{
lean_object* v___x_3322_; lean_object* v_mctx_3323_; lean_object* v___x_3324_; lean_object* v_fst_3325_; lean_object* v_snd_3326_; lean_object* v___x_3327_; lean_object* v_cache_3328_; lean_object* v_zetaDeltaFVarIds_3329_; lean_object* v_postponed_3330_; lean_object* v_diag_3331_; lean_object* v___x_3333_; uint8_t v_isShared_3334_; uint8_t v_isSharedCheck_3340_; 
v___x_3322_ = lean_st_ref_get(v___y_3318_);
v_mctx_3323_ = lean_ctor_get(v___x_3322_, 0);
lean_inc_ref(v_mctx_3323_);
lean_dec(v___x_3322_);
v___x_3324_ = l_Lean_instantiateMVarsCore(v_mctx_3323_, v_e_3317_);
v_fst_3325_ = lean_ctor_get(v___x_3324_, 0);
lean_inc(v_fst_3325_);
v_snd_3326_ = lean_ctor_get(v___x_3324_, 1);
lean_inc(v_snd_3326_);
lean_dec_ref(v___x_3324_);
v___x_3327_ = lean_st_ref_take(v___y_3318_);
v_cache_3328_ = lean_ctor_get(v___x_3327_, 1);
v_zetaDeltaFVarIds_3329_ = lean_ctor_get(v___x_3327_, 2);
v_postponed_3330_ = lean_ctor_get(v___x_3327_, 3);
v_diag_3331_ = lean_ctor_get(v___x_3327_, 4);
v_isSharedCheck_3340_ = !lean_is_exclusive(v___x_3327_);
if (v_isSharedCheck_3340_ == 0)
{
lean_object* v_unused_3341_; 
v_unused_3341_ = lean_ctor_get(v___x_3327_, 0);
lean_dec(v_unused_3341_);
v___x_3333_ = v___x_3327_;
v_isShared_3334_ = v_isSharedCheck_3340_;
goto v_resetjp_3332_;
}
else
{
lean_inc(v_diag_3331_);
lean_inc(v_postponed_3330_);
lean_inc(v_zetaDeltaFVarIds_3329_);
lean_inc(v_cache_3328_);
lean_dec(v___x_3327_);
v___x_3333_ = lean_box(0);
v_isShared_3334_ = v_isSharedCheck_3340_;
goto v_resetjp_3332_;
}
v_resetjp_3332_:
{
lean_object* v___x_3336_; 
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 0, v_snd_3326_);
v___x_3336_ = v___x_3333_;
goto v_reusejp_3335_;
}
else
{
lean_object* v_reuseFailAlloc_3339_; 
v_reuseFailAlloc_3339_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3339_, 0, v_snd_3326_);
lean_ctor_set(v_reuseFailAlloc_3339_, 1, v_cache_3328_);
lean_ctor_set(v_reuseFailAlloc_3339_, 2, v_zetaDeltaFVarIds_3329_);
lean_ctor_set(v_reuseFailAlloc_3339_, 3, v_postponed_3330_);
lean_ctor_set(v_reuseFailAlloc_3339_, 4, v_diag_3331_);
v___x_3336_ = v_reuseFailAlloc_3339_;
goto v_reusejp_3335_;
}
v_reusejp_3335_:
{
lean_object* v___x_3337_; lean_object* v___x_3338_; 
v___x_3337_ = lean_st_ref_set(v___y_3318_, v___x_3336_);
v___x_3338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3338_, 0, v_fst_3325_);
return v___x_3338_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg___boxed(lean_object* v_e_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_){
_start:
{
lean_object* v_res_3345_; 
v_res_3345_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg(v_e_3342_, v___y_3343_);
lean_dec(v___y_3343_);
return v_res_3345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0(lean_object* v_e_3346_, lean_object* v___y_3347_, lean_object* v___y_3348_, lean_object* v___y_3349_, lean_object* v___y_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_){
_start:
{
lean_object* v___x_3354_; 
v___x_3354_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg(v_e_3346_, v___y_3350_);
return v___x_3354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___boxed(lean_object* v_e_3355_, lean_object* v___y_3356_, lean_object* v___y_3357_, lean_object* v___y_3358_, lean_object* v___y_3359_, lean_object* v___y_3360_, lean_object* v___y_3361_, lean_object* v___y_3362_){
_start:
{
lean_object* v_res_3363_; 
v_res_3363_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0(v_e_3355_, v___y_3356_, v___y_3357_, v___y_3358_, v___y_3359_, v___y_3360_, v___y_3361_);
lean_dec(v___y_3361_);
lean_dec_ref(v___y_3360_);
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec(v___y_3357_);
lean_dec_ref(v___y_3356_);
return v_res_3363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg(lean_object* v_x_3364_, lean_object* v___y_3365_, lean_object* v___y_3366_, lean_object* v___y_3367_, lean_object* v___y_3368_){
_start:
{
lean_object* v___x_3370_; 
v___x_3370_ = l_Lean_Meta_saveState___redArg(v___y_3366_, v___y_3368_);
if (lean_obj_tag(v___x_3370_) == 0)
{
lean_object* v_a_3371_; lean_object* v___x_3372_; 
v_a_3371_ = lean_ctor_get(v___x_3370_, 0);
lean_inc(v_a_3371_);
lean_dec_ref_known(v___x_3370_, 1);
lean_inc(v___y_3368_);
lean_inc_ref(v___y_3367_);
lean_inc(v___y_3366_);
lean_inc_ref(v___y_3365_);
v___x_3372_ = lean_apply_5(v_x_3364_, v___y_3365_, v___y_3366_, v___y_3367_, v___y_3368_, lean_box(0));
if (lean_obj_tag(v___x_3372_) == 0)
{
lean_object* v_a_3373_; lean_object* v___x_3375_; uint8_t v_isShared_3376_; uint8_t v_isSharedCheck_3381_; 
lean_dec(v_a_3371_);
v_a_3373_ = lean_ctor_get(v___x_3372_, 0);
v_isSharedCheck_3381_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3381_ == 0)
{
v___x_3375_ = v___x_3372_;
v_isShared_3376_ = v_isSharedCheck_3381_;
goto v_resetjp_3374_;
}
else
{
lean_inc(v_a_3373_);
lean_dec(v___x_3372_);
v___x_3375_ = lean_box(0);
v_isShared_3376_ = v_isSharedCheck_3381_;
goto v_resetjp_3374_;
}
v_resetjp_3374_:
{
lean_object* v___x_3377_; lean_object* v___x_3379_; 
v___x_3377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3377_, 0, v_a_3373_);
if (v_isShared_3376_ == 0)
{
lean_ctor_set(v___x_3375_, 0, v___x_3377_);
v___x_3379_ = v___x_3375_;
goto v_reusejp_3378_;
}
else
{
lean_object* v_reuseFailAlloc_3380_; 
v_reuseFailAlloc_3380_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3380_, 0, v___x_3377_);
v___x_3379_ = v_reuseFailAlloc_3380_;
goto v_reusejp_3378_;
}
v_reusejp_3378_:
{
return v___x_3379_;
}
}
}
else
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3411_; 
v_a_3382_ = lean_ctor_get(v___x_3372_, 0);
v_isSharedCheck_3411_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3411_ == 0)
{
v___x_3384_ = v___x_3372_;
v_isShared_3385_ = v_isSharedCheck_3411_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3372_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3411_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
uint8_t v___y_3387_; uint8_t v___x_3409_; 
v___x_3409_ = l_Lean_Exception_isInterrupt(v_a_3382_);
if (v___x_3409_ == 0)
{
uint8_t v___x_3410_; 
lean_inc(v_a_3382_);
v___x_3410_ = l_Lean_Exception_isRuntime(v_a_3382_);
v___y_3387_ = v___x_3410_;
goto v___jp_3386_;
}
else
{
v___y_3387_ = v___x_3409_;
goto v___jp_3386_;
}
v___jp_3386_:
{
if (v___y_3387_ == 0)
{
lean_object* v___x_3388_; 
lean_del_object(v___x_3384_);
lean_dec(v_a_3382_);
v___x_3388_ = l_Lean_Meta_SavedState_restore___redArg(v_a_3371_, v___y_3366_, v___y_3368_);
lean_dec(v_a_3371_);
if (lean_obj_tag(v___x_3388_) == 0)
{
lean_object* v___x_3390_; uint8_t v_isShared_3391_; uint8_t v_isSharedCheck_3396_; 
v_isSharedCheck_3396_ = !lean_is_exclusive(v___x_3388_);
if (v_isSharedCheck_3396_ == 0)
{
lean_object* v_unused_3397_; 
v_unused_3397_ = lean_ctor_get(v___x_3388_, 0);
lean_dec(v_unused_3397_);
v___x_3390_ = v___x_3388_;
v_isShared_3391_ = v_isSharedCheck_3396_;
goto v_resetjp_3389_;
}
else
{
lean_dec(v___x_3388_);
v___x_3390_ = lean_box(0);
v_isShared_3391_ = v_isSharedCheck_3396_;
goto v_resetjp_3389_;
}
v_resetjp_3389_:
{
lean_object* v___x_3392_; lean_object* v___x_3394_; 
v___x_3392_ = lean_box(0);
if (v_isShared_3391_ == 0)
{
lean_ctor_set(v___x_3390_, 0, v___x_3392_);
v___x_3394_ = v___x_3390_;
goto v_reusejp_3393_;
}
else
{
lean_object* v_reuseFailAlloc_3395_; 
v_reuseFailAlloc_3395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3395_, 0, v___x_3392_);
v___x_3394_ = v_reuseFailAlloc_3395_;
goto v_reusejp_3393_;
}
v_reusejp_3393_:
{
return v___x_3394_;
}
}
}
else
{
lean_object* v_a_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3405_; 
v_a_3398_ = lean_ctor_get(v___x_3388_, 0);
v_isSharedCheck_3405_ = !lean_is_exclusive(v___x_3388_);
if (v_isSharedCheck_3405_ == 0)
{
v___x_3400_ = v___x_3388_;
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_a_3398_);
lean_dec(v___x_3388_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3403_; 
if (v_isShared_3401_ == 0)
{
v___x_3403_ = v___x_3400_;
goto v_reusejp_3402_;
}
else
{
lean_object* v_reuseFailAlloc_3404_; 
v_reuseFailAlloc_3404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3404_, 0, v_a_3398_);
v___x_3403_ = v_reuseFailAlloc_3404_;
goto v_reusejp_3402_;
}
v_reusejp_3402_:
{
return v___x_3403_;
}
}
}
}
else
{
lean_object* v___x_3407_; 
lean_dec(v_a_3371_);
if (v_isShared_3385_ == 0)
{
v___x_3407_ = v___x_3384_;
goto v_reusejp_3406_;
}
else
{
lean_object* v_reuseFailAlloc_3408_; 
v_reuseFailAlloc_3408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3408_, 0, v_a_3382_);
v___x_3407_ = v_reuseFailAlloc_3408_;
goto v_reusejp_3406_;
}
v_reusejp_3406_:
{
return v___x_3407_;
}
}
}
}
}
}
else
{
lean_object* v_a_3412_; lean_object* v___x_3414_; uint8_t v_isShared_3415_; uint8_t v_isSharedCheck_3419_; 
lean_dec_ref(v_x_3364_);
v_a_3412_ = lean_ctor_get(v___x_3370_, 0);
v_isSharedCheck_3419_ = !lean_is_exclusive(v___x_3370_);
if (v_isSharedCheck_3419_ == 0)
{
v___x_3414_ = v___x_3370_;
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
else
{
lean_inc(v_a_3412_);
lean_dec(v___x_3370_);
v___x_3414_ = lean_box(0);
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
v_resetjp_3413_:
{
lean_object* v___x_3417_; 
if (v_isShared_3415_ == 0)
{
v___x_3417_ = v___x_3414_;
goto v_reusejp_3416_;
}
else
{
lean_object* v_reuseFailAlloc_3418_; 
v_reuseFailAlloc_3418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3418_, 0, v_a_3412_);
v___x_3417_ = v_reuseFailAlloc_3418_;
goto v_reusejp_3416_;
}
v_reusejp_3416_:
{
return v___x_3417_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg___boxed(lean_object* v_x_3420_, lean_object* v___y_3421_, lean_object* v___y_3422_, lean_object* v___y_3423_, lean_object* v___y_3424_, lean_object* v___y_3425_){
_start:
{
lean_object* v_res_3426_; 
v_res_3426_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg(v_x_3420_, v___y_3421_, v___y_3422_, v___y_3423_, v___y_3424_);
lean_dec(v___y_3424_);
lean_dec_ref(v___y_3423_);
lean_dec(v___y_3422_);
lean_dec_ref(v___y_3421_);
return v_res_3426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1(lean_object* v_00_u03b1_3427_, lean_object* v_x_3428_, lean_object* v___y_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_, lean_object* v___y_3432_){
_start:
{
lean_object* v___x_3434_; 
v___x_3434_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg(v_x_3428_, v___y_3429_, v___y_3430_, v___y_3431_, v___y_3432_);
return v___x_3434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___boxed(lean_object* v_00_u03b1_3435_, lean_object* v_x_3436_, lean_object* v___y_3437_, lean_object* v___y_3438_, lean_object* v___y_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_){
_start:
{
lean_object* v_res_3442_; 
v_res_3442_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1(v_00_u03b1_3435_, v_x_3436_, v___y_3437_, v___y_3438_, v___y_3439_, v___y_3440_);
lean_dec(v___y_3440_);
lean_dec_ref(v___y_3439_);
lean_dec(v___y_3438_);
lean_dec_ref(v___y_3437_);
return v_res_3442_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_3443_; lean_object* v___x_3444_; lean_object* v___x_3445_; 
v___x_3443_ = lean_unsigned_to_nat(32u);
v___x_3444_ = lean_mk_empty_array_with_capacity(v___x_3443_);
v___x_3445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3445_, 0, v___x_3444_);
return v___x_3445_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1(void){
_start:
{
size_t v___x_3446_; lean_object* v___x_3447_; lean_object* v___x_3448_; lean_object* v___x_3449_; lean_object* v___x_3450_; lean_object* v___x_3451_; 
v___x_3446_ = ((size_t)5ULL);
v___x_3447_ = lean_unsigned_to_nat(0u);
v___x_3448_ = lean_unsigned_to_nat(32u);
v___x_3449_ = lean_mk_empty_array_with_capacity(v___x_3448_);
v___x_3450_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__0);
v___x_3451_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3451_, 0, v___x_3450_);
lean_ctor_set(v___x_3451_, 1, v___x_3449_);
lean_ctor_set(v___x_3451_, 2, v___x_3447_);
lean_ctor_set(v___x_3451_, 3, v___x_3447_);
lean_ctor_set_usize(v___x_3451_, 4, v___x_3446_);
return v___x_3451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(lean_object* v___y_3452_){
_start:
{
lean_object* v___x_3454_; lean_object* v_traceState_3455_; lean_object* v_traces_3456_; lean_object* v___x_3457_; lean_object* v_traceState_3458_; lean_object* v_env_3459_; lean_object* v_nextMacroScope_3460_; lean_object* v_ngen_3461_; lean_object* v_auxDeclNGen_3462_; lean_object* v_cache_3463_; lean_object* v_messages_3464_; lean_object* v_infoState_3465_; lean_object* v_snapshotTasks_3466_; lean_object* v___x_3468_; uint8_t v_isShared_3469_; uint8_t v_isSharedCheck_3485_; 
v___x_3454_ = lean_st_ref_get(v___y_3452_);
v_traceState_3455_ = lean_ctor_get(v___x_3454_, 4);
lean_inc_ref(v_traceState_3455_);
lean_dec(v___x_3454_);
v_traces_3456_ = lean_ctor_get(v_traceState_3455_, 0);
lean_inc_ref(v_traces_3456_);
lean_dec_ref(v_traceState_3455_);
v___x_3457_ = lean_st_ref_take(v___y_3452_);
v_traceState_3458_ = lean_ctor_get(v___x_3457_, 4);
v_env_3459_ = lean_ctor_get(v___x_3457_, 0);
v_nextMacroScope_3460_ = lean_ctor_get(v___x_3457_, 1);
v_ngen_3461_ = lean_ctor_get(v___x_3457_, 2);
v_auxDeclNGen_3462_ = lean_ctor_get(v___x_3457_, 3);
v_cache_3463_ = lean_ctor_get(v___x_3457_, 5);
v_messages_3464_ = lean_ctor_get(v___x_3457_, 6);
v_infoState_3465_ = lean_ctor_get(v___x_3457_, 7);
v_snapshotTasks_3466_ = lean_ctor_get(v___x_3457_, 8);
v_isSharedCheck_3485_ = !lean_is_exclusive(v___x_3457_);
if (v_isSharedCheck_3485_ == 0)
{
v___x_3468_ = v___x_3457_;
v_isShared_3469_ = v_isSharedCheck_3485_;
goto v_resetjp_3467_;
}
else
{
lean_inc(v_snapshotTasks_3466_);
lean_inc(v_infoState_3465_);
lean_inc(v_messages_3464_);
lean_inc(v_cache_3463_);
lean_inc(v_traceState_3458_);
lean_inc(v_auxDeclNGen_3462_);
lean_inc(v_ngen_3461_);
lean_inc(v_nextMacroScope_3460_);
lean_inc(v_env_3459_);
lean_dec(v___x_3457_);
v___x_3468_ = lean_box(0);
v_isShared_3469_ = v_isSharedCheck_3485_;
goto v_resetjp_3467_;
}
v_resetjp_3467_:
{
uint64_t v_tid_3470_; lean_object* v___x_3472_; uint8_t v_isShared_3473_; uint8_t v_isSharedCheck_3483_; 
v_tid_3470_ = lean_ctor_get_uint64(v_traceState_3458_, sizeof(void*)*1);
v_isSharedCheck_3483_ = !lean_is_exclusive(v_traceState_3458_);
if (v_isSharedCheck_3483_ == 0)
{
lean_object* v_unused_3484_; 
v_unused_3484_ = lean_ctor_get(v_traceState_3458_, 0);
lean_dec(v_unused_3484_);
v___x_3472_ = v_traceState_3458_;
v_isShared_3473_ = v_isSharedCheck_3483_;
goto v_resetjp_3471_;
}
else
{
lean_dec(v_traceState_3458_);
v___x_3472_ = lean_box(0);
v_isShared_3473_ = v_isSharedCheck_3483_;
goto v_resetjp_3471_;
}
v_resetjp_3471_:
{
lean_object* v___x_3474_; lean_object* v___x_3476_; 
v___x_3474_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___closed__1);
if (v_isShared_3473_ == 0)
{
lean_ctor_set(v___x_3472_, 0, v___x_3474_);
v___x_3476_ = v___x_3472_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3482_; 
v_reuseFailAlloc_3482_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3482_, 0, v___x_3474_);
lean_ctor_set_uint64(v_reuseFailAlloc_3482_, sizeof(void*)*1, v_tid_3470_);
v___x_3476_ = v_reuseFailAlloc_3482_;
goto v_reusejp_3475_;
}
v_reusejp_3475_:
{
lean_object* v___x_3478_; 
if (v_isShared_3469_ == 0)
{
lean_ctor_set(v___x_3468_, 4, v___x_3476_);
v___x_3478_ = v___x_3468_;
goto v_reusejp_3477_;
}
else
{
lean_object* v_reuseFailAlloc_3481_; 
v_reuseFailAlloc_3481_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3481_, 0, v_env_3459_);
lean_ctor_set(v_reuseFailAlloc_3481_, 1, v_nextMacroScope_3460_);
lean_ctor_set(v_reuseFailAlloc_3481_, 2, v_ngen_3461_);
lean_ctor_set(v_reuseFailAlloc_3481_, 3, v_auxDeclNGen_3462_);
lean_ctor_set(v_reuseFailAlloc_3481_, 4, v___x_3476_);
lean_ctor_set(v_reuseFailAlloc_3481_, 5, v_cache_3463_);
lean_ctor_set(v_reuseFailAlloc_3481_, 6, v_messages_3464_);
lean_ctor_set(v_reuseFailAlloc_3481_, 7, v_infoState_3465_);
lean_ctor_set(v_reuseFailAlloc_3481_, 8, v_snapshotTasks_3466_);
v___x_3478_ = v_reuseFailAlloc_3481_;
goto v_reusejp_3477_;
}
v_reusejp_3477_:
{
lean_object* v___x_3479_; lean_object* v___x_3480_; 
v___x_3479_ = lean_st_ref_set(v___y_3452_, v___x_3478_);
v___x_3480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3480_, 0, v_traces_3456_);
return v___x_3480_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg___boxed(lean_object* v___y_3486_, lean_object* v___y_3487_){
_start:
{
lean_object* v_res_3488_; 
v_res_3488_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(v___y_3486_);
lean_dec(v___y_3486_);
return v_res_3488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8(lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_){
_start:
{
lean_object* v___x_3494_; 
v___x_3494_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(v___y_3492_);
return v___x_3494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___boxed(lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_){
_start:
{
lean_object* v_res_3500_; 
v_res_3500_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8(v___y_3495_, v___y_3496_, v___y_3497_, v___y_3498_);
lean_dec(v___y_3498_);
lean_dec_ref(v___y_3497_);
lean_dec(v___y_3496_);
lean_dec_ref(v___y_3495_);
return v_res_3500_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(lean_object* v_opts_3501_, lean_object* v_opt_3502_){
_start:
{
lean_object* v_name_3503_; lean_object* v_defValue_3504_; lean_object* v_map_3505_; lean_object* v___x_3506_; 
v_name_3503_ = lean_ctor_get(v_opt_3502_, 0);
v_defValue_3504_ = lean_ctor_get(v_opt_3502_, 1);
v_map_3505_ = lean_ctor_get(v_opts_3501_, 0);
v___x_3506_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_3505_, v_name_3503_);
if (lean_obj_tag(v___x_3506_) == 0)
{
uint8_t v___x_3507_; 
v___x_3507_ = lean_unbox(v_defValue_3504_);
return v___x_3507_;
}
else
{
lean_object* v_val_3508_; 
v_val_3508_ = lean_ctor_get(v___x_3506_, 0);
lean_inc(v_val_3508_);
lean_dec_ref_known(v___x_3506_, 1);
if (lean_obj_tag(v_val_3508_) == 1)
{
uint8_t v_v_3509_; 
v_v_3509_ = lean_ctor_get_uint8(v_val_3508_, 0);
lean_dec_ref_known(v_val_3508_, 0);
return v_v_3509_;
}
else
{
uint8_t v___x_3510_; 
lean_dec(v_val_3508_);
v___x_3510_ = lean_unbox(v_defValue_3504_);
return v___x_3510_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9___boxed(lean_object* v_opts_3511_, lean_object* v_opt_3512_){
_start:
{
uint8_t v_res_3513_; lean_object* v_r_3514_; 
v_res_3513_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_opts_3511_, v_opt_3512_);
lean_dec_ref(v_opt_3512_);
lean_dec_ref(v_opts_3511_);
v_r_3514_ = lean_box(v_res_3513_);
return v_r_3514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3(lean_object* v_init_3515_, lean_object* v_x_3516_){
_start:
{
if (lean_obj_tag(v_x_3516_) == 0)
{
lean_object* v_k_3517_; lean_object* v_l_3518_; lean_object* v_r_3519_; lean_object* v___x_3520_; lean_object* v___x_3521_; 
v_k_3517_ = lean_ctor_get(v_x_3516_, 1);
v_l_3518_ = lean_ctor_get(v_x_3516_, 3);
v_r_3519_ = lean_ctor_get(v_x_3516_, 4);
v___x_3520_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3(v_init_3515_, v_r_3519_);
lean_inc(v_k_3517_);
v___x_3521_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3521_, 0, v_k_3517_);
lean_ctor_set(v___x_3521_, 1, v___x_3520_);
v_init_3515_ = v___x_3521_;
v_x_3516_ = v_l_3518_;
goto _start;
}
else
{
return v_init_3515_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3___boxed(lean_object* v_init_3523_, lean_object* v_x_3524_){
_start:
{
lean_object* v_res_3525_; 
v_res_3525_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3(v_init_3523_, v_x_3524_);
lean_dec(v_x_3524_);
return v_res_3525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg(lean_object* v_x_3526_, lean_object* v_x_3527_, lean_object* v___y_3528_){
_start:
{
if (lean_obj_tag(v_x_3526_) == 0)
{
lean_object* v___x_3530_; lean_object* v___x_3531_; 
v___x_3530_ = l_List_reverse___redArg(v_x_3527_);
v___x_3531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3531_, 0, v___x_3530_);
return v___x_3531_;
}
else
{
lean_object* v_head_3532_; lean_object* v_tail_3533_; lean_object* v___x_3535_; uint8_t v_isShared_3536_; uint8_t v_isSharedCheck_3553_; 
v_head_3532_ = lean_ctor_get(v_x_3526_, 0);
v_tail_3533_ = lean_ctor_get(v_x_3526_, 1);
v_isSharedCheck_3553_ = !lean_is_exclusive(v_x_3526_);
if (v_isSharedCheck_3553_ == 0)
{
v___x_3535_ = v_x_3526_;
v_isShared_3536_ = v_isSharedCheck_3553_;
goto v_resetjp_3534_;
}
else
{
lean_inc(v_tail_3533_);
lean_inc(v_head_3532_);
lean_dec(v_x_3526_);
v___x_3535_ = lean_box(0);
v_isShared_3536_ = v_isSharedCheck_3553_;
goto v_resetjp_3534_;
}
v_resetjp_3534_:
{
lean_object* v_fst_3537_; lean_object* v_snd_3538_; lean_object* v___x_3540_; uint8_t v_isShared_3541_; uint8_t v_isSharedCheck_3552_; 
v_fst_3537_ = lean_ctor_get(v_head_3532_, 0);
v_snd_3538_ = lean_ctor_get(v_head_3532_, 1);
v_isSharedCheck_3552_ = !lean_is_exclusive(v_head_3532_);
if (v_isSharedCheck_3552_ == 0)
{
v___x_3540_ = v_head_3532_;
v_isShared_3541_ = v_isSharedCheck_3552_;
goto v_resetjp_3539_;
}
else
{
lean_inc(v_snd_3538_);
lean_inc(v_fst_3537_);
lean_dec(v_head_3532_);
v___x_3540_ = lean_box(0);
v_isShared_3541_ = v_isSharedCheck_3552_;
goto v_resetjp_3539_;
}
v_resetjp_3539_:
{
lean_object* v___x_3542_; lean_object* v___x_3543_; lean_object* v___x_3544_; lean_object* v___x_3546_; 
v___x_3542_ = lean_st_ref_get(v___y_3528_);
v___x_3543_ = l_Lean_instInhabitedExpr;
v___x_3544_ = lean_array_get(v___x_3543_, v___x_3542_, v_fst_3537_);
lean_dec(v_fst_3537_);
lean_dec(v___x_3542_);
if (v_isShared_3541_ == 0)
{
lean_ctor_set(v___x_3540_, 0, v___x_3544_);
v___x_3546_ = v___x_3540_;
goto v_reusejp_3545_;
}
else
{
lean_object* v_reuseFailAlloc_3551_; 
v_reuseFailAlloc_3551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3551_, 0, v___x_3544_);
lean_ctor_set(v_reuseFailAlloc_3551_, 1, v_snd_3538_);
v___x_3546_ = v_reuseFailAlloc_3551_;
goto v_reusejp_3545_;
}
v_reusejp_3545_:
{
lean_object* v___x_3548_; 
if (v_isShared_3536_ == 0)
{
lean_ctor_set(v___x_3535_, 1, v_x_3527_);
lean_ctor_set(v___x_3535_, 0, v___x_3546_);
v___x_3548_ = v___x_3535_;
goto v_reusejp_3547_;
}
else
{
lean_object* v_reuseFailAlloc_3550_; 
v_reuseFailAlloc_3550_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3550_, 0, v___x_3546_);
lean_ctor_set(v_reuseFailAlloc_3550_, 1, v_x_3527_);
v___x_3548_ = v_reuseFailAlloc_3550_;
goto v_reusejp_3547_;
}
v_reusejp_3547_:
{
v_x_3526_ = v_tail_3533_;
v_x_3527_ = v___x_3548_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg___boxed(lean_object* v_x_3554_, lean_object* v_x_3555_, lean_object* v___y_3556_, lean_object* v___y_3557_){
_start:
{
lean_object* v_res_3558_; 
v_res_3558_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg(v_x_3554_, v_x_3555_, v___y_3556_);
lean_dec(v___y_3556_);
return v_res_3558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2(lean_object* v_x_3559_, lean_object* v_x_3560_, lean_object* v___y_3561_, lean_object* v___y_3562_, lean_object* v___y_3563_, lean_object* v___y_3564_, lean_object* v___y_3565_, lean_object* v___y_3566_){
_start:
{
if (lean_obj_tag(v_x_3560_) == 0)
{
lean_object* v___x_3568_; 
v___x_3568_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3568_, 0, v_x_3559_);
return v___x_3568_;
}
else
{
lean_object* v_head_3569_; lean_object* v_tail_3570_; lean_object* v___x_3571_; 
v_head_3569_ = lean_ctor_get(v_x_3560_, 0);
lean_inc(v_head_3569_);
v_tail_3570_ = lean_ctor_get(v_x_3560_, 1);
lean_inc(v_tail_3570_);
lean_dec_ref_known(v_x_3560_, 2);
lean_inc(v___y_3566_);
lean_inc_ref(v___y_3565_);
lean_inc(v___y_3564_);
lean_inc_ref(v___y_3563_);
v___x_3571_ = lean_infer_type(v_head_3569_, v___y_3563_, v___y_3564_, v___y_3565_, v___y_3566_);
if (lean_obj_tag(v___x_3571_) == 0)
{
lean_object* v_a_3572_; lean_object* v___x_3573_; lean_object* v_a_3574_; lean_object* v___x_3575_; 
v_a_3572_ = lean_ctor_get(v___x_3571_, 0);
lean_inc(v_a_3572_);
lean_dec_ref_known(v___x_3571_, 1);
v___x_3573_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__0___redArg(v_a_3572_, v___y_3564_);
v_a_3574_ = lean_ctor_get(v___x_3573_, 0);
lean_inc(v_a_3574_);
lean_dec_ref(v___x_3573_);
v___x_3575_ = lp_mathlib_Mathlib_Tactic_Linarith_findSquares(v_x_3559_, v_a_3574_, v___y_3561_, v___y_3562_, v___y_3563_, v___y_3564_, v___y_3565_, v___y_3566_);
if (lean_obj_tag(v___x_3575_) == 0)
{
lean_object* v_a_3576_; 
v_a_3576_ = lean_ctor_get(v___x_3575_, 0);
lean_inc(v_a_3576_);
lean_dec_ref_known(v___x_3575_, 1);
v_x_3559_ = v_a_3576_;
v_x_3560_ = v_tail_3570_;
goto _start;
}
else
{
lean_dec(v_tail_3570_);
return v___x_3575_;
}
}
else
{
lean_object* v_a_3578_; lean_object* v___x_3580_; uint8_t v_isShared_3581_; uint8_t v_isSharedCheck_3585_; 
lean_dec(v_tail_3570_);
lean_dec(v_x_3559_);
v_a_3578_ = lean_ctor_get(v___x_3571_, 0);
v_isSharedCheck_3585_ = !lean_is_exclusive(v___x_3571_);
if (v_isSharedCheck_3585_ == 0)
{
v___x_3580_ = v___x_3571_;
v_isShared_3581_ = v_isSharedCheck_3585_;
goto v_resetjp_3579_;
}
else
{
lean_inc(v_a_3578_);
lean_dec(v___x_3571_);
v___x_3580_ = lean_box(0);
v_isShared_3581_ = v_isSharedCheck_3585_;
goto v_resetjp_3579_;
}
v_resetjp_3579_:
{
lean_object* v___x_3583_; 
if (v_isShared_3581_ == 0)
{
v___x_3583_ = v___x_3580_;
goto v_reusejp_3582_;
}
else
{
lean_object* v_reuseFailAlloc_3584_; 
v_reuseFailAlloc_3584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3584_, 0, v_a_3578_);
v___x_3583_ = v_reuseFailAlloc_3584_;
goto v_reusejp_3582_;
}
v_reusejp_3582_:
{
return v___x_3583_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2___boxed(lean_object* v_x_3586_, lean_object* v_x_3587_, lean_object* v___y_3588_, lean_object* v___y_3589_, lean_object* v___y_3590_, lean_object* v___y_3591_, lean_object* v___y_3592_, lean_object* v___y_3593_, lean_object* v___y_3594_){
_start:
{
lean_object* v_res_3595_; 
v_res_3595_ = lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2(v_x_3586_, v_x_3587_, v___y_3588_, v___y_3589_, v___y_3590_, v___y_3591_, v___y_3592_, v___y_3593_);
lean_dec(v___y_3593_);
lean_dec_ref(v___y_3592_);
lean_dec(v___y_3591_);
lean_dec_ref(v___y_3590_);
lean_dec(v___y_3589_);
lean_dec_ref(v___y_3588_);
return v_res_3595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1(lean_object* v___x_3596_, lean_object* v___x_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_, lean_object* v___y_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_){
_start:
{
lean_object* v___x_3605_; 
v___x_3605_ = lp_mathlib_List_foldlM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__2(v___x_3596_, v___x_3597_, v___y_3598_, v___y_3599_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_);
if (lean_obj_tag(v___x_3605_) == 0)
{
lean_object* v_a_3606_; lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; 
v_a_3606_ = lean_ctor_get(v___x_3605_, 0);
lean_inc(v_a_3606_);
lean_dec_ref_known(v___x_3605_, 1);
v___x_3607_ = lean_box(0);
v___x_3608_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__3(v___x_3607_, v_a_3606_);
lean_dec(v_a_3606_);
v___x_3609_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg(v___x_3608_, v___x_3607_, v___y_3599_);
return v___x_3609_;
}
else
{
lean_object* v_a_3610_; lean_object* v___x_3612_; uint8_t v_isShared_3613_; uint8_t v_isSharedCheck_3617_; 
v_a_3610_ = lean_ctor_get(v___x_3605_, 0);
v_isSharedCheck_3617_ = !lean_is_exclusive(v___x_3605_);
if (v_isSharedCheck_3617_ == 0)
{
v___x_3612_ = v___x_3605_;
v_isShared_3613_ = v_isSharedCheck_3617_;
goto v_resetjp_3611_;
}
else
{
lean_inc(v_a_3610_);
lean_dec(v___x_3605_);
v___x_3612_ = lean_box(0);
v_isShared_3613_ = v_isSharedCheck_3617_;
goto v_resetjp_3611_;
}
v_resetjp_3611_:
{
lean_object* v___x_3615_; 
if (v_isShared_3613_ == 0)
{
v___x_3615_ = v___x_3612_;
goto v_reusejp_3614_;
}
else
{
lean_object* v_reuseFailAlloc_3616_; 
v_reuseFailAlloc_3616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3616_, 0, v_a_3610_);
v___x_3615_ = v_reuseFailAlloc_3616_;
goto v_reusejp_3614_;
}
v_reusejp_3614_:
{
return v___x_3615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1___boxed(lean_object* v___x_3618_, lean_object* v___x_3619_, lean_object* v___y_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_, lean_object* v___y_3623_, lean_object* v___y_3624_, lean_object* v___y_3625_, lean_object* v___y_3626_){
_start:
{
lean_object* v_res_3627_; 
v_res_3627_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1(v___x_3618_, v___x_3619_, v___y_3620_, v___y_3621_, v___y_3622_, v___y_3623_, v___y_3624_, v___y_3625_);
lean_dec(v___y_3625_);
lean_dec_ref(v___y_3624_);
lean_dec(v___y_3623_);
lean_dec_ref(v___y_3622_);
lean_dec(v___y_3621_);
lean_dec_ref(v___y_3620_);
return v_res_3627_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3629_; lean_object* v___x_3630_; 
v___x_3629_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__0));
v___x_3630_ = l_Lean_stringToMessageData(v___x_3629_);
return v___x_3630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0(lean_object* v_x_3631_, lean_object* v___y_3632_, lean_object* v___y_3633_, lean_object* v___y_3634_, lean_object* v___y_3635_){
_start:
{
lean_object* v___x_3637_; lean_object* v___x_3638_; 
v___x_3637_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___closed__1);
v___x_3638_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3638_, 0, v___x_3637_);
return v___x_3638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0___boxed(lean_object* v_x_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_){
_start:
{
lean_object* v_res_3645_; 
v_res_3645_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__0(v_x_3639_, v___y_3640_, v___y_3641_, v___y_3642_, v___y_3643_);
lean_dec(v___y_3643_);
lean_dec_ref(v___y_3642_);
lean_dec(v___y_3641_);
lean_dec_ref(v___y_3640_);
lean_dec_ref(v_x_3639_);
return v_res_3645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6(lean_object* v_msg_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_){
_start:
{
lean_object* v_ref_3652_; lean_object* v___x_3653_; lean_object* v_a_3654_; lean_object* v___x_3656_; uint8_t v_isShared_3657_; uint8_t v_isSharedCheck_3691_; 
v_ref_3652_ = lean_ctor_get(v___y_3649_, 5);
v___x_3653_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(v_msg_3646_, v___y_3647_, v___y_3648_, v___y_3649_, v___y_3650_);
v_a_3654_ = lean_ctor_get(v___x_3653_, 0);
v_isSharedCheck_3691_ = !lean_is_exclusive(v___x_3653_);
if (v_isSharedCheck_3691_ == 0)
{
v___x_3656_ = v___x_3653_;
v_isShared_3657_ = v_isSharedCheck_3691_;
goto v_resetjp_3655_;
}
else
{
lean_inc(v_a_3654_);
lean_dec(v___x_3653_);
v___x_3656_ = lean_box(0);
v_isShared_3657_ = v_isSharedCheck_3691_;
goto v_resetjp_3655_;
}
v_resetjp_3655_:
{
lean_object* v___x_3658_; lean_object* v_traceState_3659_; lean_object* v_env_3660_; lean_object* v_nextMacroScope_3661_; lean_object* v_ngen_3662_; lean_object* v_auxDeclNGen_3663_; lean_object* v_cache_3664_; lean_object* v_messages_3665_; lean_object* v_infoState_3666_; lean_object* v_snapshotTasks_3667_; lean_object* v___x_3669_; uint8_t v_isShared_3670_; uint8_t v_isSharedCheck_3690_; 
v___x_3658_ = lean_st_ref_take(v___y_3650_);
v_traceState_3659_ = lean_ctor_get(v___x_3658_, 4);
v_env_3660_ = lean_ctor_get(v___x_3658_, 0);
v_nextMacroScope_3661_ = lean_ctor_get(v___x_3658_, 1);
v_ngen_3662_ = lean_ctor_get(v___x_3658_, 2);
v_auxDeclNGen_3663_ = lean_ctor_get(v___x_3658_, 3);
v_cache_3664_ = lean_ctor_get(v___x_3658_, 5);
v_messages_3665_ = lean_ctor_get(v___x_3658_, 6);
v_infoState_3666_ = lean_ctor_get(v___x_3658_, 7);
v_snapshotTasks_3667_ = lean_ctor_get(v___x_3658_, 8);
v_isSharedCheck_3690_ = !lean_is_exclusive(v___x_3658_);
if (v_isSharedCheck_3690_ == 0)
{
v___x_3669_ = v___x_3658_;
v_isShared_3670_ = v_isSharedCheck_3690_;
goto v_resetjp_3668_;
}
else
{
lean_inc(v_snapshotTasks_3667_);
lean_inc(v_infoState_3666_);
lean_inc(v_messages_3665_);
lean_inc(v_cache_3664_);
lean_inc(v_traceState_3659_);
lean_inc(v_auxDeclNGen_3663_);
lean_inc(v_ngen_3662_);
lean_inc(v_nextMacroScope_3661_);
lean_inc(v_env_3660_);
lean_dec(v___x_3658_);
v___x_3669_ = lean_box(0);
v_isShared_3670_ = v_isSharedCheck_3690_;
goto v_resetjp_3668_;
}
v_resetjp_3668_:
{
uint64_t v_tid_3671_; lean_object* v_traces_3672_; lean_object* v___x_3674_; uint8_t v_isShared_3675_; uint8_t v_isSharedCheck_3689_; 
v_tid_3671_ = lean_ctor_get_uint64(v_traceState_3659_, sizeof(void*)*1);
v_traces_3672_ = lean_ctor_get(v_traceState_3659_, 0);
v_isSharedCheck_3689_ = !lean_is_exclusive(v_traceState_3659_);
if (v_isSharedCheck_3689_ == 0)
{
v___x_3674_ = v_traceState_3659_;
v_isShared_3675_ = v_isSharedCheck_3689_;
goto v_resetjp_3673_;
}
else
{
lean_inc(v_traces_3672_);
lean_dec(v_traceState_3659_);
v___x_3674_ = lean_box(0);
v_isShared_3675_ = v_isSharedCheck_3689_;
goto v_resetjp_3673_;
}
v_resetjp_3673_:
{
lean_object* v___x_3676_; lean_object* v___x_3677_; lean_object* v___x_3679_; 
lean_inc(v_ref_3652_);
v___x_3676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3676_, 0, v_ref_3652_);
lean_ctor_set(v___x_3676_, 1, v_a_3654_);
v___x_3677_ = l_Lean_PersistentArray_push___redArg(v_traces_3672_, v___x_3676_);
if (v_isShared_3675_ == 0)
{
lean_ctor_set(v___x_3674_, 0, v___x_3677_);
v___x_3679_ = v___x_3674_;
goto v_reusejp_3678_;
}
else
{
lean_object* v_reuseFailAlloc_3688_; 
v_reuseFailAlloc_3688_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3688_, 0, v___x_3677_);
lean_ctor_set_uint64(v_reuseFailAlloc_3688_, sizeof(void*)*1, v_tid_3671_);
v___x_3679_ = v_reuseFailAlloc_3688_;
goto v_reusejp_3678_;
}
v_reusejp_3678_:
{
lean_object* v___x_3681_; 
if (v_isShared_3670_ == 0)
{
lean_ctor_set(v___x_3669_, 4, v___x_3679_);
v___x_3681_ = v___x_3669_;
goto v_reusejp_3680_;
}
else
{
lean_object* v_reuseFailAlloc_3687_; 
v_reuseFailAlloc_3687_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3687_, 0, v_env_3660_);
lean_ctor_set(v_reuseFailAlloc_3687_, 1, v_nextMacroScope_3661_);
lean_ctor_set(v_reuseFailAlloc_3687_, 2, v_ngen_3662_);
lean_ctor_set(v_reuseFailAlloc_3687_, 3, v_auxDeclNGen_3663_);
lean_ctor_set(v_reuseFailAlloc_3687_, 4, v___x_3679_);
lean_ctor_set(v_reuseFailAlloc_3687_, 5, v_cache_3664_);
lean_ctor_set(v_reuseFailAlloc_3687_, 6, v_messages_3665_);
lean_ctor_set(v_reuseFailAlloc_3687_, 7, v_infoState_3666_);
lean_ctor_set(v_reuseFailAlloc_3687_, 8, v_snapshotTasks_3667_);
v___x_3681_ = v_reuseFailAlloc_3687_;
goto v_reusejp_3680_;
}
v_reusejp_3680_:
{
lean_object* v___x_3682_; lean_object* v___x_3683_; lean_object* v___x_3685_; 
v___x_3682_ = lean_st_ref_set(v___y_3650_, v___x_3681_);
v___x_3683_ = lean_box(0);
if (v_isShared_3657_ == 0)
{
lean_ctor_set(v___x_3656_, 0, v___x_3683_);
v___x_3685_ = v___x_3656_;
goto v_reusejp_3684_;
}
else
{
lean_object* v_reuseFailAlloc_3686_; 
v_reuseFailAlloc_3686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3686_, 0, v___x_3683_);
v___x_3685_ = v_reuseFailAlloc_3686_;
goto v_reusejp_3684_;
}
v_reusejp_3684_:
{
return v___x_3685_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6___boxed(lean_object* v_msg_3692_, lean_object* v___y_3693_, lean_object* v___y_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_, lean_object* v___y_3697_){
_start:
{
lean_object* v_res_3698_; 
v_res_3698_ = lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6(v_msg_3692_, v___y_3693_, v___y_3694_, v___y_3695_, v___y_3696_);
lean_dec(v___y_3696_);
lean_dec_ref(v___y_3695_);
lean_dec(v___y_3694_);
lean_dec_ref(v___y_3693_);
return v_res_3698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(lean_object* v_s_3699_, lean_object* v_l_3700_, lean_object* v_a_3701_, lean_object* v_a_3702_, lean_object* v_a_3703_, lean_object* v_a_3704_){
_start:
{
lean_object* v_options_3709_; uint8_t v_hasTrace_3710_; 
v_options_3709_ = lean_ctor_get(v_a_3703_, 2);
v_hasTrace_3710_ = lean_ctor_get_uint8(v_options_3709_, sizeof(void*)*1);
if (v_hasTrace_3710_ == 0)
{
lean_dec(v_l_3700_);
lean_dec_ref(v_s_3699_);
goto v___jp_3706_;
}
else
{
lean_object* v_inheritedTraceOptions_3711_; lean_object* v___x_3712_; lean_object* v___x_3713_; uint8_t v___x_3714_; 
v_inheritedTraceOptions_3711_ = lean_ctor_get(v_a_3703_, 13);
v___x_3712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_3713_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6);
v___x_3714_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3711_, v_options_3709_, v___x_3713_);
if (v___x_3714_ == 0)
{
lean_dec(v_l_3700_);
lean_dec_ref(v_s_3699_);
goto v___jp_3706_;
}
else
{
lean_object* v___x_3715_; 
v___x_3715_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(v_l_3700_, v_a_3701_, v_a_3702_, v_a_3703_, v_a_3704_);
if (lean_obj_tag(v___x_3715_) == 0)
{
lean_object* v_a_3716_; lean_object* v___x_3717_; double v___x_3718_; lean_object* v___x_3719_; lean_object* v___x_3720_; lean_object* v___x_3721_; lean_object* v___x_3722_; lean_object* v___x_3723_; lean_object* v___x_3724_; lean_object* v___x_3725_; lean_object* v___x_3726_; 
v_a_3716_ = lean_ctor_get(v___x_3715_, 0);
lean_inc(v_a_3716_);
lean_dec_ref_known(v___x_3715_, 1);
v___x_3717_ = lean_box(0);
v___x_3718_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0);
v___x_3719_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1));
v___x_3720_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3720_, 0, v___x_3712_);
lean_ctor_set(v___x_3720_, 1, v___x_3717_);
lean_ctor_set(v___x_3720_, 2, v___x_3719_);
lean_ctor_set_float(v___x_3720_, sizeof(void*)*3, v___x_3718_);
lean_ctor_set_float(v___x_3720_, sizeof(void*)*3 + 8, v___x_3718_);
lean_ctor_set_uint8(v___x_3720_, sizeof(void*)*3 + 16, v_hasTrace_3710_);
v___x_3721_ = l_Lean_stringToMessageData(v_s_3699_);
v___x_3722_ = lean_unsigned_to_nat(1u);
v___x_3723_ = lean_mk_empty_array_with_capacity(v___x_3722_);
v___x_3724_ = lean_array_push(v___x_3723_, v_a_3716_);
v___x_3725_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3725_, 0, v___x_3720_);
lean_ctor_set(v___x_3725_, 1, v___x_3721_);
lean_ctor_set(v___x_3725_, 2, v___x_3724_);
v___x_3726_ = lp_mathlib_Lean_addRawTrace___at___00Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6_spec__6(v___x_3725_, v_a_3701_, v_a_3702_, v_a_3703_, v_a_3704_);
return v___x_3726_;
}
else
{
lean_object* v_a_3727_; lean_object* v___x_3729_; uint8_t v_isShared_3730_; uint8_t v_isSharedCheck_3734_; 
lean_dec_ref(v_s_3699_);
v_a_3727_ = lean_ctor_get(v___x_3715_, 0);
v_isSharedCheck_3734_ = !lean_is_exclusive(v___x_3715_);
if (v_isSharedCheck_3734_ == 0)
{
v___x_3729_ = v___x_3715_;
v_isShared_3730_ = v_isSharedCheck_3734_;
goto v_resetjp_3728_;
}
else
{
lean_inc(v_a_3727_);
lean_dec(v___x_3715_);
v___x_3729_ = lean_box(0);
v_isShared_3730_ = v_isSharedCheck_3734_;
goto v_resetjp_3728_;
}
v_resetjp_3728_:
{
lean_object* v___x_3732_; 
if (v_isShared_3730_ == 0)
{
v___x_3732_ = v___x_3729_;
goto v_reusejp_3731_;
}
else
{
lean_object* v_reuseFailAlloc_3733_; 
v_reuseFailAlloc_3733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3733_, 0, v_a_3727_);
v___x_3732_ = v_reuseFailAlloc_3733_;
goto v_reusejp_3731_;
}
v_reusejp_3731_:
{
return v___x_3732_;
}
}
}
}
}
v___jp_3706_:
{
lean_object* v___x_3707_; lean_object* v___x_3708_; 
v___x_3707_ = lean_box(0);
v___x_3708_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3708_, 0, v___x_3707_);
return v___x_3708_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6___boxed(lean_object* v_s_3735_, lean_object* v_l_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_a_3739_, lean_object* v_a_3740_, lean_object* v_a_3741_){
_start:
{
lean_object* v_res_3742_; 
v_res_3742_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(v_s_3735_, v_l_3736_, v_a_3737_, v_a_3738_, v_a_3739_, v_a_3740_);
lean_dec(v_a_3740_);
lean_dec_ref(v_a_3739_);
lean_dec(v_a_3738_);
lean_dec_ref(v_a_3737_);
return v_res_3742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(lean_object* v_a_3744_, lean_object* v_____r_3745_, lean_object* v___y_3746_, lean_object* v___y_3747_, lean_object* v___y_3748_, lean_object* v___y_3749_){
_start:
{
lean_object* v___x_3751_; lean_object* v___x_3752_; 
v___x_3751_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0));
lean_inc(v_a_3744_);
v___x_3752_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(v___x_3751_, v_a_3744_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_);
if (lean_obj_tag(v___x_3752_) == 0)
{
lean_object* v___x_3754_; uint8_t v_isShared_3755_; uint8_t v_isSharedCheck_3759_; 
v_isSharedCheck_3759_ = !lean_is_exclusive(v___x_3752_);
if (v_isSharedCheck_3759_ == 0)
{
lean_object* v_unused_3760_; 
v_unused_3760_ = lean_ctor_get(v___x_3752_, 0);
lean_dec(v_unused_3760_);
v___x_3754_ = v___x_3752_;
v_isShared_3755_ = v_isSharedCheck_3759_;
goto v_resetjp_3753_;
}
else
{
lean_dec(v___x_3752_);
v___x_3754_ = lean_box(0);
v_isShared_3755_ = v_isSharedCheck_3759_;
goto v_resetjp_3753_;
}
v_resetjp_3753_:
{
lean_object* v___x_3757_; 
if (v_isShared_3755_ == 0)
{
lean_ctor_set(v___x_3754_, 0, v_a_3744_);
v___x_3757_ = v___x_3754_;
goto v_reusejp_3756_;
}
else
{
lean_object* v_reuseFailAlloc_3758_; 
v_reuseFailAlloc_3758_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3758_, 0, v_a_3744_);
v___x_3757_ = v_reuseFailAlloc_3758_;
goto v_reusejp_3756_;
}
v_reusejp_3756_:
{
return v___x_3757_;
}
}
}
else
{
lean_object* v_a_3761_; lean_object* v___x_3763_; uint8_t v_isShared_3764_; uint8_t v_isSharedCheck_3768_; 
lean_dec(v_a_3744_);
v_a_3761_ = lean_ctor_get(v___x_3752_, 0);
v_isSharedCheck_3768_ = !lean_is_exclusive(v___x_3752_);
if (v_isSharedCheck_3768_ == 0)
{
v___x_3763_ = v___x_3752_;
v_isShared_3764_ = v_isSharedCheck_3768_;
goto v_resetjp_3762_;
}
else
{
lean_inc(v_a_3761_);
lean_dec(v___x_3752_);
v___x_3763_ = lean_box(0);
v_isShared_3764_ = v_isSharedCheck_3768_;
goto v_resetjp_3762_;
}
v_resetjp_3762_:
{
lean_object* v___x_3766_; 
if (v_isShared_3764_ == 0)
{
v___x_3766_ = v___x_3763_;
goto v_reusejp_3765_;
}
else
{
lean_object* v_reuseFailAlloc_3767_; 
v_reuseFailAlloc_3767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3767_, 0, v_a_3761_);
v___x_3766_ = v_reuseFailAlloc_3767_;
goto v_reusejp_3765_;
}
v_reusejp_3765_:
{
return v___x_3766_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___boxed(lean_object* v_a_3769_, lean_object* v_____r_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_, lean_object* v___y_3775_){
_start:
{
lean_object* v_res_3776_; 
v_res_3776_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(v_a_3769_, v_____r_3770_, v___y_3771_, v___y_3772_, v___y_3773_, v___y_3774_);
lean_dec(v___y_3774_);
lean_dec_ref(v___y_3773_);
lean_dec(v___y_3772_);
lean_dec_ref(v___y_3771_);
return v_res_3776_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2(void){
_start:
{
lean_object* v___x_3780_; lean_object* v___x_3781_; 
v___x_3780_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__1));
v___x_3781_ = l_Lean_MessageData_ofFormat(v___x_3780_);
return v___x_3781_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3(void){
_start:
{
lean_object* v___x_3782_; lean_object* v___x_3783_; 
v___x_3782_ = lean_box(1);
v___x_3783_ = l_Lean_MessageData_ofFormat(v___x_3782_);
return v___x_3783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7(lean_object* v_a_3786_, lean_object* v_a_3787_){
_start:
{
if (lean_obj_tag(v_a_3786_) == 0)
{
lean_object* v___x_3788_; 
v___x_3788_ = l_List_reverse___redArg(v_a_3787_);
return v___x_3788_;
}
else
{
lean_object* v_head_3789_; lean_object* v_tail_3790_; lean_object* v___x_3792_; uint8_t v_isShared_3793_; uint8_t v_isSharedCheck_3820_; 
v_head_3789_ = lean_ctor_get(v_a_3786_, 0);
v_tail_3790_ = lean_ctor_get(v_a_3786_, 1);
v_isSharedCheck_3820_ = !lean_is_exclusive(v_a_3786_);
if (v_isSharedCheck_3820_ == 0)
{
v___x_3792_ = v_a_3786_;
v_isShared_3793_ = v_isSharedCheck_3820_;
goto v_resetjp_3791_;
}
else
{
lean_inc(v_tail_3790_);
lean_inc(v_head_3789_);
lean_dec(v_a_3786_);
v___x_3792_ = lean_box(0);
v_isShared_3793_ = v_isSharedCheck_3820_;
goto v_resetjp_3791_;
}
v_resetjp_3791_:
{
lean_object* v_fst_3794_; lean_object* v_snd_3795_; lean_object* v___x_3797_; uint8_t v_isShared_3798_; uint8_t v_isSharedCheck_3819_; 
v_fst_3794_ = lean_ctor_get(v_head_3789_, 0);
v_snd_3795_ = lean_ctor_get(v_head_3789_, 1);
v_isSharedCheck_3819_ = !lean_is_exclusive(v_head_3789_);
if (v_isSharedCheck_3819_ == 0)
{
v___x_3797_ = v_head_3789_;
v_isShared_3798_ = v_isSharedCheck_3819_;
goto v_resetjp_3796_;
}
else
{
lean_inc(v_snd_3795_);
lean_inc(v_fst_3794_);
lean_dec(v_head_3789_);
v___x_3797_ = lean_box(0);
v_isShared_3798_ = v_isSharedCheck_3819_;
goto v_resetjp_3796_;
}
v_resetjp_3796_:
{
lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3802_; 
v___x_3799_ = l_Lean_MessageData_ofExpr(v_fst_3794_);
v___x_3800_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__2);
if (v_isShared_3798_ == 0)
{
lean_ctor_set_tag(v___x_3797_, 7);
lean_ctor_set(v___x_3797_, 1, v___x_3800_);
lean_ctor_set(v___x_3797_, 0, v___x_3799_);
v___x_3802_ = v___x_3797_;
goto v_reusejp_3801_;
}
else
{
lean_object* v_reuseFailAlloc_3818_; 
v_reuseFailAlloc_3818_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3818_, 0, v___x_3799_);
lean_ctor_set(v_reuseFailAlloc_3818_, 1, v___x_3800_);
v___x_3802_ = v_reuseFailAlloc_3818_;
goto v_reusejp_3801_;
}
v_reusejp_3801_:
{
lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___y_3806_; uint8_t v___x_3815_; 
v___x_3803_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__3);
v___x_3804_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3804_, 0, v___x_3802_);
lean_ctor_set(v___x_3804_, 1, v___x_3803_);
v___x_3815_ = lean_unbox(v_snd_3795_);
lean_dec(v_snd_3795_);
if (v___x_3815_ == 0)
{
lean_object* v___x_3816_; 
v___x_3816_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__4));
v___y_3806_ = v___x_3816_;
goto v___jp_3805_;
}
else
{
lean_object* v___x_3817_; 
v___x_3817_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7___closed__5));
v___y_3806_ = v___x_3817_;
goto v___jp_3805_;
}
v___jp_3805_:
{
lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_3812_; 
lean_inc_ref(v___y_3806_);
v___x_3807_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3807_, 0, v___y_3806_);
v___x_3808_ = l_Lean_MessageData_ofFormat(v___x_3807_);
v___x_3809_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3809_, 0, v___x_3804_);
lean_ctor_set(v___x_3809_, 1, v___x_3808_);
v___x_3810_ = l_Lean_MessageData_paren(v___x_3809_);
if (v_isShared_3793_ == 0)
{
lean_ctor_set(v___x_3792_, 1, v_a_3787_);
lean_ctor_set(v___x_3792_, 0, v___x_3810_);
v___x_3812_ = v___x_3792_;
goto v_reusejp_3811_;
}
else
{
lean_object* v_reuseFailAlloc_3814_; 
v_reuseFailAlloc_3814_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3814_, 0, v___x_3810_);
lean_ctor_set(v_reuseFailAlloc_3814_, 1, v_a_3787_);
v___x_3812_ = v_reuseFailAlloc_3814_;
goto v_reusejp_3811_;
}
v_reusejp_3811_:
{
v_a_3786_ = v_tail_3790_;
v_a_3787_ = v___x_3812_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13(lean_object* v_e_3821_){
_start:
{
if (lean_obj_tag(v_e_3821_) == 0)
{
uint8_t v___x_3822_; 
v___x_3822_ = 2;
return v___x_3822_;
}
else
{
uint8_t v___x_3823_; 
v___x_3823_ = 0;
return v___x_3823_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13___boxed(lean_object* v_e_3824_){
_start:
{
uint8_t v_res_3825_; lean_object* v_r_3826_; 
v_res_3825_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13(v_e_3824_);
lean_dec_ref(v_e_3824_);
v_r_3826_ = lean_box(v_res_3825_);
return v_r_3826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(lean_object* v_opts_3827_, lean_object* v_opt_3828_){
_start:
{
lean_object* v_name_3829_; lean_object* v_defValue_3830_; lean_object* v_map_3831_; lean_object* v___x_3832_; 
v_name_3829_ = lean_ctor_get(v_opt_3828_, 0);
v_defValue_3830_ = lean_ctor_get(v_opt_3828_, 1);
v_map_3831_ = lean_ctor_get(v_opts_3827_, 0);
v___x_3832_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_3831_, v_name_3829_);
if (lean_obj_tag(v___x_3832_) == 0)
{
lean_inc(v_defValue_3830_);
return v_defValue_3830_;
}
else
{
lean_object* v_val_3833_; 
v_val_3833_ = lean_ctor_get(v___x_3832_, 0);
lean_inc(v_val_3833_);
lean_dec_ref_known(v___x_3832_, 1);
if (lean_obj_tag(v_val_3833_) == 3)
{
lean_object* v_v_3834_; 
v_v_3834_ = lean_ctor_get(v_val_3833_, 0);
lean_inc(v_v_3834_);
lean_dec_ref_known(v_val_3833_, 1);
return v_v_3834_;
}
else
{
lean_dec(v_val_3833_);
lean_inc(v_defValue_3830_);
return v_defValue_3830_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14___boxed(lean_object* v_opts_3835_, lean_object* v_opt_3836_){
_start:
{
lean_object* v_res_3837_; 
v_res_3837_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(v_opts_3835_, v_opt_3836_);
lean_dec_ref(v_opt_3836_);
lean_dec_ref(v_opts_3835_);
return v_res_3837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12(size_t v_sz_3838_, size_t v_i_3839_, lean_object* v_bs_3840_){
_start:
{
uint8_t v___x_3841_; 
v___x_3841_ = lean_usize_dec_lt(v_i_3839_, v_sz_3838_);
if (v___x_3841_ == 0)
{
return v_bs_3840_;
}
else
{
lean_object* v_v_3842_; lean_object* v_msg_3843_; lean_object* v___x_3844_; lean_object* v_bs_x27_3845_; size_t v___x_3846_; size_t v___x_3847_; lean_object* v___x_3848_; 
v_v_3842_ = lean_array_uget_borrowed(v_bs_3840_, v_i_3839_);
v_msg_3843_ = lean_ctor_get(v_v_3842_, 1);
lean_inc_ref(v_msg_3843_);
v___x_3844_ = lean_unsigned_to_nat(0u);
v_bs_x27_3845_ = lean_array_uset(v_bs_3840_, v_i_3839_, v___x_3844_);
v___x_3846_ = ((size_t)1ULL);
v___x_3847_ = lean_usize_add(v_i_3839_, v___x_3846_);
v___x_3848_ = lean_array_uset(v_bs_x27_3845_, v_i_3839_, v_msg_3843_);
v_i_3839_ = v___x_3847_;
v_bs_3840_ = v___x_3848_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12___boxed(lean_object* v_sz_3850_, lean_object* v_i_3851_, lean_object* v_bs_3852_){
_start:
{
size_t v_sz_boxed_3853_; size_t v_i_boxed_3854_; lean_object* v_res_3855_; 
v_sz_boxed_3853_ = lean_unbox_usize(v_sz_3850_);
lean_dec(v_sz_3850_);
v_i_boxed_3854_ = lean_unbox_usize(v_i_3851_);
lean_dec(v_i_3851_);
v_res_3855_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12(v_sz_boxed_3853_, v_i_boxed_3854_, v_bs_3852_);
return v_res_3855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11(lean_object* v_oldTraces_3856_, lean_object* v_data_3857_, lean_object* v_ref_3858_, lean_object* v_msg_3859_, lean_object* v___y_3860_, lean_object* v___y_3861_, lean_object* v___y_3862_, lean_object* v___y_3863_){
_start:
{
lean_object* v_fileName_3865_; lean_object* v_fileMap_3866_; lean_object* v_options_3867_; lean_object* v_currRecDepth_3868_; lean_object* v_maxRecDepth_3869_; lean_object* v_ref_3870_; lean_object* v_currNamespace_3871_; lean_object* v_openDecls_3872_; lean_object* v_initHeartbeats_3873_; lean_object* v_maxHeartbeats_3874_; lean_object* v_quotContext_3875_; lean_object* v_currMacroScope_3876_; uint8_t v_diag_3877_; lean_object* v_cancelTk_x3f_3878_; uint8_t v_suppressElabErrors_3879_; lean_object* v_inheritedTraceOptions_3880_; lean_object* v___x_3881_; lean_object* v_traceState_3882_; lean_object* v_traces_3883_; lean_object* v_ref_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; size_t v_sz_3887_; size_t v___x_3888_; lean_object* v___x_3889_; lean_object* v_msg_3890_; lean_object* v___x_3891_; lean_object* v_a_3892_; lean_object* v___x_3894_; uint8_t v_isShared_3895_; uint8_t v_isSharedCheck_3929_; 
v_fileName_3865_ = lean_ctor_get(v___y_3862_, 0);
v_fileMap_3866_ = lean_ctor_get(v___y_3862_, 1);
v_options_3867_ = lean_ctor_get(v___y_3862_, 2);
v_currRecDepth_3868_ = lean_ctor_get(v___y_3862_, 3);
v_maxRecDepth_3869_ = lean_ctor_get(v___y_3862_, 4);
v_ref_3870_ = lean_ctor_get(v___y_3862_, 5);
v_currNamespace_3871_ = lean_ctor_get(v___y_3862_, 6);
v_openDecls_3872_ = lean_ctor_get(v___y_3862_, 7);
v_initHeartbeats_3873_ = lean_ctor_get(v___y_3862_, 8);
v_maxHeartbeats_3874_ = lean_ctor_get(v___y_3862_, 9);
v_quotContext_3875_ = lean_ctor_get(v___y_3862_, 10);
v_currMacroScope_3876_ = lean_ctor_get(v___y_3862_, 11);
v_diag_3877_ = lean_ctor_get_uint8(v___y_3862_, sizeof(void*)*14);
v_cancelTk_x3f_3878_ = lean_ctor_get(v___y_3862_, 12);
v_suppressElabErrors_3879_ = lean_ctor_get_uint8(v___y_3862_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3880_ = lean_ctor_get(v___y_3862_, 13);
v___x_3881_ = lean_st_ref_get(v___y_3863_);
v_traceState_3882_ = lean_ctor_get(v___x_3881_, 4);
lean_inc_ref(v_traceState_3882_);
lean_dec(v___x_3881_);
v_traces_3883_ = lean_ctor_get(v_traceState_3882_, 0);
lean_inc_ref(v_traces_3883_);
lean_dec_ref(v_traceState_3882_);
v_ref_3884_ = l_Lean_replaceRef(v_ref_3858_, v_ref_3870_);
lean_inc_ref(v_inheritedTraceOptions_3880_);
lean_inc(v_cancelTk_x3f_3878_);
lean_inc(v_currMacroScope_3876_);
lean_inc(v_quotContext_3875_);
lean_inc(v_maxHeartbeats_3874_);
lean_inc(v_initHeartbeats_3873_);
lean_inc(v_openDecls_3872_);
lean_inc(v_currNamespace_3871_);
lean_inc(v_maxRecDepth_3869_);
lean_inc(v_currRecDepth_3868_);
lean_inc_ref(v_options_3867_);
lean_inc_ref(v_fileMap_3866_);
lean_inc_ref(v_fileName_3865_);
v___x_3885_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3885_, 0, v_fileName_3865_);
lean_ctor_set(v___x_3885_, 1, v_fileMap_3866_);
lean_ctor_set(v___x_3885_, 2, v_options_3867_);
lean_ctor_set(v___x_3885_, 3, v_currRecDepth_3868_);
lean_ctor_set(v___x_3885_, 4, v_maxRecDepth_3869_);
lean_ctor_set(v___x_3885_, 5, v_ref_3884_);
lean_ctor_set(v___x_3885_, 6, v_currNamespace_3871_);
lean_ctor_set(v___x_3885_, 7, v_openDecls_3872_);
lean_ctor_set(v___x_3885_, 8, v_initHeartbeats_3873_);
lean_ctor_set(v___x_3885_, 9, v_maxHeartbeats_3874_);
lean_ctor_set(v___x_3885_, 10, v_quotContext_3875_);
lean_ctor_set(v___x_3885_, 11, v_currMacroScope_3876_);
lean_ctor_set(v___x_3885_, 12, v_cancelTk_x3f_3878_);
lean_ctor_set(v___x_3885_, 13, v_inheritedTraceOptions_3880_);
lean_ctor_set_uint8(v___x_3885_, sizeof(void*)*14, v_diag_3877_);
lean_ctor_set_uint8(v___x_3885_, sizeof(void*)*14 + 1, v_suppressElabErrors_3879_);
v___x_3886_ = l_Lean_PersistentArray_toArray___redArg(v_traces_3883_);
lean_dec_ref(v_traces_3883_);
v_sz_3887_ = lean_array_size(v___x_3886_);
v___x_3888_ = ((size_t)0ULL);
v___x_3889_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11_spec__12(v_sz_3887_, v___x_3888_, v___x_3886_);
v_msg_3890_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_3890_, 0, v_data_3857_);
lean_ctor_set(v_msg_3890_, 1, v_msg_3859_);
lean_ctor_set(v_msg_3890_, 2, v___x_3889_);
v___x_3891_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0_spec__0(v_msg_3890_, v___y_3860_, v___y_3861_, v___x_3885_, v___y_3863_);
lean_dec_ref_known(v___x_3885_, 14);
v_a_3892_ = lean_ctor_get(v___x_3891_, 0);
v_isSharedCheck_3929_ = !lean_is_exclusive(v___x_3891_);
if (v_isSharedCheck_3929_ == 0)
{
v___x_3894_ = v___x_3891_;
v_isShared_3895_ = v_isSharedCheck_3929_;
goto v_resetjp_3893_;
}
else
{
lean_inc(v_a_3892_);
lean_dec(v___x_3891_);
v___x_3894_ = lean_box(0);
v_isShared_3895_ = v_isSharedCheck_3929_;
goto v_resetjp_3893_;
}
v_resetjp_3893_:
{
lean_object* v___x_3896_; lean_object* v_traceState_3897_; lean_object* v_env_3898_; lean_object* v_nextMacroScope_3899_; lean_object* v_ngen_3900_; lean_object* v_auxDeclNGen_3901_; lean_object* v_cache_3902_; lean_object* v_messages_3903_; lean_object* v_infoState_3904_; lean_object* v_snapshotTasks_3905_; lean_object* v___x_3907_; uint8_t v_isShared_3908_; uint8_t v_isSharedCheck_3928_; 
v___x_3896_ = lean_st_ref_take(v___y_3863_);
v_traceState_3897_ = lean_ctor_get(v___x_3896_, 4);
v_env_3898_ = lean_ctor_get(v___x_3896_, 0);
v_nextMacroScope_3899_ = lean_ctor_get(v___x_3896_, 1);
v_ngen_3900_ = lean_ctor_get(v___x_3896_, 2);
v_auxDeclNGen_3901_ = lean_ctor_get(v___x_3896_, 3);
v_cache_3902_ = lean_ctor_get(v___x_3896_, 5);
v_messages_3903_ = lean_ctor_get(v___x_3896_, 6);
v_infoState_3904_ = lean_ctor_get(v___x_3896_, 7);
v_snapshotTasks_3905_ = lean_ctor_get(v___x_3896_, 8);
v_isSharedCheck_3928_ = !lean_is_exclusive(v___x_3896_);
if (v_isSharedCheck_3928_ == 0)
{
v___x_3907_ = v___x_3896_;
v_isShared_3908_ = v_isSharedCheck_3928_;
goto v_resetjp_3906_;
}
else
{
lean_inc(v_snapshotTasks_3905_);
lean_inc(v_infoState_3904_);
lean_inc(v_messages_3903_);
lean_inc(v_cache_3902_);
lean_inc(v_traceState_3897_);
lean_inc(v_auxDeclNGen_3901_);
lean_inc(v_ngen_3900_);
lean_inc(v_nextMacroScope_3899_);
lean_inc(v_env_3898_);
lean_dec(v___x_3896_);
v___x_3907_ = lean_box(0);
v_isShared_3908_ = v_isSharedCheck_3928_;
goto v_resetjp_3906_;
}
v_resetjp_3906_:
{
uint64_t v_tid_3909_; lean_object* v___x_3911_; uint8_t v_isShared_3912_; uint8_t v_isSharedCheck_3926_; 
v_tid_3909_ = lean_ctor_get_uint64(v_traceState_3897_, sizeof(void*)*1);
v_isSharedCheck_3926_ = !lean_is_exclusive(v_traceState_3897_);
if (v_isSharedCheck_3926_ == 0)
{
lean_object* v_unused_3927_; 
v_unused_3927_ = lean_ctor_get(v_traceState_3897_, 0);
lean_dec(v_unused_3927_);
v___x_3911_ = v_traceState_3897_;
v_isShared_3912_ = v_isSharedCheck_3926_;
goto v_resetjp_3910_;
}
else
{
lean_dec(v_traceState_3897_);
v___x_3911_ = lean_box(0);
v_isShared_3912_ = v_isSharedCheck_3926_;
goto v_resetjp_3910_;
}
v_resetjp_3910_:
{
lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3916_; 
v___x_3913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3913_, 0, v_ref_3858_);
lean_ctor_set(v___x_3913_, 1, v_a_3892_);
v___x_3914_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_3856_, v___x_3913_);
if (v_isShared_3912_ == 0)
{
lean_ctor_set(v___x_3911_, 0, v___x_3914_);
v___x_3916_ = v___x_3911_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_3925_; 
v_reuseFailAlloc_3925_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3925_, 0, v___x_3914_);
lean_ctor_set_uint64(v_reuseFailAlloc_3925_, sizeof(void*)*1, v_tid_3909_);
v___x_3916_ = v_reuseFailAlloc_3925_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
lean_object* v___x_3918_; 
if (v_isShared_3908_ == 0)
{
lean_ctor_set(v___x_3907_, 4, v___x_3916_);
v___x_3918_ = v___x_3907_;
goto v_reusejp_3917_;
}
else
{
lean_object* v_reuseFailAlloc_3924_; 
v_reuseFailAlloc_3924_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3924_, 0, v_env_3898_);
lean_ctor_set(v_reuseFailAlloc_3924_, 1, v_nextMacroScope_3899_);
lean_ctor_set(v_reuseFailAlloc_3924_, 2, v_ngen_3900_);
lean_ctor_set(v_reuseFailAlloc_3924_, 3, v_auxDeclNGen_3901_);
lean_ctor_set(v_reuseFailAlloc_3924_, 4, v___x_3916_);
lean_ctor_set(v_reuseFailAlloc_3924_, 5, v_cache_3902_);
lean_ctor_set(v_reuseFailAlloc_3924_, 6, v_messages_3903_);
lean_ctor_set(v_reuseFailAlloc_3924_, 7, v_infoState_3904_);
lean_ctor_set(v_reuseFailAlloc_3924_, 8, v_snapshotTasks_3905_);
v___x_3918_ = v_reuseFailAlloc_3924_;
goto v_reusejp_3917_;
}
v_reusejp_3917_:
{
lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3922_; 
v___x_3919_ = lean_st_ref_set(v___y_3863_, v___x_3918_);
v___x_3920_ = lean_box(0);
if (v_isShared_3895_ == 0)
{
lean_ctor_set(v___x_3894_, 0, v___x_3920_);
v___x_3922_ = v___x_3894_;
goto v_reusejp_3921_;
}
else
{
lean_object* v_reuseFailAlloc_3923_; 
v_reuseFailAlloc_3923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3923_, 0, v___x_3920_);
v___x_3922_ = v_reuseFailAlloc_3923_;
goto v_reusejp_3921_;
}
v_reusejp_3921_:
{
return v___x_3922_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11___boxed(lean_object* v_oldTraces_3930_, lean_object* v_data_3931_, lean_object* v_ref_3932_, lean_object* v_msg_3933_, lean_object* v___y_3934_, lean_object* v___y_3935_, lean_object* v___y_3936_, lean_object* v___y_3937_, lean_object* v___y_3938_){
_start:
{
lean_object* v_res_3939_; 
v_res_3939_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11(v_oldTraces_3930_, v_data_3931_, v_ref_3932_, v_msg_3933_, v___y_3934_, v___y_3935_, v___y_3936_, v___y_3937_);
lean_dec(v___y_3937_);
lean_dec_ref(v___y_3936_);
lean_dec(v___y_3935_);
lean_dec_ref(v___y_3934_);
return v_res_3939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(lean_object* v_x_3940_){
_start:
{
if (lean_obj_tag(v_x_3940_) == 0)
{
lean_object* v_a_3942_; lean_object* v___x_3944_; uint8_t v_isShared_3945_; uint8_t v_isSharedCheck_3949_; 
v_a_3942_ = lean_ctor_get(v_x_3940_, 0);
v_isSharedCheck_3949_ = !lean_is_exclusive(v_x_3940_);
if (v_isSharedCheck_3949_ == 0)
{
v___x_3944_ = v_x_3940_;
v_isShared_3945_ = v_isSharedCheck_3949_;
goto v_resetjp_3943_;
}
else
{
lean_inc(v_a_3942_);
lean_dec(v_x_3940_);
v___x_3944_ = lean_box(0);
v_isShared_3945_ = v_isSharedCheck_3949_;
goto v_resetjp_3943_;
}
v_resetjp_3943_:
{
lean_object* v___x_3947_; 
if (v_isShared_3945_ == 0)
{
lean_ctor_set_tag(v___x_3944_, 1);
v___x_3947_ = v___x_3944_;
goto v_reusejp_3946_;
}
else
{
lean_object* v_reuseFailAlloc_3948_; 
v_reuseFailAlloc_3948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3948_, 0, v_a_3942_);
v___x_3947_ = v_reuseFailAlloc_3948_;
goto v_reusejp_3946_;
}
v_reusejp_3946_:
{
return v___x_3947_;
}
}
}
else
{
lean_object* v_a_3950_; lean_object* v___x_3952_; uint8_t v_isShared_3953_; uint8_t v_isSharedCheck_3957_; 
v_a_3950_ = lean_ctor_get(v_x_3940_, 0);
v_isSharedCheck_3957_ = !lean_is_exclusive(v_x_3940_);
if (v_isSharedCheck_3957_ == 0)
{
v___x_3952_ = v_x_3940_;
v_isShared_3953_ = v_isSharedCheck_3957_;
goto v_resetjp_3951_;
}
else
{
lean_inc(v_a_3950_);
lean_dec(v_x_3940_);
v___x_3952_ = lean_box(0);
v_isShared_3953_ = v_isSharedCheck_3957_;
goto v_resetjp_3951_;
}
v_resetjp_3951_:
{
lean_object* v___x_3955_; 
if (v_isShared_3953_ == 0)
{
lean_ctor_set_tag(v___x_3952_, 0);
v___x_3955_ = v___x_3952_;
goto v_reusejp_3954_;
}
else
{
lean_object* v_reuseFailAlloc_3956_; 
v_reuseFailAlloc_3956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3956_, 0, v_a_3950_);
v___x_3955_ = v_reuseFailAlloc_3956_;
goto v_reusejp_3954_;
}
v_reusejp_3954_:
{
return v___x_3955_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg___boxed(lean_object* v_x_3958_, lean_object* v___y_3959_){
_start:
{
lean_object* v_res_3960_; 
v_res_3960_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_x_3958_);
return v_res_3960_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1(void){
_start:
{
lean_object* v___x_3962_; lean_object* v___x_3963_; 
v___x_3962_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__0));
v___x_3963_ = l_Lean_stringToMessageData(v___x_3962_);
return v___x_3963_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2(void){
_start:
{
lean_object* v___x_3964_; double v___x_3965_; 
v___x_3964_ = lean_unsigned_to_nat(1000u);
v___x_3965_ = lean_float_of_nat(v___x_3964_);
return v___x_3965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(lean_object* v_cls_3966_, uint8_t v_collapsed_3967_, lean_object* v_tag_3968_, lean_object* v_opts_3969_, uint8_t v_clsEnabled_3970_, lean_object* v_oldTraces_3971_, lean_object* v_msg_3972_, lean_object* v_resStartStop_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_){
_start:
{
lean_object* v_fst_3979_; lean_object* v_snd_3980_; lean_object* v___y_3982_; lean_object* v___y_3983_; lean_object* v_data_3984_; lean_object* v_fst_3995_; lean_object* v_snd_3996_; lean_object* v___x_3997_; uint8_t v___x_3998_; lean_object* v___y_4000_; lean_object* v_a_4001_; uint8_t v___y_4016_; double v___y_4047_; 
v_fst_3979_ = lean_ctor_get(v_resStartStop_3973_, 0);
lean_inc(v_fst_3979_);
v_snd_3980_ = lean_ctor_get(v_resStartStop_3973_, 1);
lean_inc(v_snd_3980_);
lean_dec_ref(v_resStartStop_3973_);
v_fst_3995_ = lean_ctor_get(v_snd_3980_, 0);
lean_inc(v_fst_3995_);
v_snd_3996_ = lean_ctor_get(v_snd_3980_, 1);
lean_inc(v_snd_3996_);
lean_dec(v_snd_3980_);
v___x_3997_ = l_Lean_trace_profiler;
v___x_3998_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_opts_3969_, v___x_3997_);
if (v___x_3998_ == 0)
{
v___y_4016_ = v___x_3998_;
goto v___jp_4015_;
}
else
{
lean_object* v___x_4052_; uint8_t v___x_4053_; 
v___x_4052_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4053_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_opts_3969_, v___x_4052_);
if (v___x_4053_ == 0)
{
lean_object* v___x_4054_; lean_object* v___x_4055_; double v___x_4056_; double v___x_4057_; double v___x_4058_; 
v___x_4054_ = l_Lean_trace_profiler_threshold;
v___x_4055_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(v_opts_3969_, v___x_4054_);
v___x_4056_ = lean_float_of_nat(v___x_4055_);
v___x_4057_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2);
v___x_4058_ = lean_float_div(v___x_4056_, v___x_4057_);
v___y_4047_ = v___x_4058_;
goto v___jp_4046_;
}
else
{
lean_object* v___x_4059_; lean_object* v___x_4060_; double v___x_4061_; 
v___x_4059_ = l_Lean_trace_profiler_threshold;
v___x_4060_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(v_opts_3969_, v___x_4059_);
v___x_4061_ = lean_float_of_nat(v___x_4060_);
v___y_4047_ = v___x_4061_;
goto v___jp_4046_;
}
}
v___jp_3981_:
{
lean_object* v___x_3985_; 
lean_inc(v___y_3983_);
v___x_3985_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11(v_oldTraces_3971_, v_data_3984_, v___y_3983_, v___y_3982_, v___y_3974_, v___y_3975_, v___y_3976_, v___y_3977_);
if (lean_obj_tag(v___x_3985_) == 0)
{
lean_object* v___x_3986_; 
lean_dec_ref_known(v___x_3985_, 1);
v___x_3986_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_fst_3979_);
return v___x_3986_;
}
else
{
lean_object* v_a_3987_; lean_object* v___x_3989_; uint8_t v_isShared_3990_; uint8_t v_isSharedCheck_3994_; 
lean_dec(v_fst_3979_);
v_a_3987_ = lean_ctor_get(v___x_3985_, 0);
v_isSharedCheck_3994_ = !lean_is_exclusive(v___x_3985_);
if (v_isSharedCheck_3994_ == 0)
{
v___x_3989_ = v___x_3985_;
v_isShared_3990_ = v_isSharedCheck_3994_;
goto v_resetjp_3988_;
}
else
{
lean_inc(v_a_3987_);
lean_dec(v___x_3985_);
v___x_3989_ = lean_box(0);
v_isShared_3990_ = v_isSharedCheck_3994_;
goto v_resetjp_3988_;
}
v_resetjp_3988_:
{
lean_object* v___x_3992_; 
if (v_isShared_3990_ == 0)
{
v___x_3992_ = v___x_3989_;
goto v_reusejp_3991_;
}
else
{
lean_object* v_reuseFailAlloc_3993_; 
v_reuseFailAlloc_3993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3993_, 0, v_a_3987_);
v___x_3992_ = v_reuseFailAlloc_3993_;
goto v_reusejp_3991_;
}
v_reusejp_3991_:
{
return v___x_3992_;
}
}
}
}
v___jp_3999_:
{
uint8_t v_result_4002_; lean_object* v___x_4003_; lean_object* v___x_4004_; double v___x_4005_; lean_object* v_data_4006_; 
v_result_4002_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__13(v_fst_3979_);
v___x_4003_ = lean_box(v_result_4002_);
v___x_4004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4004_, 0, v___x_4003_);
v___x_4005_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0);
lean_inc_ref(v_tag_3968_);
lean_inc_ref(v___x_4004_);
lean_inc(v_cls_3966_);
v_data_4006_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4006_, 0, v_cls_3966_);
lean_ctor_set(v_data_4006_, 1, v___x_4004_);
lean_ctor_set(v_data_4006_, 2, v_tag_3968_);
lean_ctor_set_float(v_data_4006_, sizeof(void*)*3, v___x_4005_);
lean_ctor_set_float(v_data_4006_, sizeof(void*)*3 + 8, v___x_4005_);
lean_ctor_set_uint8(v_data_4006_, sizeof(void*)*3 + 16, v_collapsed_3967_);
if (v___x_3998_ == 0)
{
lean_dec_ref_known(v___x_4004_, 1);
lean_dec(v_snd_3996_);
lean_dec(v_fst_3995_);
lean_dec_ref(v_tag_3968_);
lean_dec(v_cls_3966_);
v___y_3982_ = v_a_4001_;
v___y_3983_ = v___y_4000_;
v_data_3984_ = v_data_4006_;
goto v___jp_3981_;
}
else
{
lean_object* v_data_4007_; double v___x_4008_; double v___x_4009_; 
lean_dec_ref_known(v_data_4006_, 3);
v_data_4007_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4007_, 0, v_cls_3966_);
lean_ctor_set(v_data_4007_, 1, v___x_4004_);
lean_ctor_set(v_data_4007_, 2, v_tag_3968_);
v___x_4008_ = lean_unbox_float(v_fst_3995_);
lean_dec(v_fst_3995_);
lean_ctor_set_float(v_data_4007_, sizeof(void*)*3, v___x_4008_);
v___x_4009_ = lean_unbox_float(v_snd_3996_);
lean_dec(v_snd_3996_);
lean_ctor_set_float(v_data_4007_, sizeof(void*)*3 + 8, v___x_4009_);
lean_ctor_set_uint8(v_data_4007_, sizeof(void*)*3 + 16, v_collapsed_3967_);
v___y_3982_ = v_a_4001_;
v___y_3983_ = v___y_4000_;
v_data_3984_ = v_data_4007_;
goto v___jp_3981_;
}
}
v___jp_4010_:
{
lean_object* v_ref_4011_; lean_object* v___x_4012_; 
v_ref_4011_ = lean_ctor_get(v___y_3976_, 5);
lean_inc(v___y_3977_);
lean_inc_ref(v___y_3976_);
lean_inc(v___y_3975_);
lean_inc_ref(v___y_3974_);
lean_inc(v_fst_3979_);
v___x_4012_ = lean_apply_6(v_msg_3972_, v_fst_3979_, v___y_3974_, v___y_3975_, v___y_3976_, v___y_3977_, lean_box(0));
if (lean_obj_tag(v___x_4012_) == 0)
{
lean_object* v_a_4013_; 
v_a_4013_ = lean_ctor_get(v___x_4012_, 0);
lean_inc(v_a_4013_);
lean_dec_ref_known(v___x_4012_, 1);
v___y_4000_ = v_ref_4011_;
v_a_4001_ = v_a_4013_;
goto v___jp_3999_;
}
else
{
lean_object* v___x_4014_; 
lean_dec_ref_known(v___x_4012_, 1);
v___x_4014_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1);
v___y_4000_ = v_ref_4011_;
v_a_4001_ = v___x_4014_;
goto v___jp_3999_;
}
}
v___jp_4015_:
{
if (v_clsEnabled_3970_ == 0)
{
if (v___y_4016_ == 0)
{
lean_object* v___x_4017_; lean_object* v_traceState_4018_; lean_object* v_env_4019_; lean_object* v_nextMacroScope_4020_; lean_object* v_ngen_4021_; lean_object* v_auxDeclNGen_4022_; lean_object* v_cache_4023_; lean_object* v_messages_4024_; lean_object* v_infoState_4025_; lean_object* v_snapshotTasks_4026_; lean_object* v___x_4028_; uint8_t v_isShared_4029_; uint8_t v_isSharedCheck_4045_; 
lean_dec(v_snd_3996_);
lean_dec(v_fst_3995_);
lean_dec_ref(v_msg_3972_);
lean_dec_ref(v_tag_3968_);
lean_dec(v_cls_3966_);
v___x_4017_ = lean_st_ref_take(v___y_3977_);
v_traceState_4018_ = lean_ctor_get(v___x_4017_, 4);
v_env_4019_ = lean_ctor_get(v___x_4017_, 0);
v_nextMacroScope_4020_ = lean_ctor_get(v___x_4017_, 1);
v_ngen_4021_ = lean_ctor_get(v___x_4017_, 2);
v_auxDeclNGen_4022_ = lean_ctor_get(v___x_4017_, 3);
v_cache_4023_ = lean_ctor_get(v___x_4017_, 5);
v_messages_4024_ = lean_ctor_get(v___x_4017_, 6);
v_infoState_4025_ = lean_ctor_get(v___x_4017_, 7);
v_snapshotTasks_4026_ = lean_ctor_get(v___x_4017_, 8);
v_isSharedCheck_4045_ = !lean_is_exclusive(v___x_4017_);
if (v_isSharedCheck_4045_ == 0)
{
v___x_4028_ = v___x_4017_;
v_isShared_4029_ = v_isSharedCheck_4045_;
goto v_resetjp_4027_;
}
else
{
lean_inc(v_snapshotTasks_4026_);
lean_inc(v_infoState_4025_);
lean_inc(v_messages_4024_);
lean_inc(v_cache_4023_);
lean_inc(v_traceState_4018_);
lean_inc(v_auxDeclNGen_4022_);
lean_inc(v_ngen_4021_);
lean_inc(v_nextMacroScope_4020_);
lean_inc(v_env_4019_);
lean_dec(v___x_4017_);
v___x_4028_ = lean_box(0);
v_isShared_4029_ = v_isSharedCheck_4045_;
goto v_resetjp_4027_;
}
v_resetjp_4027_:
{
uint64_t v_tid_4030_; lean_object* v_traces_4031_; lean_object* v___x_4033_; uint8_t v_isShared_4034_; uint8_t v_isSharedCheck_4044_; 
v_tid_4030_ = lean_ctor_get_uint64(v_traceState_4018_, sizeof(void*)*1);
v_traces_4031_ = lean_ctor_get(v_traceState_4018_, 0);
v_isSharedCheck_4044_ = !lean_is_exclusive(v_traceState_4018_);
if (v_isSharedCheck_4044_ == 0)
{
v___x_4033_ = v_traceState_4018_;
v_isShared_4034_ = v_isSharedCheck_4044_;
goto v_resetjp_4032_;
}
else
{
lean_inc(v_traces_4031_);
lean_dec(v_traceState_4018_);
v___x_4033_ = lean_box(0);
v_isShared_4034_ = v_isSharedCheck_4044_;
goto v_resetjp_4032_;
}
v_resetjp_4032_:
{
lean_object* v___x_4035_; lean_object* v___x_4037_; 
v___x_4035_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_3971_, v_traces_4031_);
lean_dec_ref(v_traces_4031_);
if (v_isShared_4034_ == 0)
{
lean_ctor_set(v___x_4033_, 0, v___x_4035_);
v___x_4037_ = v___x_4033_;
goto v_reusejp_4036_;
}
else
{
lean_object* v_reuseFailAlloc_4043_; 
v_reuseFailAlloc_4043_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_4043_, 0, v___x_4035_);
lean_ctor_set_uint64(v_reuseFailAlloc_4043_, sizeof(void*)*1, v_tid_4030_);
v___x_4037_ = v_reuseFailAlloc_4043_;
goto v_reusejp_4036_;
}
v_reusejp_4036_:
{
lean_object* v___x_4039_; 
if (v_isShared_4029_ == 0)
{
lean_ctor_set(v___x_4028_, 4, v___x_4037_);
v___x_4039_ = v___x_4028_;
goto v_reusejp_4038_;
}
else
{
lean_object* v_reuseFailAlloc_4042_; 
v_reuseFailAlloc_4042_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4042_, 0, v_env_4019_);
lean_ctor_set(v_reuseFailAlloc_4042_, 1, v_nextMacroScope_4020_);
lean_ctor_set(v_reuseFailAlloc_4042_, 2, v_ngen_4021_);
lean_ctor_set(v_reuseFailAlloc_4042_, 3, v_auxDeclNGen_4022_);
lean_ctor_set(v_reuseFailAlloc_4042_, 4, v___x_4037_);
lean_ctor_set(v_reuseFailAlloc_4042_, 5, v_cache_4023_);
lean_ctor_set(v_reuseFailAlloc_4042_, 6, v_messages_4024_);
lean_ctor_set(v_reuseFailAlloc_4042_, 7, v_infoState_4025_);
lean_ctor_set(v_reuseFailAlloc_4042_, 8, v_snapshotTasks_4026_);
v___x_4039_ = v_reuseFailAlloc_4042_;
goto v_reusejp_4038_;
}
v_reusejp_4038_:
{
lean_object* v___x_4040_; lean_object* v___x_4041_; 
v___x_4040_ = lean_st_ref_set(v___y_3977_, v___x_4039_);
v___x_4041_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_fst_3979_);
return v___x_4041_;
}
}
}
}
}
else
{
goto v___jp_4010_;
}
}
else
{
goto v___jp_4010_;
}
}
v___jp_4046_:
{
double v___x_4048_; double v___x_4049_; double v___x_4050_; uint8_t v___x_4051_; 
v___x_4048_ = lean_unbox_float(v_snd_3996_);
v___x_4049_ = lean_unbox_float(v_fst_3995_);
v___x_4050_ = lean_float_sub(v___x_4048_, v___x_4049_);
v___x_4051_ = lean_float_decLt(v___y_4047_, v___x_4050_);
v___y_4016_ = v___x_4051_;
goto v___jp_4015_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___boxed(lean_object* v_cls_4062_, lean_object* v_collapsed_4063_, lean_object* v_tag_4064_, lean_object* v_opts_4065_, lean_object* v_clsEnabled_4066_, lean_object* v_oldTraces_4067_, lean_object* v_msg_4068_, lean_object* v_resStartStop_4069_, lean_object* v___y_4070_, lean_object* v___y_4071_, lean_object* v___y_4072_, lean_object* v___y_4073_, lean_object* v___y_4074_){
_start:
{
uint8_t v_collapsed_boxed_4075_; uint8_t v_clsEnabled_boxed_4076_; lean_object* v_res_4077_; 
v_collapsed_boxed_4075_ = lean_unbox(v_collapsed_4063_);
v_clsEnabled_boxed_4076_ = lean_unbox(v_clsEnabled_4066_);
v_res_4077_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(v_cls_4062_, v_collapsed_boxed_4075_, v_tag_4064_, v_opts_4065_, v_clsEnabled_boxed_4076_, v_oldTraces_4067_, v_msg_4068_, v_resStartStop_4069_, v___y_4070_, v___y_4071_, v___y_4072_, v___y_4073_);
lean_dec(v___y_4073_);
lean_dec_ref(v___y_4072_);
lean_dec(v___y_4071_);
lean_dec_ref(v___y_4070_);
lean_dec_ref(v_opts_4065_);
return v_res_4077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(lean_object* v_x_4084_, lean_object* v_x_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_){
_start:
{
if (lean_obj_tag(v_x_4084_) == 0)
{
lean_object* v___x_4091_; lean_object* v___x_4092_; 
v___x_4091_ = l_List_reverse___redArg(v_x_4085_);
v___x_4092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4092_, 0, v___x_4091_);
return v___x_4092_;
}
else
{
lean_object* v_head_4093_; lean_object* v_tail_4094_; lean_object* v___x_4096_; uint8_t v_isShared_4097_; uint8_t v_isSharedCheck_4125_; 
v_head_4093_ = lean_ctor_get(v_x_4084_, 0);
v_tail_4094_ = lean_ctor_get(v_x_4084_, 1);
v_isSharedCheck_4125_ = !lean_is_exclusive(v_x_4084_);
if (v_isSharedCheck_4125_ == 0)
{
v___x_4096_ = v_x_4084_;
v_isShared_4097_ = v_isSharedCheck_4125_;
goto v_resetjp_4095_;
}
else
{
lean_inc(v_tail_4094_);
lean_inc(v_head_4093_);
lean_dec(v_x_4084_);
v___x_4096_ = lean_box(0);
v_isShared_4097_ = v_isSharedCheck_4125_;
goto v_resetjp_4095_;
}
v_resetjp_4095_:
{
lean_object* v_fst_4098_; lean_object* v_snd_4099_; lean_object* v___y_4101_; uint8_t v___x_4122_; 
v_fst_4098_ = lean_ctor_get(v_head_4093_, 0);
lean_inc(v_fst_4098_);
v_snd_4099_ = lean_ctor_get(v_head_4093_, 1);
lean_inc(v_snd_4099_);
lean_dec(v_head_4093_);
v___x_4122_ = lean_unbox(v_snd_4099_);
lean_dec(v_snd_4099_);
if (v___x_4122_ == 0)
{
lean_object* v___x_4123_; 
v___x_4123_ = ((lean_object*)(lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__1));
v___y_4101_ = v___x_4123_;
goto v___jp_4100_;
}
else
{
lean_object* v___x_4124_; 
v___x_4124_ = ((lean_object*)(lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___closed__3));
v___y_4101_ = v___x_4124_;
goto v___jp_4100_;
}
v___jp_4100_:
{
lean_object* v___x_4102_; lean_object* v___x_4103_; lean_object* v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; 
v___x_4102_ = lean_unsigned_to_nat(1u);
v___x_4103_ = lean_mk_empty_array_with_capacity(v___x_4102_);
v___x_4104_ = lean_array_push(v___x_4103_, v_fst_4098_);
lean_inc(v___y_4101_);
v___x_4105_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_4105_, 0, v___y_4101_);
lean_closure_set(v___x_4105_, 1, v___x_4104_);
v___x_4106_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__1___redArg(v___x_4105_, v___y_4086_, v___y_4087_, v___y_4088_, v___y_4089_);
if (lean_obj_tag(v___x_4106_) == 0)
{
lean_object* v_a_4107_; 
v_a_4107_ = lean_ctor_get(v___x_4106_, 0);
lean_inc(v_a_4107_);
lean_dec_ref_known(v___x_4106_, 1);
if (lean_obj_tag(v_a_4107_) == 0)
{
lean_del_object(v___x_4096_);
v_x_4084_ = v_tail_4094_;
goto _start;
}
else
{
lean_object* v_val_4109_; lean_object* v___x_4111_; 
v_val_4109_ = lean_ctor_get(v_a_4107_, 0);
lean_inc(v_val_4109_);
lean_dec_ref_known(v_a_4107_, 1);
if (v_isShared_4097_ == 0)
{
lean_ctor_set(v___x_4096_, 1, v_x_4085_);
lean_ctor_set(v___x_4096_, 0, v_val_4109_);
v___x_4111_ = v___x_4096_;
goto v_reusejp_4110_;
}
else
{
lean_object* v_reuseFailAlloc_4113_; 
v_reuseFailAlloc_4113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4113_, 0, v_val_4109_);
lean_ctor_set(v_reuseFailAlloc_4113_, 1, v_x_4085_);
v___x_4111_ = v_reuseFailAlloc_4113_;
goto v_reusejp_4110_;
}
v_reusejp_4110_:
{
v_x_4084_ = v_tail_4094_;
v_x_4085_ = v___x_4111_;
goto _start;
}
}
}
else
{
lean_object* v_a_4114_; lean_object* v___x_4116_; uint8_t v_isShared_4117_; uint8_t v_isSharedCheck_4121_; 
lean_del_object(v___x_4096_);
lean_dec(v_tail_4094_);
lean_dec(v_x_4085_);
v_a_4114_ = lean_ctor_get(v___x_4106_, 0);
v_isSharedCheck_4121_ = !lean_is_exclusive(v___x_4106_);
if (v_isSharedCheck_4121_ == 0)
{
v___x_4116_ = v___x_4106_;
v_isShared_4117_ = v_isSharedCheck_4121_;
goto v_resetjp_4115_;
}
else
{
lean_inc(v_a_4114_);
lean_dec(v___x_4106_);
v___x_4116_ = lean_box(0);
v_isShared_4117_ = v_isSharedCheck_4121_;
goto v_resetjp_4115_;
}
v_resetjp_4115_:
{
lean_object* v___x_4119_; 
if (v_isShared_4117_ == 0)
{
v___x_4119_ = v___x_4116_;
goto v_reusejp_4118_;
}
else
{
lean_object* v_reuseFailAlloc_4120_; 
v_reuseFailAlloc_4120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4120_, 0, v_a_4114_);
v___x_4119_ = v_reuseFailAlloc_4120_;
goto v_reusejp_4118_;
}
v_reusejp_4118_:
{
return v___x_4119_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5___boxed(lean_object* v_x_4126_, lean_object* v_x_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_, lean_object* v___y_4132_){
_start:
{
lean_object* v_res_4133_; 
v_res_4133_ = lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(v_x_4126_, v_x_4127_, v___y_4128_, v___y_4129_, v___y_4130_, v___y_4131_);
lean_dec(v___y_4131_);
lean_dec_ref(v___y_4130_);
lean_dec(v___y_4129_);
lean_dec_ref(v___y_4128_);
return v_res_4133_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0(void){
_start:
{
lean_object* v___x_4134_; lean_object* v___x_4135_; 
v___x_4134_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_compWithZero));
v___x_4135_ = lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(v___x_4134_);
return v___x_4135_;
}
}
static double _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2(void){
_start:
{
lean_object* v___x_4137_; double v___x_4138_; 
v___x_4137_ = lean_unsigned_to_nat(1000000000u);
v___x_4138_ = lean_float_of_nat(v___x_4137_);
return v___x_4138_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4(void){
_start:
{
lean_object* v___x_4140_; lean_object* v___x_4141_; 
v___x_4140_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__3));
v___x_4141_ = l_Lean_stringToMessageData(v___x_4140_);
return v___x_4141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs(lean_object* v_ls_4142_, lean_object* v_a_4143_, lean_object* v_a_4144_, lean_object* v_a_4145_, lean_object* v_a_4146_){
_start:
{
lean_object* v_options_4148_; lean_object* v_inheritedTraceOptions_4149_; uint8_t v_hasTrace_4150_; lean_object* v___f_4151_; uint8_t v___x_4152_; lean_object* v___x_4153_; lean_object* v___x_4154_; lean_object* v___f_4155_; 
v_options_4148_ = lean_ctor_get(v_a_4145_, 2);
v_inheritedTraceOptions_4149_ = lean_ctor_get(v_a_4145_, 13);
v_hasTrace_4150_ = lean_ctor_get_uint8(v_options_4148_, sizeof(void*)*1);
v___f_4151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_natToInt___closed__0));
v___x_4152_ = 2;
v___x_4153_ = lean_box(1);
v___x_4154_ = l_List_reverse___redArg(v_ls_4142_);
v___f_4155_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__1___boxed), 9, 2);
lean_closure_set(v___f_4155_, 0, v___x_4153_);
lean_closure_set(v___f_4155_, 1, v___x_4154_);
if (v_hasTrace_4150_ == 0)
{
lean_object* v___x_4156_; 
v___x_4156_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_4152_, v___f_4155_, v___f_4151_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4156_) == 0)
{
lean_object* v_a_4157_; lean_object* v___x_4158_; lean_object* v___x_4159_; 
v_a_4157_ = lean_ctor_get(v___x_4156_, 0);
lean_inc(v_a_4157_);
lean_dec_ref_known(v___x_4156_, 1);
v___x_4158_ = lean_box(0);
v___x_4159_ = lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(v_a_4157_, v___x_4158_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4159_) == 0)
{
lean_object* v_a_4160_; lean_object* v___x_4161_; lean_object* v_transform_4162_; lean_object* v___x_4163_; 
v_a_4160_ = lean_ctor_get(v___x_4159_, 0);
lean_inc(v_a_4160_);
lean_dec_ref_known(v___x_4159_, 1);
v___x_4161_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4162_ = lean_ctor_get(v___x_4161_, 1);
lean_inc_ref(v_transform_4162_);
lean_inc(v_a_4146_);
lean_inc_ref(v_a_4145_);
lean_inc(v_a_4144_);
lean_inc_ref(v_a_4143_);
v___x_4163_ = lean_apply_6(v_transform_4162_, v_a_4160_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, lean_box(0));
if (lean_obj_tag(v___x_4163_) == 0)
{
lean_object* v_a_4164_; lean_object* v___x_4165_; lean_object* v___x_4166_; 
v_a_4164_ = lean_ctor_get(v___x_4163_, 0);
lean_inc_n(v_a_4164_, 2);
lean_dec_ref_known(v___x_4163_, 1);
v___x_4165_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0));
v___x_4166_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(v___x_4165_, v_a_4164_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4166_) == 0)
{
lean_object* v___x_4168_; uint8_t v_isShared_4169_; uint8_t v_isSharedCheck_4173_; 
v_isSharedCheck_4173_ = !lean_is_exclusive(v___x_4166_);
if (v_isSharedCheck_4173_ == 0)
{
lean_object* v_unused_4174_; 
v_unused_4174_ = lean_ctor_get(v___x_4166_, 0);
lean_dec(v_unused_4174_);
v___x_4168_ = v___x_4166_;
v_isShared_4169_ = v_isSharedCheck_4173_;
goto v_resetjp_4167_;
}
else
{
lean_dec(v___x_4166_);
v___x_4168_ = lean_box(0);
v_isShared_4169_ = v_isSharedCheck_4173_;
goto v_resetjp_4167_;
}
v_resetjp_4167_:
{
lean_object* v___x_4171_; 
if (v_isShared_4169_ == 0)
{
lean_ctor_set(v___x_4168_, 0, v_a_4164_);
v___x_4171_ = v___x_4168_;
goto v_reusejp_4170_;
}
else
{
lean_object* v_reuseFailAlloc_4172_; 
v_reuseFailAlloc_4172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4172_, 0, v_a_4164_);
v___x_4171_ = v_reuseFailAlloc_4172_;
goto v_reusejp_4170_;
}
v_reusejp_4170_:
{
return v___x_4171_;
}
}
}
else
{
lean_object* v_a_4175_; lean_object* v___x_4177_; uint8_t v_isShared_4178_; uint8_t v_isSharedCheck_4182_; 
lean_dec(v_a_4164_);
v_a_4175_ = lean_ctor_get(v___x_4166_, 0);
v_isSharedCheck_4182_ = !lean_is_exclusive(v___x_4166_);
if (v_isSharedCheck_4182_ == 0)
{
v___x_4177_ = v___x_4166_;
v_isShared_4178_ = v_isSharedCheck_4182_;
goto v_resetjp_4176_;
}
else
{
lean_inc(v_a_4175_);
lean_dec(v___x_4166_);
v___x_4177_ = lean_box(0);
v_isShared_4178_ = v_isSharedCheck_4182_;
goto v_resetjp_4176_;
}
v_resetjp_4176_:
{
lean_object* v___x_4180_; 
if (v_isShared_4178_ == 0)
{
v___x_4180_ = v___x_4177_;
goto v_reusejp_4179_;
}
else
{
lean_object* v_reuseFailAlloc_4181_; 
v_reuseFailAlloc_4181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4181_, 0, v_a_4175_);
v___x_4180_ = v_reuseFailAlloc_4181_;
goto v_reusejp_4179_;
}
v_reusejp_4179_:
{
return v___x_4180_;
}
}
}
}
else
{
return v___x_4163_;
}
}
else
{
return v___x_4159_;
}
}
else
{
lean_object* v_a_4183_; lean_object* v___x_4185_; uint8_t v_isShared_4186_; uint8_t v_isSharedCheck_4190_; 
v_a_4183_ = lean_ctor_get(v___x_4156_, 0);
v_isSharedCheck_4190_ = !lean_is_exclusive(v___x_4156_);
if (v_isSharedCheck_4190_ == 0)
{
v___x_4185_ = v___x_4156_;
v_isShared_4186_ = v_isSharedCheck_4190_;
goto v_resetjp_4184_;
}
else
{
lean_inc(v_a_4183_);
lean_dec(v___x_4156_);
v___x_4185_ = lean_box(0);
v_isShared_4186_ = v_isSharedCheck_4190_;
goto v_resetjp_4184_;
}
v_resetjp_4184_:
{
lean_object* v___x_4188_; 
if (v_isShared_4186_ == 0)
{
v___x_4188_ = v___x_4185_;
goto v_reusejp_4187_;
}
else
{
lean_object* v_reuseFailAlloc_4189_; 
v_reuseFailAlloc_4189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4189_, 0, v_a_4183_);
v___x_4188_ = v_reuseFailAlloc_4189_;
goto v_reusejp_4187_;
}
v_reusejp_4187_:
{
return v___x_4188_;
}
}
}
}
else
{
lean_object* v___f_4191_; lean_object* v_cls_4192_; lean_object* v___x_4193_; lean_object* v___x_4194_; uint8_t v___x_4195_; lean_object* v___y_4197_; lean_object* v___y_4198_; lean_object* v_a_4199_; lean_object* v___y_4209_; lean_object* v___y_4210_; lean_object* v_a_4211_; lean_object* v___y_4214_; lean_object* v___y_4215_; lean_object* v___y_4216_; lean_object* v___y_4227_; lean_object* v___y_4228_; lean_object* v_a_4229_; lean_object* v___y_4242_; lean_object* v___y_4243_; lean_object* v_a_4244_; lean_object* v___y_4247_; lean_object* v___y_4248_; lean_object* v___y_4249_; 
v___f_4191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__1));
v_cls_4192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_4193_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1));
v___x_4194_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6);
v___x_4195_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4149_, v_options_4148_, v___x_4194_);
if (v___x_4195_ == 0)
{
lean_object* v___x_4312_; uint8_t v___x_4313_; 
v___x_4312_ = l_Lean_trace_profiler;
v___x_4313_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_4148_, v___x_4312_);
if (v___x_4313_ == 0)
{
lean_object* v___x_4314_; 
v___x_4314_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_4152_, v___f_4155_, v___f_4151_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4314_) == 0)
{
lean_object* v_a_4315_; lean_object* v___x_4316_; lean_object* v___x_4317_; 
v_a_4315_ = lean_ctor_get(v___x_4314_, 0);
lean_inc_n(v_a_4315_, 2);
lean_dec_ref_known(v___x_4314_, 1);
v___x_4316_ = lean_box(0);
v___x_4317_ = lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(v_a_4315_, v___x_4316_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4317_) == 0)
{
lean_object* v_a_4318_; lean_object* v___x_4319_; lean_object* v_transform_4320_; lean_object* v___x_4321_; 
v_a_4318_ = lean_ctor_get(v___x_4317_, 0);
lean_inc(v_a_4318_);
lean_dec_ref_known(v___x_4317_, 1);
v___x_4319_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4320_ = lean_ctor_get(v___x_4319_, 1);
lean_inc_ref(v_transform_4320_);
lean_inc(v_a_4146_);
lean_inc_ref(v_a_4145_);
lean_inc(v_a_4144_);
lean_inc_ref(v_a_4143_);
v___x_4321_ = lean_apply_6(v_transform_4320_, v_a_4318_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, lean_box(0));
if (lean_obj_tag(v___x_4321_) == 0)
{
lean_object* v_a_4322_; lean_object* v___y_4324_; lean_object* v___y_4325_; lean_object* v___y_4326_; lean_object* v___y_4327_; 
v_a_4322_ = lean_ctor_get(v___x_4321_, 0);
lean_inc(v_a_4322_);
lean_dec_ref_known(v___x_4321_, 1);
if (v___x_4195_ == 0)
{
lean_dec(v_a_4315_);
v___y_4324_ = v_a_4143_;
v___y_4325_ = v_a_4144_;
v___y_4326_ = v_a_4145_;
v___y_4327_ = v_a_4146_;
goto v___jp_4323_;
}
else
{
lean_object* v___x_4346_; lean_object* v___x_4347_; lean_object* v___x_4348_; lean_object* v___x_4349_; lean_object* v___x_4350_; lean_object* v___x_4351_; 
v___x_4346_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4);
v___x_4347_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7(v_a_4315_, v___x_4316_);
v___x_4348_ = l_Lean_MessageData_ofList(v___x_4347_);
v___x_4349_ = l_Lean_indentD(v___x_4348_);
v___x_4350_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4350_, 0, v___x_4346_);
lean_ctor_set(v___x_4350_, 1, v___x_4349_);
v___x_4351_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(v_cls_4192_, v___x_4350_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4351_) == 0)
{
lean_dec_ref_known(v___x_4351_, 1);
v___y_4324_ = v_a_4143_;
v___y_4325_ = v_a_4144_;
v___y_4326_ = v_a_4145_;
v___y_4327_ = v_a_4146_;
goto v___jp_4323_;
}
else
{
lean_object* v_a_4352_; lean_object* v___x_4354_; uint8_t v_isShared_4355_; uint8_t v_isSharedCheck_4359_; 
lean_dec(v_a_4322_);
v_a_4352_ = lean_ctor_get(v___x_4351_, 0);
v_isSharedCheck_4359_ = !lean_is_exclusive(v___x_4351_);
if (v_isSharedCheck_4359_ == 0)
{
v___x_4354_ = v___x_4351_;
v_isShared_4355_ = v_isSharedCheck_4359_;
goto v_resetjp_4353_;
}
else
{
lean_inc(v_a_4352_);
lean_dec(v___x_4351_);
v___x_4354_ = lean_box(0);
v_isShared_4355_ = v_isSharedCheck_4359_;
goto v_resetjp_4353_;
}
v_resetjp_4353_:
{
lean_object* v___x_4357_; 
if (v_isShared_4355_ == 0)
{
v___x_4357_ = v___x_4354_;
goto v_reusejp_4356_;
}
else
{
lean_object* v_reuseFailAlloc_4358_; 
v_reuseFailAlloc_4358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4358_, 0, v_a_4352_);
v___x_4357_ = v_reuseFailAlloc_4358_;
goto v_reusejp_4356_;
}
v_reusejp_4356_:
{
return v___x_4357_;
}
}
}
}
v___jp_4323_:
{
lean_object* v___x_4328_; lean_object* v___x_4329_; 
v___x_4328_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2___closed__0));
lean_inc(v_a_4322_);
v___x_4329_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__6(v___x_4328_, v_a_4322_, v___y_4324_, v___y_4325_, v___y_4326_, v___y_4327_);
if (lean_obj_tag(v___x_4329_) == 0)
{
lean_object* v___x_4331_; uint8_t v_isShared_4332_; uint8_t v_isSharedCheck_4336_; 
v_isSharedCheck_4336_ = !lean_is_exclusive(v___x_4329_);
if (v_isSharedCheck_4336_ == 0)
{
lean_object* v_unused_4337_; 
v_unused_4337_ = lean_ctor_get(v___x_4329_, 0);
lean_dec(v_unused_4337_);
v___x_4331_ = v___x_4329_;
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
else
{
lean_dec(v___x_4329_);
v___x_4331_ = lean_box(0);
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
v_resetjp_4330_:
{
lean_object* v___x_4334_; 
if (v_isShared_4332_ == 0)
{
lean_ctor_set(v___x_4331_, 0, v_a_4322_);
v___x_4334_ = v___x_4331_;
goto v_reusejp_4333_;
}
else
{
lean_object* v_reuseFailAlloc_4335_; 
v_reuseFailAlloc_4335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4335_, 0, v_a_4322_);
v___x_4334_ = v_reuseFailAlloc_4335_;
goto v_reusejp_4333_;
}
v_reusejp_4333_:
{
return v___x_4334_;
}
}
}
else
{
lean_object* v_a_4338_; lean_object* v___x_4340_; uint8_t v_isShared_4341_; uint8_t v_isSharedCheck_4345_; 
lean_dec(v_a_4322_);
v_a_4338_ = lean_ctor_get(v___x_4329_, 0);
v_isSharedCheck_4345_ = !lean_is_exclusive(v___x_4329_);
if (v_isSharedCheck_4345_ == 0)
{
v___x_4340_ = v___x_4329_;
v_isShared_4341_ = v_isSharedCheck_4345_;
goto v_resetjp_4339_;
}
else
{
lean_inc(v_a_4338_);
lean_dec(v___x_4329_);
v___x_4340_ = lean_box(0);
v_isShared_4341_ = v_isSharedCheck_4345_;
goto v_resetjp_4339_;
}
v_resetjp_4339_:
{
lean_object* v___x_4343_; 
if (v_isShared_4341_ == 0)
{
v___x_4343_ = v___x_4340_;
goto v_reusejp_4342_;
}
else
{
lean_object* v_reuseFailAlloc_4344_; 
v_reuseFailAlloc_4344_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4344_, 0, v_a_4338_);
v___x_4343_ = v_reuseFailAlloc_4344_;
goto v_reusejp_4342_;
}
v_reusejp_4342_:
{
return v___x_4343_;
}
}
}
}
}
else
{
lean_dec(v_a_4315_);
return v___x_4321_;
}
}
else
{
lean_dec(v_a_4315_);
return v___x_4317_;
}
}
else
{
lean_object* v_a_4360_; lean_object* v___x_4362_; uint8_t v_isShared_4363_; uint8_t v_isSharedCheck_4367_; 
v_a_4360_ = lean_ctor_get(v___x_4314_, 0);
v_isSharedCheck_4367_ = !lean_is_exclusive(v___x_4314_);
if (v_isSharedCheck_4367_ == 0)
{
v___x_4362_ = v___x_4314_;
v_isShared_4363_ = v_isSharedCheck_4367_;
goto v_resetjp_4361_;
}
else
{
lean_inc(v_a_4360_);
lean_dec(v___x_4314_);
v___x_4362_ = lean_box(0);
v_isShared_4363_ = v_isSharedCheck_4367_;
goto v_resetjp_4361_;
}
v_resetjp_4361_:
{
lean_object* v___x_4365_; 
if (v_isShared_4363_ == 0)
{
v___x_4365_ = v___x_4362_;
goto v_reusejp_4364_;
}
else
{
lean_object* v_reuseFailAlloc_4366_; 
v_reuseFailAlloc_4366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4366_, 0, v_a_4360_);
v___x_4365_ = v_reuseFailAlloc_4366_;
goto v_reusejp_4364_;
}
v_reusejp_4364_:
{
return v___x_4365_;
}
}
}
}
else
{
goto v___jp_4259_;
}
}
else
{
goto v___jp_4259_;
}
v___jp_4196_:
{
lean_object* v___x_4200_; double v___x_4201_; double v___x_4202_; lean_object* v___x_4203_; lean_object* v___x_4204_; lean_object* v___x_4205_; lean_object* v___x_4206_; lean_object* v___x_4207_; 
v___x_4200_ = lean_io_get_num_heartbeats();
v___x_4201_ = lean_float_of_nat(v___y_4198_);
v___x_4202_ = lean_float_of_nat(v___x_4200_);
v___x_4203_ = lean_box_float(v___x_4201_);
v___x_4204_ = lean_box_float(v___x_4202_);
v___x_4205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4205_, 0, v___x_4203_);
lean_ctor_set(v___x_4205_, 1, v___x_4204_);
v___x_4206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4206_, 0, v_a_4199_);
lean_ctor_set(v___x_4206_, 1, v___x_4205_);
v___x_4207_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(v_cls_4192_, v_hasTrace_4150_, v___x_4193_, v_options_4148_, v___x_4195_, v___y_4197_, v___f_4191_, v___x_4206_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
return v___x_4207_;
}
v___jp_4208_:
{
lean_object* v___x_4212_; 
v___x_4212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4212_, 0, v_a_4211_);
v___y_4197_ = v___y_4210_;
v___y_4198_ = v___y_4209_;
v_a_4199_ = v___x_4212_;
goto v___jp_4196_;
}
v___jp_4213_:
{
if (lean_obj_tag(v___y_4216_) == 0)
{
lean_object* v_a_4217_; lean_object* v___x_4219_; uint8_t v_isShared_4220_; uint8_t v_isSharedCheck_4224_; 
v_a_4217_ = lean_ctor_get(v___y_4216_, 0);
v_isSharedCheck_4224_ = !lean_is_exclusive(v___y_4216_);
if (v_isSharedCheck_4224_ == 0)
{
v___x_4219_ = v___y_4216_;
v_isShared_4220_ = v_isSharedCheck_4224_;
goto v_resetjp_4218_;
}
else
{
lean_inc(v_a_4217_);
lean_dec(v___y_4216_);
v___x_4219_ = lean_box(0);
v_isShared_4220_ = v_isSharedCheck_4224_;
goto v_resetjp_4218_;
}
v_resetjp_4218_:
{
lean_object* v___x_4222_; 
if (v_isShared_4220_ == 0)
{
lean_ctor_set_tag(v___x_4219_, 1);
v___x_4222_ = v___x_4219_;
goto v_reusejp_4221_;
}
else
{
lean_object* v_reuseFailAlloc_4223_; 
v_reuseFailAlloc_4223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4223_, 0, v_a_4217_);
v___x_4222_ = v_reuseFailAlloc_4223_;
goto v_reusejp_4221_;
}
v_reusejp_4221_:
{
v___y_4197_ = v___y_4215_;
v___y_4198_ = v___y_4214_;
v_a_4199_ = v___x_4222_;
goto v___jp_4196_;
}
}
}
else
{
lean_object* v_a_4225_; 
v_a_4225_ = lean_ctor_get(v___y_4216_, 0);
lean_inc(v_a_4225_);
lean_dec_ref_known(v___y_4216_, 1);
v___y_4209_ = v___y_4214_;
v___y_4210_ = v___y_4215_;
v_a_4211_ = v_a_4225_;
goto v___jp_4208_;
}
}
v___jp_4226_:
{
lean_object* v___x_4230_; double v___x_4231_; double v___x_4232_; double v___x_4233_; double v___x_4234_; double v___x_4235_; lean_object* v___x_4236_; lean_object* v___x_4237_; lean_object* v___x_4238_; lean_object* v___x_4239_; lean_object* v___x_4240_; 
v___x_4230_ = lean_io_mono_nanos_now();
v___x_4231_ = lean_float_of_nat(v___y_4227_);
v___x_4232_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2);
v___x_4233_ = lean_float_div(v___x_4231_, v___x_4232_);
v___x_4234_ = lean_float_of_nat(v___x_4230_);
v___x_4235_ = lean_float_div(v___x_4234_, v___x_4232_);
v___x_4236_ = lean_box_float(v___x_4233_);
v___x_4237_ = lean_box_float(v___x_4235_);
v___x_4238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4238_, 0, v___x_4236_);
lean_ctor_set(v___x_4238_, 1, v___x_4237_);
v___x_4239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4239_, 0, v_a_4229_);
lean_ctor_set(v___x_4239_, 1, v___x_4238_);
v___x_4240_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(v_cls_4192_, v_hasTrace_4150_, v___x_4193_, v_options_4148_, v___x_4195_, v___y_4228_, v___f_4191_, v___x_4239_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
return v___x_4240_;
}
v___jp_4241_:
{
lean_object* v___x_4245_; 
v___x_4245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4245_, 0, v_a_4244_);
v___y_4227_ = v___y_4242_;
v___y_4228_ = v___y_4243_;
v_a_4229_ = v___x_4245_;
goto v___jp_4226_;
}
v___jp_4246_:
{
if (lean_obj_tag(v___y_4249_) == 0)
{
lean_object* v_a_4250_; lean_object* v___x_4252_; uint8_t v_isShared_4253_; uint8_t v_isSharedCheck_4257_; 
v_a_4250_ = lean_ctor_get(v___y_4249_, 0);
v_isSharedCheck_4257_ = !lean_is_exclusive(v___y_4249_);
if (v_isSharedCheck_4257_ == 0)
{
v___x_4252_ = v___y_4249_;
v_isShared_4253_ = v_isSharedCheck_4257_;
goto v_resetjp_4251_;
}
else
{
lean_inc(v_a_4250_);
lean_dec(v___y_4249_);
v___x_4252_ = lean_box(0);
v_isShared_4253_ = v_isSharedCheck_4257_;
goto v_resetjp_4251_;
}
v_resetjp_4251_:
{
lean_object* v___x_4255_; 
if (v_isShared_4253_ == 0)
{
lean_ctor_set_tag(v___x_4252_, 1);
v___x_4255_ = v___x_4252_;
goto v_reusejp_4254_;
}
else
{
lean_object* v_reuseFailAlloc_4256_; 
v_reuseFailAlloc_4256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4256_, 0, v_a_4250_);
v___x_4255_ = v_reuseFailAlloc_4256_;
goto v_reusejp_4254_;
}
v_reusejp_4254_:
{
v___y_4227_ = v___y_4247_;
v___y_4228_ = v___y_4248_;
v_a_4229_ = v___x_4255_;
goto v___jp_4226_;
}
}
}
else
{
lean_object* v_a_4258_; 
v_a_4258_ = lean_ctor_get(v___y_4249_, 0);
lean_inc(v_a_4258_);
lean_dec_ref_known(v___y_4249_, 1);
v___y_4242_ = v___y_4247_;
v___y_4243_ = v___y_4248_;
v_a_4244_ = v_a_4258_;
goto v___jp_4241_;
}
}
v___jp_4259_:
{
lean_object* v___x_4260_; lean_object* v_a_4261_; lean_object* v___x_4262_; uint8_t v___x_4263_; 
v___x_4260_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(v_a_4146_);
v_a_4261_ = lean_ctor_get(v___x_4260_, 0);
lean_inc(v_a_4261_);
lean_dec_ref(v___x_4260_);
v___x_4262_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4263_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_4148_, v___x_4262_);
if (v___x_4263_ == 0)
{
lean_object* v___x_4264_; lean_object* v___x_4265_; 
v___x_4264_ = lean_io_mono_nanos_now();
v___x_4265_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_4152_, v___f_4155_, v___f_4151_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4265_) == 0)
{
lean_object* v_a_4266_; lean_object* v___x_4267_; lean_object* v___x_4268_; 
v_a_4266_ = lean_ctor_get(v___x_4265_, 0);
lean_inc_n(v_a_4266_, 2);
lean_dec_ref_known(v___x_4265_, 1);
v___x_4267_ = lean_box(0);
v___x_4268_ = lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(v_a_4266_, v___x_4267_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4268_) == 0)
{
lean_object* v_a_4269_; lean_object* v___x_4270_; lean_object* v_transform_4271_; lean_object* v___x_4272_; 
v_a_4269_ = lean_ctor_get(v___x_4268_, 0);
lean_inc(v_a_4269_);
lean_dec_ref_known(v___x_4268_, 1);
v___x_4270_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4271_ = lean_ctor_get(v___x_4270_, 1);
lean_inc_ref(v_transform_4271_);
lean_inc(v_a_4146_);
lean_inc_ref(v_a_4145_);
lean_inc(v_a_4144_);
lean_inc_ref(v_a_4143_);
v___x_4272_ = lean_apply_6(v_transform_4271_, v_a_4269_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, lean_box(0));
if (lean_obj_tag(v___x_4272_) == 0)
{
if (v___x_4195_ == 0)
{
lean_object* v_a_4273_; lean_object* v___x_4274_; lean_object* v___x_4275_; 
lean_dec(v_a_4266_);
v_a_4273_ = lean_ctor_get(v___x_4272_, 0);
lean_inc(v_a_4273_);
lean_dec_ref_known(v___x_4272_, 1);
v___x_4274_ = lean_box(0);
v___x_4275_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(v_a_4273_, v___x_4274_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
v___y_4247_ = v___x_4264_;
v___y_4248_ = v_a_4261_;
v___y_4249_ = v___x_4275_;
goto v___jp_4246_;
}
else
{
lean_object* v_a_4276_; lean_object* v___x_4277_; lean_object* v___x_4278_; lean_object* v___x_4279_; lean_object* v___x_4280_; lean_object* v___x_4281_; lean_object* v___x_4282_; 
v_a_4276_ = lean_ctor_get(v___x_4272_, 0);
lean_inc(v_a_4276_);
lean_dec_ref_known(v___x_4272_, 1);
v___x_4277_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4);
v___x_4278_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7(v_a_4266_, v___x_4267_);
v___x_4279_ = l_Lean_MessageData_ofList(v___x_4278_);
v___x_4280_ = l_Lean_indentD(v___x_4279_);
v___x_4281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4281_, 0, v___x_4277_);
lean_ctor_set(v___x_4281_, 1, v___x_4280_);
v___x_4282_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(v_cls_4192_, v___x_4281_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4282_) == 0)
{
lean_object* v_a_4283_; lean_object* v___x_4284_; 
v_a_4283_ = lean_ctor_get(v___x_4282_, 0);
lean_inc(v_a_4283_);
lean_dec_ref_known(v___x_4282_, 1);
v___x_4284_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(v_a_4276_, v_a_4283_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
v___y_4247_ = v___x_4264_;
v___y_4248_ = v_a_4261_;
v___y_4249_ = v___x_4284_;
goto v___jp_4246_;
}
else
{
lean_object* v_a_4285_; 
lean_dec(v_a_4276_);
v_a_4285_ = lean_ctor_get(v___x_4282_, 0);
lean_inc(v_a_4285_);
lean_dec_ref_known(v___x_4282_, 1);
v___y_4242_ = v___x_4264_;
v___y_4243_ = v_a_4261_;
v_a_4244_ = v_a_4285_;
goto v___jp_4241_;
}
}
}
else
{
lean_object* v_a_4286_; 
lean_dec(v_a_4266_);
v_a_4286_ = lean_ctor_get(v___x_4272_, 0);
lean_inc(v_a_4286_);
lean_dec_ref_known(v___x_4272_, 1);
v___y_4242_ = v___x_4264_;
v___y_4243_ = v_a_4261_;
v_a_4244_ = v_a_4286_;
goto v___jp_4241_;
}
}
else
{
lean_dec(v_a_4266_);
v___y_4247_ = v___x_4264_;
v___y_4248_ = v_a_4261_;
v___y_4249_ = v___x_4268_;
goto v___jp_4246_;
}
}
else
{
lean_object* v_a_4287_; 
v_a_4287_ = lean_ctor_get(v___x_4265_, 0);
lean_inc(v_a_4287_);
lean_dec_ref_known(v___x_4265_, 1);
v___y_4242_ = v___x_4264_;
v___y_4243_ = v_a_4261_;
v_a_4244_ = v_a_4287_;
goto v___jp_4241_;
}
}
else
{
lean_object* v___x_4288_; lean_object* v___x_4289_; 
v___x_4288_ = lean_io_get_num_heartbeats();
v___x_4289_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_4152_, v___f_4155_, v___f_4151_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4289_) == 0)
{
lean_object* v_a_4290_; lean_object* v___x_4291_; lean_object* v___x_4292_; 
v_a_4290_ = lean_ctor_get(v___x_4289_, 0);
lean_inc_n(v_a_4290_, 2);
lean_dec_ref_known(v___x_4289_, 1);
v___x_4291_ = lean_box(0);
v___x_4292_ = lp_mathlib_List_filterMapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__5(v_a_4290_, v___x_4291_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4292_) == 0)
{
lean_object* v_a_4293_; lean_object* v___x_4294_; lean_object* v_transform_4295_; lean_object* v___x_4296_; 
v_a_4293_ = lean_ctor_get(v___x_4292_, 0);
lean_inc(v_a_4293_);
lean_dec_ref_known(v___x_4292_, 1);
v___x_4294_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4295_ = lean_ctor_get(v___x_4294_, 1);
lean_inc_ref(v_transform_4295_);
lean_inc(v_a_4146_);
lean_inc_ref(v_a_4145_);
lean_inc(v_a_4144_);
lean_inc_ref(v_a_4143_);
v___x_4296_ = lean_apply_6(v_transform_4295_, v_a_4293_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, lean_box(0));
if (lean_obj_tag(v___x_4296_) == 0)
{
if (v___x_4195_ == 0)
{
lean_object* v_a_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; 
lean_dec(v_a_4290_);
v_a_4297_ = lean_ctor_get(v___x_4296_, 0);
lean_inc(v_a_4297_);
lean_dec_ref_known(v___x_4296_, 1);
v___x_4298_ = lean_box(0);
v___x_4299_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(v_a_4297_, v___x_4298_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
v___y_4214_ = v___x_4288_;
v___y_4215_ = v_a_4261_;
v___y_4216_ = v___x_4299_;
goto v___jp_4213_;
}
else
{
lean_object* v_a_4300_; lean_object* v___x_4301_; lean_object* v___x_4302_; lean_object* v___x_4303_; lean_object* v___x_4304_; lean_object* v___x_4305_; lean_object* v___x_4306_; 
v_a_4300_ = lean_ctor_get(v___x_4296_, 0);
lean_inc(v_a_4300_);
lean_dec_ref_known(v___x_4296_, 1);
v___x_4301_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__4);
v___x_4302_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__7(v_a_4290_, v___x_4291_);
v___x_4303_ = l_Lean_MessageData_ofList(v___x_4302_);
v___x_4304_ = l_Lean_indentD(v___x_4303_);
v___x_4305_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4305_, 0, v___x_4301_);
lean_ctor_set(v___x_4305_, 1, v___x_4304_);
v___x_4306_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1(v_cls_4192_, v___x_4305_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
if (lean_obj_tag(v___x_4306_) == 0)
{
lean_object* v_a_4307_; lean_object* v___x_4308_; 
v_a_4307_ = lean_ctor_get(v___x_4306_, 0);
lean_inc(v_a_4307_);
lean_dec_ref_known(v___x_4306_, 1);
v___x_4308_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___lam__2(v_a_4300_, v_a_4307_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_);
v___y_4214_ = v___x_4288_;
v___y_4215_ = v_a_4261_;
v___y_4216_ = v___x_4308_;
goto v___jp_4213_;
}
else
{
lean_object* v_a_4309_; 
lean_dec(v_a_4300_);
v_a_4309_ = lean_ctor_get(v___x_4306_, 0);
lean_inc(v_a_4309_);
lean_dec_ref_known(v___x_4306_, 1);
v___y_4209_ = v___x_4288_;
v___y_4210_ = v_a_4261_;
v_a_4211_ = v_a_4309_;
goto v___jp_4208_;
}
}
}
else
{
lean_object* v_a_4310_; 
lean_dec(v_a_4290_);
v_a_4310_ = lean_ctor_get(v___x_4296_, 0);
lean_inc(v_a_4310_);
lean_dec_ref_known(v___x_4296_, 1);
v___y_4209_ = v___x_4288_;
v___y_4210_ = v_a_4261_;
v_a_4211_ = v_a_4310_;
goto v___jp_4208_;
}
}
else
{
lean_dec(v_a_4290_);
v___y_4214_ = v___x_4288_;
v___y_4215_ = v_a_4261_;
v___y_4216_ = v___x_4292_;
goto v___jp_4213_;
}
}
else
{
lean_object* v_a_4311_; 
v_a_4311_ = lean_ctor_get(v___x_4289_, 0);
lean_inc(v_a_4311_);
lean_dec_ref_known(v___x_4289_, 1);
v___y_4209_ = v___x_4288_;
v___y_4210_ = v_a_4261_;
v_a_4211_ = v_a_4311_;
goto v___jp_4208_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___boxed(lean_object* v_ls_4368_, lean_object* v_a_4369_, lean_object* v_a_4370_, lean_object* v_a_4371_, lean_object* v_a_4372_, lean_object* v_a_4373_){
_start:
{
lean_object* v_res_4374_; 
v_res_4374_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs(v_ls_4368_, v_a_4369_, v_a_4370_, v_a_4371_, v_a_4372_);
lean_dec(v_a_4372_);
lean_dec_ref(v_a_4371_);
lean_dec(v_a_4370_);
lean_dec_ref(v_a_4369_);
return v_res_4374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4(lean_object* v_x_4375_, lean_object* v_x_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_){
_start:
{
lean_object* v___x_4384_; 
v___x_4384_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___redArg(v_x_4375_, v_x_4376_, v___y_4378_);
return v___x_4384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4___boxed(lean_object* v_x_4385_, lean_object* v_x_4386_, lean_object* v___y_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_, lean_object* v___y_4392_, lean_object* v___y_4393_){
_start:
{
lean_object* v_res_4394_; 
v_res_4394_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__4(v_x_4385_, v_x_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, v___y_4392_);
lean_dec(v___y_4392_);
lean_dec_ref(v___y_4391_);
lean_dec(v___y_4390_);
lean_dec_ref(v___y_4389_);
lean_dec(v___y_4388_);
lean_dec_ref(v___y_4387_);
return v_res_4394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12(lean_object* v_00_u03b1_4395_, lean_object* v_x_4396_, lean_object* v___y_4397_, lean_object* v___y_4398_, lean_object* v___y_4399_, lean_object* v___y_4400_){
_start:
{
lean_object* v___x_4402_; 
v___x_4402_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_x_4396_);
return v___x_4402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___boxed(lean_object* v_00_u03b1_4403_, lean_object* v_x_4404_, lean_object* v___y_4405_, lean_object* v___y_4406_, lean_object* v___y_4407_, lean_object* v___y_4408_, lean_object* v___y_4409_){
_start:
{
lean_object* v_res_4410_; 
v_res_4410_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12(v_00_u03b1_4403_, v_x_4404_, v___y_4405_, v___y_4406_, v___y_4407_, v___y_4408_);
lean_dec(v___y_4408_);
lean_dec_ref(v___y_4407_);
lean_dec(v___y_4406_);
lean_dec_ref(v___y_4405_);
return v_res_4410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0(lean_object* v_snd_4417_, lean_object* v_snd_4418_, uint8_t v_x_4419_, lean_object* v___y_4420_, lean_object* v___y_4421_, lean_object* v___y_4422_, lean_object* v___y_4423_){
_start:
{
lean_object* v___x_4425_; lean_object* v___x_4426_; lean_object* v___x_4427_; lean_object* v___x_4428_; lean_object* v___x_4429_; lean_object* v___x_4430_; 
v___x_4425_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___closed__1));
v___x_4426_ = lean_unsigned_to_nat(2u);
v___x_4427_ = lean_mk_empty_array_with_capacity(v___x_4426_);
v___x_4428_ = lean_array_push(v___x_4427_, v_snd_4417_);
v___x_4429_ = lean_array_push(v___x_4428_, v_snd_4418_);
v___x_4430_ = l_Lean_Meta_mkAppM(v___x_4425_, v___x_4429_, v___y_4420_, v___y_4421_, v___y_4422_, v___y_4423_);
return v___x_4430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0___boxed(lean_object* v_snd_4431_, lean_object* v_snd_4432_, lean_object* v_x_4433_, lean_object* v___y_4434_, lean_object* v___y_4435_, lean_object* v___y_4436_, lean_object* v___y_4437_, lean_object* v___y_4438_){
_start:
{
uint8_t v_x_12840__boxed_4439_; lean_object* v_res_4440_; 
v_x_12840__boxed_4439_ = lean_unbox(v_x_4433_);
v_res_4440_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0(v_snd_4431_, v_snd_4432_, v_x_12840__boxed_4439_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_);
lean_dec(v___y_4437_);
lean_dec_ref(v___y_4436_);
lean_dec(v___y_4435_);
lean_dec_ref(v___y_4434_);
return v_res_4440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1(lean_object* v_x_4456_, lean_object* v_x_4457_, lean_object* v___y_4458_, lean_object* v___y_4459_, lean_object* v___y_4460_, lean_object* v___y_4461_){
_start:
{
lean_object* v___y_4464_; uint8_t v___y_4465_; lean_object* v___y_4469_; lean_object* v_fst_4489_; uint8_t v___x_4490_; 
v_fst_4489_ = lean_ctor_get(v_x_4456_, 0);
v___x_4490_ = lean_unbox(v_fst_4489_);
switch(v___x_4490_)
{
case 0:
{
lean_object* v_snd_4491_; lean_object* v_snd_4492_; lean_object* v___x_4493_; lean_object* v___x_4494_; lean_object* v___x_4495_; lean_object* v___x_4496_; lean_object* v___x_4497_; lean_object* v___x_4498_; 
v_snd_4491_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4491_);
lean_dec_ref(v_x_4456_);
v_snd_4492_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4492_);
lean_dec_ref(v_x_4457_);
v___x_4493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__1));
v___x_4494_ = lean_unsigned_to_nat(2u);
v___x_4495_ = lean_mk_empty_array_with_capacity(v___x_4494_);
v___x_4496_ = lean_array_push(v___x_4495_, v_snd_4491_);
v___x_4497_ = lean_array_push(v___x_4496_, v_snd_4492_);
v___x_4498_ = l_Lean_Meta_mkAppM(v___x_4493_, v___x_4497_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4498_;
goto v___jp_4468_;
}
case 1:
{
lean_object* v_fst_4499_; uint8_t v___x_4500_; 
v_fst_4499_ = lean_ctor_get(v_x_4457_, 0);
v___x_4500_ = lean_unbox(v_fst_4499_);
switch(v___x_4500_)
{
case 0:
{
lean_object* v_snd_4501_; lean_object* v_snd_4502_; uint8_t v___x_4503_; lean_object* v___x_4504_; 
lean_inc(v_fst_4489_);
v_snd_4501_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4501_);
lean_dec_ref(v_x_4456_);
v_snd_4502_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4502_);
lean_dec_ref(v_x_4457_);
v___x_4503_ = lean_unbox(v_fst_4489_);
lean_dec(v_fst_4489_);
v___x_4504_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0(v_snd_4501_, v_snd_4502_, v___x_4503_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4504_;
goto v___jp_4468_;
}
case 1:
{
lean_object* v_snd_4505_; lean_object* v_snd_4506_; lean_object* v___x_4507_; lean_object* v___x_4508_; lean_object* v___x_4509_; lean_object* v___x_4510_; lean_object* v___x_4511_; lean_object* v___x_4512_; 
v_snd_4505_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4505_);
lean_dec_ref(v_x_4456_);
v_snd_4506_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4506_);
lean_dec_ref(v_x_4457_);
v___x_4507_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3));
v___x_4508_ = lean_unsigned_to_nat(2u);
v___x_4509_ = lean_mk_empty_array_with_capacity(v___x_4508_);
v___x_4510_ = lean_array_push(v___x_4509_, v_snd_4505_);
v___x_4511_ = lean_array_push(v___x_4510_, v_snd_4506_);
v___x_4512_ = l_Lean_Meta_mkAppM(v___x_4507_, v___x_4511_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4512_;
goto v___jp_4468_;
}
default: 
{
lean_object* v_snd_4513_; lean_object* v_snd_4514_; lean_object* v___x_4515_; lean_object* v___x_4516_; lean_object* v___x_4517_; lean_object* v___x_4518_; lean_object* v___x_4519_; 
v_snd_4513_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4513_);
lean_dec_ref(v_x_4456_);
v_snd_4514_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4514_);
lean_dec_ref(v_x_4457_);
v___x_4515_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__5));
v___x_4516_ = lean_unsigned_to_nat(1u);
v___x_4517_ = lean_mk_empty_array_with_capacity(v___x_4516_);
v___x_4518_ = lean_array_push(v___x_4517_, v_snd_4514_);
v___x_4519_ = l_Lean_Meta_mkAppM(v___x_4515_, v___x_4518_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
if (lean_obj_tag(v___x_4519_) == 0)
{
lean_object* v_a_4520_; lean_object* v___x_4521_; lean_object* v___x_4522_; lean_object* v___x_4523_; lean_object* v___x_4524_; lean_object* v___x_4525_; lean_object* v___x_4526_; 
v_a_4520_ = lean_ctor_get(v___x_4519_, 0);
lean_inc(v_a_4520_);
lean_dec_ref_known(v___x_4519_, 1);
v___x_4521_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3));
v___x_4522_ = lean_unsigned_to_nat(2u);
v___x_4523_ = lean_mk_empty_array_with_capacity(v___x_4522_);
v___x_4524_ = lean_array_push(v___x_4523_, v_snd_4513_);
v___x_4525_ = lean_array_push(v___x_4524_, v_a_4520_);
v___x_4526_ = l_Lean_Meta_mkAppM(v___x_4521_, v___x_4525_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4526_;
goto v___jp_4468_;
}
else
{
lean_dec(v_snd_4513_);
v___y_4469_ = v___x_4519_;
goto v___jp_4468_;
}
}
}
}
default: 
{
lean_object* v_fst_4527_; uint8_t v___x_4528_; 
v_fst_4527_ = lean_ctor_get(v_x_4457_, 0);
v___x_4528_ = lean_unbox(v_fst_4527_);
switch(v___x_4528_)
{
case 0:
{
lean_object* v_snd_4529_; lean_object* v_snd_4530_; uint8_t v___x_4531_; lean_object* v___x_4532_; 
lean_inc(v_fst_4489_);
v_snd_4529_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4529_);
lean_dec_ref(v_x_4456_);
v_snd_4530_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4530_);
lean_dec_ref(v_x_4457_);
v___x_4531_ = lean_unbox(v_fst_4489_);
lean_dec(v_fst_4489_);
v___x_4532_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__0(v_snd_4529_, v_snd_4530_, v___x_4531_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4532_;
goto v___jp_4468_;
}
case 1:
{
lean_object* v_snd_4533_; lean_object* v_snd_4534_; lean_object* v___x_4535_; lean_object* v___x_4536_; lean_object* v___x_4537_; lean_object* v___x_4538_; lean_object* v___x_4539_; 
v_snd_4533_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4533_);
lean_dec_ref(v_x_4456_);
v_snd_4534_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4534_);
lean_dec_ref(v_x_4457_);
v___x_4535_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__5));
v___x_4536_ = lean_unsigned_to_nat(1u);
v___x_4537_ = lean_mk_empty_array_with_capacity(v___x_4536_);
v___x_4538_ = lean_array_push(v___x_4537_, v_snd_4533_);
v___x_4539_ = l_Lean_Meta_mkAppM(v___x_4535_, v___x_4538_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
if (lean_obj_tag(v___x_4539_) == 0)
{
lean_object* v_a_4540_; lean_object* v___x_4541_; lean_object* v___x_4542_; lean_object* v___x_4543_; lean_object* v___x_4544_; lean_object* v___x_4545_; lean_object* v___x_4546_; 
v_a_4540_ = lean_ctor_get(v___x_4539_, 0);
lean_inc(v_a_4540_);
lean_dec_ref_known(v___x_4539_, 1);
v___x_4541_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__3));
v___x_4542_ = lean_unsigned_to_nat(2u);
v___x_4543_ = lean_mk_empty_array_with_capacity(v___x_4542_);
v___x_4544_ = lean_array_push(v___x_4543_, v_a_4540_);
v___x_4545_ = lean_array_push(v___x_4544_, v_snd_4534_);
v___x_4546_ = l_Lean_Meta_mkAppM(v___x_4541_, v___x_4545_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4546_;
goto v___jp_4468_;
}
else
{
lean_dec(v_snd_4534_);
v___y_4469_ = v___x_4539_;
goto v___jp_4468_;
}
}
default: 
{
lean_object* v_snd_4547_; lean_object* v_snd_4548_; lean_object* v___x_4549_; lean_object* v___x_4550_; lean_object* v___x_4551_; lean_object* v___x_4552_; lean_object* v___x_4553_; lean_object* v___x_4554_; 
v_snd_4547_ = lean_ctor_get(v_x_4456_, 1);
lean_inc(v_snd_4547_);
lean_dec_ref(v_x_4456_);
v_snd_4548_ = lean_ctor_get(v_x_4457_, 1);
lean_inc(v_snd_4548_);
lean_dec_ref(v_x_4457_);
v___x_4549_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___closed__7));
v___x_4550_ = lean_unsigned_to_nat(2u);
v___x_4551_ = lean_mk_empty_array_with_capacity(v___x_4550_);
v___x_4552_ = lean_array_push(v___x_4551_, v_snd_4547_);
v___x_4553_ = lean_array_push(v___x_4552_, v_snd_4548_);
v___x_4554_ = l_Lean_Meta_mkAppM(v___x_4549_, v___x_4553_, v___y_4458_, v___y_4459_, v___y_4460_, v___y_4461_);
v___y_4469_ = v___x_4554_;
goto v___jp_4468_;
}
}
}
}
v___jp_4463_:
{
if (v___y_4465_ == 0)
{
lean_object* v___x_4466_; lean_object* v___x_4467_; 
lean_dec_ref(v___y_4464_);
v___x_4466_ = lean_box(0);
v___x_4467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4467_, 0, v___x_4466_);
return v___x_4467_;
}
else
{
return v___y_4464_;
}
}
v___jp_4468_:
{
if (lean_obj_tag(v___y_4469_) == 0)
{
lean_object* v_a_4470_; lean_object* v___x_4472_; uint8_t v_isShared_4473_; uint8_t v_isSharedCheck_4478_; 
v_a_4470_ = lean_ctor_get(v___y_4469_, 0);
v_isSharedCheck_4478_ = !lean_is_exclusive(v___y_4469_);
if (v_isSharedCheck_4478_ == 0)
{
v___x_4472_ = v___y_4469_;
v_isShared_4473_ = v_isSharedCheck_4478_;
goto v_resetjp_4471_;
}
else
{
lean_inc(v_a_4470_);
lean_dec(v___y_4469_);
v___x_4472_ = lean_box(0);
v_isShared_4473_ = v_isSharedCheck_4478_;
goto v_resetjp_4471_;
}
v_resetjp_4471_:
{
lean_object* v___x_4474_; lean_object* v___x_4476_; 
v___x_4474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4474_, 0, v_a_4470_);
if (v_isShared_4473_ == 0)
{
lean_ctor_set(v___x_4472_, 0, v___x_4474_);
v___x_4476_ = v___x_4472_;
goto v_reusejp_4475_;
}
else
{
lean_object* v_reuseFailAlloc_4477_; 
v_reuseFailAlloc_4477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4477_, 0, v___x_4474_);
v___x_4476_ = v_reuseFailAlloc_4477_;
goto v_reusejp_4475_;
}
v_reusejp_4475_:
{
return v___x_4476_;
}
}
}
else
{
lean_object* v_a_4479_; lean_object* v___x_4481_; uint8_t v_isShared_4482_; uint8_t v_isSharedCheck_4488_; 
v_a_4479_ = lean_ctor_get(v___y_4469_, 0);
v_isSharedCheck_4488_ = !lean_is_exclusive(v___y_4469_);
if (v_isSharedCheck_4488_ == 0)
{
v___x_4481_ = v___y_4469_;
v_isShared_4482_ = v_isSharedCheck_4488_;
goto v_resetjp_4480_;
}
else
{
lean_inc(v_a_4479_);
lean_dec(v___y_4469_);
v___x_4481_ = lean_box(0);
v_isShared_4482_ = v_isSharedCheck_4488_;
goto v_resetjp_4480_;
}
v_resetjp_4480_:
{
lean_object* v___x_4484_; 
lean_inc(v_a_4479_);
if (v_isShared_4482_ == 0)
{
v___x_4484_ = v___x_4481_;
goto v_reusejp_4483_;
}
else
{
lean_object* v_reuseFailAlloc_4487_; 
v_reuseFailAlloc_4487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4487_, 0, v_a_4479_);
v___x_4484_ = v_reuseFailAlloc_4487_;
goto v_reusejp_4483_;
}
v_reusejp_4483_:
{
uint8_t v___x_4485_; 
v___x_4485_ = l_Lean_Exception_isInterrupt(v_a_4479_);
if (v___x_4485_ == 0)
{
uint8_t v___x_4486_; 
v___x_4486_ = l_Lean_Exception_isRuntime(v_a_4479_);
v___y_4464_ = v___x_4484_;
v___y_4465_ = v___x_4486_;
goto v___jp_4463_;
}
else
{
lean_dec(v_a_4479_);
v___y_4464_ = v___x_4484_;
v___y_4465_ = v___x_4485_;
goto v___jp_4463_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1___boxed(lean_object* v_x_4555_, lean_object* v_x_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_, lean_object* v___y_4560_, lean_object* v___y_4561_){
_start:
{
lean_object* v_res_4562_; 
v_res_4562_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__1(v_x_4555_, v_x_4556_, v___y_4557_, v___y_4558_, v___y_4559_, v___y_4560_);
lean_dec(v___y_4560_);
lean_dec_ref(v___y_4559_);
lean_dec(v___y_4558_);
lean_dec_ref(v___y_4557_);
return v_res_4562_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1(void){
_start:
{
lean_object* v___x_4564_; lean_object* v___x_4565_; 
v___x_4564_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__0));
v___x_4565_ = l_Lean_stringToMessageData(v___x_4564_);
return v___x_4565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2(lean_object* v_x_4566_, lean_object* v___y_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_){
_start:
{
lean_object* v___x_4572_; lean_object* v___x_4573_; 
v___x_4572_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___closed__1);
v___x_4573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4573_, 0, v___x_4572_);
return v___x_4573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2___boxed(lean_object* v_x_4574_, lean_object* v___y_4575_, lean_object* v___y_4576_, lean_object* v___y_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_){
_start:
{
lean_object* v_res_4580_; 
v_res_4580_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___lam__2(v_x_4574_, v___y_4575_, v___y_4576_, v___y_4577_, v___y_4578_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___y_4576_);
lean_dec_ref(v___y_4575_);
lean_dec_ref(v_x_4574_);
return v_res_4580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg(lean_object* v_f_4581_, lean_object* v_head_4582_, lean_object* v_x_4583_, lean_object* v_x_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_, lean_object* v___y_4587_, lean_object* v___y_4588_){
_start:
{
if (lean_obj_tag(v_x_4584_) == 0)
{
lean_object* v___x_4590_; 
lean_dec(v_head_4582_);
lean_dec_ref(v_f_4581_);
v___x_4590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4590_, 0, v_x_4583_);
return v___x_4590_;
}
else
{
lean_object* v_head_4591_; lean_object* v_tail_4592_; lean_object* v___x_4593_; 
v_head_4591_ = lean_ctor_get(v_x_4584_, 0);
lean_inc(v_head_4591_);
v_tail_4592_ = lean_ctor_get(v_x_4584_, 1);
lean_inc(v_tail_4592_);
lean_dec_ref_known(v_x_4584_, 2);
lean_inc_ref(v_f_4581_);
lean_inc(v___y_4588_);
lean_inc_ref(v___y_4587_);
lean_inc(v___y_4586_);
lean_inc_ref(v___y_4585_);
lean_inc(v_head_4582_);
v___x_4593_ = lean_apply_7(v_f_4581_, v_head_4582_, v_head_4591_, v___y_4585_, v___y_4586_, v___y_4587_, v___y_4588_, lean_box(0));
if (lean_obj_tag(v___x_4593_) == 0)
{
lean_object* v_a_4594_; lean_object* v___x_4595_; 
v_a_4594_ = lean_ctor_get(v___x_4593_, 0);
lean_inc(v_a_4594_);
lean_dec_ref_known(v___x_4593_, 1);
v___x_4595_ = lean_array_push(v_x_4583_, v_a_4594_);
v_x_4583_ = v___x_4595_;
v_x_4584_ = v_tail_4592_;
goto _start;
}
else
{
lean_object* v_a_4597_; lean_object* v___x_4599_; uint8_t v_isShared_4600_; uint8_t v_isSharedCheck_4604_; 
lean_dec(v_tail_4592_);
lean_dec_ref(v_x_4583_);
lean_dec(v_head_4582_);
lean_dec_ref(v_f_4581_);
v_a_4597_ = lean_ctor_get(v___x_4593_, 0);
v_isSharedCheck_4604_ = !lean_is_exclusive(v___x_4593_);
if (v_isSharedCheck_4604_ == 0)
{
v___x_4599_ = v___x_4593_;
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
else
{
lean_inc(v_a_4597_);
lean_dec(v___x_4593_);
v___x_4599_ = lean_box(0);
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
v_resetjp_4598_:
{
lean_object* v___x_4602_; 
if (v_isShared_4600_ == 0)
{
v___x_4602_ = v___x_4599_;
goto v_reusejp_4601_;
}
else
{
lean_object* v_reuseFailAlloc_4603_; 
v_reuseFailAlloc_4603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4603_, 0, v_a_4597_);
v___x_4602_ = v_reuseFailAlloc_4603_;
goto v_reusejp_4601_;
}
v_reusejp_4601_:
{
return v___x_4602_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_f_4605_, lean_object* v_head_4606_, lean_object* v_x_4607_, lean_object* v_x_4608_, lean_object* v___y_4609_, lean_object* v___y_4610_, lean_object* v___y_4611_, lean_object* v___y_4612_, lean_object* v___y_4613_){
_start:
{
lean_object* v_res_4614_; 
v_res_4614_ = lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg(v_f_4605_, v_head_4606_, v_x_4607_, v_x_4608_, v___y_4609_, v___y_4610_, v___y_4611_, v___y_4612_);
lean_dec(v___y_4612_);
lean_dec_ref(v___y_4611_);
lean_dec(v___y_4610_);
lean_dec_ref(v___y_4609_);
return v_res_4614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg(lean_object* v_f_4615_, lean_object* v_a_4616_, lean_object* v_a_4617_, lean_object* v___y_4618_, lean_object* v___y_4619_, lean_object* v___y_4620_, lean_object* v___y_4621_){
_start:
{
if (lean_obj_tag(v_a_4616_) == 0)
{
lean_object* v___x_4623_; lean_object* v___x_4624_; 
lean_dec_ref(v_f_4615_);
v___x_4623_ = lean_array_to_list(v_a_4617_);
v___x_4624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4624_, 0, v___x_4623_);
return v___x_4624_;
}
else
{
lean_object* v_head_4625_; lean_object* v_tail_4626_; lean_object* v___x_4627_; 
v_head_4625_ = lean_ctor_get(v_a_4616_, 0);
lean_inc_n(v_head_4625_, 3);
v_tail_4626_ = lean_ctor_get(v_a_4616_, 1);
lean_inc(v_tail_4626_);
lean_dec_ref_known(v_a_4616_, 2);
lean_inc_ref(v_f_4615_);
lean_inc(v___y_4621_);
lean_inc_ref(v___y_4620_);
lean_inc(v___y_4619_);
lean_inc_ref(v___y_4618_);
v___x_4627_ = lean_apply_7(v_f_4615_, v_head_4625_, v_head_4625_, v___y_4618_, v___y_4619_, v___y_4620_, v___y_4621_, lean_box(0));
if (lean_obj_tag(v___x_4627_) == 0)
{
lean_object* v_a_4628_; lean_object* v___x_4629_; lean_object* v___x_4630_; 
v_a_4628_ = lean_ctor_get(v___x_4627_, 0);
lean_inc(v_a_4628_);
lean_dec_ref_known(v___x_4627_, 1);
v___x_4629_ = lean_array_push(v_a_4617_, v_a_4628_);
lean_inc(v_tail_4626_);
lean_inc_ref(v_f_4615_);
v___x_4630_ = lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg(v_f_4615_, v_head_4625_, v___x_4629_, v_tail_4626_, v___y_4618_, v___y_4619_, v___y_4620_, v___y_4621_);
if (lean_obj_tag(v___x_4630_) == 0)
{
lean_object* v_a_4631_; 
v_a_4631_ = lean_ctor_get(v___x_4630_, 0);
lean_inc(v_a_4631_);
lean_dec_ref_known(v___x_4630_, 1);
v_a_4616_ = v_tail_4626_;
v_a_4617_ = v_a_4631_;
goto _start;
}
else
{
lean_object* v_a_4633_; lean_object* v___x_4635_; uint8_t v_isShared_4636_; uint8_t v_isSharedCheck_4640_; 
lean_dec(v_tail_4626_);
lean_dec_ref(v_f_4615_);
v_a_4633_ = lean_ctor_get(v___x_4630_, 0);
v_isSharedCheck_4640_ = !lean_is_exclusive(v___x_4630_);
if (v_isSharedCheck_4640_ == 0)
{
v___x_4635_ = v___x_4630_;
v_isShared_4636_ = v_isSharedCheck_4640_;
goto v_resetjp_4634_;
}
else
{
lean_inc(v_a_4633_);
lean_dec(v___x_4630_);
v___x_4635_ = lean_box(0);
v_isShared_4636_ = v_isSharedCheck_4640_;
goto v_resetjp_4634_;
}
v_resetjp_4634_:
{
lean_object* v___x_4638_; 
if (v_isShared_4636_ == 0)
{
v___x_4638_ = v___x_4635_;
goto v_reusejp_4637_;
}
else
{
lean_object* v_reuseFailAlloc_4639_; 
v_reuseFailAlloc_4639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4639_, 0, v_a_4633_);
v___x_4638_ = v_reuseFailAlloc_4639_;
goto v_reusejp_4637_;
}
v_reusejp_4637_:
{
return v___x_4638_;
}
}
}
}
else
{
lean_object* v_a_4641_; lean_object* v___x_4643_; uint8_t v_isShared_4644_; uint8_t v_isSharedCheck_4648_; 
lean_dec(v_tail_4626_);
lean_dec(v_head_4625_);
lean_dec_ref(v_a_4617_);
lean_dec_ref(v_f_4615_);
v_a_4641_ = lean_ctor_get(v___x_4627_, 0);
v_isSharedCheck_4648_ = !lean_is_exclusive(v___x_4627_);
if (v_isSharedCheck_4648_ == 0)
{
v___x_4643_ = v___x_4627_;
v_isShared_4644_ = v_isSharedCheck_4648_;
goto v_resetjp_4642_;
}
else
{
lean_inc(v_a_4641_);
lean_dec(v___x_4627_);
v___x_4643_ = lean_box(0);
v_isShared_4644_ = v_isSharedCheck_4648_;
goto v_resetjp_4642_;
}
v_resetjp_4642_:
{
lean_object* v___x_4646_; 
if (v_isShared_4644_ == 0)
{
v___x_4646_ = v___x_4643_;
goto v_reusejp_4645_;
}
else
{
lean_object* v_reuseFailAlloc_4647_; 
v_reuseFailAlloc_4647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4647_, 0, v_a_4641_);
v___x_4646_ = v_reuseFailAlloc_4647_;
goto v_reusejp_4645_;
}
v_reusejp_4645_:
{
return v___x_4646_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg___boxed(lean_object* v_f_4649_, lean_object* v_a_4650_, lean_object* v_a_4651_, lean_object* v___y_4652_, lean_object* v___y_4653_, lean_object* v___y_4654_, lean_object* v___y_4655_, lean_object* v___y_4656_){
_start:
{
lean_object* v_res_4657_; 
v_res_4657_ = lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg(v_f_4649_, v_a_4650_, v_a_4651_, v___y_4652_, v___y_4653_, v___y_4654_, v___y_4655_);
lean_dec(v___y_4655_);
lean_dec_ref(v___y_4654_);
lean_dec(v___y_4653_);
lean_dec_ref(v___y_4652_);
return v_res_4657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(lean_object* v_f_4658_, lean_object* v_l_4659_, lean_object* v___y_4660_, lean_object* v___y_4661_, lean_object* v___y_4662_, lean_object* v___y_4663_){
_start:
{
lean_object* v___x_4665_; lean_object* v___x_4666_; 
v___x_4665_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_natToInt_spec__4___closed__1));
v___x_4666_ = lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg(v_f_4658_, v_l_4659_, v___x_4665_, v___y_4660_, v___y_4661_, v___y_4662_, v___y_4663_);
return v___x_4666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg___boxed(lean_object* v_f_4667_, lean_object* v_l_4668_, lean_object* v___y_4669_, lean_object* v___y_4670_, lean_object* v___y_4671_, lean_object* v___y_4672_, lean_object* v___y_4673_){
_start:
{
lean_object* v_res_4674_; 
v_res_4674_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v_f_4667_, v_l_4668_, v___y_4669_, v___y_4670_, v___y_4671_, v___y_4672_);
lean_dec(v___y_4672_);
lean_dec_ref(v___y_4671_);
lean_dec(v___y_4670_);
lean_dec_ref(v___y_4669_);
return v_res_4674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(lean_object* v_a_4675_, lean_object* v_a_4676_){
_start:
{
if (lean_obj_tag(v_a_4675_) == 0)
{
lean_object* v___x_4677_; 
v___x_4677_ = lean_array_to_list(v_a_4676_);
return v___x_4677_;
}
else
{
lean_object* v_head_4678_; 
v_head_4678_ = lean_ctor_get(v_a_4675_, 0);
if (lean_obj_tag(v_head_4678_) == 0)
{
lean_object* v_tail_4679_; 
v_tail_4679_ = lean_ctor_get(v_a_4675_, 1);
lean_inc(v_tail_4679_);
lean_dec_ref_known(v_a_4675_, 2);
v_a_4675_ = v_tail_4679_;
goto _start;
}
else
{
lean_object* v_tail_4681_; lean_object* v_val_4682_; lean_object* v___x_4683_; 
lean_inc_ref(v_head_4678_);
v_tail_4681_ = lean_ctor_get(v_a_4675_, 1);
lean_inc(v_tail_4681_);
lean_dec_ref_known(v_a_4675_, 2);
v_val_4682_ = lean_ctor_get(v_head_4678_, 0);
lean_inc(v_val_4682_);
lean_dec_ref_known(v_head_4678_, 1);
v___x_4683_ = lean_array_push(v_a_4676_, v_val_4682_);
v_a_4675_ = v_tail_4681_;
v_a_4676_ = v___x_4683_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(lean_object* v_x_4685_, lean_object* v_x_4686_, lean_object* v___y_4687_, lean_object* v___y_4688_, lean_object* v___y_4689_, lean_object* v___y_4690_){
_start:
{
if (lean_obj_tag(v_x_4685_) == 0)
{
lean_object* v___x_4692_; lean_object* v___x_4693_; 
v___x_4692_ = l_List_reverse___redArg(v_x_4686_);
v___x_4693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4693_, 0, v___x_4692_);
return v___x_4693_;
}
else
{
lean_object* v_head_4694_; lean_object* v_tail_4695_; lean_object* v___x_4697_; uint8_t v_isShared_4698_; uint8_t v_isSharedCheck_4745_; 
v_head_4694_ = lean_ctor_get(v_x_4685_, 0);
v_tail_4695_ = lean_ctor_get(v_x_4685_, 1);
v_isSharedCheck_4745_ = !lean_is_exclusive(v_x_4685_);
if (v_isSharedCheck_4745_ == 0)
{
v___x_4697_ = v_x_4685_;
v_isShared_4698_ = v_isSharedCheck_4745_;
goto v_resetjp_4696_;
}
else
{
lean_inc(v_tail_4695_);
lean_inc(v_head_4694_);
lean_dec(v_x_4685_);
v___x_4697_ = lean_box(0);
v_isShared_4698_ = v_isSharedCheck_4745_;
goto v_resetjp_4696_;
}
v_resetjp_4696_:
{
lean_object* v_a_4700_; lean_object* v___y_4706_; lean_object* v___x_4716_; 
lean_inc(v___y_4690_);
lean_inc_ref(v___y_4689_);
lean_inc(v___y_4688_);
lean_inc_ref(v___y_4687_);
lean_inc(v_head_4694_);
v___x_4716_ = lean_infer_type(v_head_4694_, v___y_4687_, v___y_4688_, v___y_4689_, v___y_4690_);
if (lean_obj_tag(v___x_4716_) == 0)
{
lean_object* v_a_4717_; lean_object* v___x_4718_; 
v_a_4717_ = lean_ctor_get(v___x_4716_, 0);
lean_inc(v_a_4717_);
lean_dec_ref_known(v___x_4716_, 1);
v___x_4718_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_a_4717_, v___y_4687_, v___y_4688_, v___y_4689_, v___y_4690_);
if (lean_obj_tag(v___x_4718_) == 0)
{
lean_object* v_a_4719_; lean_object* v_fst_4720_; lean_object* v___x_4722_; uint8_t v_isShared_4723_; uint8_t v_isSharedCheck_4727_; 
v_a_4719_ = lean_ctor_get(v___x_4718_, 0);
lean_inc(v_a_4719_);
lean_dec_ref_known(v___x_4718_, 1);
v_fst_4720_ = lean_ctor_get(v_a_4719_, 0);
v_isSharedCheck_4727_ = !lean_is_exclusive(v_a_4719_);
if (v_isSharedCheck_4727_ == 0)
{
lean_object* v_unused_4728_; 
v_unused_4728_ = lean_ctor_get(v_a_4719_, 1);
lean_dec(v_unused_4728_);
v___x_4722_ = v_a_4719_;
v_isShared_4723_ = v_isSharedCheck_4727_;
goto v_resetjp_4721_;
}
else
{
lean_inc(v_fst_4720_);
lean_dec(v_a_4719_);
v___x_4722_ = lean_box(0);
v_isShared_4723_ = v_isSharedCheck_4727_;
goto v_resetjp_4721_;
}
v_resetjp_4721_:
{
lean_object* v___x_4725_; 
if (v_isShared_4723_ == 0)
{
lean_ctor_set(v___x_4722_, 1, v_head_4694_);
v___x_4725_ = v___x_4722_;
goto v_reusejp_4724_;
}
else
{
lean_object* v_reuseFailAlloc_4726_; 
v_reuseFailAlloc_4726_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4726_, 0, v_fst_4720_);
lean_ctor_set(v_reuseFailAlloc_4726_, 1, v_head_4694_);
v___x_4725_ = v_reuseFailAlloc_4726_;
goto v_reusejp_4724_;
}
v_reusejp_4724_:
{
v_a_4700_ = v___x_4725_;
goto v___jp_4699_;
}
}
}
else
{
if (lean_obj_tag(v___x_4718_) == 0)
{
lean_dec(v_head_4694_);
v___y_4706_ = v___x_4718_;
goto v___jp_4705_;
}
else
{
lean_object* v_a_4729_; uint8_t v___y_4731_; uint8_t v___x_4735_; 
v_a_4729_ = lean_ctor_get(v___x_4718_, 0);
lean_inc(v_a_4729_);
v___x_4735_ = l_Lean_Exception_isInterrupt(v_a_4729_);
if (v___x_4735_ == 0)
{
uint8_t v___x_4736_; 
v___x_4736_ = l_Lean_Exception_isRuntime(v_a_4729_);
v___y_4731_ = v___x_4736_;
goto v___jp_4730_;
}
else
{
lean_dec(v_a_4729_);
v___y_4731_ = v___x_4735_;
goto v___jp_4730_;
}
v___jp_4730_:
{
if (v___y_4731_ == 0)
{
uint8_t v___x_4732_; lean_object* v___x_4733_; lean_object* v___x_4734_; 
lean_dec_ref_known(v___x_4718_, 1);
v___x_4732_ = 2;
v___x_4733_ = lean_box(v___x_4732_);
v___x_4734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4734_, 0, v___x_4733_);
lean_ctor_set(v___x_4734_, 1, v_head_4694_);
v_a_4700_ = v___x_4734_;
goto v___jp_4699_;
}
else
{
lean_dec(v_head_4694_);
v___y_4706_ = v___x_4718_;
goto v___jp_4705_;
}
}
}
}
}
else
{
lean_object* v_a_4737_; lean_object* v___x_4739_; uint8_t v_isShared_4740_; uint8_t v_isSharedCheck_4744_; 
lean_del_object(v___x_4697_);
lean_dec(v_tail_4695_);
lean_dec(v_head_4694_);
lean_dec(v_x_4686_);
v_a_4737_ = lean_ctor_get(v___x_4716_, 0);
v_isSharedCheck_4744_ = !lean_is_exclusive(v___x_4716_);
if (v_isSharedCheck_4744_ == 0)
{
v___x_4739_ = v___x_4716_;
v_isShared_4740_ = v_isSharedCheck_4744_;
goto v_resetjp_4738_;
}
else
{
lean_inc(v_a_4737_);
lean_dec(v___x_4716_);
v___x_4739_ = lean_box(0);
v_isShared_4740_ = v_isSharedCheck_4744_;
goto v_resetjp_4738_;
}
v_resetjp_4738_:
{
lean_object* v___x_4742_; 
if (v_isShared_4740_ == 0)
{
v___x_4742_ = v___x_4739_;
goto v_reusejp_4741_;
}
else
{
lean_object* v_reuseFailAlloc_4743_; 
v_reuseFailAlloc_4743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4743_, 0, v_a_4737_);
v___x_4742_ = v_reuseFailAlloc_4743_;
goto v_reusejp_4741_;
}
v_reusejp_4741_:
{
return v___x_4742_;
}
}
}
v___jp_4699_:
{
lean_object* v___x_4702_; 
if (v_isShared_4698_ == 0)
{
lean_ctor_set(v___x_4697_, 1, v_x_4686_);
lean_ctor_set(v___x_4697_, 0, v_a_4700_);
v___x_4702_ = v___x_4697_;
goto v_reusejp_4701_;
}
else
{
lean_object* v_reuseFailAlloc_4704_; 
v_reuseFailAlloc_4704_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4704_, 0, v_a_4700_);
lean_ctor_set(v_reuseFailAlloc_4704_, 1, v_x_4686_);
v___x_4702_ = v_reuseFailAlloc_4704_;
goto v_reusejp_4701_;
}
v_reusejp_4701_:
{
v_x_4685_ = v_tail_4695_;
v_x_4686_ = v___x_4702_;
goto _start;
}
}
v___jp_4705_:
{
if (lean_obj_tag(v___y_4706_) == 0)
{
lean_object* v_a_4707_; 
v_a_4707_ = lean_ctor_get(v___y_4706_, 0);
lean_inc(v_a_4707_);
lean_dec_ref_known(v___y_4706_, 1);
v_a_4700_ = v_a_4707_;
goto v___jp_4699_;
}
else
{
lean_object* v_a_4708_; lean_object* v___x_4710_; uint8_t v_isShared_4711_; uint8_t v_isSharedCheck_4715_; 
lean_del_object(v___x_4697_);
lean_dec(v_tail_4695_);
lean_dec(v_x_4686_);
v_a_4708_ = lean_ctor_get(v___y_4706_, 0);
v_isSharedCheck_4715_ = !lean_is_exclusive(v___y_4706_);
if (v_isSharedCheck_4715_ == 0)
{
v___x_4710_ = v___y_4706_;
v_isShared_4711_ = v_isSharedCheck_4715_;
goto v_resetjp_4709_;
}
else
{
lean_inc(v_a_4708_);
lean_dec(v___y_4706_);
v___x_4710_ = lean_box(0);
v_isShared_4711_ = v_isSharedCheck_4715_;
goto v_resetjp_4709_;
}
v_resetjp_4709_:
{
lean_object* v___x_4713_; 
if (v_isShared_4711_ == 0)
{
v___x_4713_ = v___x_4710_;
goto v_reusejp_4712_;
}
else
{
lean_object* v_reuseFailAlloc_4714_; 
v_reuseFailAlloc_4714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4714_, 0, v_a_4708_);
v___x_4713_ = v_reuseFailAlloc_4714_;
goto v_reusejp_4712_;
}
v_reusejp_4712_:
{
return v___x_4713_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0___boxed(lean_object* v_x_4746_, lean_object* v_x_4747_, lean_object* v___y_4748_, lean_object* v___y_4749_, lean_object* v___y_4750_, lean_object* v___y_4751_, lean_object* v___y_4752_){
_start:
{
lean_object* v_res_4753_; 
v_res_4753_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(v_x_4746_, v_x_4747_, v___y_4748_, v___y_4749_, v___y_4750_, v___y_4751_);
lean_dec(v___y_4751_);
lean_dec_ref(v___y_4750_);
lean_dec(v___y_4749_);
lean_dec_ref(v___y_4748_);
return v_res_4753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs(lean_object* v_ls_4758_, lean_object* v_a_4759_, lean_object* v_a_4760_, lean_object* v_a_4761_, lean_object* v_a_4762_){
_start:
{
lean_object* v_options_4764_; lean_object* v_inheritedTraceOptions_4765_; uint8_t v_hasTrace_4766_; lean_object* v___x_4767_; 
v_options_4764_ = lean_ctor_get(v_a_4761_, 2);
v_inheritedTraceOptions_4765_ = lean_ctor_get(v_a_4761_, 13);
v_hasTrace_4766_ = lean_ctor_get_uint8(v_options_4764_, sizeof(void*)*1);
v___x_4767_ = lean_box(0);
if (v_hasTrace_4766_ == 0)
{
lean_object* v___x_4768_; 
v___x_4768_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(v_ls_4758_, v___x_4767_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4768_) == 0)
{
lean_object* v_a_4769_; lean_object* v___f_4770_; lean_object* v___x_4771_; 
v_a_4769_ = lean_ctor_get(v___x_4768_, 0);
lean_inc(v_a_4769_);
lean_dec_ref_known(v___x_4768_, 1);
v___f_4770_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__0));
v___x_4771_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v___f_4770_, v_a_4769_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4771_) == 0)
{
lean_object* v_a_4772_; lean_object* v___x_4773_; lean_object* v_transform_4774_; lean_object* v___x_4775_; lean_object* v___x_4776_; lean_object* v___x_4777_; 
v_a_4772_ = lean_ctor_get(v___x_4771_, 0);
lean_inc(v_a_4772_);
lean_dec_ref_known(v___x_4771_, 1);
v___x_4773_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4774_ = lean_ctor_get(v___x_4773_, 1);
v___x_4775_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1));
v___x_4776_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(v_a_4772_, v___x_4775_);
lean_inc_ref(v_transform_4774_);
lean_inc(v_a_4762_);
lean_inc_ref(v_a_4761_);
lean_inc(v_a_4760_);
lean_inc_ref(v_a_4759_);
v___x_4777_ = lean_apply_6(v_transform_4774_, v___x_4776_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, lean_box(0));
return v___x_4777_;
}
else
{
lean_object* v_a_4778_; lean_object* v___x_4780_; uint8_t v_isShared_4781_; uint8_t v_isSharedCheck_4785_; 
v_a_4778_ = lean_ctor_get(v___x_4771_, 0);
v_isSharedCheck_4785_ = !lean_is_exclusive(v___x_4771_);
if (v_isSharedCheck_4785_ == 0)
{
v___x_4780_ = v___x_4771_;
v_isShared_4781_ = v_isSharedCheck_4785_;
goto v_resetjp_4779_;
}
else
{
lean_inc(v_a_4778_);
lean_dec(v___x_4771_);
v___x_4780_ = lean_box(0);
v_isShared_4781_ = v_isSharedCheck_4785_;
goto v_resetjp_4779_;
}
v_resetjp_4779_:
{
lean_object* v___x_4783_; 
if (v_isShared_4781_ == 0)
{
v___x_4783_ = v___x_4780_;
goto v_reusejp_4782_;
}
else
{
lean_object* v_reuseFailAlloc_4784_; 
v_reuseFailAlloc_4784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4784_, 0, v_a_4778_);
v___x_4783_ = v_reuseFailAlloc_4784_;
goto v_reusejp_4782_;
}
v_reusejp_4782_:
{
return v___x_4783_;
}
}
}
}
else
{
lean_object* v_a_4786_; lean_object* v___x_4788_; uint8_t v_isShared_4789_; uint8_t v_isSharedCheck_4793_; 
v_a_4786_ = lean_ctor_get(v___x_4768_, 0);
v_isSharedCheck_4793_ = !lean_is_exclusive(v___x_4768_);
if (v_isSharedCheck_4793_ == 0)
{
v___x_4788_ = v___x_4768_;
v_isShared_4789_ = v_isSharedCheck_4793_;
goto v_resetjp_4787_;
}
else
{
lean_inc(v_a_4786_);
lean_dec(v___x_4768_);
v___x_4788_ = lean_box(0);
v_isShared_4789_ = v_isSharedCheck_4793_;
goto v_resetjp_4787_;
}
v_resetjp_4787_:
{
lean_object* v___x_4791_; 
if (v_isShared_4789_ == 0)
{
v___x_4791_ = v___x_4788_;
goto v_reusejp_4790_;
}
else
{
lean_object* v_reuseFailAlloc_4792_; 
v_reuseFailAlloc_4792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4792_, 0, v_a_4786_);
v___x_4791_ = v_reuseFailAlloc_4792_;
goto v_reusejp_4790_;
}
v_reusejp_4790_:
{
return v___x_4791_;
}
}
}
}
else
{
lean_object* v___f_4794_; lean_object* v___f_4795_; lean_object* v___x_4796_; lean_object* v___x_4797_; lean_object* v___x_4798_; uint8_t v___x_4799_; lean_object* v___y_4801_; lean_object* v___y_4802_; lean_object* v_a_4803_; lean_object* v___y_4816_; lean_object* v___y_4817_; lean_object* v_a_4818_; lean_object* v___y_4821_; lean_object* v___y_4822_; lean_object* v_a_4823_; lean_object* v___y_4833_; lean_object* v___y_4834_; lean_object* v_a_4835_; 
v___f_4794_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__2));
v___f_4795_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__0));
v___x_4796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_4797_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1));
v___x_4798_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6);
v___x_4799_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4765_, v_options_4764_, v___x_4798_);
if (v___x_4799_ == 0)
{
lean_object* v___x_4884_; uint8_t v___x_4885_; 
v___x_4884_ = l_Lean_trace_profiler;
v___x_4885_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_4764_, v___x_4884_);
if (v___x_4885_ == 0)
{
lean_object* v___x_4886_; 
v___x_4886_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(v_ls_4758_, v___x_4767_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4886_) == 0)
{
lean_object* v_a_4887_; lean_object* v___x_4888_; 
v_a_4887_ = lean_ctor_get(v___x_4886_, 0);
lean_inc(v_a_4887_);
lean_dec_ref_known(v___x_4886_, 1);
v___x_4888_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v___f_4795_, v_a_4887_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4888_) == 0)
{
lean_object* v_a_4889_; lean_object* v___x_4890_; lean_object* v_transform_4891_; lean_object* v___x_4892_; lean_object* v___x_4893_; lean_object* v___x_4894_; 
v_a_4889_ = lean_ctor_get(v___x_4888_, 0);
lean_inc(v_a_4889_);
lean_dec_ref_known(v___x_4888_, 1);
v___x_4890_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4891_ = lean_ctor_get(v___x_4890_, 1);
v___x_4892_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1));
v___x_4893_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(v_a_4889_, v___x_4892_);
lean_inc_ref(v_transform_4891_);
lean_inc(v_a_4762_);
lean_inc_ref(v_a_4761_);
lean_inc(v_a_4760_);
lean_inc_ref(v_a_4759_);
v___x_4894_ = lean_apply_6(v_transform_4891_, v___x_4893_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, lean_box(0));
return v___x_4894_;
}
else
{
lean_object* v_a_4895_; lean_object* v___x_4897_; uint8_t v_isShared_4898_; uint8_t v_isSharedCheck_4902_; 
v_a_4895_ = lean_ctor_get(v___x_4888_, 0);
v_isSharedCheck_4902_ = !lean_is_exclusive(v___x_4888_);
if (v_isSharedCheck_4902_ == 0)
{
v___x_4897_ = v___x_4888_;
v_isShared_4898_ = v_isSharedCheck_4902_;
goto v_resetjp_4896_;
}
else
{
lean_inc(v_a_4895_);
lean_dec(v___x_4888_);
v___x_4897_ = lean_box(0);
v_isShared_4898_ = v_isSharedCheck_4902_;
goto v_resetjp_4896_;
}
v_resetjp_4896_:
{
lean_object* v___x_4900_; 
if (v_isShared_4898_ == 0)
{
v___x_4900_ = v___x_4897_;
goto v_reusejp_4899_;
}
else
{
lean_object* v_reuseFailAlloc_4901_; 
v_reuseFailAlloc_4901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4901_, 0, v_a_4895_);
v___x_4900_ = v_reuseFailAlloc_4901_;
goto v_reusejp_4899_;
}
v_reusejp_4899_:
{
return v___x_4900_;
}
}
}
}
else
{
lean_object* v_a_4903_; lean_object* v___x_4905_; uint8_t v_isShared_4906_; uint8_t v_isSharedCheck_4910_; 
v_a_4903_ = lean_ctor_get(v___x_4886_, 0);
v_isSharedCheck_4910_ = !lean_is_exclusive(v___x_4886_);
if (v_isSharedCheck_4910_ == 0)
{
v___x_4905_ = v___x_4886_;
v_isShared_4906_ = v_isSharedCheck_4910_;
goto v_resetjp_4904_;
}
else
{
lean_inc(v_a_4903_);
lean_dec(v___x_4886_);
v___x_4905_ = lean_box(0);
v_isShared_4906_ = v_isSharedCheck_4910_;
goto v_resetjp_4904_;
}
v_resetjp_4904_:
{
lean_object* v___x_4908_; 
if (v_isShared_4906_ == 0)
{
v___x_4908_ = v___x_4905_;
goto v_reusejp_4907_;
}
else
{
lean_object* v_reuseFailAlloc_4909_; 
v_reuseFailAlloc_4909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4909_, 0, v_a_4903_);
v___x_4908_ = v_reuseFailAlloc_4909_;
goto v_reusejp_4907_;
}
v_reusejp_4907_:
{
return v___x_4908_;
}
}
}
}
else
{
goto v___jp_4837_;
}
}
else
{
goto v___jp_4837_;
}
v___jp_4800_:
{
lean_object* v___x_4804_; double v___x_4805_; double v___x_4806_; double v___x_4807_; double v___x_4808_; double v___x_4809_; lean_object* v___x_4810_; lean_object* v___x_4811_; lean_object* v___x_4812_; lean_object* v___x_4813_; lean_object* v___x_4814_; 
v___x_4804_ = lean_io_mono_nanos_now();
v___x_4805_ = lean_float_of_nat(v___y_4801_);
v___x_4806_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2);
v___x_4807_ = lean_float_div(v___x_4805_, v___x_4806_);
v___x_4808_ = lean_float_of_nat(v___x_4804_);
v___x_4809_ = lean_float_div(v___x_4808_, v___x_4806_);
v___x_4810_ = lean_box_float(v___x_4807_);
v___x_4811_ = lean_box_float(v___x_4809_);
v___x_4812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4812_, 0, v___x_4810_);
lean_ctor_set(v___x_4812_, 1, v___x_4811_);
v___x_4813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4813_, 0, v_a_4803_);
lean_ctor_set(v___x_4813_, 1, v___x_4812_);
v___x_4814_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(v___x_4796_, v_hasTrace_4766_, v___x_4797_, v_options_4764_, v___x_4799_, v___y_4802_, v___f_4794_, v___x_4813_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
return v___x_4814_;
}
v___jp_4815_:
{
lean_object* v___x_4819_; 
v___x_4819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4819_, 0, v_a_4818_);
v___y_4801_ = v___y_4816_;
v___y_4802_ = v___y_4817_;
v_a_4803_ = v___x_4819_;
goto v___jp_4800_;
}
v___jp_4820_:
{
lean_object* v___x_4824_; double v___x_4825_; double v___x_4826_; lean_object* v___x_4827_; lean_object* v___x_4828_; lean_object* v___x_4829_; lean_object* v___x_4830_; lean_object* v___x_4831_; 
v___x_4824_ = lean_io_get_num_heartbeats();
v___x_4825_ = lean_float_of_nat(v___y_4822_);
v___x_4826_ = lean_float_of_nat(v___x_4824_);
v___x_4827_ = lean_box_float(v___x_4825_);
v___x_4828_ = lean_box_float(v___x_4826_);
v___x_4829_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4829_, 0, v___x_4827_);
lean_ctor_set(v___x_4829_, 1, v___x_4828_);
v___x_4830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4830_, 0, v_a_4823_);
lean_ctor_set(v___x_4830_, 1, v___x_4829_);
v___x_4831_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10(v___x_4796_, v_hasTrace_4766_, v___x_4797_, v_options_4764_, v___x_4799_, v___y_4821_, v___f_4794_, v___x_4830_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
return v___x_4831_;
}
v___jp_4832_:
{
lean_object* v___x_4836_; 
v___x_4836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4836_, 0, v_a_4835_);
v___y_4821_ = v___y_4834_;
v___y_4822_ = v___y_4833_;
v_a_4823_ = v___x_4836_;
goto v___jp_4820_;
}
v___jp_4837_:
{
lean_object* v___x_4838_; lean_object* v_a_4839_; lean_object* v___x_4840_; uint8_t v___x_4841_; 
v___x_4838_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(v_a_4762_);
v_a_4839_ = lean_ctor_get(v___x_4838_, 0);
lean_inc(v_a_4839_);
lean_dec_ref(v___x_4838_);
v___x_4840_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4841_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_4764_, v___x_4840_);
if (v___x_4841_ == 0)
{
lean_object* v___x_4842_; lean_object* v___x_4843_; 
v___x_4842_ = lean_io_mono_nanos_now();
v___x_4843_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(v_ls_4758_, v___x_4767_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4843_) == 0)
{
lean_object* v_a_4844_; lean_object* v___x_4845_; 
v_a_4844_ = lean_ctor_get(v___x_4843_, 0);
lean_inc(v_a_4844_);
lean_dec_ref_known(v___x_4843_, 1);
v___x_4845_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v___f_4795_, v_a_4844_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4845_) == 0)
{
lean_object* v_a_4846_; lean_object* v___x_4847_; lean_object* v_transform_4848_; lean_object* v___x_4849_; lean_object* v___x_4850_; lean_object* v___x_4851_; 
v_a_4846_ = lean_ctor_get(v___x_4845_, 0);
lean_inc(v_a_4846_);
lean_dec_ref_known(v___x_4845_, 1);
v___x_4847_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4848_ = lean_ctor_get(v___x_4847_, 1);
v___x_4849_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1));
v___x_4850_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(v_a_4846_, v___x_4849_);
lean_inc_ref(v_transform_4848_);
lean_inc(v_a_4762_);
lean_inc_ref(v_a_4761_);
lean_inc(v_a_4760_);
lean_inc_ref(v_a_4759_);
v___x_4851_ = lean_apply_6(v_transform_4848_, v___x_4850_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, lean_box(0));
if (lean_obj_tag(v___x_4851_) == 0)
{
lean_object* v_a_4852_; lean_object* v___x_4854_; uint8_t v_isShared_4855_; uint8_t v_isSharedCheck_4859_; 
v_a_4852_ = lean_ctor_get(v___x_4851_, 0);
v_isSharedCheck_4859_ = !lean_is_exclusive(v___x_4851_);
if (v_isSharedCheck_4859_ == 0)
{
v___x_4854_ = v___x_4851_;
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
else
{
lean_inc(v_a_4852_);
lean_dec(v___x_4851_);
v___x_4854_ = lean_box(0);
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
v_resetjp_4853_:
{
lean_object* v___x_4857_; 
if (v_isShared_4855_ == 0)
{
lean_ctor_set_tag(v___x_4854_, 1);
v___x_4857_ = v___x_4854_;
goto v_reusejp_4856_;
}
else
{
lean_object* v_reuseFailAlloc_4858_; 
v_reuseFailAlloc_4858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4858_, 0, v_a_4852_);
v___x_4857_ = v_reuseFailAlloc_4858_;
goto v_reusejp_4856_;
}
v_reusejp_4856_:
{
v___y_4801_ = v___x_4842_;
v___y_4802_ = v_a_4839_;
v_a_4803_ = v___x_4857_;
goto v___jp_4800_;
}
}
}
else
{
lean_object* v_a_4860_; 
v_a_4860_ = lean_ctor_get(v___x_4851_, 0);
lean_inc(v_a_4860_);
lean_dec_ref_known(v___x_4851_, 1);
v___y_4816_ = v___x_4842_;
v___y_4817_ = v_a_4839_;
v_a_4818_ = v_a_4860_;
goto v___jp_4815_;
}
}
else
{
lean_object* v_a_4861_; 
v_a_4861_ = lean_ctor_get(v___x_4845_, 0);
lean_inc(v_a_4861_);
lean_dec_ref_known(v___x_4845_, 1);
v___y_4816_ = v___x_4842_;
v___y_4817_ = v_a_4839_;
v_a_4818_ = v_a_4861_;
goto v___jp_4815_;
}
}
else
{
lean_object* v_a_4862_; 
v_a_4862_ = lean_ctor_get(v___x_4843_, 0);
lean_inc(v_a_4862_);
lean_dec_ref_known(v___x_4843_, 1);
v___y_4816_ = v___x_4842_;
v___y_4817_ = v_a_4839_;
v_a_4818_ = v_a_4862_;
goto v___jp_4815_;
}
}
else
{
lean_object* v___x_4863_; lean_object* v___x_4864_; 
v___x_4863_ = lean_io_get_num_heartbeats();
v___x_4864_ = lp_mathlib_List_mapM_loop___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__0(v_ls_4758_, v___x_4767_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4864_) == 0)
{
lean_object* v_a_4865_; lean_object* v___x_4866_; 
v_a_4865_ = lean_ctor_get(v___x_4864_, 0);
lean_inc(v_a_4865_);
lean_dec_ref_known(v___x_4864_, 1);
v___x_4866_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v___f_4795_, v_a_4865_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_);
if (lean_obj_tag(v___x_4866_) == 0)
{
lean_object* v_a_4867_; lean_object* v___x_4868_; lean_object* v_transform_4869_; lean_object* v___x_4870_; lean_object* v___x_4871_; lean_object* v___x_4872_; 
v_a_4867_ = lean_ctor_get(v___x_4866_, 0);
lean_inc(v_a_4867_);
lean_dec_ref_known(v___x_4866_, 1);
v___x_4868_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v_transform_4869_ = lean_ctor_get(v___x_4868_, 1);
v___x_4870_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___closed__1));
v___x_4871_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__2(v_a_4867_, v___x_4870_);
lean_inc_ref(v_transform_4869_);
lean_inc(v_a_4762_);
lean_inc_ref(v_a_4761_);
lean_inc(v_a_4760_);
lean_inc_ref(v_a_4759_);
v___x_4872_ = lean_apply_6(v_transform_4869_, v___x_4871_, v_a_4759_, v_a_4760_, v_a_4761_, v_a_4762_, lean_box(0));
if (lean_obj_tag(v___x_4872_) == 0)
{
lean_object* v_a_4873_; lean_object* v___x_4875_; uint8_t v_isShared_4876_; uint8_t v_isSharedCheck_4880_; 
v_a_4873_ = lean_ctor_get(v___x_4872_, 0);
v_isSharedCheck_4880_ = !lean_is_exclusive(v___x_4872_);
if (v_isSharedCheck_4880_ == 0)
{
v___x_4875_ = v___x_4872_;
v_isShared_4876_ = v_isSharedCheck_4880_;
goto v_resetjp_4874_;
}
else
{
lean_inc(v_a_4873_);
lean_dec(v___x_4872_);
v___x_4875_ = lean_box(0);
v_isShared_4876_ = v_isSharedCheck_4880_;
goto v_resetjp_4874_;
}
v_resetjp_4874_:
{
lean_object* v___x_4878_; 
if (v_isShared_4876_ == 0)
{
lean_ctor_set_tag(v___x_4875_, 1);
v___x_4878_ = v___x_4875_;
goto v_reusejp_4877_;
}
else
{
lean_object* v_reuseFailAlloc_4879_; 
v_reuseFailAlloc_4879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4879_, 0, v_a_4873_);
v___x_4878_ = v_reuseFailAlloc_4879_;
goto v_reusejp_4877_;
}
v_reusejp_4877_:
{
v___y_4821_ = v_a_4839_;
v___y_4822_ = v___x_4863_;
v_a_4823_ = v___x_4878_;
goto v___jp_4820_;
}
}
}
else
{
lean_object* v_a_4881_; 
v_a_4881_ = lean_ctor_get(v___x_4872_, 0);
lean_inc(v_a_4881_);
lean_dec_ref_known(v___x_4872_, 1);
v___y_4833_ = v___x_4863_;
v___y_4834_ = v_a_4839_;
v_a_4835_ = v_a_4881_;
goto v___jp_4832_;
}
}
else
{
lean_object* v_a_4882_; 
v_a_4882_ = lean_ctor_get(v___x_4866_, 0);
lean_inc(v_a_4882_);
lean_dec_ref_known(v___x_4866_, 1);
v___y_4833_ = v___x_4863_;
v___y_4834_ = v_a_4839_;
v_a_4835_ = v_a_4882_;
goto v___jp_4832_;
}
}
else
{
lean_object* v_a_4883_; 
v_a_4883_ = lean_ctor_get(v___x_4864_, 0);
lean_inc(v_a_4883_);
lean_dec_ref_known(v___x_4864_, 1);
v___y_4833_ = v___x_4863_;
v___y_4834_ = v_a_4839_;
v_a_4835_ = v_a_4883_;
goto v___jp_4832_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs___boxed(lean_object* v_ls_4911_, lean_object* v_a_4912_, lean_object* v_a_4913_, lean_object* v_a_4914_, lean_object* v_a_4915_, lean_object* v_a_4916_){
_start:
{
lean_object* v_res_4917_; 
v_res_4917_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs(v_ls_4911_, v_a_4912_, v_a_4913_, v_a_4914_, v_a_4915_);
lean_dec(v_a_4915_);
lean_dec_ref(v_a_4914_);
lean_dec(v_a_4913_);
lean_dec_ref(v_a_4912_);
return v_res_4917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1(lean_object* v_00_u03b1_4918_, lean_object* v_00_u03b2_4919_, lean_object* v_f_4920_, lean_object* v_l_4921_, lean_object* v___y_4922_, lean_object* v___y_4923_, lean_object* v___y_4924_, lean_object* v___y_4925_){
_start:
{
lean_object* v___x_4927_; 
v___x_4927_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___redArg(v_f_4920_, v_l_4921_, v___y_4922_, v___y_4923_, v___y_4924_, v___y_4925_);
return v___x_4927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1___boxed(lean_object* v_00_u03b1_4928_, lean_object* v_00_u03b2_4929_, lean_object* v_f_4930_, lean_object* v_l_4931_, lean_object* v___y_4932_, lean_object* v___y_4933_, lean_object* v___y_4934_, lean_object* v___y_4935_, lean_object* v___y_4936_){
_start:
{
lean_object* v_res_4937_; 
v_res_4937_ = lp_mathlib_List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1(v_00_u03b1_4928_, v_00_u03b2_4929_, v_f_4930_, v_l_4931_, v___y_4932_, v___y_4933_, v___y_4934_, v___y_4935_);
lean_dec(v___y_4935_);
lean_dec_ref(v___y_4934_);
lean_dec(v___y_4933_);
lean_dec_ref(v___y_4932_);
return v_res_4937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1(lean_object* v_00_u03b1_4938_, lean_object* v_00_u03b2_4939_, lean_object* v_f_4940_, lean_object* v_a_4941_, lean_object* v_a_4942_, lean_object* v___y_4943_, lean_object* v___y_4944_, lean_object* v___y_4945_, lean_object* v___y_4946_){
_start:
{
lean_object* v___x_4948_; 
v___x_4948_ = lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___redArg(v_f_4940_, v_a_4941_, v_a_4942_, v___y_4943_, v___y_4944_, v___y_4945_, v___y_4946_);
return v___x_4948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1___boxed(lean_object* v_00_u03b1_4949_, lean_object* v_00_u03b2_4950_, lean_object* v_f_4951_, lean_object* v_a_4952_, lean_object* v_a_4953_, lean_object* v___y_4954_, lean_object* v___y_4955_, lean_object* v___y_4956_, lean_object* v___y_4957_, lean_object* v___y_4958_){
_start:
{
lean_object* v_res_4959_; 
v_res_4959_ = lp_mathlib_List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1(v_00_u03b1_4949_, v_00_u03b2_4950_, v_f_4951_, v_a_4952_, v_a_4953_, v___y_4954_, v___y_4955_, v___y_4956_, v___y_4957_);
lean_dec(v___y_4957_);
lean_dec_ref(v___y_4956_);
lean_dec(v___y_4955_);
lean_dec_ref(v___y_4954_);
return v_res_4959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_4960_, lean_object* v_00_u03b1_4961_, lean_object* v_f_4962_, lean_object* v_head_4963_, lean_object* v_x_4964_, lean_object* v_x_4965_, lean_object* v___y_4966_, lean_object* v___y_4967_, lean_object* v___y_4968_, lean_object* v___y_4969_){
_start:
{
lean_object* v___x_4971_; 
v___x_4971_ = lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___redArg(v_f_4962_, v_head_4963_, v_x_4964_, v_x_4965_, v___y_4966_, v___y_4967_, v___y_4968_, v___y_4969_);
return v___x_4971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_4972_, lean_object* v_00_u03b1_4973_, lean_object* v_f_4974_, lean_object* v_head_4975_, lean_object* v_x_4976_, lean_object* v_x_4977_, lean_object* v___y_4978_, lean_object* v___y_4979_, lean_object* v___y_4980_, lean_object* v___y_4981_, lean_object* v___y_4982_){
_start:
{
lean_object* v_res_4983_; 
v_res_4983_ = lp_mathlib_List_foldlM___at___00List_mapDiagM_go___at___00List_mapDiagM___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs_spec__1_spec__1_spec__2(v_00_u03b2_4972_, v_00_u03b1_4973_, v_f_4974_, v_head_4975_, v_x_4976_, v_x_4977_, v___y_4978_, v___y_4979_, v___y_4980_, v___y_4981_);
lean_dec(v___y_4981_);
lean_dec_ref(v___y_4980_);
lean_dec(v___y_4979_);
lean_dec_ref(v___y_4978_);
return v_res_4983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0(lean_object* v_ls_4984_, lean_object* v___y_4985_, lean_object* v___y_4986_, lean_object* v___y_4987_, lean_object* v___y_4988_){
_start:
{
lean_object* v___x_4990_; 
lean_inc(v_ls_4984_);
v___x_4990_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs(v_ls_4984_, v___y_4985_, v___y_4986_, v___y_4987_, v___y_4988_);
if (lean_obj_tag(v___x_4990_) == 0)
{
lean_object* v_a_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; 
v_a_4991_ = lean_ctor_get(v___x_4990_, 0);
lean_inc(v_a_4991_);
lean_dec_ref_known(v___x_4990_, 1);
v___x_4992_ = l_List_appendTR___redArg(v_a_4991_, v_ls_4984_);
lean_inc(v___x_4992_);
v___x_4993_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetProductsProofs(v___x_4992_, v___y_4985_, v___y_4986_, v___y_4987_, v___y_4988_);
if (lean_obj_tag(v___x_4993_) == 0)
{
lean_object* v_a_4994_; lean_object* v___x_4996_; uint8_t v_isShared_4997_; uint8_t v_isSharedCheck_5002_; 
v_a_4994_ = lean_ctor_get(v___x_4993_, 0);
v_isSharedCheck_5002_ = !lean_is_exclusive(v___x_4993_);
if (v_isSharedCheck_5002_ == 0)
{
v___x_4996_ = v___x_4993_;
v_isShared_4997_ = v_isSharedCheck_5002_;
goto v_resetjp_4995_;
}
else
{
lean_inc(v_a_4994_);
lean_dec(v___x_4993_);
v___x_4996_ = lean_box(0);
v_isShared_4997_ = v_isSharedCheck_5002_;
goto v_resetjp_4995_;
}
v_resetjp_4995_:
{
lean_object* v___x_4998_; lean_object* v___x_5000_; 
v___x_4998_ = l_List_appendTR___redArg(v___x_4992_, v_a_4994_);
if (v_isShared_4997_ == 0)
{
lean_ctor_set(v___x_4996_, 0, v___x_4998_);
v___x_5000_ = v___x_4996_;
goto v_reusejp_4999_;
}
else
{
lean_object* v_reuseFailAlloc_5001_; 
v_reuseFailAlloc_5001_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5001_, 0, v___x_4998_);
v___x_5000_ = v_reuseFailAlloc_5001_;
goto v_reusejp_4999_;
}
v_reusejp_4999_:
{
return v___x_5000_;
}
}
}
else
{
lean_dec(v___x_4992_);
return v___x_4993_;
}
}
else
{
lean_dec(v_ls_4984_);
return v___x_4990_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0___boxed(lean_object* v_ls_5003_, lean_object* v___y_5004_, lean_object* v___y_5005_, lean_object* v___y_5006_, lean_object* v___y_5007_, lean_object* v___y_5008_){
_start:
{
lean_object* v_res_5009_; 
v_res_5009_ = lp_mathlib_Mathlib_Tactic_Linarith_nlinarithExtras___lam__0(v_ls_5003_, v___y_5004_, v___y_5005_, v___y_5006_, v___y_5007_);
lean_dec(v___y_5007_);
lean_dec_ref(v___y_5006_);
lean_dec(v___y_5005_);
lean_dec_ref(v___y_5004_);
return v_res_5009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(lean_object* v_mvarId_5025_, lean_object* v_x_5026_, lean_object* v___y_5027_, lean_object* v___y_5028_, lean_object* v___y_5029_, lean_object* v___y_5030_){
_start:
{
lean_object* v___x_5032_; 
v___x_5032_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_5025_, v_x_5026_, v___y_5027_, v___y_5028_, v___y_5029_, v___y_5030_);
if (lean_obj_tag(v___x_5032_) == 0)
{
lean_object* v_a_5033_; lean_object* v___x_5035_; uint8_t v_isShared_5036_; uint8_t v_isSharedCheck_5040_; 
v_a_5033_ = lean_ctor_get(v___x_5032_, 0);
v_isSharedCheck_5040_ = !lean_is_exclusive(v___x_5032_);
if (v_isSharedCheck_5040_ == 0)
{
v___x_5035_ = v___x_5032_;
v_isShared_5036_ = v_isSharedCheck_5040_;
goto v_resetjp_5034_;
}
else
{
lean_inc(v_a_5033_);
lean_dec(v___x_5032_);
v___x_5035_ = lean_box(0);
v_isShared_5036_ = v_isSharedCheck_5040_;
goto v_resetjp_5034_;
}
v_resetjp_5034_:
{
lean_object* v___x_5038_; 
if (v_isShared_5036_ == 0)
{
v___x_5038_ = v___x_5035_;
goto v_reusejp_5037_;
}
else
{
lean_object* v_reuseFailAlloc_5039_; 
v_reuseFailAlloc_5039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5039_, 0, v_a_5033_);
v___x_5038_ = v_reuseFailAlloc_5039_;
goto v_reusejp_5037_;
}
v_reusejp_5037_:
{
return v___x_5038_;
}
}
}
else
{
lean_object* v_a_5041_; lean_object* v___x_5043_; uint8_t v_isShared_5044_; uint8_t v_isSharedCheck_5048_; 
v_a_5041_ = lean_ctor_get(v___x_5032_, 0);
v_isSharedCheck_5048_ = !lean_is_exclusive(v___x_5032_);
if (v_isSharedCheck_5048_ == 0)
{
v___x_5043_ = v___x_5032_;
v_isShared_5044_ = v_isSharedCheck_5048_;
goto v_resetjp_5042_;
}
else
{
lean_inc(v_a_5041_);
lean_dec(v___x_5032_);
v___x_5043_ = lean_box(0);
v_isShared_5044_ = v_isSharedCheck_5048_;
goto v_resetjp_5042_;
}
v_resetjp_5042_:
{
lean_object* v___x_5046_; 
if (v_isShared_5044_ == 0)
{
v___x_5046_ = v___x_5043_;
goto v_reusejp_5045_;
}
else
{
lean_object* v_reuseFailAlloc_5047_; 
v_reuseFailAlloc_5047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5047_, 0, v_a_5041_);
v___x_5046_ = v_reuseFailAlloc_5047_;
goto v_reusejp_5045_;
}
v_reusejp_5045_:
{
return v___x_5046_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg___boxed(lean_object* v_mvarId_5049_, lean_object* v_x_5050_, lean_object* v___y_5051_, lean_object* v___y_5052_, lean_object* v___y_5053_, lean_object* v___y_5054_, lean_object* v___y_5055_){
_start:
{
lean_object* v_res_5056_; 
v_res_5056_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_mvarId_5049_, v_x_5050_, v___y_5051_, v___y_5052_, v___y_5053_, v___y_5054_);
lean_dec(v___y_5054_);
lean_dec_ref(v___y_5053_);
lean_dec(v___y_5052_);
lean_dec_ref(v___y_5051_);
return v_res_5056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3(lean_object* v_00_u03b1_5057_, lean_object* v_mvarId_5058_, lean_object* v_x_5059_, lean_object* v___y_5060_, lean_object* v___y_5061_, lean_object* v___y_5062_, lean_object* v___y_5063_){
_start:
{
lean_object* v___x_5065_; 
v___x_5065_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_mvarId_5058_, v_x_5059_, v___y_5060_, v___y_5061_, v___y_5062_, v___y_5063_);
return v___x_5065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___boxed(lean_object* v_00_u03b1_5066_, lean_object* v_mvarId_5067_, lean_object* v_x_5068_, lean_object* v___y_5069_, lean_object* v___y_5070_, lean_object* v___y_5071_, lean_object* v___y_5072_, lean_object* v___y_5073_){
_start:
{
lean_object* v_res_5074_; 
v_res_5074_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3(v_00_u03b1_5066_, v_mvarId_5067_, v_x_5068_, v___y_5069_, v___y_5070_, v___y_5071_, v___y_5072_);
lean_dec(v___y_5072_);
lean_dec_ref(v___y_5071_);
lean_dec(v___y_5070_);
lean_dec_ref(v___y_5069_);
return v_res_5074_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1(lean_object* v_a_5075_, lean_object* v_x_5076_){
_start:
{
if (lean_obj_tag(v_x_5076_) == 0)
{
uint8_t v___x_5077_; 
v___x_5077_ = 0;
return v___x_5077_;
}
else
{
lean_object* v_head_5078_; lean_object* v_tail_5079_; uint8_t v___x_5080_; 
v_head_5078_ = lean_ctor_get(v_x_5076_, 0);
v_tail_5079_ = lean_ctor_get(v_x_5076_, 1);
v___x_5080_ = lean_expr_eqv(v_a_5075_, v_head_5078_);
if (v___x_5080_ == 0)
{
v_x_5076_ = v_tail_5079_;
goto _start;
}
else
{
return v___x_5080_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1___boxed(lean_object* v_a_5082_, lean_object* v_x_5083_){
_start:
{
uint8_t v_res_5084_; lean_object* v_r_5085_; 
v_res_5084_ = lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1(v_a_5082_, v_x_5083_);
lean_dec(v_x_5083_);
lean_dec_ref(v_a_5082_);
v_r_5085_ = lean_box(v_res_5084_);
return v_r_5085_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0(lean_object* v_ys_5086_, lean_object* v_x_5087_){
_start:
{
uint8_t v___x_5088_; 
v___x_5088_ = lp_mathlib_List_elem___at___00List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1_spec__1(v_x_5087_, v_ys_5086_);
if (v___x_5088_ == 0)
{
uint8_t v___x_5089_; 
v___x_5089_ = 1;
return v___x_5089_;
}
else
{
uint8_t v___x_5090_; 
v___x_5090_ = 0;
return v___x_5090_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0___boxed(lean_object* v_ys_5091_, lean_object* v_x_5092_){
_start:
{
uint8_t v_res_5093_; lean_object* v_r_5094_; 
v_res_5093_ = lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0(v_ys_5091_, v_x_5092_);
lean_dec_ref(v_x_5092_);
lean_dec(v_ys_5091_);
v_r_5094_ = lean_box(v_res_5093_);
return v_r_5094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1(lean_object* v_xs_5095_, lean_object* v_ys_5096_){
_start:
{
lean_object* v___f_5097_; lean_object* v___x_5098_; 
v___f_5097_ = lean_alloc_closure((void*)(lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1___lam__0___boxed), 2, 1);
lean_closure_set(v___f_5097_, 0, v_ys_5096_);
v___x_5098_ = l_List_filter___redArg(v___f_5097_, v_xs_5095_);
return v___x_5098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0(lean_object* v_x_5102_, lean_object* v___y_5103_, lean_object* v___y_5104_, lean_object* v___y_5105_, lean_object* v___y_5106_){
_start:
{
if (lean_obj_tag(v_x_5102_) == 0)
{
lean_object* v___x_5108_; lean_object* v___x_5109_; 
v___x_5108_ = lean_box(0);
v___x_5109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5109_, 0, v___x_5108_);
return v___x_5109_;
}
else
{
lean_object* v_head_5110_; lean_object* v_tail_5111_; lean_object* v___x_5113_; uint8_t v_isShared_5114_; uint8_t v_isSharedCheck_5175_; 
v_head_5110_ = lean_ctor_get(v_x_5102_, 0);
v_tail_5111_ = lean_ctor_get(v_x_5102_, 1);
v_isSharedCheck_5175_ = !lean_is_exclusive(v_x_5102_);
if (v_isSharedCheck_5175_ == 0)
{
v___x_5113_ = v_x_5102_;
v_isShared_5114_ = v_isSharedCheck_5175_;
goto v_resetjp_5112_;
}
else
{
lean_inc(v_tail_5111_);
lean_inc(v_head_5110_);
lean_dec(v_x_5102_);
v___x_5113_ = lean_box(0);
v_isShared_5114_ = v_isSharedCheck_5175_;
goto v_resetjp_5112_;
}
v_resetjp_5112_:
{
lean_object* v___x_5115_; 
lean_inc(v___y_5106_);
lean_inc_ref(v___y_5105_);
lean_inc(v___y_5104_);
lean_inc_ref(v___y_5103_);
lean_inc(v_head_5110_);
v___x_5115_ = lean_infer_type(v_head_5110_, v___y_5103_, v___y_5104_, v___y_5105_, v___y_5106_);
if (lean_obj_tag(v___x_5115_) == 0)
{
lean_object* v_a_5116_; lean_object* v___x_5117_; lean_object* v_a_5118_; lean_object* v___x_5119_; 
v_a_5116_ = lean_ctor_get(v___x_5115_, 0);
lean_inc(v_a_5116_);
lean_dec_ref_known(v___x_5115_, 1);
v___x_5117_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_splitConjunctions_aux_spec__0___redArg(v_a_5116_, v___y_5104_);
v_a_5118_ = lean_ctor_get(v___x_5117_, 0);
lean_inc(v_a_5118_);
lean_dec_ref(v___x_5117_);
v___x_5119_ = lp_mathlib_Lean_Expr_ne_x3f_x27(v_a_5118_);
lean_dec(v_a_5118_);
if (lean_obj_tag(v___x_5119_) == 1)
{
lean_object* v_val_5120_; lean_object* v_fst_5121_; lean_object* v___x_5122_; lean_object* v___x_5123_; lean_object* v___x_5124_; lean_object* v___x_5125_; lean_object* v___x_5126_; 
v_val_5120_ = lean_ctor_get(v___x_5119_, 0);
lean_inc(v_val_5120_);
lean_dec_ref_known(v___x_5119_, 1);
v_fst_5121_ = lean_ctor_get(v_val_5120_, 0);
v___x_5122_ = ((lean_object*)(lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___closed__1));
v___x_5123_ = lean_unsigned_to_nat(1u);
v___x_5124_ = lean_mk_empty_array_with_capacity(v___x_5123_);
lean_inc(v_fst_5121_);
v___x_5125_ = lean_array_push(v___x_5124_, v_fst_5121_);
v___x_5126_ = l_Lean_Meta_mkAppM(v___x_5122_, v___x_5125_, v___y_5103_, v___y_5104_, v___y_5105_, v___y_5106_);
if (lean_obj_tag(v___x_5126_) == 0)
{
lean_object* v_a_5127_; lean_object* v___x_5128_; lean_object* v___x_5129_; 
v_a_5127_ = lean_ctor_get(v___x_5126_, 0);
lean_inc(v_a_5127_);
lean_dec_ref_known(v___x_5126_, 1);
v___x_5128_ = lean_box(0);
v___x_5129_ = l_Lean_Meta_synthInstance_x3f(v_a_5127_, v___x_5128_, v___y_5103_, v___y_5104_, v___y_5105_, v___y_5106_);
if (lean_obj_tag(v___x_5129_) == 0)
{
lean_object* v_a_5130_; lean_object* v___x_5132_; uint8_t v_isShared_5133_; uint8_t v_isSharedCheck_5149_; 
v_a_5130_ = lean_ctor_get(v___x_5129_, 0);
v_isSharedCheck_5149_ = !lean_is_exclusive(v___x_5129_);
if (v_isSharedCheck_5149_ == 0)
{
v___x_5132_ = v___x_5129_;
v_isShared_5133_ = v_isSharedCheck_5149_;
goto v_resetjp_5131_;
}
else
{
lean_inc(v_a_5130_);
lean_dec(v___x_5129_);
v___x_5132_ = lean_box(0);
v_isShared_5133_ = v_isSharedCheck_5149_;
goto v_resetjp_5131_;
}
v_resetjp_5131_:
{
if (lean_obj_tag(v_a_5130_) == 0)
{
lean_del_object(v___x_5132_);
lean_dec(v_val_5120_);
lean_del_object(v___x_5113_);
lean_dec(v_head_5110_);
v_x_5102_ = v_tail_5111_;
goto _start;
}
else
{
lean_object* v___x_5136_; uint8_t v_isShared_5137_; uint8_t v_isSharedCheck_5147_; 
lean_dec(v_tail_5111_);
v_isSharedCheck_5147_ = !lean_is_exclusive(v_a_5130_);
if (v_isSharedCheck_5147_ == 0)
{
lean_object* v_unused_5148_; 
v_unused_5148_ = lean_ctor_get(v_a_5130_, 0);
lean_dec(v_unused_5148_);
v___x_5136_ = v_a_5130_;
v_isShared_5137_ = v_isSharedCheck_5147_;
goto v_resetjp_5135_;
}
else
{
lean_dec(v_a_5130_);
v___x_5136_ = lean_box(0);
v_isShared_5137_ = v_isSharedCheck_5147_;
goto v_resetjp_5135_;
}
v_resetjp_5135_:
{
lean_object* v___x_5139_; 
if (v_isShared_5114_ == 0)
{
lean_ctor_set_tag(v___x_5113_, 0);
lean_ctor_set(v___x_5113_, 1, v_val_5120_);
v___x_5139_ = v___x_5113_;
goto v_reusejp_5138_;
}
else
{
lean_object* v_reuseFailAlloc_5146_; 
v_reuseFailAlloc_5146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5146_, 0, v_head_5110_);
lean_ctor_set(v_reuseFailAlloc_5146_, 1, v_val_5120_);
v___x_5139_ = v_reuseFailAlloc_5146_;
goto v_reusejp_5138_;
}
v_reusejp_5138_:
{
lean_object* v___x_5141_; 
if (v_isShared_5137_ == 0)
{
lean_ctor_set(v___x_5136_, 0, v___x_5139_);
v___x_5141_ = v___x_5136_;
goto v_reusejp_5140_;
}
else
{
lean_object* v_reuseFailAlloc_5145_; 
v_reuseFailAlloc_5145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5145_, 0, v___x_5139_);
v___x_5141_ = v_reuseFailAlloc_5145_;
goto v_reusejp_5140_;
}
v_reusejp_5140_:
{
lean_object* v___x_5143_; 
if (v_isShared_5133_ == 0)
{
lean_ctor_set(v___x_5132_, 0, v___x_5141_);
v___x_5143_ = v___x_5132_;
goto v_reusejp_5142_;
}
else
{
lean_object* v_reuseFailAlloc_5144_; 
v_reuseFailAlloc_5144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5144_, 0, v___x_5141_);
v___x_5143_ = v_reuseFailAlloc_5144_;
goto v_reusejp_5142_;
}
v_reusejp_5142_:
{
return v___x_5143_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_5150_; lean_object* v___x_5152_; uint8_t v_isShared_5153_; uint8_t v_isSharedCheck_5157_; 
lean_dec(v_val_5120_);
lean_del_object(v___x_5113_);
lean_dec(v_tail_5111_);
lean_dec(v_head_5110_);
v_a_5150_ = lean_ctor_get(v___x_5129_, 0);
v_isSharedCheck_5157_ = !lean_is_exclusive(v___x_5129_);
if (v_isSharedCheck_5157_ == 0)
{
v___x_5152_ = v___x_5129_;
v_isShared_5153_ = v_isSharedCheck_5157_;
goto v_resetjp_5151_;
}
else
{
lean_inc(v_a_5150_);
lean_dec(v___x_5129_);
v___x_5152_ = lean_box(0);
v_isShared_5153_ = v_isSharedCheck_5157_;
goto v_resetjp_5151_;
}
v_resetjp_5151_:
{
lean_object* v___x_5155_; 
if (v_isShared_5153_ == 0)
{
v___x_5155_ = v___x_5152_;
goto v_reusejp_5154_;
}
else
{
lean_object* v_reuseFailAlloc_5156_; 
v_reuseFailAlloc_5156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5156_, 0, v_a_5150_);
v___x_5155_ = v_reuseFailAlloc_5156_;
goto v_reusejp_5154_;
}
v_reusejp_5154_:
{
return v___x_5155_;
}
}
}
}
else
{
lean_object* v_a_5158_; lean_object* v___x_5160_; uint8_t v_isShared_5161_; uint8_t v_isSharedCheck_5165_; 
lean_dec(v_val_5120_);
lean_del_object(v___x_5113_);
lean_dec(v_tail_5111_);
lean_dec(v_head_5110_);
v_a_5158_ = lean_ctor_get(v___x_5126_, 0);
v_isSharedCheck_5165_ = !lean_is_exclusive(v___x_5126_);
if (v_isSharedCheck_5165_ == 0)
{
v___x_5160_ = v___x_5126_;
v_isShared_5161_ = v_isSharedCheck_5165_;
goto v_resetjp_5159_;
}
else
{
lean_inc(v_a_5158_);
lean_dec(v___x_5126_);
v___x_5160_ = lean_box(0);
v_isShared_5161_ = v_isSharedCheck_5165_;
goto v_resetjp_5159_;
}
v_resetjp_5159_:
{
lean_object* v___x_5163_; 
if (v_isShared_5161_ == 0)
{
v___x_5163_ = v___x_5160_;
goto v_reusejp_5162_;
}
else
{
lean_object* v_reuseFailAlloc_5164_; 
v_reuseFailAlloc_5164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5164_, 0, v_a_5158_);
v___x_5163_ = v_reuseFailAlloc_5164_;
goto v_reusejp_5162_;
}
v_reusejp_5162_:
{
return v___x_5163_;
}
}
}
}
else
{
lean_dec(v___x_5119_);
lean_del_object(v___x_5113_);
lean_dec(v_head_5110_);
v_x_5102_ = v_tail_5111_;
goto _start;
}
}
else
{
lean_object* v_a_5167_; lean_object* v___x_5169_; uint8_t v_isShared_5170_; uint8_t v_isSharedCheck_5174_; 
lean_del_object(v___x_5113_);
lean_dec(v_tail_5111_);
lean_dec(v_head_5110_);
v_a_5167_ = lean_ctor_get(v___x_5115_, 0);
v_isSharedCheck_5174_ = !lean_is_exclusive(v___x_5115_);
if (v_isSharedCheck_5174_ == 0)
{
v___x_5169_ = v___x_5115_;
v_isShared_5170_ = v_isSharedCheck_5174_;
goto v_resetjp_5168_;
}
else
{
lean_inc(v_a_5167_);
lean_dec(v___x_5115_);
v___x_5169_ = lean_box(0);
v_isShared_5170_ = v_isSharedCheck_5174_;
goto v_resetjp_5168_;
}
v_resetjp_5168_:
{
lean_object* v___x_5172_; 
if (v_isShared_5170_ == 0)
{
v___x_5172_ = v___x_5169_;
goto v_reusejp_5171_;
}
else
{
lean_object* v_reuseFailAlloc_5173_; 
v_reuseFailAlloc_5173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5173_, 0, v_a_5167_);
v___x_5172_ = v_reuseFailAlloc_5173_;
goto v_reusejp_5171_;
}
v_reusejp_5171_:
{
return v___x_5172_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0___boxed(lean_object* v_x_5176_, lean_object* v___y_5177_, lean_object* v___y_5178_, lean_object* v___y_5179_, lean_object* v___y_5180_, lean_object* v___y_5181_){
_start:
{
lean_object* v_res_5182_; 
v_res_5182_ = lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0(v_x_5176_, v___y_5177_, v___y_5178_, v___y_5179_, v___y_5180_);
lean_dec(v___y_5180_);
lean_dec_ref(v___y_5179_);
lean_dec(v___y_5178_);
lean_dec_ref(v___y_5177_);
return v_res_5182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__2(lean_object* v_fst_5183_, lean_object* v_a_5184_, lean_object* v_a_5185_){
_start:
{
if (lean_obj_tag(v_a_5184_) == 0)
{
lean_object* v___x_5186_; 
lean_dec(v_fst_5183_);
v___x_5186_ = l_List_reverse___redArg(v_a_5185_);
return v___x_5186_;
}
else
{
lean_object* v_head_5187_; lean_object* v_tail_5188_; lean_object* v___x_5190_; uint8_t v_isShared_5191_; uint8_t v_isSharedCheck_5207_; 
v_head_5187_ = lean_ctor_get(v_a_5184_, 0);
v_tail_5188_ = lean_ctor_get(v_a_5184_, 1);
v_isSharedCheck_5207_ = !lean_is_exclusive(v_a_5184_);
if (v_isSharedCheck_5207_ == 0)
{
v___x_5190_ = v_a_5184_;
v_isShared_5191_ = v_isSharedCheck_5207_;
goto v_resetjp_5189_;
}
else
{
lean_inc(v_tail_5188_);
lean_inc(v_head_5187_);
lean_dec(v_a_5184_);
v___x_5190_ = lean_box(0);
v_isShared_5191_ = v_isSharedCheck_5207_;
goto v_resetjp_5189_;
}
v_resetjp_5189_:
{
lean_object* v_fst_5192_; lean_object* v_snd_5193_; lean_object* v___x_5195_; uint8_t v_isShared_5196_; uint8_t v_isSharedCheck_5206_; 
v_fst_5192_ = lean_ctor_get(v_head_5187_, 0);
v_snd_5193_ = lean_ctor_get(v_head_5187_, 1);
v_isSharedCheck_5206_ = !lean_is_exclusive(v_head_5187_);
if (v_isSharedCheck_5206_ == 0)
{
v___x_5195_ = v_head_5187_;
v_isShared_5196_ = v_isSharedCheck_5206_;
goto v_resetjp_5194_;
}
else
{
lean_inc(v_snd_5193_);
lean_inc(v_fst_5192_);
lean_dec(v_head_5187_);
v___x_5195_ = lean_box(0);
v_isShared_5196_ = v_isSharedCheck_5206_;
goto v_resetjp_5194_;
}
v_resetjp_5194_:
{
lean_object* v___x_5197_; lean_object* v___x_5199_; 
lean_inc(v_fst_5183_);
v___x_5197_ = l_Lean_Expr_fvar___override(v_fst_5183_);
if (v_isShared_5191_ == 0)
{
lean_ctor_set(v___x_5190_, 1, v_snd_5193_);
lean_ctor_set(v___x_5190_, 0, v___x_5197_);
v___x_5199_ = v___x_5190_;
goto v_reusejp_5198_;
}
else
{
lean_object* v_reuseFailAlloc_5205_; 
v_reuseFailAlloc_5205_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5205_, 0, v___x_5197_);
lean_ctor_set(v_reuseFailAlloc_5205_, 1, v_snd_5193_);
v___x_5199_ = v_reuseFailAlloc_5205_;
goto v_reusejp_5198_;
}
v_reusejp_5198_:
{
lean_object* v___x_5201_; 
if (v_isShared_5196_ == 0)
{
lean_ctor_set(v___x_5195_, 1, v___x_5199_);
v___x_5201_ = v___x_5195_;
goto v_reusejp_5200_;
}
else
{
lean_object* v_reuseFailAlloc_5204_; 
v_reuseFailAlloc_5204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5204_, 0, v_fst_5192_);
lean_ctor_set(v_reuseFailAlloc_5204_, 1, v___x_5199_);
v___x_5201_ = v_reuseFailAlloc_5204_;
goto v_reusejp_5200_;
}
v_reusejp_5200_:
{
lean_object* v___x_5202_; 
v___x_5202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5202_, 0, v___x_5201_);
lean_ctor_set(v___x_5202_, 1, v_a_5185_);
v_a_5184_ = v_tail_5188_;
v_a_5185_ = v___x_5202_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5(void){
_start:
{
lean_object* v___x_5216_; lean_object* v___x_5217_; lean_object* v___x_5218_; lean_object* v___x_5219_; 
v___x_5216_ = lean_box(0);
v___x_5217_ = lean_unsigned_to_nat(4u);
v___x_5218_ = lean_mk_empty_array_with_capacity(v___x_5217_);
v___x_5219_ = lean_array_push(v___x_5218_, v___x_5216_);
return v___x_5219_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6(void){
_start:
{
lean_object* v___x_5220_; lean_object* v___x_5221_; lean_object* v___x_5222_; 
v___x_5220_ = lean_box(0);
v___x_5221_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__5);
v___x_5222_ = lean_array_push(v___x_5221_, v___x_5220_);
return v___x_5222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0___boxed(lean_object* v_snd_5227_, lean_object* v___x_5228_, lean_object* v_fst_5229_, lean_object* v___y_5230_, lean_object* v___y_5231_, lean_object* v___y_5232_, lean_object* v___y_5233_, lean_object* v___y_5234_){
_start:
{
lean_object* v_res_5235_; 
v_res_5235_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0(v_snd_5227_, v___x_5228_, v_fst_5229_, v___y_5230_, v___y_5231_, v___y_5232_, v___y_5233_);
lean_dec(v___y_5233_);
lean_dec_ref(v___y_5232_);
lean_dec(v___y_5231_);
lean_dec_ref(v___y_5230_);
return v_res_5235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1(lean_object* v_fst_5236_, lean_object* v_hs_5237_, lean_object* v_g_5238_, lean_object* v___y_5239_, lean_object* v___y_5240_, lean_object* v___y_5241_, lean_object* v___y_5242_){
_start:
{
uint8_t v___x_5244_; lean_object* v___x_5245_; 
v___x_5244_ = 0;
v___x_5245_ = l_Lean_Meta_intro1Core(v_g_5238_, v___x_5244_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_);
if (lean_obj_tag(v___x_5245_) == 0)
{
lean_object* v_a_5246_; lean_object* v_fst_5247_; lean_object* v_snd_5248_; lean_object* v___x_5250_; uint8_t v_isShared_5251_; uint8_t v_isSharedCheck_5259_; 
v_a_5246_ = lean_ctor_get(v___x_5245_, 0);
lean_inc(v_a_5246_);
lean_dec_ref_known(v___x_5245_, 1);
v_fst_5247_ = lean_ctor_get(v_a_5246_, 0);
v_snd_5248_ = lean_ctor_get(v_a_5246_, 1);
v_isSharedCheck_5259_ = !lean_is_exclusive(v_a_5246_);
if (v_isSharedCheck_5259_ == 0)
{
v___x_5250_ = v_a_5246_;
v_isShared_5251_ = v_isSharedCheck_5259_;
goto v_resetjp_5249_;
}
else
{
lean_inc(v_snd_5248_);
lean_inc(v_fst_5247_);
lean_dec(v_a_5246_);
v___x_5250_ = lean_box(0);
v_isShared_5251_ = v_isSharedCheck_5259_;
goto v_resetjp_5249_;
}
v_resetjp_5249_:
{
lean_object* v___x_5252_; lean_object* v___x_5254_; 
v___x_5252_ = lean_box(0);
if (v_isShared_5251_ == 0)
{
lean_ctor_set_tag(v___x_5250_, 1);
lean_ctor_set(v___x_5250_, 1, v___x_5252_);
lean_ctor_set(v___x_5250_, 0, v_fst_5236_);
v___x_5254_ = v___x_5250_;
goto v_reusejp_5253_;
}
else
{
lean_object* v_reuseFailAlloc_5258_; 
v_reuseFailAlloc_5258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5258_, 0, v_fst_5236_);
lean_ctor_set(v_reuseFailAlloc_5258_, 1, v___x_5252_);
v___x_5254_ = v_reuseFailAlloc_5258_;
goto v_reusejp_5253_;
}
v_reusejp_5253_:
{
lean_object* v___x_5255_; lean_object* v___f_5256_; lean_object* v___x_5257_; 
v___x_5255_ = lp_mathlib_List_removeAll___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__1(v_hs_5237_, v___x_5254_);
lean_inc(v_snd_5248_);
v___f_5256_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0___boxed), 8, 3);
lean_closure_set(v___f_5256_, 0, v_snd_5248_);
lean_closure_set(v___f_5256_, 1, v___x_5255_);
lean_closure_set(v___f_5256_, 2, v_fst_5247_);
v___x_5257_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_snd_5248_, v___f_5256_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_);
return v___x_5257_;
}
}
}
else
{
lean_object* v_a_5260_; lean_object* v___x_5262_; uint8_t v_isShared_5263_; uint8_t v_isSharedCheck_5267_; 
lean_dec(v_hs_5237_);
lean_dec_ref(v_fst_5236_);
v_a_5260_ = lean_ctor_get(v___x_5245_, 0);
v_isSharedCheck_5267_ = !lean_is_exclusive(v___x_5245_);
if (v_isSharedCheck_5267_ == 0)
{
v___x_5262_ = v___x_5245_;
v_isShared_5263_ = v_isSharedCheck_5267_;
goto v_resetjp_5261_;
}
else
{
lean_inc(v_a_5260_);
lean_dec(v___x_5245_);
v___x_5262_ = lean_box(0);
v_isShared_5263_ = v_isSharedCheck_5267_;
goto v_resetjp_5261_;
}
v_resetjp_5261_:
{
lean_object* v___x_5265_; 
if (v_isShared_5263_ == 0)
{
v___x_5265_ = v___x_5262_;
goto v_reusejp_5264_;
}
else
{
lean_object* v_reuseFailAlloc_5266_; 
v_reuseFailAlloc_5266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5266_, 0, v_a_5260_);
v___x_5265_ = v_reuseFailAlloc_5266_;
goto v_reusejp_5264_;
}
v_reusejp_5264_:
{
return v___x_5265_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux(lean_object* v_g_5268_, lean_object* v_hs_5269_, lean_object* v_a_5270_, lean_object* v_a_5271_, lean_object* v_a_5272_, lean_object* v_a_5273_){
_start:
{
lean_object* v___x_5275_; 
lean_inc(v_hs_5269_);
v___x_5275_ = lp_mathlib_List_findSomeM_x3f___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__0(v_hs_5269_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5275_) == 0)
{
lean_object* v_a_5276_; lean_object* v___x_5278_; uint8_t v_isShared_5279_; uint8_t v_isSharedCheck_5382_; 
v_a_5276_ = lean_ctor_get(v___x_5275_, 0);
v_isSharedCheck_5382_ = !lean_is_exclusive(v___x_5275_);
if (v_isSharedCheck_5382_ == 0)
{
v___x_5278_ = v___x_5275_;
v_isShared_5279_ = v_isSharedCheck_5382_;
goto v_resetjp_5277_;
}
else
{
lean_inc(v_a_5276_);
lean_dec(v___x_5275_);
v___x_5278_ = lean_box(0);
v_isShared_5279_ = v_isSharedCheck_5382_;
goto v_resetjp_5277_;
}
v_resetjp_5277_:
{
if (lean_obj_tag(v_a_5276_) == 1)
{
lean_object* v_val_5280_; lean_object* v___x_5282_; uint8_t v_isShared_5283_; uint8_t v_isSharedCheck_5375_; 
lean_del_object(v___x_5278_);
v_val_5280_ = lean_ctor_get(v_a_5276_, 0);
v_isSharedCheck_5375_ = !lean_is_exclusive(v_a_5276_);
if (v_isSharedCheck_5375_ == 0)
{
v___x_5282_ = v_a_5276_;
v_isShared_5283_ = v_isSharedCheck_5375_;
goto v_resetjp_5281_;
}
else
{
lean_inc(v_val_5280_);
lean_dec(v_a_5276_);
v___x_5282_ = lean_box(0);
v_isShared_5283_ = v_isSharedCheck_5375_;
goto v_resetjp_5281_;
}
v_resetjp_5281_:
{
lean_object* v_snd_5284_; lean_object* v_snd_5285_; lean_object* v_fst_5286_; lean_object* v_fst_5287_; lean_object* v_fst_5288_; lean_object* v_snd_5289_; lean_object* v___x_5290_; 
v_snd_5284_ = lean_ctor_get(v_val_5280_, 1);
lean_inc(v_snd_5284_);
v_snd_5285_ = lean_ctor_get(v_snd_5284_, 1);
lean_inc(v_snd_5285_);
v_fst_5286_ = lean_ctor_get(v_val_5280_, 0);
lean_inc(v_fst_5286_);
lean_dec(v_val_5280_);
v_fst_5287_ = lean_ctor_get(v_snd_5284_, 0);
lean_inc(v_fst_5287_);
lean_dec(v_snd_5284_);
v_fst_5288_ = lean_ctor_get(v_snd_5285_, 0);
lean_inc(v_fst_5288_);
v_snd_5289_ = lean_ctor_get(v_snd_5285_, 1);
lean_inc(v_snd_5289_);
lean_dec(v_snd_5285_);
lean_inc(v_g_5268_);
v___x_5290_ = l_Lean_MVarId_getType(v_g_5268_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5290_) == 0)
{
lean_object* v_a_5291_; lean_object* v___x_5292_; lean_object* v___x_5294_; 
v_a_5291_ = lean_ctor_get(v___x_5290_, 0);
lean_inc(v_a_5291_);
lean_dec_ref_known(v___x_5290_, 1);
v___x_5292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__1));
if (v_isShared_5283_ == 0)
{
lean_ctor_set(v___x_5282_, 0, v_fst_5287_);
v___x_5294_ = v___x_5282_;
goto v_reusejp_5293_;
}
else
{
lean_object* v_reuseFailAlloc_5366_; 
v_reuseFailAlloc_5366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5366_, 0, v_fst_5287_);
v___x_5294_ = v_reuseFailAlloc_5366_;
goto v_reusejp_5293_;
}
v_reusejp_5293_:
{
lean_object* v___x_5295_; lean_object* v___x_5296_; lean_object* v___x_5297_; lean_object* v___x_5298_; lean_object* v___x_5299_; lean_object* v___x_5300_; lean_object* v___x_5301_; lean_object* v___x_5302_; lean_object* v___x_5303_; lean_object* v___x_5304_; lean_object* v___x_5305_; lean_object* v___x_5306_; 
v___x_5295_ = lean_box(0);
v___x_5296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5296_, 0, v_fst_5288_);
v___x_5297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5297_, 0, v_snd_5289_);
lean_inc(v_fst_5286_);
v___x_5298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5298_, 0, v_fst_5286_);
v___x_5299_ = lean_unsigned_to_nat(5u);
v___x_5300_ = lean_mk_empty_array_with_capacity(v___x_5299_);
v___x_5301_ = lean_array_push(v___x_5300_, v___x_5294_);
v___x_5302_ = lean_array_push(v___x_5301_, v___x_5295_);
v___x_5303_ = lean_array_push(v___x_5302_, v___x_5296_);
v___x_5304_ = lean_array_push(v___x_5303_, v___x_5297_);
v___x_5305_ = lean_array_push(v___x_5304_, v___x_5298_);
v___x_5306_ = l_Lean_Meta_mkAppOptM(v___x_5292_, v___x_5305_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5306_) == 0)
{
lean_object* v_a_5307_; lean_object* v___x_5308_; lean_object* v___x_5309_; lean_object* v___x_5310_; lean_object* v___x_5311_; lean_object* v___x_5312_; lean_object* v___x_5313_; lean_object* v___x_5314_; 
v_a_5307_ = lean_ctor_get(v___x_5306_, 0);
lean_inc(v_a_5307_);
lean_dec_ref_known(v___x_5306_, 1);
v___x_5308_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__4));
v___x_5309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5309_, 0, v_a_5291_);
v___x_5310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5310_, 0, v_a_5307_);
v___x_5311_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__6);
v___x_5312_ = lean_array_push(v___x_5311_, v___x_5309_);
v___x_5313_ = lean_array_push(v___x_5312_, v___x_5310_);
v___x_5314_ = l_Lean_Meta_mkAppOptM(v___x_5308_, v___x_5313_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5314_) == 0)
{
lean_object* v_a_5315_; lean_object* v___x_5316_; lean_object* v___x_5317_; 
v_a_5315_ = lean_ctor_get(v___x_5314_, 0);
lean_inc(v_a_5315_);
lean_dec_ref_known(v___x_5314_, 1);
v___x_5316_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___closed__7));
v___x_5317_ = l_Lean_MVarId_apply(v_g_5268_, v_a_5315_, v___x_5316_, v___x_5295_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5317_) == 0)
{
lean_object* v_a_5318_; lean_object* v___y_5320_; lean_object* v___y_5321_; lean_object* v___y_5322_; lean_object* v___y_5323_; 
v_a_5318_ = lean_ctor_get(v___x_5317_, 0);
lean_inc(v_a_5318_);
lean_dec_ref_known(v___x_5317_, 1);
if (lean_obj_tag(v_a_5318_) == 1)
{
lean_object* v_tail_5326_; 
v_tail_5326_ = lean_ctor_get(v_a_5318_, 1);
lean_inc(v_tail_5326_);
if (lean_obj_tag(v_tail_5326_) == 1)
{
lean_object* v_tail_5327_; 
v_tail_5327_ = lean_ctor_get(v_tail_5326_, 1);
if (lean_obj_tag(v_tail_5327_) == 0)
{
lean_object* v_head_5328_; lean_object* v_head_5329_; lean_object* v___x_5330_; 
v_head_5328_ = lean_ctor_get(v_a_5318_, 0);
lean_inc(v_head_5328_);
lean_dec_ref_known(v_a_5318_, 2);
v_head_5329_ = lean_ctor_get(v_tail_5326_, 0);
lean_inc(v_head_5329_);
lean_dec_ref_known(v_tail_5326_, 2);
lean_inc(v_hs_5269_);
lean_inc(v_fst_5286_);
v___x_5330_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1(v_fst_5286_, v_hs_5269_, v_head_5328_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5330_) == 0)
{
lean_object* v_a_5331_; lean_object* v___x_5332_; 
v_a_5331_ = lean_ctor_get(v___x_5330_, 0);
lean_inc(v_a_5331_);
lean_dec_ref_known(v___x_5330_, 1);
v___x_5332_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1(v_fst_5286_, v_hs_5269_, v_head_5329_, v_a_5270_, v_a_5271_, v_a_5272_, v_a_5273_);
if (lean_obj_tag(v___x_5332_) == 0)
{
lean_object* v_a_5333_; lean_object* v___x_5335_; uint8_t v_isShared_5336_; uint8_t v_isSharedCheck_5341_; 
v_a_5333_ = lean_ctor_get(v___x_5332_, 0);
v_isSharedCheck_5341_ = !lean_is_exclusive(v___x_5332_);
if (v_isSharedCheck_5341_ == 0)
{
v___x_5335_ = v___x_5332_;
v_isShared_5336_ = v_isSharedCheck_5341_;
goto v_resetjp_5334_;
}
else
{
lean_inc(v_a_5333_);
lean_dec(v___x_5332_);
v___x_5335_ = lean_box(0);
v_isShared_5336_ = v_isSharedCheck_5341_;
goto v_resetjp_5334_;
}
v_resetjp_5334_:
{
lean_object* v___x_5337_; lean_object* v___x_5339_; 
v___x_5337_ = l_List_appendTR___redArg(v_a_5331_, v_a_5333_);
if (v_isShared_5336_ == 0)
{
lean_ctor_set(v___x_5335_, 0, v___x_5337_);
v___x_5339_ = v___x_5335_;
goto v_reusejp_5338_;
}
else
{
lean_object* v_reuseFailAlloc_5340_; 
v_reuseFailAlloc_5340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5340_, 0, v___x_5337_);
v___x_5339_ = v_reuseFailAlloc_5340_;
goto v_reusejp_5338_;
}
v_reusejp_5338_:
{
return v___x_5339_;
}
}
}
else
{
lean_dec(v_a_5331_);
return v___x_5332_;
}
}
else
{
lean_dec(v_head_5329_);
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
return v___x_5330_;
}
}
else
{
lean_dec_ref_known(v_tail_5326_, 2);
lean_dec_ref_known(v_a_5318_, 2);
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
v___y_5320_ = v_a_5270_;
v___y_5321_ = v_a_5271_;
v___y_5322_ = v_a_5272_;
v___y_5323_ = v_a_5273_;
goto v___jp_5319_;
}
}
else
{
lean_dec(v_tail_5326_);
lean_dec_ref_known(v_a_5318_, 2);
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
v___y_5320_ = v_a_5270_;
v___y_5321_ = v_a_5271_;
v___y_5322_ = v_a_5272_;
v___y_5323_ = v_a_5273_;
goto v___jp_5319_;
}
}
else
{
lean_dec(v_a_5318_);
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
v___y_5320_ = v_a_5270_;
v___y_5321_ = v_a_5271_;
v___y_5322_ = v_a_5272_;
v___y_5323_ = v_a_5273_;
goto v___jp_5319_;
}
v___jp_5319_:
{
lean_object* v___x_5324_; lean_object* v___x_5325_; 
v___x_5324_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_isNatProp___lam__0___closed__1);
v___x_5325_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_isNatProp_spec__0___redArg(v___x_5324_, v___y_5320_, v___y_5321_, v___y_5322_, v___y_5323_);
return v___x_5325_;
}
}
else
{
lean_object* v_a_5342_; lean_object* v___x_5344_; uint8_t v_isShared_5345_; uint8_t v_isSharedCheck_5349_; 
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
v_a_5342_ = lean_ctor_get(v___x_5317_, 0);
v_isSharedCheck_5349_ = !lean_is_exclusive(v___x_5317_);
if (v_isSharedCheck_5349_ == 0)
{
v___x_5344_ = v___x_5317_;
v_isShared_5345_ = v_isSharedCheck_5349_;
goto v_resetjp_5343_;
}
else
{
lean_inc(v_a_5342_);
lean_dec(v___x_5317_);
v___x_5344_ = lean_box(0);
v_isShared_5345_ = v_isSharedCheck_5349_;
goto v_resetjp_5343_;
}
v_resetjp_5343_:
{
lean_object* v___x_5347_; 
if (v_isShared_5345_ == 0)
{
v___x_5347_ = v___x_5344_;
goto v_reusejp_5346_;
}
else
{
lean_object* v_reuseFailAlloc_5348_; 
v_reuseFailAlloc_5348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5348_, 0, v_a_5342_);
v___x_5347_ = v_reuseFailAlloc_5348_;
goto v_reusejp_5346_;
}
v_reusejp_5346_:
{
return v___x_5347_;
}
}
}
}
else
{
lean_object* v_a_5350_; lean_object* v___x_5352_; uint8_t v_isShared_5353_; uint8_t v_isSharedCheck_5357_; 
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
lean_dec(v_g_5268_);
v_a_5350_ = lean_ctor_get(v___x_5314_, 0);
v_isSharedCheck_5357_ = !lean_is_exclusive(v___x_5314_);
if (v_isSharedCheck_5357_ == 0)
{
v___x_5352_ = v___x_5314_;
v_isShared_5353_ = v_isSharedCheck_5357_;
goto v_resetjp_5351_;
}
else
{
lean_inc(v_a_5350_);
lean_dec(v___x_5314_);
v___x_5352_ = lean_box(0);
v_isShared_5353_ = v_isSharedCheck_5357_;
goto v_resetjp_5351_;
}
v_resetjp_5351_:
{
lean_object* v___x_5355_; 
if (v_isShared_5353_ == 0)
{
v___x_5355_ = v___x_5352_;
goto v_reusejp_5354_;
}
else
{
lean_object* v_reuseFailAlloc_5356_; 
v_reuseFailAlloc_5356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5356_, 0, v_a_5350_);
v___x_5355_ = v_reuseFailAlloc_5356_;
goto v_reusejp_5354_;
}
v_reusejp_5354_:
{
return v___x_5355_;
}
}
}
}
else
{
lean_object* v_a_5358_; lean_object* v___x_5360_; uint8_t v_isShared_5361_; uint8_t v_isSharedCheck_5365_; 
lean_dec(v_a_5291_);
lean_dec(v_fst_5286_);
lean_dec(v_hs_5269_);
lean_dec(v_g_5268_);
v_a_5358_ = lean_ctor_get(v___x_5306_, 0);
v_isSharedCheck_5365_ = !lean_is_exclusive(v___x_5306_);
if (v_isSharedCheck_5365_ == 0)
{
v___x_5360_ = v___x_5306_;
v_isShared_5361_ = v_isSharedCheck_5365_;
goto v_resetjp_5359_;
}
else
{
lean_inc(v_a_5358_);
lean_dec(v___x_5306_);
v___x_5360_ = lean_box(0);
v_isShared_5361_ = v_isSharedCheck_5365_;
goto v_resetjp_5359_;
}
v_resetjp_5359_:
{
lean_object* v___x_5363_; 
if (v_isShared_5361_ == 0)
{
v___x_5363_ = v___x_5360_;
goto v_reusejp_5362_;
}
else
{
lean_object* v_reuseFailAlloc_5364_; 
v_reuseFailAlloc_5364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5364_, 0, v_a_5358_);
v___x_5363_ = v_reuseFailAlloc_5364_;
goto v_reusejp_5362_;
}
v_reusejp_5362_:
{
return v___x_5363_;
}
}
}
}
}
else
{
lean_object* v_a_5367_; lean_object* v___x_5369_; uint8_t v_isShared_5370_; uint8_t v_isSharedCheck_5374_; 
lean_dec(v_snd_5289_);
lean_dec(v_fst_5288_);
lean_dec(v_fst_5287_);
lean_dec(v_fst_5286_);
lean_del_object(v___x_5282_);
lean_dec(v_hs_5269_);
lean_dec(v_g_5268_);
v_a_5367_ = lean_ctor_get(v___x_5290_, 0);
v_isSharedCheck_5374_ = !lean_is_exclusive(v___x_5290_);
if (v_isSharedCheck_5374_ == 0)
{
v___x_5369_ = v___x_5290_;
v_isShared_5370_ = v_isSharedCheck_5374_;
goto v_resetjp_5368_;
}
else
{
lean_inc(v_a_5367_);
lean_dec(v___x_5290_);
v___x_5369_ = lean_box(0);
v_isShared_5370_ = v_isSharedCheck_5374_;
goto v_resetjp_5368_;
}
v_resetjp_5368_:
{
lean_object* v___x_5372_; 
if (v_isShared_5370_ == 0)
{
v___x_5372_ = v___x_5369_;
goto v_reusejp_5371_;
}
else
{
lean_object* v_reuseFailAlloc_5373_; 
v_reuseFailAlloc_5373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5373_, 0, v_a_5367_);
v___x_5372_ = v_reuseFailAlloc_5373_;
goto v_reusejp_5371_;
}
v_reusejp_5371_:
{
return v___x_5372_;
}
}
}
}
}
else
{
lean_object* v___x_5376_; lean_object* v___x_5377_; lean_object* v___x_5378_; lean_object* v___x_5380_; 
lean_dec(v_a_5276_);
v___x_5376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5376_, 0, v_g_5268_);
lean_ctor_set(v___x_5376_, 1, v_hs_5269_);
v___x_5377_ = lean_box(0);
v___x_5378_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5378_, 0, v___x_5376_);
lean_ctor_set(v___x_5378_, 1, v___x_5377_);
if (v_isShared_5279_ == 0)
{
lean_ctor_set(v___x_5278_, 0, v___x_5378_);
v___x_5380_ = v___x_5278_;
goto v_reusejp_5379_;
}
else
{
lean_object* v_reuseFailAlloc_5381_; 
v_reuseFailAlloc_5381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5381_, 0, v___x_5378_);
v___x_5380_ = v_reuseFailAlloc_5381_;
goto v_reusejp_5379_;
}
v_reusejp_5379_:
{
return v___x_5380_;
}
}
}
}
else
{
lean_object* v_a_5383_; lean_object* v___x_5385_; uint8_t v_isShared_5386_; uint8_t v_isSharedCheck_5390_; 
lean_dec(v_hs_5269_);
lean_dec(v_g_5268_);
v_a_5383_ = lean_ctor_get(v___x_5275_, 0);
v_isSharedCheck_5390_ = !lean_is_exclusive(v___x_5275_);
if (v_isSharedCheck_5390_ == 0)
{
v___x_5385_ = v___x_5275_;
v_isShared_5386_ = v_isSharedCheck_5390_;
goto v_resetjp_5384_;
}
else
{
lean_inc(v_a_5383_);
lean_dec(v___x_5275_);
v___x_5385_ = lean_box(0);
v_isShared_5386_ = v_isSharedCheck_5390_;
goto v_resetjp_5384_;
}
v_resetjp_5384_:
{
lean_object* v___x_5388_; 
if (v_isShared_5386_ == 0)
{
v___x_5388_ = v___x_5385_;
goto v_reusejp_5387_;
}
else
{
lean_object* v_reuseFailAlloc_5389_; 
v_reuseFailAlloc_5389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5389_, 0, v_a_5383_);
v___x_5388_ = v_reuseFailAlloc_5389_;
goto v_reusejp_5387_;
}
v_reusejp_5387_:
{
return v___x_5388_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__0(lean_object* v_snd_5391_, lean_object* v___x_5392_, lean_object* v_fst_5393_, lean_object* v___y_5394_, lean_object* v___y_5395_, lean_object* v___y_5396_, lean_object* v___y_5397_){
_start:
{
lean_object* v___x_5399_; 
v___x_5399_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux(v_snd_5391_, v___x_5392_, v___y_5394_, v___y_5395_, v___y_5396_, v___y_5397_);
if (lean_obj_tag(v___x_5399_) == 0)
{
lean_object* v_a_5400_; lean_object* v___x_5402_; uint8_t v_isShared_5403_; uint8_t v_isSharedCheck_5409_; 
v_a_5400_ = lean_ctor_get(v___x_5399_, 0);
v_isSharedCheck_5409_ = !lean_is_exclusive(v___x_5399_);
if (v_isSharedCheck_5409_ == 0)
{
v___x_5402_ = v___x_5399_;
v_isShared_5403_ = v_isSharedCheck_5409_;
goto v_resetjp_5401_;
}
else
{
lean_inc(v_a_5400_);
lean_dec(v___x_5399_);
v___x_5402_ = lean_box(0);
v_isShared_5403_ = v_isSharedCheck_5409_;
goto v_resetjp_5401_;
}
v_resetjp_5401_:
{
lean_object* v___x_5404_; lean_object* v___x_5405_; lean_object* v___x_5407_; 
v___x_5404_ = lean_box(0);
v___x_5405_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__2(v_fst_5393_, v_a_5400_, v___x_5404_);
if (v_isShared_5403_ == 0)
{
lean_ctor_set(v___x_5402_, 0, v___x_5405_);
v___x_5407_ = v___x_5402_;
goto v_reusejp_5406_;
}
else
{
lean_object* v_reuseFailAlloc_5408_; 
v_reuseFailAlloc_5408_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5408_, 0, v___x_5405_);
v___x_5407_ = v_reuseFailAlloc_5408_;
goto v_reusejp_5406_;
}
v_reusejp_5406_:
{
return v___x_5407_;
}
}
}
else
{
lean_dec(v_fst_5393_);
return v___x_5399_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1___boxed(lean_object* v_fst_5410_, lean_object* v_hs_5411_, lean_object* v_g_5412_, lean_object* v___y_5413_, lean_object* v___y_5414_, lean_object* v___y_5415_, lean_object* v___y_5416_, lean_object* v___y_5417_){
_start:
{
lean_object* v_res_5418_; 
v_res_5418_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___lam__1(v_fst_5410_, v_hs_5411_, v_g_5412_, v___y_5413_, v___y_5414_, v___y_5415_, v___y_5416_);
lean_dec(v___y_5416_);
lean_dec_ref(v___y_5415_);
lean_dec(v___y_5414_);
lean_dec_ref(v___y_5413_);
return v_res_5418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux___boxed(lean_object* v_g_5419_, lean_object* v_hs_5420_, lean_object* v_a_5421_, lean_object* v_a_5422_, lean_object* v_a_5423_, lean_object* v_a_5424_, lean_object* v_a_5425_){
_start:
{
lean_object* v_res_5426_; 
v_res_5426_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux(v_g_5419_, v_hs_5420_, v_a_5421_, v_a_5422_, v_a_5423_, v_a_5424_);
lean_dec(v_a_5424_);
lean_dec_ref(v_a_5423_);
lean_dec(v_a_5422_);
lean_dec_ref(v_a_5421_);
return v_res_5426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe__aux(lean_object* v_a_5427_, lean_object* v_a_5428_, lean_object* v_a_5429_, lean_object* v_a_5430_, lean_object* v_a_5431_, lean_object* v_a_5432_){
_start:
{
lean_object* v___x_5434_; 
v___x_5434_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNeAux(v_a_5427_, v_a_5428_, v_a_5429_, v_a_5430_, v_a_5431_, v_a_5432_);
return v___x_5434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_removeNe__aux___boxed(lean_object* v_a_5435_, lean_object* v_a_5436_, lean_object* v_a_5437_, lean_object* v_a_5438_, lean_object* v_a_5439_, lean_object* v_a_5440_, lean_object* v_a_5441_){
_start:
{
lean_object* v_res_5442_; 
v_res_5442_ = lp_mathlib_Mathlib_Tactic_Linarith_removeNe__aux(v_a_5435_, v_a_5436_, v_a_5437_, v_a_5438_, v_a_5439_, v_a_5440_);
lean_dec(v_a_5440_);
lean_dec_ref(v_a_5439_);
lean_dec(v_a_5438_);
lean_dec_ref(v_a_5437_);
return v_res_5442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_(lean_object* v___y_5458_, lean_object* v___y_5459_, lean_object* v___y_5460_, lean_object* v___y_5461_, lean_object* v___y_5462_){
_start:
{
lean_object* v___x_5464_; 
v___x_5464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5464_, 0, v___y_5458_);
return v___x_5464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2____boxed(lean_object* v___y_5465_, lean_object* v___y_5466_, lean_object* v___y_5467_, lean_object* v___y_5468_, lean_object* v___y_5469_, lean_object* v___y_5470_){
_start:
{
lean_object* v_res_5471_; 
v_res_5471_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___lam__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_(v___y_5465_, v___y_5466_, v___y_5467_, v___y_5468_, v___y_5469_);
lean_dec(v___y_5469_);
lean_dec_ref(v___y_5468_);
lean_dec(v___y_5467_);
lean_dec_ref(v___y_5466_);
return v_res_5471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_5474_; lean_object* v___x_5475_; lean_object* v___x_5476_; 
v___f_5474_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_));
v___x_5475_ = lean_st_mk_ref(v___f_5474_);
v___x_5476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5476_, 0, v___x_5475_);
return v___x_5476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2____boxed(lean_object* v_a_5477_){
_start:
{
lean_object* v_res_5478_; 
v_res_5478_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_();
return v_res_5478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0(lean_object* v_l_5479_, lean_object* v___y_5480_, lean_object* v___y_5481_, lean_object* v___y_5482_, lean_object* v___y_5483_){
_start:
{
lean_object* v___x_5485_; lean_object* v___x_5486_; lean_object* v___x_5487_; 
v___x_5485_ = lp_mathlib_Mathlib_Tactic_Linarith_nnrealToRealTransform;
v___x_5486_ = lean_st_ref_get(v___x_5485_);
lean_inc(v___y_5483_);
lean_inc_ref(v___y_5482_);
lean_inc(v___y_5481_);
lean_inc_ref(v___y_5480_);
v___x_5487_ = lean_apply_6(v___x_5486_, v_l_5479_, v___y_5480_, v___y_5481_, v___y_5482_, v___y_5483_, lean_box(0));
return v___x_5487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0___boxed(lean_object* v_l_5488_, lean_object* v___y_5489_, lean_object* v___y_5490_, lean_object* v___y_5491_, lean_object* v___y_5492_, lean_object* v___y_5493_){
_start:
{
lean_object* v_res_5494_; 
v_res_5494_ = lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal___lam__0(v_l_5488_, v___y_5489_, v___y_5490_, v___y_5491_, v___y_5492_);
lean_dec(v___y_5492_);
lean_dec_ref(v___y_5491_);
lean_dec(v___y_5490_);
lean_dec_ref(v___y_5489_);
return v_res_5494_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0(void){
_start:
{
lean_object* v___x_5510_; lean_object* v___x_5511_; 
v___x_5510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_filterComparisons));
v___x_5511_ = lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(v___x_5510_);
return v___x_5511_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1(void){
_start:
{
lean_object* v___x_5512_; lean_object* v___x_5513_; 
v___x_5512_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__0);
v___x_5513_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(v___x_5512_);
return v___x_5513_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2(void){
_start:
{
lean_object* v___x_5514_; lean_object* v___x_5515_; 
v___x_5514_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_nnrealToReal));
v___x_5515_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(v___x_5514_);
return v___x_5515_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3(void){
_start:
{
lean_object* v___x_5516_; lean_object* v___x_5517_; 
v___x_5516_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_strengthenStrictInt));
v___x_5517_ = lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(v___x_5516_);
return v___x_5517_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4(void){
_start:
{
lean_object* v___x_5518_; lean_object* v___x_5519_; 
v___x_5518_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__3);
v___x_5519_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(v___x_5518_);
return v___x_5519_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5(void){
_start:
{
lean_object* v___x_5520_; lean_object* v___x_5521_; 
v___x_5520_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__0);
v___x_5521_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(v___x_5520_);
return v___x_5521_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6(void){
_start:
{
lean_object* v___x_5522_; lean_object* v___x_5523_; 
v___x_5522_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_cancelDenoms));
v___x_5523_ = lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(v___x_5522_);
return v___x_5523_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7(void){
_start:
{
lean_object* v___x_5524_; lean_object* v___x_5525_; 
v___x_5524_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__6);
v___x_5525_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(v___x_5524_);
return v___x_5525_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8(void){
_start:
{
lean_object* v___x_5526_; lean_object* v___x_5527_; lean_object* v___x_5528_; 
v___x_5526_ = lean_box(0);
v___x_5527_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__7);
v___x_5528_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5528_, 0, v___x_5527_);
lean_ctor_set(v___x_5528_, 1, v___x_5526_);
return v___x_5528_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9(void){
_start:
{
lean_object* v___x_5529_; lean_object* v___x_5530_; lean_object* v___x_5531_; 
v___x_5529_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__8);
v___x_5530_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__5);
v___x_5531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5531_, 0, v___x_5530_);
lean_ctor_set(v___x_5531_, 1, v___x_5529_);
return v___x_5531_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10(void){
_start:
{
lean_object* v___x_5532_; lean_object* v___x_5533_; lean_object* v___x_5534_; 
v___x_5532_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__9);
v___x_5533_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__4);
v___x_5534_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5534_, 0, v___x_5533_);
lean_ctor_set(v___x_5534_, 1, v___x_5532_);
return v___x_5534_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11(void){
_start:
{
lean_object* v___x_5535_; lean_object* v___x_5536_; lean_object* v___x_5537_; 
v___x_5535_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__10);
v___x_5536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_natToInt));
v___x_5537_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5537_, 0, v___x_5536_);
lean_ctor_set(v___x_5537_, 1, v___x_5535_);
return v___x_5537_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12(void){
_start:
{
lean_object* v___x_5538_; lean_object* v___x_5539_; lean_object* v___x_5540_; 
v___x_5538_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__11);
v___x_5539_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__2);
v___x_5540_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5540_, 0, v___x_5539_);
lean_ctor_set(v___x_5540_, 1, v___x_5538_);
return v___x_5540_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13(void){
_start:
{
lean_object* v___x_5541_; lean_object* v___x_5542_; lean_object* v___x_5543_; 
v___x_5541_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__12);
v___x_5542_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__1);
v___x_5543_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5543_, 0, v___x_5542_);
lean_ctor_set(v___x_5543_, 1, v___x_5541_);
return v___x_5543_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors(void){
_start:
{
lean_object* v___x_5544_; 
v___x_5544_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors___closed__13);
return v___x_5544_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1(void){
_start:
{
lean_object* v___x_5546_; lean_object* v___x_5547_; 
v___x_5546_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__0));
v___x_5547_ = l_Lean_stringToMessageData(v___x_5546_);
return v___x_5547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0(lean_object* v_x_5548_, lean_object* v___y_5549_, lean_object* v___y_5550_, lean_object* v___y_5551_, lean_object* v___y_5552_){
_start:
{
lean_object* v___x_5554_; lean_object* v___x_5555_; 
v___x_5554_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___closed__1);
v___x_5555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5555_, 0, v___x_5554_);
return v___x_5555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0___boxed(lean_object* v_x_5556_, lean_object* v___y_5557_, lean_object* v___y_5558_, lean_object* v___y_5559_, lean_object* v___y_5560_, lean_object* v___y_5561_){
_start:
{
lean_object* v_res_5562_; 
v_res_5562_ = lp_mathlib_Mathlib_Tactic_Linarith_preprocess___lam__0(v_x_5556_, v___y_5557_, v___y_5558_, v___y_5559_, v___y_5560_);
lean_dec(v___y_5560_);
lean_dec_ref(v___y_5559_);
lean_dec(v___y_5558_);
lean_dec_ref(v___y_5557_);
lean_dec_ref(v_x_5556_);
return v_res_5562_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3(lean_object* v_e_5563_){
_start:
{
if (lean_obj_tag(v_e_5563_) == 0)
{
uint8_t v___x_5564_; 
v___x_5564_ = 2;
return v___x_5564_;
}
else
{
uint8_t v___x_5565_; 
v___x_5565_ = 0;
return v___x_5565_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3___boxed(lean_object* v_e_5566_){
_start:
{
uint8_t v_res_5567_; lean_object* v_r_5568_; 
v_res_5567_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3(v_e_5566_);
lean_dec_ref(v_e_5566_);
v_r_5568_ = lean_box(v_res_5567_);
return v_r_5568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3(lean_object* v_cls_5569_, uint8_t v_collapsed_5570_, lean_object* v_tag_5571_, lean_object* v_opts_5572_, uint8_t v_clsEnabled_5573_, lean_object* v_oldTraces_5574_, lean_object* v_msg_5575_, lean_object* v_resStartStop_5576_, lean_object* v___y_5577_, lean_object* v___y_5578_, lean_object* v___y_5579_, lean_object* v___y_5580_){
_start:
{
lean_object* v_fst_5582_; lean_object* v_snd_5583_; lean_object* v___y_5585_; lean_object* v___y_5586_; lean_object* v_data_5587_; lean_object* v_fst_5598_; lean_object* v_snd_5599_; lean_object* v___x_5600_; uint8_t v___x_5601_; lean_object* v___y_5603_; lean_object* v_a_5604_; uint8_t v___y_5619_; double v___y_5650_; 
v_fst_5582_ = lean_ctor_get(v_resStartStop_5576_, 0);
lean_inc(v_fst_5582_);
v_snd_5583_ = lean_ctor_get(v_resStartStop_5576_, 1);
lean_inc(v_snd_5583_);
lean_dec_ref(v_resStartStop_5576_);
v_fst_5598_ = lean_ctor_get(v_snd_5583_, 0);
lean_inc(v_fst_5598_);
v_snd_5599_ = lean_ctor_get(v_snd_5583_, 1);
lean_inc(v_snd_5599_);
lean_dec(v_snd_5583_);
v___x_5600_ = l_Lean_trace_profiler;
v___x_5601_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_opts_5572_, v___x_5600_);
if (v___x_5601_ == 0)
{
v___y_5619_ = v___x_5601_;
goto v___jp_5618_;
}
else
{
lean_object* v___x_5655_; uint8_t v___x_5656_; 
v___x_5655_ = l_Lean_trace_profiler_useHeartbeats;
v___x_5656_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_opts_5572_, v___x_5655_);
if (v___x_5656_ == 0)
{
lean_object* v___x_5657_; lean_object* v___x_5658_; double v___x_5659_; double v___x_5660_; double v___x_5661_; 
v___x_5657_ = l_Lean_trace_profiler_threshold;
v___x_5658_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(v_opts_5572_, v___x_5657_);
v___x_5659_ = lean_float_of_nat(v___x_5658_);
v___x_5660_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__2);
v___x_5661_ = lean_float_div(v___x_5659_, v___x_5660_);
v___y_5650_ = v___x_5661_;
goto v___jp_5649_;
}
else
{
lean_object* v___x_5662_; lean_object* v___x_5663_; double v___x_5664_; 
v___x_5662_ = l_Lean_trace_profiler_threshold;
v___x_5663_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__14(v_opts_5572_, v___x_5662_);
v___x_5664_ = lean_float_of_nat(v___x_5663_);
v___y_5650_ = v___x_5664_;
goto v___jp_5649_;
}
}
v___jp_5584_:
{
lean_object* v___x_5588_; 
lean_inc(v___y_5586_);
v___x_5588_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__11(v_oldTraces_5574_, v_data_5587_, v___y_5586_, v___y_5585_, v___y_5577_, v___y_5578_, v___y_5579_, v___y_5580_);
if (lean_obj_tag(v___x_5588_) == 0)
{
lean_object* v___x_5589_; 
lean_dec_ref_known(v___x_5588_, 1);
v___x_5589_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_fst_5582_);
return v___x_5589_;
}
else
{
lean_object* v_a_5590_; lean_object* v___x_5592_; uint8_t v_isShared_5593_; uint8_t v_isSharedCheck_5597_; 
lean_dec(v_fst_5582_);
v_a_5590_ = lean_ctor_get(v___x_5588_, 0);
v_isSharedCheck_5597_ = !lean_is_exclusive(v___x_5588_);
if (v_isSharedCheck_5597_ == 0)
{
v___x_5592_ = v___x_5588_;
v_isShared_5593_ = v_isSharedCheck_5597_;
goto v_resetjp_5591_;
}
else
{
lean_inc(v_a_5590_);
lean_dec(v___x_5588_);
v___x_5592_ = lean_box(0);
v_isShared_5593_ = v_isSharedCheck_5597_;
goto v_resetjp_5591_;
}
v_resetjp_5591_:
{
lean_object* v___x_5595_; 
if (v_isShared_5593_ == 0)
{
v___x_5595_ = v___x_5592_;
goto v_reusejp_5594_;
}
else
{
lean_object* v_reuseFailAlloc_5596_; 
v_reuseFailAlloc_5596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5596_, 0, v_a_5590_);
v___x_5595_ = v_reuseFailAlloc_5596_;
goto v_reusejp_5594_;
}
v_reusejp_5594_:
{
return v___x_5595_;
}
}
}
}
v___jp_5602_:
{
uint8_t v_result_5605_; lean_object* v___x_5606_; lean_object* v___x_5607_; double v___x_5608_; lean_object* v_data_5609_; 
v_result_5605_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3_spec__3(v_fst_5582_);
v___x_5606_ = lean_box(v_result_5605_);
v___x_5607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5607_, 0, v___x_5606_);
v___x_5608_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__0);
lean_inc_ref(v_tag_5571_);
lean_inc_ref(v___x_5607_);
lean_inc(v_cls_5569_);
v_data_5609_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_5609_, 0, v_cls_5569_);
lean_ctor_set(v_data_5609_, 1, v___x_5607_);
lean_ctor_set(v_data_5609_, 2, v_tag_5571_);
lean_ctor_set_float(v_data_5609_, sizeof(void*)*3, v___x_5608_);
lean_ctor_set_float(v_data_5609_, sizeof(void*)*3 + 8, v___x_5608_);
lean_ctor_set_uint8(v_data_5609_, sizeof(void*)*3 + 16, v_collapsed_5570_);
if (v___x_5601_ == 0)
{
lean_dec_ref_known(v___x_5607_, 1);
lean_dec(v_snd_5599_);
lean_dec(v_fst_5598_);
lean_dec_ref(v_tag_5571_);
lean_dec(v_cls_5569_);
v___y_5585_ = v_a_5604_;
v___y_5586_ = v___y_5603_;
v_data_5587_ = v_data_5609_;
goto v___jp_5584_;
}
else
{
lean_object* v_data_5610_; double v___x_5611_; double v___x_5612_; 
lean_dec_ref_known(v_data_5609_, 3);
v_data_5610_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_5610_, 0, v_cls_5569_);
lean_ctor_set(v_data_5610_, 1, v___x_5607_);
lean_ctor_set(v_data_5610_, 2, v_tag_5571_);
v___x_5611_ = lean_unbox_float(v_fst_5598_);
lean_dec(v_fst_5598_);
lean_ctor_set_float(v_data_5610_, sizeof(void*)*3, v___x_5611_);
v___x_5612_ = lean_unbox_float(v_snd_5599_);
lean_dec(v_snd_5599_);
lean_ctor_set_float(v_data_5610_, sizeof(void*)*3 + 8, v___x_5612_);
lean_ctor_set_uint8(v_data_5610_, sizeof(void*)*3 + 16, v_collapsed_5570_);
v___y_5585_ = v_a_5604_;
v___y_5586_ = v___y_5603_;
v_data_5587_ = v_data_5610_;
goto v___jp_5584_;
}
}
v___jp_5613_:
{
lean_object* v_ref_5614_; lean_object* v___x_5615_; 
v_ref_5614_ = lean_ctor_get(v___y_5579_, 5);
lean_inc(v___y_5580_);
lean_inc_ref(v___y_5579_);
lean_inc(v___y_5578_);
lean_inc_ref(v___y_5577_);
lean_inc(v_fst_5582_);
v___x_5615_ = lean_apply_6(v_msg_5575_, v_fst_5582_, v___y_5577_, v___y_5578_, v___y_5579_, v___y_5580_, lean_box(0));
if (lean_obj_tag(v___x_5615_) == 0)
{
lean_object* v_a_5616_; 
v_a_5616_ = lean_ctor_get(v___x_5615_, 0);
lean_inc(v_a_5616_);
lean_dec_ref_known(v___x_5615_, 1);
v___y_5603_ = v_ref_5614_;
v_a_5604_ = v_a_5616_;
goto v___jp_5602_;
}
else
{
lean_object* v___x_5617_; 
lean_dec_ref_known(v___x_5615_, 1);
v___x_5617_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10___closed__1);
v___y_5603_ = v_ref_5614_;
v_a_5604_ = v___x_5617_;
goto v___jp_5602_;
}
}
v___jp_5618_:
{
if (v_clsEnabled_5573_ == 0)
{
if (v___y_5619_ == 0)
{
lean_object* v___x_5620_; lean_object* v_traceState_5621_; lean_object* v_env_5622_; lean_object* v_nextMacroScope_5623_; lean_object* v_ngen_5624_; lean_object* v_auxDeclNGen_5625_; lean_object* v_cache_5626_; lean_object* v_messages_5627_; lean_object* v_infoState_5628_; lean_object* v_snapshotTasks_5629_; lean_object* v___x_5631_; uint8_t v_isShared_5632_; uint8_t v_isSharedCheck_5648_; 
lean_dec(v_snd_5599_);
lean_dec(v_fst_5598_);
lean_dec_ref(v_msg_5575_);
lean_dec_ref(v_tag_5571_);
lean_dec(v_cls_5569_);
v___x_5620_ = lean_st_ref_take(v___y_5580_);
v_traceState_5621_ = lean_ctor_get(v___x_5620_, 4);
v_env_5622_ = lean_ctor_get(v___x_5620_, 0);
v_nextMacroScope_5623_ = lean_ctor_get(v___x_5620_, 1);
v_ngen_5624_ = lean_ctor_get(v___x_5620_, 2);
v_auxDeclNGen_5625_ = lean_ctor_get(v___x_5620_, 3);
v_cache_5626_ = lean_ctor_get(v___x_5620_, 5);
v_messages_5627_ = lean_ctor_get(v___x_5620_, 6);
v_infoState_5628_ = lean_ctor_get(v___x_5620_, 7);
v_snapshotTasks_5629_ = lean_ctor_get(v___x_5620_, 8);
v_isSharedCheck_5648_ = !lean_is_exclusive(v___x_5620_);
if (v_isSharedCheck_5648_ == 0)
{
v___x_5631_ = v___x_5620_;
v_isShared_5632_ = v_isSharedCheck_5648_;
goto v_resetjp_5630_;
}
else
{
lean_inc(v_snapshotTasks_5629_);
lean_inc(v_infoState_5628_);
lean_inc(v_messages_5627_);
lean_inc(v_cache_5626_);
lean_inc(v_traceState_5621_);
lean_inc(v_auxDeclNGen_5625_);
lean_inc(v_ngen_5624_);
lean_inc(v_nextMacroScope_5623_);
lean_inc(v_env_5622_);
lean_dec(v___x_5620_);
v___x_5631_ = lean_box(0);
v_isShared_5632_ = v_isSharedCheck_5648_;
goto v_resetjp_5630_;
}
v_resetjp_5630_:
{
uint64_t v_tid_5633_; lean_object* v_traces_5634_; lean_object* v___x_5636_; uint8_t v_isShared_5637_; uint8_t v_isSharedCheck_5647_; 
v_tid_5633_ = lean_ctor_get_uint64(v_traceState_5621_, sizeof(void*)*1);
v_traces_5634_ = lean_ctor_get(v_traceState_5621_, 0);
v_isSharedCheck_5647_ = !lean_is_exclusive(v_traceState_5621_);
if (v_isSharedCheck_5647_ == 0)
{
v___x_5636_ = v_traceState_5621_;
v_isShared_5637_ = v_isSharedCheck_5647_;
goto v_resetjp_5635_;
}
else
{
lean_inc(v_traces_5634_);
lean_dec(v_traceState_5621_);
v___x_5636_ = lean_box(0);
v_isShared_5637_ = v_isSharedCheck_5647_;
goto v_resetjp_5635_;
}
v_resetjp_5635_:
{
lean_object* v___x_5638_; lean_object* v___x_5640_; 
v___x_5638_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_5574_, v_traces_5634_);
lean_dec_ref(v_traces_5634_);
if (v_isShared_5637_ == 0)
{
lean_ctor_set(v___x_5636_, 0, v___x_5638_);
v___x_5640_ = v___x_5636_;
goto v_reusejp_5639_;
}
else
{
lean_object* v_reuseFailAlloc_5646_; 
v_reuseFailAlloc_5646_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_5646_, 0, v___x_5638_);
lean_ctor_set_uint64(v_reuseFailAlloc_5646_, sizeof(void*)*1, v_tid_5633_);
v___x_5640_ = v_reuseFailAlloc_5646_;
goto v_reusejp_5639_;
}
v_reusejp_5639_:
{
lean_object* v___x_5642_; 
if (v_isShared_5632_ == 0)
{
lean_ctor_set(v___x_5631_, 4, v___x_5640_);
v___x_5642_ = v___x_5631_;
goto v_reusejp_5641_;
}
else
{
lean_object* v_reuseFailAlloc_5645_; 
v_reuseFailAlloc_5645_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_5645_, 0, v_env_5622_);
lean_ctor_set(v_reuseFailAlloc_5645_, 1, v_nextMacroScope_5623_);
lean_ctor_set(v_reuseFailAlloc_5645_, 2, v_ngen_5624_);
lean_ctor_set(v_reuseFailAlloc_5645_, 3, v_auxDeclNGen_5625_);
lean_ctor_set(v_reuseFailAlloc_5645_, 4, v___x_5640_);
lean_ctor_set(v_reuseFailAlloc_5645_, 5, v_cache_5626_);
lean_ctor_set(v_reuseFailAlloc_5645_, 6, v_messages_5627_);
lean_ctor_set(v_reuseFailAlloc_5645_, 7, v_infoState_5628_);
lean_ctor_set(v_reuseFailAlloc_5645_, 8, v_snapshotTasks_5629_);
v___x_5642_ = v_reuseFailAlloc_5645_;
goto v_reusejp_5641_;
}
v_reusejp_5641_:
{
lean_object* v___x_5643_; lean_object* v___x_5644_; 
v___x_5643_ = lean_st_ref_set(v___y_5580_, v___x_5642_);
v___x_5644_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__10_spec__12___redArg(v_fst_5582_);
return v___x_5644_;
}
}
}
}
}
else
{
goto v___jp_5613_;
}
}
else
{
goto v___jp_5613_;
}
}
v___jp_5649_:
{
double v___x_5651_; double v___x_5652_; double v___x_5653_; uint8_t v___x_5654_; 
v___x_5651_ = lean_unbox_float(v_snd_5599_);
v___x_5652_ = lean_unbox_float(v_fst_5598_);
v___x_5653_ = lean_float_sub(v___x_5651_, v___x_5652_);
v___x_5654_ = lean_float_decLt(v___y_5650_, v___x_5653_);
v___y_5619_ = v___x_5654_;
goto v___jp_5618_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3___boxed(lean_object* v_cls_5665_, lean_object* v_collapsed_5666_, lean_object* v_tag_5667_, lean_object* v_opts_5668_, lean_object* v_clsEnabled_5669_, lean_object* v_oldTraces_5670_, lean_object* v_msg_5671_, lean_object* v_resStartStop_5672_, lean_object* v___y_5673_, lean_object* v___y_5674_, lean_object* v___y_5675_, lean_object* v___y_5676_, lean_object* v___y_5677_){
_start:
{
uint8_t v_collapsed_boxed_5678_; uint8_t v_clsEnabled_boxed_5679_; lean_object* v_res_5680_; 
v_collapsed_boxed_5678_ = lean_unbox(v_collapsed_5666_);
v_clsEnabled_boxed_5679_ = lean_unbox(v_clsEnabled_5669_);
v_res_5680_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3(v_cls_5665_, v_collapsed_boxed_5678_, v_tag_5667_, v_opts_5668_, v_clsEnabled_boxed_5679_, v_oldTraces_5670_, v_msg_5671_, v_resStartStop_5672_, v___y_5673_, v___y_5674_, v___y_5675_, v___y_5676_);
lean_dec(v___y_5676_);
lean_dec_ref(v___y_5675_);
lean_dec(v___y_5674_);
lean_dec_ref(v___y_5673_);
lean_dec_ref(v_opts_5668_);
return v_res_5680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0(lean_object* v_pp_5681_, lean_object* v_x_5682_, lean_object* v_x_5683_, lean_object* v___y_5684_, lean_object* v___y_5685_, lean_object* v___y_5686_, lean_object* v___y_5687_){
_start:
{
if (lean_obj_tag(v_x_5682_) == 0)
{
lean_object* v___x_5689_; lean_object* v___x_5690_; 
lean_dec_ref(v_pp_5681_);
v___x_5689_ = l_List_reverse___redArg(v_x_5683_);
v___x_5690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5690_, 0, v___x_5689_);
return v___x_5690_;
}
else
{
lean_object* v_head_5691_; lean_object* v_tail_5692_; lean_object* v___x_5694_; uint8_t v_isShared_5695_; uint8_t v_isSharedCheck_5712_; 
v_head_5691_ = lean_ctor_get(v_x_5682_, 0);
v_tail_5692_ = lean_ctor_get(v_x_5682_, 1);
v_isSharedCheck_5712_ = !lean_is_exclusive(v_x_5682_);
if (v_isSharedCheck_5712_ == 0)
{
v___x_5694_ = v_x_5682_;
v_isShared_5695_ = v_isSharedCheck_5712_;
goto v_resetjp_5693_;
}
else
{
lean_inc(v_tail_5692_);
lean_inc(v_head_5691_);
lean_dec(v_x_5682_);
v___x_5694_ = lean_box(0);
v_isShared_5695_ = v_isSharedCheck_5712_;
goto v_resetjp_5693_;
}
v_resetjp_5693_:
{
lean_object* v_fst_5696_; lean_object* v_snd_5697_; lean_object* v___x_5698_; 
v_fst_5696_ = lean_ctor_get(v_head_5691_, 0);
lean_inc(v_fst_5696_);
v_snd_5697_ = lean_ctor_get(v_head_5691_, 1);
lean_inc(v_snd_5697_);
lean_dec(v_head_5691_);
lean_inc_ref(v_pp_5681_);
v___x_5698_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process(v_pp_5681_, v_fst_5696_, v_snd_5697_, v___y_5684_, v___y_5685_, v___y_5686_, v___y_5687_);
if (lean_obj_tag(v___x_5698_) == 0)
{
lean_object* v_a_5699_; lean_object* v___x_5701_; 
v_a_5699_ = lean_ctor_get(v___x_5698_, 0);
lean_inc(v_a_5699_);
lean_dec_ref_known(v___x_5698_, 1);
if (v_isShared_5695_ == 0)
{
lean_ctor_set(v___x_5694_, 1, v_x_5683_);
lean_ctor_set(v___x_5694_, 0, v_a_5699_);
v___x_5701_ = v___x_5694_;
goto v_reusejp_5700_;
}
else
{
lean_object* v_reuseFailAlloc_5703_; 
v_reuseFailAlloc_5703_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5703_, 0, v_a_5699_);
lean_ctor_set(v_reuseFailAlloc_5703_, 1, v_x_5683_);
v___x_5701_ = v_reuseFailAlloc_5703_;
goto v_reusejp_5700_;
}
v_reusejp_5700_:
{
v_x_5682_ = v_tail_5692_;
v_x_5683_ = v___x_5701_;
goto _start;
}
}
else
{
lean_object* v_a_5704_; lean_object* v___x_5706_; uint8_t v_isShared_5707_; uint8_t v_isSharedCheck_5711_; 
lean_del_object(v___x_5694_);
lean_dec(v_tail_5692_);
lean_dec(v_x_5683_);
lean_dec_ref(v_pp_5681_);
v_a_5704_ = lean_ctor_get(v___x_5698_, 0);
v_isSharedCheck_5711_ = !lean_is_exclusive(v___x_5698_);
if (v_isSharedCheck_5711_ == 0)
{
v___x_5706_ = v___x_5698_;
v_isShared_5707_ = v_isSharedCheck_5711_;
goto v_resetjp_5705_;
}
else
{
lean_inc(v_a_5704_);
lean_dec(v___x_5698_);
v___x_5706_ = lean_box(0);
v_isShared_5707_ = v_isSharedCheck_5711_;
goto v_resetjp_5705_;
}
v_resetjp_5705_:
{
lean_object* v___x_5709_; 
if (v_isShared_5707_ == 0)
{
v___x_5709_ = v___x_5706_;
goto v_reusejp_5708_;
}
else
{
lean_object* v_reuseFailAlloc_5710_; 
v_reuseFailAlloc_5710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5710_, 0, v_a_5704_);
v___x_5709_ = v_reuseFailAlloc_5710_;
goto v_reusejp_5708_;
}
v_reusejp_5708_:
{
return v___x_5709_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0___boxed(lean_object* v_pp_5713_, lean_object* v_x_5714_, lean_object* v_x_5715_, lean_object* v___y_5716_, lean_object* v___y_5717_, lean_object* v___y_5718_, lean_object* v___y_5719_, lean_object* v___y_5720_){
_start:
{
lean_object* v_res_5721_; 
v_res_5721_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0(v_pp_5713_, v_x_5714_, v_x_5715_, v___y_5716_, v___y_5717_, v___y_5718_, v___y_5719_);
lean_dec(v___y_5719_);
lean_dec_ref(v___y_5718_);
lean_dec(v___y_5717_);
lean_dec_ref(v___y_5716_);
return v_res_5721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_Linarith_preprocess_spec__1(lean_object* v_a_5722_, lean_object* v_a_5723_){
_start:
{
if (lean_obj_tag(v_a_5722_) == 0)
{
lean_object* v___x_5724_; 
v___x_5724_ = lean_array_to_list(v_a_5723_);
return v___x_5724_;
}
else
{
lean_object* v_head_5725_; lean_object* v_tail_5726_; lean_object* v___x_5727_; 
v_head_5725_ = lean_ctor_get(v_a_5722_, 0);
lean_inc(v_head_5725_);
v_tail_5726_ = lean_ctor_get(v_a_5722_, 1);
lean_inc(v_tail_5726_);
lean_dec_ref_known(v_a_5722_, 2);
v___x_5727_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_5723_, v_head_5725_);
v_a_5722_ = v_tail_5726_;
v_a_5723_ = v___x_5727_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2(lean_object* v_x_5731_, lean_object* v_x_5732_, lean_object* v___y_5733_, lean_object* v___y_5734_, lean_object* v___y_5735_, lean_object* v___y_5736_){
_start:
{
if (lean_obj_tag(v_x_5732_) == 0)
{
lean_object* v___x_5738_; 
v___x_5738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5738_, 0, v_x_5731_);
return v___x_5738_;
}
else
{
lean_object* v_head_5739_; lean_object* v_tail_5740_; lean_object* v___x_5741_; lean_object* v___x_5742_; 
v_head_5739_ = lean_ctor_get(v_x_5732_, 0);
lean_inc(v_head_5739_);
v_tail_5740_ = lean_ctor_get(v_x_5732_, 1);
lean_inc(v_tail_5740_);
lean_dec_ref_known(v_x_5732_, 2);
v___x_5741_ = lean_box(0);
v___x_5742_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_preprocess_spec__0(v_head_5739_, v_x_5731_, v___x_5741_, v___y_5733_, v___y_5734_, v___y_5735_, v___y_5736_);
if (lean_obj_tag(v___x_5742_) == 0)
{
lean_object* v_a_5743_; lean_object* v___x_5744_; lean_object* v___x_5745_; 
v_a_5743_ = lean_ctor_get(v___x_5742_, 0);
lean_inc(v_a_5743_);
lean_dec_ref_known(v___x_5742_, 1);
v___x_5744_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___closed__0));
v___x_5745_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_Linarith_preprocess_spec__1(v_a_5743_, v___x_5744_);
v_x_5731_ = v___x_5745_;
v_x_5732_ = v_tail_5740_;
goto _start;
}
else
{
lean_object* v_a_5747_; lean_object* v___x_5749_; uint8_t v_isShared_5750_; uint8_t v_isSharedCheck_5754_; 
lean_dec(v_tail_5740_);
v_a_5747_ = lean_ctor_get(v___x_5742_, 0);
v_isSharedCheck_5754_ = !lean_is_exclusive(v___x_5742_);
if (v_isSharedCheck_5754_ == 0)
{
v___x_5749_ = v___x_5742_;
v_isShared_5750_ = v_isSharedCheck_5754_;
goto v_resetjp_5748_;
}
else
{
lean_inc(v_a_5747_);
lean_dec(v___x_5742_);
v___x_5749_ = lean_box(0);
v_isShared_5750_ = v_isSharedCheck_5754_;
goto v_resetjp_5748_;
}
v_resetjp_5748_:
{
lean_object* v___x_5752_; 
if (v_isShared_5750_ == 0)
{
v___x_5752_ = v___x_5749_;
goto v_reusejp_5751_;
}
else
{
lean_object* v_reuseFailAlloc_5753_; 
v_reuseFailAlloc_5753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5753_, 0, v_a_5747_);
v___x_5752_ = v_reuseFailAlloc_5753_;
goto v_reusejp_5751_;
}
v_reusejp_5751_:
{
return v___x_5752_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___boxed(lean_object* v_x_5755_, lean_object* v_x_5756_, lean_object* v___y_5757_, lean_object* v___y_5758_, lean_object* v___y_5759_, lean_object* v___y_5760_, lean_object* v___y_5761_){
_start:
{
lean_object* v_res_5762_; 
v_res_5762_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2(v_x_5755_, v_x_5756_, v___y_5757_, v___y_5758_, v___y_5759_, v___y_5760_);
lean_dec(v___y_5760_);
lean_dec_ref(v___y_5759_);
lean_dec(v___y_5758_);
lean_dec_ref(v___y_5757_);
return v_res_5762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess(lean_object* v_pps_5764_, lean_object* v_g_5765_, lean_object* v_l_5766_, lean_object* v_a_5767_, lean_object* v_a_5768_, lean_object* v_a_5769_, lean_object* v_a_5770_){
_start:
{
lean_object* v_options_5772_; lean_object* v_inheritedTraceOptions_5773_; uint8_t v_hasTrace_5774_; lean_object* v___x_5775_; lean_object* v___x_5776_; lean_object* v___x_5777_; lean_object* v___x_5778_; 
v_options_5772_ = lean_ctor_get(v_a_5769_, 2);
v_inheritedTraceOptions_5773_ = lean_ctor_get(v_a_5769_, 13);
v_hasTrace_5774_ = lean_ctor_get_uint8(v_options_5772_, sizeof(void*)*1);
lean_inc(v_g_5765_);
v___x_5775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5775_, 0, v_g_5765_);
lean_ctor_set(v___x_5775_, 1, v_l_5766_);
v___x_5776_ = lean_box(0);
v___x_5777_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5777_, 0, v___x_5775_);
lean_ctor_set(v___x_5777_, 1, v___x_5776_);
v___x_5778_ = lean_alloc_closure((void*)(lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_preprocess_spec__2___boxed), 7, 2);
lean_closure_set(v___x_5778_, 0, v___x_5777_);
lean_closure_set(v___x_5778_, 1, v_pps_5764_);
if (v_hasTrace_5774_ == 0)
{
lean_object* v___x_5779_; 
v___x_5779_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_g_5765_, v___x_5778_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
return v___x_5779_;
}
else
{
lean_object* v___f_5780_; lean_object* v___x_5781_; lean_object* v___x_5782_; lean_object* v___x_5783_; uint8_t v___x_5784_; lean_object* v___y_5786_; lean_object* v___y_5787_; lean_object* v_a_5788_; lean_object* v___y_5801_; lean_object* v___y_5802_; lean_object* v_a_5803_; 
v___f_5780_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_preprocess___closed__0));
v___x_5781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__3));
v___x_5782_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f_spec__1___closed__1));
v___x_5783_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkNatCastNonnegProof_x3f___closed__6);
v___x_5784_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_5773_, v_options_5772_, v___x_5783_);
if (v___x_5784_ == 0)
{
lean_object* v___x_5853_; uint8_t v___x_5854_; 
v___x_5853_ = l_Lean_trace_profiler;
v___x_5854_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_5772_, v___x_5853_);
if (v___x_5854_ == 0)
{
lean_object* v___x_5855_; 
v___x_5855_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_g_5765_, v___x_5778_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
return v___x_5855_;
}
else
{
goto v___jp_5812_;
}
}
else
{
goto v___jp_5812_;
}
v___jp_5785_:
{
lean_object* v___x_5789_; double v___x_5790_; double v___x_5791_; double v___x_5792_; double v___x_5793_; double v___x_5794_; lean_object* v___x_5795_; lean_object* v___x_5796_; lean_object* v___x_5797_; lean_object* v___x_5798_; lean_object* v___x_5799_; 
v___x_5789_ = lean_io_mono_nanos_now();
v___x_5790_ = lean_float_of_nat(v___y_5787_);
v___x_5791_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2, &lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs___closed__2);
v___x_5792_ = lean_float_div(v___x_5790_, v___x_5791_);
v___x_5793_ = lean_float_of_nat(v___x_5789_);
v___x_5794_ = lean_float_div(v___x_5793_, v___x_5791_);
v___x_5795_ = lean_box_float(v___x_5792_);
v___x_5796_ = lean_box_float(v___x_5794_);
v___x_5797_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5797_, 0, v___x_5795_);
lean_ctor_set(v___x_5797_, 1, v___x_5796_);
v___x_5798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5798_, 0, v_a_5788_);
lean_ctor_set(v___x_5798_, 1, v___x_5797_);
v___x_5799_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3(v___x_5781_, v_hasTrace_5774_, v___x_5782_, v_options_5772_, v___x_5784_, v___y_5786_, v___f_5780_, v___x_5798_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
return v___x_5799_;
}
v___jp_5800_:
{
lean_object* v___x_5804_; double v___x_5805_; double v___x_5806_; lean_object* v___x_5807_; lean_object* v___x_5808_; lean_object* v___x_5809_; lean_object* v___x_5810_; lean_object* v___x_5811_; 
v___x_5804_ = lean_io_get_num_heartbeats();
v___x_5805_ = lean_float_of_nat(v___y_5802_);
v___x_5806_ = lean_float_of_nat(v___x_5804_);
v___x_5807_ = lean_box_float(v___x_5805_);
v___x_5808_ = lean_box_float(v___x_5806_);
v___x_5809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5809_, 0, v___x_5807_);
lean_ctor_set(v___x_5809_, 1, v___x_5808_);
v___x_5810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5810_, 0, v_a_5803_);
lean_ctor_set(v___x_5810_, 1, v___x_5809_);
v___x_5811_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_preprocess_spec__3(v___x_5781_, v_hasTrace_5774_, v___x_5782_, v_options_5772_, v___x_5784_, v___y_5801_, v___f_5780_, v___x_5810_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
return v___x_5811_;
}
v___jp_5812_:
{
lean_object* v___x_5813_; lean_object* v_a_5814_; lean_object* v___x_5815_; uint8_t v___x_5816_; 
v___x_5813_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__8___redArg(v_a_5770_);
v_a_5814_ = lean_ctor_get(v___x_5813_, 0);
lean_inc(v_a_5814_);
lean_dec_ref(v___x_5813_);
v___x_5815_ = l_Lean_trace_profiler_useHeartbeats;
v___x_5816_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_nlinarithGetSquareProofs_spec__9(v_options_5772_, v___x_5815_);
if (v___x_5816_ == 0)
{
lean_object* v___x_5817_; lean_object* v___x_5818_; 
v___x_5817_ = lean_io_mono_nanos_now();
v___x_5818_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_g_5765_, v___x_5778_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
if (lean_obj_tag(v___x_5818_) == 0)
{
lean_object* v_a_5819_; lean_object* v___x_5821_; uint8_t v_isShared_5822_; uint8_t v_isSharedCheck_5826_; 
v_a_5819_ = lean_ctor_get(v___x_5818_, 0);
v_isSharedCheck_5826_ = !lean_is_exclusive(v___x_5818_);
if (v_isSharedCheck_5826_ == 0)
{
v___x_5821_ = v___x_5818_;
v_isShared_5822_ = v_isSharedCheck_5826_;
goto v_resetjp_5820_;
}
else
{
lean_inc(v_a_5819_);
lean_dec(v___x_5818_);
v___x_5821_ = lean_box(0);
v_isShared_5822_ = v_isSharedCheck_5826_;
goto v_resetjp_5820_;
}
v_resetjp_5820_:
{
lean_object* v___x_5824_; 
if (v_isShared_5822_ == 0)
{
lean_ctor_set_tag(v___x_5821_, 1);
v___x_5824_ = v___x_5821_;
goto v_reusejp_5823_;
}
else
{
lean_object* v_reuseFailAlloc_5825_; 
v_reuseFailAlloc_5825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5825_, 0, v_a_5819_);
v___x_5824_ = v_reuseFailAlloc_5825_;
goto v_reusejp_5823_;
}
v_reusejp_5823_:
{
v___y_5786_ = v_a_5814_;
v___y_5787_ = v___x_5817_;
v_a_5788_ = v___x_5824_;
goto v___jp_5785_;
}
}
}
else
{
lean_object* v_a_5827_; lean_object* v___x_5829_; uint8_t v_isShared_5830_; uint8_t v_isSharedCheck_5834_; 
v_a_5827_ = lean_ctor_get(v___x_5818_, 0);
v_isSharedCheck_5834_ = !lean_is_exclusive(v___x_5818_);
if (v_isSharedCheck_5834_ == 0)
{
v___x_5829_ = v___x_5818_;
v_isShared_5830_ = v_isSharedCheck_5834_;
goto v_resetjp_5828_;
}
else
{
lean_inc(v_a_5827_);
lean_dec(v___x_5818_);
v___x_5829_ = lean_box(0);
v_isShared_5830_ = v_isSharedCheck_5834_;
goto v_resetjp_5828_;
}
v_resetjp_5828_:
{
lean_object* v___x_5832_; 
if (v_isShared_5830_ == 0)
{
lean_ctor_set_tag(v___x_5829_, 0);
v___x_5832_ = v___x_5829_;
goto v_reusejp_5831_;
}
else
{
lean_object* v_reuseFailAlloc_5833_; 
v_reuseFailAlloc_5833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5833_, 0, v_a_5827_);
v___x_5832_ = v_reuseFailAlloc_5833_;
goto v_reusejp_5831_;
}
v_reusejp_5831_:
{
v___y_5786_ = v_a_5814_;
v___y_5787_ = v___x_5817_;
v_a_5788_ = v___x_5832_;
goto v___jp_5785_;
}
}
}
}
else
{
lean_object* v___x_5835_; lean_object* v___x_5836_; 
v___x_5835_ = lean_io_get_num_heartbeats();
v___x_5836_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_removeNeAux_spec__3___redArg(v_g_5765_, v___x_5778_, v_a_5767_, v_a_5768_, v_a_5769_, v_a_5770_);
if (lean_obj_tag(v___x_5836_) == 0)
{
lean_object* v_a_5837_; lean_object* v___x_5839_; uint8_t v_isShared_5840_; uint8_t v_isSharedCheck_5844_; 
v_a_5837_ = lean_ctor_get(v___x_5836_, 0);
v_isSharedCheck_5844_ = !lean_is_exclusive(v___x_5836_);
if (v_isSharedCheck_5844_ == 0)
{
v___x_5839_ = v___x_5836_;
v_isShared_5840_ = v_isSharedCheck_5844_;
goto v_resetjp_5838_;
}
else
{
lean_inc(v_a_5837_);
lean_dec(v___x_5836_);
v___x_5839_ = lean_box(0);
v_isShared_5840_ = v_isSharedCheck_5844_;
goto v_resetjp_5838_;
}
v_resetjp_5838_:
{
lean_object* v___x_5842_; 
if (v_isShared_5840_ == 0)
{
lean_ctor_set_tag(v___x_5839_, 1);
v___x_5842_ = v___x_5839_;
goto v_reusejp_5841_;
}
else
{
lean_object* v_reuseFailAlloc_5843_; 
v_reuseFailAlloc_5843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5843_, 0, v_a_5837_);
v___x_5842_ = v_reuseFailAlloc_5843_;
goto v_reusejp_5841_;
}
v_reusejp_5841_:
{
v___y_5801_ = v_a_5814_;
v___y_5802_ = v___x_5835_;
v_a_5803_ = v___x_5842_;
goto v___jp_5800_;
}
}
}
else
{
lean_object* v_a_5845_; lean_object* v___x_5847_; uint8_t v_isShared_5848_; uint8_t v_isSharedCheck_5852_; 
v_a_5845_ = lean_ctor_get(v___x_5836_, 0);
v_isSharedCheck_5852_ = !lean_is_exclusive(v___x_5836_);
if (v_isSharedCheck_5852_ == 0)
{
v___x_5847_ = v___x_5836_;
v_isShared_5848_ = v_isSharedCheck_5852_;
goto v_resetjp_5846_;
}
else
{
lean_inc(v_a_5845_);
lean_dec(v___x_5836_);
v___x_5847_ = lean_box(0);
v_isShared_5848_ = v_isSharedCheck_5852_;
goto v_resetjp_5846_;
}
v_resetjp_5846_:
{
lean_object* v___x_5850_; 
if (v_isShared_5848_ == 0)
{
lean_ctor_set_tag(v___x_5847_, 0);
v___x_5850_ = v___x_5847_;
goto v_reusejp_5849_;
}
else
{
lean_object* v_reuseFailAlloc_5851_; 
v_reuseFailAlloc_5851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5851_, 0, v_a_5845_);
v___x_5850_ = v_reuseFailAlloc_5851_;
goto v_reusejp_5849_;
}
v_reusejp_5849_:
{
v___y_5801_ = v_a_5814_;
v___y_5802_ = v___x_5835_;
v_a_5803_ = v___x_5850_;
goto v___jp_5800_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_preprocess___boxed(lean_object* v_pps_5856_, lean_object* v_g_5857_, lean_object* v_l_5858_, lean_object* v_a_5859_, lean_object* v_a_5860_, lean_object* v_a_5861_, lean_object* v_a_5862_, lean_object* v_a_5863_){
_start:
{
lean_object* v_res_5864_; 
v_res_5864_ = lp_mathlib_Mathlib_Tactic_Linarith_preprocess(v_pps_5856_, v_g_5857_, v_l_5858_, v_a_5859_, v_a_5860_, v_a_5861_, v_a_5862_);
lean_dec(v_a_5862_);
lean_dec_ref(v_a_5861_);
lean_dec(v_a_5860_);
lean_dec_ref(v_a_5859_);
return v_res_5864_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Zify(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linarith_Preprocessing_0__Mathlib_Tactic_Linarith_initFn_00___x40_Mathlib_Tactic_Linarith_Preprocessing_1010411728____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Linarith_nnrealToRealTransform = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_nnrealToRealTransform);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors = _init_lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_defaultPreprocessors);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_Tactic_Rewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Zify(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Preprocessing(builtin);
}
#ifdef __cplusplus
}
#endif
