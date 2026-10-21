// Lean compiler output
// Module: Aesop.Util.EqualUpToIds
// Imports: public import Init public meta import Init public import Batteries.Lean.Meta.Basic
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
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_instMonadControlReaderT(lean_object*, lean_object*);
lean_object* l_instMonadControlStateRefT_x27(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withMCtx___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_ExprStructEq_beq___boxed(lean_object*, lean_object*);
lean_object* l_instBEqProd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ExprStructEq_hash___boxed(lean_object*);
lean_object* l_instHashableProd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonad___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonadExceptOf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonadRef___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonadLift___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
lean_object* l_Lean_instExceptToTraceResultBool___lam__0___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_io_get_num_heartbeats();
double lean_float_of_nat(lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
extern lean_object* l_Lean_Meta_instMonadEnvMetaM;
lean_object* l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMessageContextFull___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
lean_object* l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableMVarId_hash___boxed(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_local_ctx_num_indices(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isLet(lean_object*, uint8_t);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
uint8_t l_Lean_LocalDecl_binderInfo(lean_object*);
uint8_t l_Lean_LocalDecl_kind(lean_object*);
uint8_t l_Lean_instDecidableEqLocalDeclKind(uint8_t, uint8_t);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_LocalDecl_value(lean_object*, uint8_t);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_batteries_Lean_MetavarContext_isExprMVarDeclared(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_instBEqFVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableFVarId_hash___boxed(lean_object*);
uint8_t l_Option_instBEq_beq___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqLevelMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_instBEqLevelMVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableLevelMVarId_hash___boxed(lean_object*);
uint8_t l_Lean_PersistentHashMap_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_forIn_x27_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t l_Lean_instBEqLiteral_beq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
uint8_t lean_expr_equal(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwMaxRecDepthAt___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "EqualUpToIds"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(83, 136, 252, 104, 104, 29, 21, 73)}};
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(150, 245, 193, 33, 71, 243, 152, 171)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__4_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__5_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(213, 96, 250, 13, 195, 1, 48, 100)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__7_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__6_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(111, 34, 111, 91, 49, 75, 171, 204)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__7_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__7_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__8_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__7_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(170, 232, 34, 76, 85, 24, 179, 16)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__8_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__8_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__9_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__8_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(171, 16, 1, 134, 26, 169, 80, 1)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__9_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__9_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__10_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__9_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(17, 35, 22, 59, 31, 197, 85, 65)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__10_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__10_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__11_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__11_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__11_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__12_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__10_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__11_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(232, 198, 242, 76, 157, 143, 125, 83)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__12_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__12_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__13_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__13_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__13_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__14_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__12_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__13_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(153, 198, 140, 105, 108, 191, 184, 12)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__14_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__14_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__15_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__14_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__0_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(43, 168, 101, 121, 82, 248, 116, 180)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__15_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__15_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__16_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__15_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__1_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(145, 46, 40, 34, 209, 201, 26, 36)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__16_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__16_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__17_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__16_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__2_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(44, 115, 111, 242, 161, 222, 179, 163)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__17_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__17_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__18_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__17_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1192428161) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(69, 110, 194, 91, 230, 128, 174, 189)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__18_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__18_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__19_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__19_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__19_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__20_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__18_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__19_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 193, 11, 108, 243, 181, 207, 154)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__20_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__20_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__21_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__21_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__21_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__22_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__20_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__21_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(170, 221, 239, 138, 83, 8, 53, 222)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__22_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__22_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__23_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__22_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(147, 110, 86, 148, 77, 211, 132, 72)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__23_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__23_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2____boxed(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1;
static const lean_closure_object lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2 = (const lean_object*)&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3 = (const lean_object*)&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4 = (const lean_object*)&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5 = (const lean_object*)&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadEqualUpToIdsM;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqLevelMVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableLevelMVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_mvarId_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_mvarId_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_expr_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_expr_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_delayedAssignment_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_delayedAssignment_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_namesEqualUpToMacroScopes(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_namesEqualUpToMacroScopes___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__4_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__5_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__6_value;
static const lean_closure_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__1_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__8_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__3_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__4_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__5_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__6_value)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__9_value;
static const lean_ctor_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__9_value),((lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__7_value)}};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_ExprStructEq_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0 = (const lean_object*)&lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0_value;
static const lean_closure_object lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_ExprStructEq_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1 = (const lean_object*)&lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structurally equal"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "pointer-equal"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__7 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResultBool___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__26 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__26_value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__28 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__28_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__28_value)} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__29 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__29_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__29_value)} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≟ "};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__31 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__31_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqMVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableMVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__1_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "comparing targets"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "common mvars are "};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__4 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "different"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "identical"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mvar "};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__8 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = " known to be equal to different mvar "};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__10 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "mvars already known to be equal"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__12 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__0_value;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "number of hyps differs"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__14 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "comparing hyps "};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "unknown metavariable '\?"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__14 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__16 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__16_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17;
static const lean_string_object lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "comparing mvars "};
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__18 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__18_value;
static lean_once_cell_t lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 104, .m_capacity = 104, .m_length = 101, .m_data = "_private.Aesop.Util.EqualUpToIds.0.Aesop.EqualUpToIds.Unsafe.exprsEqualUpToIdsCore₃.compareMVarValues"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Aesop.Util.EqualUpToIds"};
static const lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqFVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableFVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__1 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0___boxed, .m_arity = 13, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__2 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0___boxed, .m_arity = 12, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds_x27(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_59_; uint8_t v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_59_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_60_ = 0;
v___x_61_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__23_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_62_ = l_Lean_registerTraceClass(v___x_59_, v___x_60_, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2____boxed(lean_object* v_a_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_();
return v_res_64_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0(void){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = l_instMonadEIO(lean_box(0));
return v___x_65_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__0);
v___x_67_ = l_StateRefT_x27_instMonad___redArg(v___x_66_);
return v___x_67_;
}
}
static lean_object* _init_lp_aesop_Aesop_instMonadEqualUpToIdsM(void){
_start:
{
lean_object* v___x_72_; lean_object* v_toApplicative_73_; lean_object* v_toFunctor_74_; lean_object* v_toSeq_75_; lean_object* v_toSeqLeft_76_; lean_object* v_toSeqRight_77_; lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_82_; lean_object* v___f_83_; lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v_toApplicative_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_118_; 
v___x_72_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_73_ = lean_ctor_get(v___x_72_, 0);
v_toFunctor_74_ = lean_ctor_get(v_toApplicative_73_, 0);
v_toSeq_75_ = lean_ctor_get(v_toApplicative_73_, 2);
v_toSeqLeft_76_ = lean_ctor_get(v_toApplicative_73_, 3);
v_toSeqRight_77_ = lean_ctor_get(v_toApplicative_73_, 4);
v___f_78_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_79_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_74_, 2);
v___f_80_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_80_, 0, v_toFunctor_74_);
v___f_81_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_81_, 0, v_toFunctor_74_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v___f_80_);
lean_ctor_set(v___x_82_, 1, v___f_81_);
lean_inc(v_toSeqRight_77_);
v___f_83_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_83_, 0, v_toSeqRight_77_);
lean_inc(v_toSeqLeft_76_);
v___f_84_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_84_, 0, v_toSeqLeft_76_);
lean_inc(v_toSeq_75_);
v___f_85_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_85_, 0, v_toSeq_75_);
v___x_86_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_86_, 0, v___x_82_);
lean_ctor_set(v___x_86_, 1, v___f_78_);
lean_ctor_set(v___x_86_, 2, v___f_85_);
lean_ctor_set(v___x_86_, 3, v___f_84_);
lean_ctor_set(v___x_86_, 4, v___f_83_);
v___x_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___f_79_);
v___x_88_ = l_StateRefT_x27_instMonad___redArg(v___x_87_);
v_toApplicative_89_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_118_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_118_ == 0)
{
lean_object* v_unused_119_; 
v_unused_119_ = lean_ctor_get(v___x_88_, 1);
lean_dec(v_unused_119_);
v___x_91_ = v___x_88_;
v_isShared_92_ = v_isSharedCheck_118_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_toApplicative_89_);
lean_dec(v___x_88_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_118_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v_toFunctor_93_; lean_object* v_toSeq_94_; lean_object* v_toSeqLeft_95_; lean_object* v_toSeqRight_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_116_; 
v_toFunctor_93_ = lean_ctor_get(v_toApplicative_89_, 0);
v_toSeq_94_ = lean_ctor_get(v_toApplicative_89_, 2);
v_toSeqLeft_95_ = lean_ctor_get(v_toApplicative_89_, 3);
v_toSeqRight_96_ = lean_ctor_get(v_toApplicative_89_, 4);
v_isSharedCheck_116_ = !lean_is_exclusive(v_toApplicative_89_);
if (v_isSharedCheck_116_ == 0)
{
lean_object* v_unused_117_; 
v_unused_117_ = lean_ctor_get(v_toApplicative_89_, 1);
lean_dec(v_unused_117_);
v___x_98_ = v_toApplicative_89_;
v_isShared_99_ = v_isSharedCheck_116_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_toSeqRight_96_);
lean_inc(v_toSeqLeft_95_);
lean_inc(v_toSeq_94_);
lean_inc(v_toFunctor_93_);
lean_dec(v_toApplicative_89_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_116_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___f_100_; lean_object* v___f_101_; lean_object* v___f_102_; lean_object* v___f_103_; lean_object* v___x_104_; lean_object* v___f_105_; lean_object* v___f_106_; lean_object* v___f_107_; lean_object* v___x_109_; 
v___f_100_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_101_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_93_);
v___f_102_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_102_, 0, v_toFunctor_93_);
v___f_103_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_103_, 0, v_toFunctor_93_);
v___x_104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_104_, 0, v___f_102_);
lean_ctor_set(v___x_104_, 1, v___f_103_);
v___f_105_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_105_, 0, v_toSeqRight_96_);
v___f_106_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_106_, 0, v_toSeqLeft_95_);
v___f_107_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_107_, 0, v_toSeq_94_);
if (v_isShared_99_ == 0)
{
lean_ctor_set(v___x_98_, 4, v___f_105_);
lean_ctor_set(v___x_98_, 3, v___f_106_);
lean_ctor_set(v___x_98_, 2, v___f_107_);
lean_ctor_set(v___x_98_, 1, v___f_100_);
lean_ctor_set(v___x_98_, 0, v___x_104_);
v___x_109_ = v___x_98_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_104_);
lean_ctor_set(v_reuseFailAlloc_115_, 1, v___f_100_);
lean_ctor_set(v_reuseFailAlloc_115_, 2, v___f_107_);
lean_ctor_set(v_reuseFailAlloc_115_, 3, v___f_106_);
lean_ctor_set(v_reuseFailAlloc_115_, 4, v___f_105_);
v___x_109_ = v_reuseFailAlloc_115_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
lean_object* v___x_111_; 
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 1, v___f_101_);
lean_ctor_set(v___x_91_, 0, v___x_109_);
v___x_111_ = v___x_91_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v___x_109_);
lean_ctor_set(v_reuseFailAlloc_114_, 1, v___f_101_);
v___x_111_ = v_reuseFailAlloc_114_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = l_StateRefT_x27_instMonad___redArg(v___x_111_);
v___x_113_ = l_ReaderT_instMonad___redArg(v___x_112_);
return v___x_113_;
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_120_ = lean_box(0);
v___x_121_ = lean_unsigned_to_nat(16u);
v___x_122_ = lean_mk_array(v___x_121_, v___x_120_);
return v___x_122_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1(void){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_123_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0, &lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0_once, _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__0);
v___x_124_ = lean_unsigned_to_nat(0u);
v___x_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v___x_123_);
return v___x_125_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1, &lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1);
v___x_127_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___x_126_);
lean_ctor_set(v___x_127_, 2, v___x_126_);
lean_ctor_set(v___x_127_, 3, v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(lean_object* v_x_128_, lean_object* v_commonMCtx_x3f_129_, lean_object* v_mctx_u2081_130_, lean_object* v_mctx_u2082_131_, uint8_t v_allowAssignmentDiff_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_138_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2, &lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2_once, _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__2);
v___x_139_ = lean_st_mk_ref(v___x_138_);
v___x_140_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_140_, 0, v_commonMCtx_x3f_129_);
lean_ctor_set(v___x_140_, 1, v_mctx_u2081_130_);
lean_ctor_set(v___x_140_, 2, v_mctx_u2082_131_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*3, v_allowAssignmentDiff_132_);
lean_inc(v_a_136_);
lean_inc_ref(v_a_135_);
lean_inc(v_a_134_);
lean_inc_ref(v_a_133_);
lean_inc(v___x_139_);
v___x_141_ = lean_apply_7(v_x_128_, v___x_140_, v___x_139_, v_a_133_, v_a_134_, v_a_135_, v_a_136_, lean_box(0));
if (lean_obj_tag(v___x_141_) == 0)
{
lean_object* v_a_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_151_; 
v_a_142_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_151_ == 0)
{
v___x_144_ = v___x_141_;
v_isShared_145_ = v_isSharedCheck_151_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_a_142_);
lean_dec(v___x_141_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_151_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_149_; 
v___x_146_ = lean_st_ref_get(v___x_139_);
lean_dec(v___x_139_);
v___x_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_147_, 0, v_a_142_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_147_);
v___x_149_ = v___x_144_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v___x_147_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
lean_dec(v___x_139_);
v_a_152_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_141_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_141_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___boxed(lean_object* v_x_160_, lean_object* v_commonMCtx_x3f_161_, lean_object* v_mctx_u2081_162_, lean_object* v_mctx_u2082_163_, lean_object* v_allowAssignmentDiff_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_170_; lean_object* v_res_171_; 
v_allowAssignmentDiff_boxed_170_ = lean_unbox(v_allowAssignmentDiff_164_);
v_res_171_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(v_x_160_, v_commonMCtx_x3f_161_, v_mctx_u2081_162_, v_mctx_u2082_163_, v_allowAssignmentDiff_boxed_170_, v_a_165_, v_a_166_, v_a_167_, v_a_168_);
lean_dec(v_a_168_);
lean_dec_ref(v_a_167_);
lean_dec(v_a_166_);
lean_dec_ref(v_a_165_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27(lean_object* v_00_u03b1_172_, lean_object* v_x_173_, lean_object* v_commonMCtx_x3f_174_, lean_object* v_mctx_u2081_175_, lean_object* v_mctx_u2082_176_, uint8_t v_allowAssignmentDiff_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(v_x_173_, v_commonMCtx_x3f_174_, v_mctx_u2081_175_, v_mctx_u2082_176_, v_allowAssignmentDiff_177_, v_a_178_, v_a_179_, v_a_180_, v_a_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run_x27___boxed(lean_object* v_00_u03b1_184_, lean_object* v_x_185_, lean_object* v_commonMCtx_x3f_186_, lean_object* v_mctx_u2081_187_, lean_object* v_mctx_u2082_188_, lean_object* v_allowAssignmentDiff_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_195_; lean_object* v_res_196_; 
v_allowAssignmentDiff_boxed_195_ = lean_unbox(v_allowAssignmentDiff_189_);
v_res_196_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27(v_00_u03b1_184_, v_x_185_, v_commonMCtx_x3f_186_, v_mctx_u2081_187_, v_mctx_u2082_188_, v_allowAssignmentDiff_boxed_195_, v_a_190_, v_a_191_, v_a_192_, v_a_193_);
lean_dec(v_a_193_);
lean_dec_ref(v_a_192_);
lean_dec(v_a_191_);
lean_dec_ref(v_a_190_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___redArg(lean_object* v_x_197_, lean_object* v_commonMCtx_x3f_198_, lean_object* v_mctx_u2081_199_, lean_object* v_mctx_u2082_200_, uint8_t v_allowAssignmentDiff_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_, lean_object* v_a_205_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(v_x_197_, v_commonMCtx_x3f_198_, v_mctx_u2081_199_, v_mctx_u2082_200_, v_allowAssignmentDiff_201_, v_a_202_, v_a_203_, v_a_204_, v_a_205_);
if (lean_obj_tag(v___x_207_) == 0)
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_216_; 
v_a_208_ = lean_ctor_get(v___x_207_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_207_);
if (v_isSharedCheck_216_ == 0)
{
v___x_210_ = v___x_207_;
v_isShared_211_ = v_isSharedCheck_216_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_207_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_216_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v_fst_212_; lean_object* v___x_214_; 
v_fst_212_ = lean_ctor_get(v_a_208_, 0);
lean_inc(v_fst_212_);
lean_dec(v_a_208_);
if (v_isShared_211_ == 0)
{
lean_ctor_set(v___x_210_, 0, v_fst_212_);
v___x_214_ = v___x_210_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_fst_212_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
else
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
v_a_217_ = lean_ctor_get(v___x_207_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_207_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_207_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_207_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___redArg___boxed(lean_object* v_x_225_, lean_object* v_commonMCtx_x3f_226_, lean_object* v_mctx_u2081_227_, lean_object* v_mctx_u2082_228_, lean_object* v_allowAssignmentDiff_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_235_; lean_object* v_res_236_; 
v_allowAssignmentDiff_boxed_235_ = lean_unbox(v_allowAssignmentDiff_229_);
v_res_236_ = lp_aesop_Aesop_EqualUpToIdsM_run___redArg(v_x_225_, v_commonMCtx_x3f_226_, v_mctx_u2081_227_, v_mctx_u2082_228_, v_allowAssignmentDiff_boxed_235_, v_a_230_, v_a_231_, v_a_232_, v_a_233_);
lean_dec(v_a_233_);
lean_dec_ref(v_a_232_);
lean_dec(v_a_231_);
lean_dec_ref(v_a_230_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run(lean_object* v_00_u03b1_237_, lean_object* v_x_238_, lean_object* v_commonMCtx_x3f_239_, lean_object* v_mctx_u2081_240_, lean_object* v_mctx_u2082_241_, uint8_t v_allowAssignmentDiff_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_aesop_Aesop_EqualUpToIdsM_run___redArg(v_x_238_, v_commonMCtx_x3f_239_, v_mctx_u2081_240_, v_mctx_u2082_241_, v_allowAssignmentDiff_242_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIdsM_run___boxed(lean_object* v_00_u03b1_249_, lean_object* v_x_250_, lean_object* v_commonMCtx_x3f_251_, lean_object* v_mctx_u2081_252_, lean_object* v_mctx_u2082_253_, lean_object* v_allowAssignmentDiff_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_260_; lean_object* v_res_261_; 
v_allowAssignmentDiff_boxed_260_ = lean_unbox(v_allowAssignmentDiff_254_);
v_res_261_ = lp_aesop_Aesop_EqualUpToIdsM_run(v_00_u03b1_249_, v_x_250_, v_commonMCtx_x3f_251_, v_mctx_u2081_252_, v_mctx_u2082_253_, v_allowAssignmentDiff_boxed_260_, v_a_255_, v_a_256_, v_a_257_, v_a_258_);
lean_dec(v_a_258_);
lean_dec_ref(v_a_257_);
lean_dec(v_a_256_);
lean_dec_ref(v_a_255_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___redArg(lean_object* v_a_262_){
_start:
{
lean_object* v_commonMCtx_x3f_264_; lean_object* v___x_265_; 
v_commonMCtx_x3f_264_ = lean_ctor_get(v_a_262_, 0);
lean_inc(v_commonMCtx_x3f_264_);
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v_commonMCtx_x3f_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___redArg___boxed(lean_object* v_a_266_, lean_object* v_a_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___redArg(v_a_266_);
lean_dec_ref(v_a_266_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f(lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_commonMCtx_x3f_276_; lean_object* v___x_277_; 
v_commonMCtx_x3f_276_ = lean_ctor_get(v_a_269_, 0);
lean_inc(v_commonMCtx_x3f_276_);
v___x_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_277_, 0, v_commonMCtx_x3f_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f___boxed(lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_aesop_Aesop_EqualUpToIds_readCommonMCtx_x3f(v_a_278_, v_a_279_, v_a_280_, v_a_281_, v_a_282_, v_a_283_);
lean_dec(v_a_283_);
lean_dec_ref(v_a_282_);
lean_dec(v_a_281_);
lean_dec_ref(v_a_280_);
lean_dec(v_a_279_);
lean_dec_ref(v_a_278_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___redArg(lean_object* v_a_286_){
_start:
{
lean_object* v_mctx_u2081_288_; lean_object* v___x_289_; 
v_mctx_u2081_288_ = lean_ctor_get(v_a_286_, 1);
lean_inc_ref(v_mctx_u2081_288_);
v___x_289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_289_, 0, v_mctx_u2081_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___redArg___boxed(lean_object* v_a_290_, lean_object* v_a_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___redArg(v_a_290_);
lean_dec_ref(v_a_290_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081(lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_){
_start:
{
lean_object* v_mctx_u2081_300_; lean_object* v___x_301_; 
v_mctx_u2081_300_ = lean_ctor_get(v_a_293_, 1);
lean_inc_ref(v_mctx_u2081_300_);
v___x_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_301_, 0, v_mctx_u2081_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081___boxed(lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_, lean_object* v_a_307_, lean_object* v_a_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_aesop_Aesop_EqualUpToIds_readMCtx_u2081(v_a_302_, v_a_303_, v_a_304_, v_a_305_, v_a_306_, v_a_307_);
lean_dec(v_a_307_);
lean_dec_ref(v_a_306_);
lean_dec(v_a_305_);
lean_dec_ref(v_a_304_);
lean_dec(v_a_303_);
lean_dec_ref(v_a_302_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___redArg(lean_object* v_a_310_){
_start:
{
lean_object* v_mctx_u2082_312_; lean_object* v___x_313_; 
v_mctx_u2082_312_ = lean_ctor_get(v_a_310_, 2);
lean_inc_ref(v_mctx_u2082_312_);
v___x_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_313_, 0, v_mctx_u2082_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___redArg___boxed(lean_object* v_a_314_, lean_object* v_a_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___redArg(v_a_314_);
lean_dec_ref(v_a_314_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082(lean_object* v_a_317_, lean_object* v_a_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_, lean_object* v_a_322_){
_start:
{
lean_object* v_mctx_u2082_324_; lean_object* v___x_325_; 
v_mctx_u2082_324_ = lean_ctor_get(v_a_317_, 2);
lean_inc_ref(v_mctx_u2082_324_);
v___x_325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_325_, 0, v_mctx_u2082_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082___boxed(lean_object* v_a_326_, lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_, lean_object* v_a_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_aesop_Aesop_EqualUpToIds_readMCtx_u2082(v_a_326_, v_a_327_, v_a_328_, v_a_329_, v_a_330_, v_a_331_);
lean_dec(v_a_331_);
lean_dec_ref(v_a_330_);
lean_dec(v_a_329_);
lean_dec_ref(v_a_328_);
lean_dec(v_a_327_);
lean_dec_ref(v_a_326_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___redArg(lean_object* v_a_334_){
_start:
{
uint8_t v_allowAssignmentDiff_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v_allowAssignmentDiff_336_ = lean_ctor_get_uint8(v_a_334_, sizeof(void*)*3);
v___x_337_ = lean_box(v_allowAssignmentDiff_336_);
v___x_338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___redArg___boxed(lean_object* v_a_339_, lean_object* v_a_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___redArg(v_a_339_);
lean_dec_ref(v_a_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff(lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_){
_start:
{
uint8_t v_allowAssignmentDiff_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v_allowAssignmentDiff_349_ = lean_ctor_get_uint8(v_a_342_, sizeof(void*)*3);
v___x_350_ = lean_box(v_allowAssignmentDiff_349_);
v___x_351_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff___boxed(lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_aesop_Aesop_EqualUpToIds_readAllowAssignmentDiff(v_a_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_, v_a_357_);
lean_dec(v_a_357_);
lean_dec_ref(v_a_356_);
lean_dec(v_a_355_);
lean_dec_ref(v_a_354_);
lean_dec(v_a_353_);
lean_dec_ref(v_a_352_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg(lean_object* v_lmvarId_u2081_362_, lean_object* v_lmvarId_u2082_363_, lean_object* v_a_364_){
_start:
{
uint8_t v___y_367_; lean_object* v_commonMCtx_x3f_374_; 
v_commonMCtx_x3f_374_ = lean_ctor_get(v_a_364_, 0);
if (lean_obj_tag(v_commonMCtx_x3f_374_) == 0)
{
lean_object* v___x_375_; lean_object* v___x_376_; 
lean_dec(v_lmvarId_u2082_363_);
lean_dec(v_lmvarId_u2081_362_);
v___x_375_ = lean_box(0);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
else
{
lean_object* v_val_377_; lean_object* v_lDecls_378_; lean_object* v___x_379_; lean_object* v___x_380_; uint8_t v___x_381_; 
v_val_377_ = lean_ctor_get(v_commonMCtx_x3f_374_, 0);
v_lDecls_378_ = lean_ctor_get(v_val_377_, 4);
v___x_379_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__0));
v___x_380_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__1));
lean_inc(v_lmvarId_u2081_362_);
lean_inc_ref(v_lDecls_378_);
v___x_381_ = l_Lean_PersistentHashMap_contains___redArg(v___x_379_, v___x_380_, v_lDecls_378_, v_lmvarId_u2081_362_);
if (v___x_381_ == 0)
{
uint8_t v___x_382_; 
lean_inc(v_lmvarId_u2082_363_);
lean_inc_ref(v_lDecls_378_);
v___x_382_ = l_Lean_PersistentHashMap_contains___redArg(v___x_379_, v___x_380_, v_lDecls_378_, v_lmvarId_u2082_363_);
v___y_367_ = v___x_382_;
goto v___jp_366_;
}
else
{
v___y_367_ = v___x_381_;
goto v___jp_366_;
}
}
v___jp_366_:
{
if (v___y_367_ == 0)
{
lean_object* v___x_368_; lean_object* v___x_369_; 
lean_dec(v_lmvarId_u2082_363_);
lean_dec(v_lmvarId_u2081_362_);
v___x_368_ = lean_box(0);
v___x_369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
return v___x_369_;
}
else
{
uint8_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_370_ = l_Lean_instBEqLevelMVarId_beq(v_lmvarId_u2081_362_, v_lmvarId_u2082_363_);
lean_dec(v_lmvarId_u2082_363_);
lean_dec(v_lmvarId_u2081_362_);
v___x_371_ = lean_box(v___x_370_);
v___x_372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
v___x_373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
return v___x_373_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___boxed(lean_object* v_lmvarId_u2081_383_, lean_object* v_lmvarId_u2082_384_, lean_object* v_a_385_, lean_object* v_a_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg(v_lmvarId_u2081_383_, v_lmvarId_u2082_384_, v_a_385_);
lean_dec_ref(v_a_385_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f(lean_object* v_lmvarId_u2081_388_, lean_object* v_lmvarId_u2082_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg(v_lmvarId_u2081_388_, v_lmvarId_u2082_389_, v_a_390_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___boxed(lean_object* v_lmvarId_u2081_398_, lean_object* v_lmvarId_u2082_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f(v_lmvarId_u2081_398_, v_lmvarId_u2082_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_);
lean_dec(v_a_405_);
lean_dec_ref(v_a_404_);
lean_dec(v_a_403_);
lean_dec_ref(v_a_402_);
lean_dec(v_a_401_);
lean_dec_ref(v_a_400_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(lean_object* v_mvarId_u2081_408_, lean_object* v_mvarId_u2082_409_, lean_object* v_a_410_){
_start:
{
uint8_t v___y_413_; lean_object* v_commonMCtx_x3f_420_; 
v_commonMCtx_x3f_420_ = lean_ctor_get(v_a_410_, 0);
if (lean_obj_tag(v_commonMCtx_x3f_420_) == 0)
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = lean_box(0);
v___x_422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
return v___x_422_;
}
else
{
lean_object* v_val_423_; uint8_t v___x_424_; 
v_val_423_ = lean_ctor_get(v_commonMCtx_x3f_420_, 0);
v___x_424_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_val_423_, v_mvarId_u2081_408_);
if (v___x_424_ == 0)
{
uint8_t v___x_425_; 
v___x_425_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_val_423_, v_mvarId_u2082_409_);
v___y_413_ = v___x_425_;
goto v___jp_412_;
}
else
{
v___y_413_ = v___x_424_;
goto v___jp_412_;
}
}
v___jp_412_:
{
if (v___y_413_ == 0)
{
lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_414_ = lean_box(0);
v___x_415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
return v___x_415_;
}
else
{
uint8_t v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_416_ = l_Lean_instBEqMVarId_beq(v_mvarId_u2081_408_, v_mvarId_u2082_409_);
v___x_417_ = lean_box(v___x_416_);
v___x_418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
v___x_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
return v___x_419_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg___boxed(lean_object* v_mvarId_u2081_426_, lean_object* v_mvarId_u2082_427_, lean_object* v_a_428_, lean_object* v_a_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_426_, v_mvarId_u2082_427_, v_a_428_);
lean_dec_ref(v_a_428_);
lean_dec(v_mvarId_u2082_427_);
lean_dec(v_mvarId_u2081_426_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f(lean_object* v_mvarId_u2081_431_, lean_object* v_mvarId_u2082_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_431_, v_mvarId_u2082_432_, v_a_433_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___boxed(lean_object* v_mvarId_u2081_441_, lean_object* v_mvarId_u2082_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f(v_mvarId_u2081_441_, v_mvarId_u2082_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
lean_dec(v_a_448_);
lean_dec_ref(v_a_447_);
lean_dec(v_a_446_);
lean_dec_ref(v_a_445_);
lean_dec(v_a_444_);
lean_dec_ref(v_a_443_);
lean_dec(v_mvarId_u2082_442_);
lean_dec(v_mvarId_u2081_441_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorIdx(lean_object* v_x_451_){
_start:
{
switch(lean_obj_tag(v_x_451_))
{
case 0:
{
lean_object* v___x_452_; 
v___x_452_ = lean_unsigned_to_nat(0u);
return v___x_452_;
}
case 1:
{
lean_object* v___x_453_; 
v___x_453_ = lean_unsigned_to_nat(1u);
return v___x_453_;
}
default: 
{
lean_object* v___x_454_; 
v___x_454_ = lean_unsigned_to_nat(2u);
return v___x_454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorIdx___boxed(lean_object* v_x_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorIdx(v_x_455_);
lean_dec_ref(v_x_455_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(lean_object* v_t_457_, lean_object* v_k_458_){
_start:
{
if (lean_obj_tag(v_t_457_) == 0)
{
lean_object* v_mvarId_459_; lean_object* v___x_460_; 
v_mvarId_459_ = lean_ctor_get(v_t_457_, 0);
lean_inc(v_mvarId_459_);
lean_dec_ref_known(v_t_457_, 1);
v___x_460_ = lean_apply_1(v_k_458_, v_mvarId_459_);
return v___x_460_;
}
else
{
lean_object* v_e_461_; lean_object* v___x_462_; 
v_e_461_ = lean_ctor_get(v_t_457_, 0);
lean_inc_ref(v_e_461_);
lean_dec_ref(v_t_457_);
v___x_462_ = lean_apply_1(v_k_458_, v_e_461_);
return v___x_462_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim(lean_object* v_motive_463_, lean_object* v_ctorIdx_464_, lean_object* v_t_465_, lean_object* v_h_466_, lean_object* v_k_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_465_, v_k_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___boxed(lean_object* v_motive_469_, lean_object* v_ctorIdx_470_, lean_object* v_t_471_, lean_object* v_h_472_, lean_object* v_k_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim(v_motive_469_, v_ctorIdx_470_, v_t_471_, v_h_472_, v_k_473_);
lean_dec(v_ctorIdx_470_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_mvarId_elim___redArg(lean_object* v_t_475_, lean_object* v_mvarId_476_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_475_, v_mvarId_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_mvarId_elim(lean_object* v_motive_478_, lean_object* v_t_479_, lean_object* v_h_480_, lean_object* v_mvarId_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_479_, v_mvarId_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_expr_elim___redArg(lean_object* v_t_483_, lean_object* v_expr_484_){
_start:
{
lean_object* v___x_485_; 
v___x_485_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_483_, v_expr_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_expr_elim(lean_object* v_motive_486_, lean_object* v_t_487_, lean_object* v_h_488_, lean_object* v_expr_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_487_, v_expr_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_delayedAssignment_elim___redArg(lean_object* v_t_491_, lean_object* v_delayedAssignment_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_491_, v_delayedAssignment_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_MVarValue_delayedAssignment_elim(lean_object* v_motive_494_, lean_object* v_t_495_, lean_object* v_h_496_, lean_object* v_delayedAssignment_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_aesop_Aesop_EqualUpToIds_MVarValue_ctorElim___redArg(v_t_495_, v_delayedAssignment_497_);
return v___x_498_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_namesEqualUpToMacroScopes(lean_object* v_n_u2081_499_, lean_object* v_n_u2082_500_){
_start:
{
uint8_t v___y_506_; uint8_t v___x_507_; uint8_t v___x_508_; 
v___x_507_ = l_Lean_Name_hasMacroScopes(v_n_u2081_499_);
v___x_508_ = l_Lean_Name_hasMacroScopes(v_n_u2082_500_);
if (v___x_507_ == 0)
{
if (v___x_508_ == 0)
{
goto v___jp_501_;
}
else
{
v___y_506_ = v___x_507_;
goto v___jp_505_;
}
}
else
{
v___y_506_ = v___x_508_;
goto v___jp_505_;
}
v___jp_501_:
{
lean_object* v___x_502_; lean_object* v___x_503_; uint8_t v___x_504_; 
v___x_502_ = l_Lean_Name_eraseMacroScopes(v_n_u2081_499_);
v___x_503_ = l_Lean_Name_eraseMacroScopes(v_n_u2082_500_);
v___x_504_ = lean_name_eq(v___x_502_, v___x_503_);
lean_dec(v___x_503_);
lean_dec(v___x_502_);
return v___x_504_;
}
v___jp_505_:
{
if (v___y_506_ == 0)
{
return v___y_506_;
}
else
{
goto v___jp_501_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_namesEqualUpToMacroScopes___boxed(lean_object* v_n_u2081_509_, lean_object* v_n_u2082_510_){
_start:
{
uint8_t v_res_511_; lean_object* v_r_512_; 
v_res_511_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_namesEqualUpToMacroScopes(v_n_u2081_509_, v_n_u2082_510_);
lean_dec(v_n_u2082_510_);
lean_dec(v_n_u2081_509_);
v_r_512_ = lean_box(v_res_511_);
return v_r_512_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27(lean_object* v_x_513_, lean_object* v_x_514_, lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
lean_object* v_l_u2081_527_; lean_object* v_m_u2081_528_; lean_object* v_l_u2082_529_; lean_object* v_m_u2082_530_; lean_object* v___y_531_; lean_object* v___y_532_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_535_; lean_object* v___y_536_; 
switch(lean_obj_tag(v_x_513_))
{
case 0:
{
if (lean_obj_tag(v_x_514_) == 0)
{
goto v___jp_522_;
}
else
{
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
case 1:
{
if (lean_obj_tag(v_x_514_) == 1)
{
lean_object* v_a_545_; lean_object* v_a_546_; lean_object* v___x_547_; 
v_a_545_ = lean_ctor_get(v_x_513_, 0);
lean_inc(v_a_545_);
lean_dec_ref_known(v_x_513_, 1);
v_a_546_ = lean_ctor_get(v_x_514_, 0);
lean_inc(v_a_546_);
lean_dec_ref_known(v_x_514_, 1);
v___x_547_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_a_545_, v_a_546_, v_a_515_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_);
return v___x_547_;
}
else
{
lean_dec_ref_known(v_x_513_, 1);
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
case 2:
{
if (lean_obj_tag(v_x_514_) == 2)
{
lean_object* v_a_548_; lean_object* v_a_549_; lean_object* v_a_550_; lean_object* v_a_551_; 
v_a_548_ = lean_ctor_get(v_x_513_, 0);
lean_inc(v_a_548_);
v_a_549_ = lean_ctor_get(v_x_513_, 1);
lean_inc(v_a_549_);
lean_dec_ref_known(v_x_513_, 2);
v_a_550_ = lean_ctor_get(v_x_514_, 0);
lean_inc(v_a_550_);
v_a_551_ = lean_ctor_get(v_x_514_, 1);
lean_inc(v_a_551_);
lean_dec_ref_known(v_x_514_, 2);
v_l_u2081_527_ = v_a_548_;
v_m_u2081_528_ = v_a_549_;
v_l_u2082_529_ = v_a_550_;
v_m_u2082_530_ = v_a_551_;
v___y_531_ = v_a_515_;
v___y_532_ = v_a_516_;
v___y_533_ = v_a_517_;
v___y_534_ = v_a_518_;
v___y_535_ = v_a_519_;
v___y_536_ = v_a_520_;
goto v___jp_526_;
}
else
{
lean_dec_ref_known(v_x_513_, 2);
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
case 3:
{
if (lean_obj_tag(v_x_514_) == 3)
{
lean_object* v_a_552_; lean_object* v_a_553_; lean_object* v_a_554_; lean_object* v_a_555_; 
v_a_552_ = lean_ctor_get(v_x_513_, 0);
lean_inc(v_a_552_);
v_a_553_ = lean_ctor_get(v_x_513_, 1);
lean_inc(v_a_553_);
lean_dec_ref_known(v_x_513_, 2);
v_a_554_ = lean_ctor_get(v_x_514_, 0);
lean_inc(v_a_554_);
v_a_555_ = lean_ctor_get(v_x_514_, 1);
lean_inc(v_a_555_);
lean_dec_ref_known(v_x_514_, 2);
v_l_u2081_527_ = v_a_552_;
v_m_u2081_528_ = v_a_553_;
v_l_u2082_529_ = v_a_554_;
v_m_u2082_530_ = v_a_555_;
v___y_531_ = v_a_515_;
v___y_532_ = v_a_516_;
v___y_533_ = v_a_517_;
v___y_534_ = v_a_518_;
v___y_535_ = v_a_519_;
v___y_536_ = v_a_520_;
goto v___jp_526_;
}
else
{
lean_dec_ref_known(v_x_513_, 2);
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
case 4:
{
if (lean_obj_tag(v_x_514_) == 4)
{
lean_object* v_a_556_; lean_object* v_a_557_; uint8_t v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v_a_556_ = lean_ctor_get(v_x_513_, 0);
lean_inc(v_a_556_);
lean_dec_ref_known(v_x_513_, 1);
v_a_557_ = lean_ctor_get(v_x_514_, 0);
lean_inc(v_a_557_);
lean_dec_ref_known(v_x_514_, 1);
v___x_558_ = lean_name_eq(v_a_556_, v_a_557_);
lean_dec(v_a_557_);
lean_dec(v_a_556_);
v___x_559_ = lean_box(v___x_558_);
v___x_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_560_, 0, v___x_559_);
return v___x_560_;
}
else
{
lean_dec_ref_known(v_x_513_, 1);
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
default: 
{
if (lean_obj_tag(v_x_514_) == 5)
{
lean_object* v_a_561_; lean_object* v_a_562_; lean_object* v___x_563_; 
v_a_561_ = lean_ctor_get(v_x_513_, 0);
lean_inc_n(v_a_561_, 2);
lean_dec_ref_known(v_x_513_, 1);
v_a_562_ = lean_ctor_get(v_x_514_, 0);
lean_inc_n(v_a_562_, 2);
lean_dec_ref_known(v_x_514_, 1);
v___x_563_ = lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg(v_a_561_, v_a_562_, v_a_515_);
if (lean_obj_tag(v___x_563_) == 0)
{
lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_597_; 
v_a_564_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_597_ == 0)
{
v___x_566_ = v___x_563_;
v_isShared_567_ = v_isSharedCheck_597_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_563_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_597_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
if (lean_obj_tag(v_a_564_) == 1)
{
lean_object* v_val_568_; lean_object* v___x_570_; 
lean_dec(v_a_562_);
lean_dec(v_a_561_);
v_val_568_ = lean_ctor_get(v_a_564_, 0);
lean_inc(v_val_568_);
lean_dec_ref_known(v_a_564_, 1);
if (v_isShared_567_ == 0)
{
lean_ctor_set(v___x_566_, 0, v_val_568_);
v___x_570_ = v___x_566_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_val_568_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
return v___x_570_;
}
}
else
{
lean_object* v___x_572_; lean_object* v_equalLMVarIds_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
lean_dec(v_a_564_);
v___x_572_ = lean_st_ref_get(v_a_516_);
v_equalLMVarIds_573_ = lean_ctor_get(v___x_572_, 1);
lean_inc_ref(v_equalLMVarIds_573_);
lean_dec(v___x_572_);
v___x_574_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__0));
v___x_575_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_equalCommonLMVars_x3f___redArg___closed__1));
lean_inc(v_a_561_);
v___x_576_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_574_, v___x_575_, v_equalLMVarIds_573_, v_a_561_);
lean_dec_ref(v_equalLMVarIds_573_);
if (lean_obj_tag(v___x_576_) == 1)
{
lean_object* v_val_577_; uint8_t v___x_578_; lean_object* v___x_579_; lean_object* v___x_581_; 
lean_dec(v_a_561_);
v_val_577_ = lean_ctor_get(v___x_576_, 0);
lean_inc(v_val_577_);
lean_dec_ref_known(v___x_576_, 1);
v___x_578_ = l_Lean_instBEqLevelMVarId_beq(v_val_577_, v_a_562_);
lean_dec(v_a_562_);
lean_dec(v_val_577_);
v___x_579_ = lean_box(v___x_578_);
if (v_isShared_567_ == 0)
{
lean_ctor_set(v___x_566_, 0, v___x_579_);
v___x_581_ = v___x_566_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v___x_579_);
v___x_581_ = v_reuseFailAlloc_582_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
return v___x_581_;
}
}
else
{
lean_object* v___x_583_; lean_object* v_equalMVarIds_584_; lean_object* v_equalLMVarIds_585_; lean_object* v_leftUnassignedMVarValues_586_; lean_object* v_rightUnassignedMVarValues_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_596_; 
lean_dec(v___x_576_);
lean_del_object(v___x_566_);
v___x_583_ = lean_st_ref_take(v_a_516_);
v_equalMVarIds_584_ = lean_ctor_get(v___x_583_, 0);
v_equalLMVarIds_585_ = lean_ctor_get(v___x_583_, 1);
v_leftUnassignedMVarValues_586_ = lean_ctor_get(v___x_583_, 2);
v_rightUnassignedMVarValues_587_ = lean_ctor_get(v___x_583_, 3);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_596_ == 0)
{
v___x_589_ = v___x_583_;
v_isShared_590_ = v_isSharedCheck_596_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_587_);
lean_inc(v_leftUnassignedMVarValues_586_);
lean_inc(v_equalLMVarIds_585_);
lean_inc(v_equalMVarIds_584_);
lean_dec(v___x_583_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_596_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_591_; lean_object* v___x_593_; 
v___x_591_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_574_, v___x_575_, v_equalLMVarIds_585_, v_a_561_, v_a_562_);
if (v_isShared_590_ == 0)
{
lean_ctor_set(v___x_589_, 1, v___x_591_);
v___x_593_ = v___x_589_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_equalMVarIds_584_);
lean_ctor_set(v_reuseFailAlloc_595_, 1, v___x_591_);
lean_ctor_set(v_reuseFailAlloc_595_, 2, v_leftUnassignedMVarValues_586_);
lean_ctor_set(v_reuseFailAlloc_595_, 3, v_rightUnassignedMVarValues_587_);
v___x_593_ = v_reuseFailAlloc_595_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
lean_object* v___x_594_; 
v___x_594_ = lean_st_ref_set(v_a_516_, v___x_593_);
goto v___jp_522_;
}
}
}
}
}
}
else
{
lean_object* v_a_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_605_; 
lean_dec(v_a_562_);
lean_dec(v_a_561_);
v_a_598_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_605_ == 0)
{
v___x_600_ = v___x_563_;
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_a_598_);
lean_dec(v___x_563_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_a_598_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
else
{
lean_dec_ref_known(v_x_513_, 1);
lean_dec(v_x_514_);
goto v___jp_541_;
}
}
}
v___jp_522_:
{
uint8_t v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_523_ = 1;
v___x_524_ = lean_box(v___x_523_);
v___x_525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
return v___x_525_;
}
v___jp_526_:
{
lean_object* v___x_537_; 
v___x_537_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_l_u2081_527_, v_l_u2082_529_, v___y_531_, v___y_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_);
if (lean_obj_tag(v___x_537_) == 0)
{
lean_object* v_a_538_; uint8_t v___x_539_; 
v_a_538_ = lean_ctor_get(v___x_537_, 0);
lean_inc(v_a_538_);
v___x_539_ = lean_unbox(v_a_538_);
lean_dec(v_a_538_);
if (v___x_539_ == 0)
{
lean_dec(v_m_u2082_530_);
lean_dec(v_m_u2081_528_);
return v___x_537_;
}
else
{
lean_object* v___x_540_; 
lean_dec_ref_known(v___x_537_, 1);
v___x_540_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_m_u2081_528_, v_m_u2082_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_);
return v___x_540_;
}
}
else
{
lean_dec(v_m_u2082_530_);
lean_dec(v_m_u2081_528_);
return v___x_537_;
}
}
v___jp_541_:
{
uint8_t v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_542_ = 0;
v___x_543_ = lean_box(v___x_542_);
v___x_544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_544_, 0, v___x_543_);
return v___x_544_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(lean_object* v_l_u2081_606_, lean_object* v_l_u2082_607_, lean_object* v_a_608_, lean_object* v_a_609_, lean_object* v_a_610_, lean_object* v_a_611_, lean_object* v_a_612_, lean_object* v_a_613_){
_start:
{
size_t v___x_615_; size_t v___x_616_; uint8_t v___x_617_; 
v___x_615_ = lean_ptr_addr(v_l_u2081_606_);
v___x_616_ = lean_ptr_addr(v_l_u2082_607_);
v___x_617_ = lean_usize_dec_eq(v___x_615_, v___x_616_);
if (v___x_617_ == 0)
{
lean_object* v___x_618_; 
v___x_618_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27(v_l_u2081_606_, v_l_u2082_607_, v_a_608_, v_a_609_, v_a_610_, v_a_611_, v_a_612_, v_a_613_);
return v___x_618_;
}
else
{
lean_object* v___x_619_; lean_object* v___x_620_; 
lean_dec(v_l_u2082_607_);
lean_dec(v_l_u2081_606_);
v___x_619_ = lean_box(v___x_617_);
v___x_620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
return v___x_620_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore___boxed(lean_object* v_l_u2081_621_, lean_object* v_l_u2082_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_l_u2081_621_, v_l_u2082_622_, v_a_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_);
lean_dec(v_a_628_);
lean_dec_ref(v_a_627_);
lean_dec(v_a_626_);
lean_dec_ref(v_a_625_);
lean_dec(v_a_624_);
lean_dec_ref(v_a_623_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27___boxed(lean_object* v_x_631_, lean_object* v_x_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_){
_start:
{
lean_object* v_res_640_; 
v_res_640_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore_x27(v_x_631_, v_x_632_, v_a_633_, v_a_634_, v_a_635_, v_a_636_, v_a_637_, v_a_638_);
lean_dec(v_a_638_);
lean_dec_ref(v_a_637_);
lean_dec(v_a_636_);
lean_dec_ref(v_a_635_);
lean_dec(v_a_634_);
lean_dec_ref(v_a_633_);
return v_res_640_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___lam__0(lean_object* v_x1_641_, lean_object* v_x2_642_){
_start:
{
uint8_t v___x_643_; 
v___x_643_ = l_Lean_LocalDecl_isImplementationDetail(v_x2_642_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; 
v___x_644_ = lean_array_push(v_x1_641_, v_x2_642_);
return v___x_644_;
}
else
{
lean_dec_ref(v_x2_642_);
return v_x1_641_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg(lean_object* v_lctx_665_){
_start:
{
lean_object* v___f_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v___f_667_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__0));
lean_inc_ref(v_lctx_665_);
v___x_668_ = lean_local_ctx_num_indices(v_lctx_665_);
v___x_669_ = lean_mk_empty_array_with_capacity(v___x_668_);
lean_dec(v___x_668_);
v___x_670_ = lean_unsigned_to_nat(0u);
v___x_671_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10));
v___x_672_ = l_Lean_LocalContext_foldlM___redArg(v___x_671_, v_lctx_665_, v___f_667_, v___x_669_, v___x_670_);
v___x_673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_673_, 0, v___x_672_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___boxed(lean_object* v_lctx_674_, lean_object* v_a_675_){
_start:
{
lean_object* v_res_676_; 
v_res_676_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg(v_lctx_674_);
return v_res_676_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls(lean_object* v_lctx_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_, lean_object* v_a_681_, lean_object* v_a_682_, lean_object* v_a_683_){
_start:
{
lean_object* v___f_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; 
v___f_685_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__0));
lean_inc_ref(v_lctx_677_);
v___x_686_ = lean_local_ctx_num_indices(v_lctx_677_);
v___x_687_ = lean_mk_empty_array_with_capacity(v___x_686_);
lean_dec(v___x_686_);
v___x_688_ = lean_unsigned_to_nat(0u);
v___x_689_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10));
v___x_690_ = l_Lean_LocalContext_foldlM___redArg(v___x_689_, v_lctx_677_, v___f_685_, v___x_687_, v___x_688_);
v___x_691_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_691_, 0, v___x_690_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___boxed(lean_object* v_lctx_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls(v_lctx_692_, v_a_693_, v_a_694_, v_a_695_, v_a_696_, v_a_697_, v_a_698_);
lean_dec(v_a_698_);
lean_dec_ref(v_a_697_);
lean_dec(v_a_696_);
lean_dec_ref(v_a_695_);
lean_dec(v_a_694_);
lean_dec_ref(v_a_693_);
return v_res_700_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg(lean_object* v_mvarId_701_, lean_object* v___y_702_){
_start:
{
lean_object* v___x_704_; lean_object* v_mctx_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_704_ = lean_st_ref_get(v___y_702_);
v_mctx_705_ = lean_ctor_get(v___x_704_, 0);
lean_inc_ref(v_mctx_705_);
lean_dec(v___x_704_);
v___x_706_ = l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(v_mctx_705_, v_mvarId_701_);
lean_dec_ref(v_mctx_705_);
v___x_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg___boxed(lean_object* v_mvarId_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg(v_mvarId_708_, v___y_709_);
lean_dec(v___y_709_);
lean_dec(v_mvarId_708_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0(lean_object* v_mvarId_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_){
_start:
{
lean_object* v___x_718_; 
v___x_718_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg(v_mvarId_712_, v___y_714_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___boxed(lean_object* v_mvarId_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0(v_mvarId_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
lean_dec(v___y_723_);
lean_dec_ref(v___y_722_);
lean_dec(v___y_721_);
lean_dec_ref(v___y_720_);
lean_dec(v_mvarId_719_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(lean_object* v_mctx_726_, lean_object* v_x_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), v_mctx_726_, v_x_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
if (lean_obj_tag(v___x_733_) == 0)
{
lean_object* v_a_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_741_; 
v_a_734_ = lean_ctor_get(v___x_733_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_733_);
if (v_isSharedCheck_741_ == 0)
{
v___x_736_ = v___x_733_;
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_a_734_);
lean_dec(v___x_733_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v___x_739_; 
if (v_isShared_737_ == 0)
{
v___x_739_ = v___x_736_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_a_734_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
v_a_742_ = lean_ctor_get(v___x_733_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_733_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_733_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_733_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg___boxed(lean_object* v_mctx_750_, lean_object* v_x_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(v_mctx_750_, v_x_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_);
lean_dec(v___y_755_);
lean_dec_ref(v___y_754_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
return v_res_757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1(lean_object* v_00_u03b1_758_, lean_object* v_mctx_759_, lean_object* v_x_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
lean_object* v___x_766_; 
v___x_766_ = lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(v_mctx_759_, v_x_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___boxed(lean_object* v_00_u03b1_767_, lean_object* v_mctx_768_, lean_object* v_x_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1(v_00_u03b1_767_, v_mctx_768_, v_x_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0(lean_object* v_m_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v___x_782_; 
v___x_782_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__0___redArg(v_m_776_, v___y_778_);
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; lean_object* v___x_785_; uint8_t v_isShared_786_; uint8_t v_isSharedCheck_802_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_802_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_802_ == 0)
{
v___x_785_ = v___x_782_;
v_isShared_786_ = v_isSharedCheck_802_;
goto v_resetjp_784_;
}
else
{
lean_inc(v_a_783_);
lean_dec(v___x_782_);
v___x_785_ = lean_box(0);
v_isShared_786_ = v_isSharedCheck_802_;
goto v_resetjp_784_;
}
v_resetjp_784_:
{
if (lean_obj_tag(v_a_783_) == 1)
{
lean_object* v_val_787_; lean_object* v___x_789_; uint8_t v_isShared_790_; uint8_t v_isSharedCheck_797_; 
lean_dec(v_m_776_);
v_val_787_ = lean_ctor_get(v_a_783_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v_a_783_);
if (v_isSharedCheck_797_ == 0)
{
v___x_789_ = v_a_783_;
v_isShared_790_ = v_isSharedCheck_797_;
goto v_resetjp_788_;
}
else
{
lean_inc(v_val_787_);
lean_dec(v_a_783_);
v___x_789_ = lean_box(0);
v_isShared_790_ = v_isSharedCheck_797_;
goto v_resetjp_788_;
}
v_resetjp_788_:
{
lean_object* v___x_792_; 
if (v_isShared_790_ == 0)
{
lean_ctor_set_tag(v___x_789_, 2);
v___x_792_ = v___x_789_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_val_787_);
v___x_792_ = v_reuseFailAlloc_796_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
lean_object* v___x_794_; 
if (v_isShared_786_ == 0)
{
lean_ctor_set(v___x_785_, 0, v___x_792_);
v___x_794_ = v___x_785_;
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
}
else
{
lean_object* v___x_798_; lean_object* v___x_800_; 
lean_dec(v_a_783_);
v___x_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_798_, 0, v_m_776_);
if (v_isShared_786_ == 0)
{
lean_ctor_set(v___x_785_, 0, v___x_798_);
v___x_800_ = v___x_785_;
goto v_reusejp_799_;
}
else
{
lean_object* v_reuseFailAlloc_801_; 
v_reuseFailAlloc_801_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_801_, 0, v___x_798_);
v___x_800_ = v_reuseFailAlloc_801_;
goto v_reusejp_799_;
}
v_reusejp_799_:
{
return v___x_800_;
}
}
}
}
else
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_810_; 
lean_dec(v_m_776_);
v_a_803_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_810_ == 0)
{
v___x_805_ = v___x_782_;
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___x_782_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v___x_808_; 
if (v_isShared_806_ == 0)
{
v___x_808_ = v___x_805_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_a_803_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0___boxed(lean_object* v_m_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_){
_start:
{
lean_object* v_res_817_; 
v_res_817_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0(v_m_811_, v___y_812_, v___y_813_, v___y_814_, v___y_815_);
lean_dec(v___y_815_);
lean_dec_ref(v___y_814_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
return v_res_817_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(lean_object* v_mctx_818_, lean_object* v_m_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_, lean_object* v_a_823_){
_start:
{
lean_object* v___f_825_; lean_object* v___x_826_; 
v___f_825_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___lam__0___boxed), 6, 1);
lean_closure_set(v___f_825_, 0, v_m_819_);
v___x_826_ = lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(v_mctx_818_, v___f_825_, v_a_820_, v_a_821_, v_a_822_, v_a_823_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar___boxed(lean_object* v_mctx_827_, lean_object* v_m_828_, lean_object* v_a_829_, lean_object* v_a_830_, lean_object* v_a_831_, lean_object* v_a_832_, lean_object* v_a_833_){
_start:
{
lean_object* v_res_834_; 
v_res_834_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(v_mctx_827_, v_m_828_, v_a_829_, v_a_830_, v_a_831_, v_a_832_);
lean_dec(v_a_832_);
lean_dec_ref(v_a_831_);
lean_dec(v_a_830_);
lean_dec_ref(v_a_829_);
return v_res_834_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0(lean_object* v_msgData_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v___x_841_; lean_object* v_env_842_; lean_object* v___x_843_; lean_object* v_mctx_844_; lean_object* v_lctx_845_; lean_object* v_options_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
v___x_841_ = lean_st_ref_get(v___y_839_);
v_env_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc_ref(v_env_842_);
lean_dec(v___x_841_);
v___x_843_ = lean_st_ref_get(v___y_837_);
v_mctx_844_ = lean_ctor_get(v___x_843_, 0);
lean_inc_ref(v_mctx_844_);
lean_dec(v___x_843_);
v_lctx_845_ = lean_ctor_get(v___y_836_, 2);
v_options_846_ = lean_ctor_get(v___y_838_, 2);
lean_inc_ref(v_options_846_);
lean_inc_ref(v_lctx_845_);
v___x_847_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_847_, 0, v_env_842_);
lean_ctor_set(v___x_847_, 1, v_mctx_844_);
lean_ctor_set(v___x_847_, 2, v_lctx_845_);
lean_ctor_set(v___x_847_, 3, v_options_846_);
v___x_848_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_848_, 0, v___x_847_);
lean_ctor_set(v___x_848_, 1, v_msgData_835_);
v___x_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0___boxed(lean_object* v_msgData_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_){
_start:
{
lean_object* v_res_856_; 
v_res_856_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0(v_msgData_850_, v___y_851_, v___y_852_, v___y_853_, v___y_854_);
lean_dec(v___y_854_);
lean_dec_ref(v___y_853_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
return v_res_856_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg(lean_object* v_lctx_857_, lean_object* v_localInsts_858_, lean_object* v_x_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_){
_start:
{
lean_object* v___x_865_; 
v___x_865_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_857_, v_localInsts_858_, v_x_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_);
if (lean_obj_tag(v___x_865_) == 0)
{
lean_object* v_a_866_; lean_object* v___x_868_; uint8_t v_isShared_869_; uint8_t v_isSharedCheck_873_; 
v_a_866_ = lean_ctor_get(v___x_865_, 0);
v_isSharedCheck_873_ = !lean_is_exclusive(v___x_865_);
if (v_isSharedCheck_873_ == 0)
{
v___x_868_ = v___x_865_;
v_isShared_869_ = v_isSharedCheck_873_;
goto v_resetjp_867_;
}
else
{
lean_inc(v_a_866_);
lean_dec(v___x_865_);
v___x_868_ = lean_box(0);
v_isShared_869_ = v_isSharedCheck_873_;
goto v_resetjp_867_;
}
v_resetjp_867_:
{
lean_object* v___x_871_; 
if (v_isShared_869_ == 0)
{
v___x_871_ = v___x_868_;
goto v_reusejp_870_;
}
else
{
lean_object* v_reuseFailAlloc_872_; 
v_reuseFailAlloc_872_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_872_, 0, v_a_866_);
v___x_871_ = v_reuseFailAlloc_872_;
goto v_reusejp_870_;
}
v_reusejp_870_:
{
return v___x_871_;
}
}
}
else
{
lean_object* v_a_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_881_; 
v_a_874_ = lean_ctor_get(v___x_865_, 0);
v_isSharedCheck_881_ = !lean_is_exclusive(v___x_865_);
if (v_isSharedCheck_881_ == 0)
{
v___x_876_ = v___x_865_;
v_isShared_877_ = v_isSharedCheck_881_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_a_874_);
lean_dec(v___x_865_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_881_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v___x_879_; 
if (v_isShared_877_ == 0)
{
v___x_879_ = v___x_876_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_880_; 
v_reuseFailAlloc_880_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_880_, 0, v_a_874_);
v___x_879_ = v_reuseFailAlloc_880_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
return v___x_879_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg___boxed(lean_object* v_lctx_882_, lean_object* v_localInsts_883_, lean_object* v_x_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_){
_start:
{
lean_object* v_res_890_; 
v_res_890_ = lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg(v_lctx_882_, v_localInsts_883_, v_x_884_, v___y_885_, v___y_886_, v___y_887_, v___y_888_);
lean_dec(v___y_888_);
lean_dec_ref(v___y_887_);
lean_dec(v___y_886_);
lean_dec_ref(v___y_885_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1(lean_object* v_00_u03b1_891_, lean_object* v_lctx_892_, lean_object* v_localInsts_893_, lean_object* v_x_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___redArg(v_lctx_892_, v_localInsts_893_, v_x_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___boxed(lean_object* v_00_u03b1_901_, lean_object* v_lctx_902_, lean_object* v_localInsts_903_, lean_object* v_x_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1(v_00_u03b1_901_, v_lctx_902_, v_localInsts_903_, v_x_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr(lean_object* v_mctx_911_, lean_object* v_lctx_912_, lean_object* v_localInstances_913_, lean_object* v_e_914_, lean_object* v_a_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v___x_920_ = l_Lean_MessageData_ofExpr(v_e_914_);
v___x_921_ = lean_alloc_closure((void*)(lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0___boxed), 6, 1);
lean_closure_set(v___x_921_, 0, v___x_920_);
v___x_922_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withLCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__1___boxed), 9, 4);
lean_closure_set(v___x_922_, 0, lean_box(0));
lean_closure_set(v___x_922_, 1, v_lctx_912_);
lean_closure_set(v___x_922_, 2, v_localInstances_913_);
lean_closure_set(v___x_922_, 3, v___x_921_);
v___x_923_ = lp_aesop_Lean_Meta_withMCtx___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar_spec__1___redArg(v_mctx_911_, v___x_922_, v_a_915_, v_a_916_, v_a_917_, v_a_918_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr___boxed(lean_object* v_mctx_924_, lean_object* v_lctx_925_, lean_object* v_localInstances_926_, lean_object* v_e_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_, lean_object* v_a_931_, lean_object* v_a_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr(v_mctx_924_, v_lctx_925_, v_localInstances_926_, v_e_927_, v_a_928_, v_a_929_, v_a_930_, v_a_931_);
lean_dec(v_a_931_);
lean_dec_ref(v_a_930_);
lean_dec(v_a_929_);
lean_dec_ref(v_a_928_);
return v_res_933_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0(lean_object* v___x_934_, lean_object* v_a_935_, lean_object* v_x_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_){
_start:
{
lean_object* v_snd_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_988_; 
v_snd_947_ = lean_ctor_get(v___y_937_, 1);
v_isSharedCheck_988_ = !lean_is_exclusive(v___y_937_);
if (v_isSharedCheck_988_ == 0)
{
lean_object* v_unused_989_; 
v_unused_989_ = lean_ctor_get(v___y_937_, 0);
lean_dec(v_unused_989_);
v___x_949_ = v___y_937_;
v_isShared_950_ = v_isSharedCheck_988_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_snd_947_);
lean_dec(v___y_937_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_988_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
if (lean_obj_tag(v_snd_947_) == 0)
{
lean_object* v___x_952_; 
lean_dec(v_a_935_);
if (v_isShared_950_ == 0)
{
lean_ctor_set(v___x_949_, 0, v___x_934_);
v___x_952_ = v___x_949_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v___x_934_);
lean_ctor_set(v_reuseFailAlloc_955_, 1, v_snd_947_);
v___x_952_ = v_reuseFailAlloc_955_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
v___x_954_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_954_, 0, v___x_953_);
return v___x_954_;
}
}
else
{
lean_object* v_head_956_; lean_object* v_tail_957_; lean_object* v___x_958_; 
v_head_956_ = lean_ctor_get(v_snd_947_, 0);
lean_inc(v_head_956_);
v_tail_957_ = lean_ctor_get(v_snd_947_, 1);
lean_inc(v_tail_957_);
lean_dec_ref_known(v_snd_947_, 2);
v___x_958_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_a_935_, v_head_956_, v___y_940_, v___y_941_, v___y_942_, v___y_943_, v___y_944_, v___y_945_);
if (lean_obj_tag(v___x_958_) == 0)
{
lean_object* v_a_959_; lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_979_; 
v_a_959_ = lean_ctor_get(v___x_958_, 0);
v_isSharedCheck_979_ = !lean_is_exclusive(v___x_958_);
if (v_isSharedCheck_979_ == 0)
{
v___x_961_ = v___x_958_;
v_isShared_962_ = v_isSharedCheck_979_;
goto v_resetjp_960_;
}
else
{
lean_inc(v_a_959_);
lean_dec(v___x_958_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_979_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
uint8_t v___x_963_; 
v___x_963_ = lean_unbox(v_a_959_);
if (v___x_963_ == 0)
{
lean_object* v___x_964_; lean_object* v___x_966_; 
lean_dec(v___x_934_);
v___x_964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_964_, 0, v_a_959_);
if (v_isShared_950_ == 0)
{
lean_ctor_set(v___x_949_, 1, v_tail_957_);
lean_ctor_set(v___x_949_, 0, v___x_964_);
v___x_966_ = v___x_949_;
goto v_reusejp_965_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v___x_964_);
lean_ctor_set(v_reuseFailAlloc_971_, 1, v_tail_957_);
v___x_966_ = v_reuseFailAlloc_971_;
goto v_reusejp_965_;
}
v_reusejp_965_:
{
lean_object* v___x_967_; lean_object* v___x_969_; 
v___x_967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_967_, 0, v___x_966_);
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 0, v___x_967_);
v___x_969_ = v___x_961_;
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
}
else
{
lean_object* v___x_973_; 
lean_dec(v_a_959_);
if (v_isShared_950_ == 0)
{
lean_ctor_set(v___x_949_, 1, v_tail_957_);
lean_ctor_set(v___x_949_, 0, v___x_934_);
v___x_973_ = v___x_949_;
goto v_reusejp_972_;
}
else
{
lean_object* v_reuseFailAlloc_978_; 
v_reuseFailAlloc_978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_978_, 0, v___x_934_);
lean_ctor_set(v_reuseFailAlloc_978_, 1, v_tail_957_);
v___x_973_ = v_reuseFailAlloc_978_;
goto v_reusejp_972_;
}
v_reusejp_972_:
{
lean_object* v___x_974_; lean_object* v___x_976_; 
v___x_974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 0, v___x_974_);
v___x_976_ = v___x_961_;
goto v_reusejp_975_;
}
else
{
lean_object* v_reuseFailAlloc_977_; 
v_reuseFailAlloc_977_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_977_, 0, v___x_974_);
v___x_976_ = v_reuseFailAlloc_977_;
goto v_reusejp_975_;
}
v_reusejp_975_:
{
return v___x_976_;
}
}
}
}
}
else
{
lean_object* v_a_980_; lean_object* v___x_982_; uint8_t v_isShared_983_; uint8_t v_isSharedCheck_987_; 
lean_dec(v_tail_957_);
lean_del_object(v___x_949_);
lean_dec(v___x_934_);
v_a_980_ = lean_ctor_get(v___x_958_, 0);
v_isSharedCheck_987_ = !lean_is_exclusive(v___x_958_);
if (v_isSharedCheck_987_ == 0)
{
v___x_982_ = v___x_958_;
v_isShared_983_ = v_isSharedCheck_987_;
goto v_resetjp_981_;
}
else
{
lean_inc(v_a_980_);
lean_dec(v___x_958_);
v___x_982_ = lean_box(0);
v_isShared_983_ = v_isSharedCheck_987_;
goto v_resetjp_981_;
}
v_resetjp_981_:
{
lean_object* v___x_985_; 
if (v_isShared_983_ == 0)
{
v___x_985_ = v___x_982_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v_a_980_);
v___x_985_ = v_reuseFailAlloc_986_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
return v___x_985_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0___boxed(lean_object* v___x_990_, lean_object* v_a_991_, lean_object* v_x_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_){
_start:
{
lean_object* v_res_1003_; 
v_res_1003_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___lam__0(v___x_990_, v_a_991_, v_x_992_, v___y_993_, v___y_994_, v___y_995_, v___y_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_);
lean_dec(v___y_1001_);
lean_dec_ref(v___y_1000_);
lean_dec(v___y_999_);
lean_dec_ref(v___y_998_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec_ref(v___y_995_);
lean_dec(v___y_994_);
return v_res_1003_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27___redArg(lean_object* v_x_1004_, lean_object* v_x_1005_){
_start:
{
if (lean_obj_tag(v_x_1005_) == 0)
{
return v_x_1004_;
}
else
{
lean_object* v_key_1006_; lean_object* v_value_1007_; lean_object* v_tail_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1031_; 
v_key_1006_ = lean_ctor_get(v_x_1005_, 0);
v_value_1007_ = lean_ctor_get(v_x_1005_, 1);
v_tail_1008_ = lean_ctor_get(v_x_1005_, 2);
v_isSharedCheck_1031_ = !lean_is_exclusive(v_x_1005_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_1010_ = v_x_1005_;
v_isShared_1011_ = v_isSharedCheck_1031_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_tail_1008_);
lean_inc(v_value_1007_);
lean_inc(v_key_1006_);
lean_dec(v_x_1005_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1031_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1012_; uint64_t v___x_1013_; uint64_t v___x_1014_; uint64_t v___x_1015_; uint64_t v_fold_1016_; uint64_t v___x_1017_; uint64_t v___x_1018_; uint64_t v___x_1019_; size_t v___x_1020_; size_t v___x_1021_; size_t v___x_1022_; size_t v___x_1023_; size_t v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1027_; 
v___x_1012_ = lean_array_get_size(v_x_1004_);
v___x_1013_ = l_Lean_instHashableMVarId_hash(v_key_1006_);
v___x_1014_ = 32ULL;
v___x_1015_ = lean_uint64_shift_right(v___x_1013_, v___x_1014_);
v_fold_1016_ = lean_uint64_xor(v___x_1013_, v___x_1015_);
v___x_1017_ = 16ULL;
v___x_1018_ = lean_uint64_shift_right(v_fold_1016_, v___x_1017_);
v___x_1019_ = lean_uint64_xor(v_fold_1016_, v___x_1018_);
v___x_1020_ = lean_uint64_to_usize(v___x_1019_);
v___x_1021_ = lean_usize_of_nat(v___x_1012_);
v___x_1022_ = ((size_t)1ULL);
v___x_1023_ = lean_usize_sub(v___x_1021_, v___x_1022_);
v___x_1024_ = lean_usize_land(v___x_1020_, v___x_1023_);
v___x_1025_ = lean_array_uget_borrowed(v_x_1004_, v___x_1024_);
lean_inc(v___x_1025_);
if (v_isShared_1011_ == 0)
{
lean_ctor_set(v___x_1010_, 2, v___x_1025_);
v___x_1027_ = v___x_1010_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v_key_1006_);
lean_ctor_set(v_reuseFailAlloc_1030_, 1, v_value_1007_);
lean_ctor_set(v_reuseFailAlloc_1030_, 2, v___x_1025_);
v___x_1027_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
lean_object* v___x_1028_; 
v___x_1028_ = lean_array_uset(v_x_1004_, v___x_1024_, v___x_1027_);
v_x_1004_ = v___x_1028_;
v_x_1005_ = v_tail_1008_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25___redArg(lean_object* v_i_1032_, lean_object* v_source_1033_, lean_object* v_target_1034_){
_start:
{
lean_object* v___x_1035_; uint8_t v___x_1036_; 
v___x_1035_ = lean_array_get_size(v_source_1033_);
v___x_1036_ = lean_nat_dec_lt(v_i_1032_, v___x_1035_);
if (v___x_1036_ == 0)
{
lean_dec_ref(v_source_1033_);
lean_dec(v_i_1032_);
return v_target_1034_;
}
else
{
lean_object* v_es_1037_; lean_object* v___x_1038_; lean_object* v_source_1039_; lean_object* v_target_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; 
v_es_1037_ = lean_array_fget(v_source_1033_, v_i_1032_);
v___x_1038_ = lean_box(0);
v_source_1039_ = lean_array_fset(v_source_1033_, v_i_1032_, v___x_1038_);
v_target_1040_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27___redArg(v_target_1034_, v_es_1037_);
v___x_1041_ = lean_unsigned_to_nat(1u);
v___x_1042_ = lean_nat_add(v_i_1032_, v___x_1041_);
lean_dec(v_i_1032_);
v_i_1032_ = v___x_1042_;
v_source_1033_ = v_source_1039_;
v_target_1034_ = v_target_1040_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21___redArg(lean_object* v_data_1044_){
_start:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v_nbuckets_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; 
v___x_1045_ = lean_array_get_size(v_data_1044_);
v___x_1046_ = lean_unsigned_to_nat(2u);
v_nbuckets_1047_ = lean_nat_mul(v___x_1045_, v___x_1046_);
v___x_1048_ = lean_unsigned_to_nat(0u);
v___x_1049_ = lean_box(0);
v___x_1050_ = lean_mk_array(v_nbuckets_1047_, v___x_1049_);
v___x_1051_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25___redArg(v___x_1048_, v_data_1044_, v___x_1050_);
return v___x_1051_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22___redArg(lean_object* v_a_1052_, lean_object* v_b_1053_, lean_object* v_x_1054_){
_start:
{
if (lean_obj_tag(v_x_1054_) == 0)
{
lean_dec(v_b_1053_);
lean_dec(v_a_1052_);
return v_x_1054_;
}
else
{
lean_object* v_key_1055_; lean_object* v_value_1056_; lean_object* v_tail_1057_; lean_object* v___x_1059_; uint8_t v_isShared_1060_; uint8_t v_isSharedCheck_1069_; 
v_key_1055_ = lean_ctor_get(v_x_1054_, 0);
v_value_1056_ = lean_ctor_get(v_x_1054_, 1);
v_tail_1057_ = lean_ctor_get(v_x_1054_, 2);
v_isSharedCheck_1069_ = !lean_is_exclusive(v_x_1054_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1059_ = v_x_1054_;
v_isShared_1060_ = v_isSharedCheck_1069_;
goto v_resetjp_1058_;
}
else
{
lean_inc(v_tail_1057_);
lean_inc(v_value_1056_);
lean_inc(v_key_1055_);
lean_dec(v_x_1054_);
v___x_1059_ = lean_box(0);
v_isShared_1060_ = v_isSharedCheck_1069_;
goto v_resetjp_1058_;
}
v_resetjp_1058_:
{
uint8_t v___x_1061_; 
v___x_1061_ = l_Lean_instBEqMVarId_beq(v_key_1055_, v_a_1052_);
if (v___x_1061_ == 0)
{
lean_object* v___x_1062_; lean_object* v___x_1064_; 
v___x_1062_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22___redArg(v_a_1052_, v_b_1053_, v_tail_1057_);
if (v_isShared_1060_ == 0)
{
lean_ctor_set(v___x_1059_, 2, v___x_1062_);
v___x_1064_ = v___x_1059_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_key_1055_);
lean_ctor_set(v_reuseFailAlloc_1065_, 1, v_value_1056_);
lean_ctor_set(v_reuseFailAlloc_1065_, 2, v___x_1062_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
else
{
lean_object* v___x_1067_; 
lean_dec(v_value_1056_);
lean_dec(v_key_1055_);
if (v_isShared_1060_ == 0)
{
lean_ctor_set(v___x_1059_, 1, v_b_1053_);
lean_ctor_set(v___x_1059_, 0, v_a_1052_);
v___x_1067_ = v___x_1059_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v_a_1052_);
lean_ctor_set(v_reuseFailAlloc_1068_, 1, v_b_1053_);
lean_ctor_set(v_reuseFailAlloc_1068_, 2, v_tail_1057_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg(lean_object* v_a_1070_, lean_object* v_x_1071_){
_start:
{
if (lean_obj_tag(v_x_1071_) == 0)
{
uint8_t v___x_1072_; 
v___x_1072_ = 0;
return v___x_1072_;
}
else
{
lean_object* v_key_1073_; lean_object* v_tail_1074_; uint8_t v___x_1075_; 
v_key_1073_ = lean_ctor_get(v_x_1071_, 0);
v_tail_1074_ = lean_ctor_get(v_x_1071_, 2);
v___x_1075_ = l_Lean_instBEqMVarId_beq(v_key_1073_, v_a_1070_);
if (v___x_1075_ == 0)
{
v_x_1071_ = v_tail_1074_;
goto _start;
}
else
{
return v___x_1075_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg___boxed(lean_object* v_a_1077_, lean_object* v_x_1078_){
_start:
{
uint8_t v_res_1079_; lean_object* v_r_1080_; 
v_res_1079_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg(v_a_1077_, v_x_1078_);
lean_dec(v_x_1078_);
lean_dec(v_a_1077_);
v_r_1080_ = lean_box(v_res_1079_);
return v_r_1080_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10___redArg(lean_object* v_m_1081_, lean_object* v_a_1082_, lean_object* v_b_1083_){
_start:
{
lean_object* v_size_1084_; lean_object* v_buckets_1085_; lean_object* v___x_1087_; uint8_t v_isShared_1088_; uint8_t v_isSharedCheck_1128_; 
v_size_1084_ = lean_ctor_get(v_m_1081_, 0);
v_buckets_1085_ = lean_ctor_get(v_m_1081_, 1);
v_isSharedCheck_1128_ = !lean_is_exclusive(v_m_1081_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1087_ = v_m_1081_;
v_isShared_1088_ = v_isSharedCheck_1128_;
goto v_resetjp_1086_;
}
else
{
lean_inc(v_buckets_1085_);
lean_inc(v_size_1084_);
lean_dec(v_m_1081_);
v___x_1087_ = lean_box(0);
v_isShared_1088_ = v_isSharedCheck_1128_;
goto v_resetjp_1086_;
}
v_resetjp_1086_:
{
lean_object* v___x_1089_; uint64_t v___x_1090_; uint64_t v___x_1091_; uint64_t v___x_1092_; uint64_t v_fold_1093_; uint64_t v___x_1094_; uint64_t v___x_1095_; uint64_t v___x_1096_; size_t v___x_1097_; size_t v___x_1098_; size_t v___x_1099_; size_t v___x_1100_; size_t v___x_1101_; lean_object* v_bkt_1102_; uint8_t v___x_1103_; 
v___x_1089_ = lean_array_get_size(v_buckets_1085_);
v___x_1090_ = l_Lean_instHashableMVarId_hash(v_a_1082_);
v___x_1091_ = 32ULL;
v___x_1092_ = lean_uint64_shift_right(v___x_1090_, v___x_1091_);
v_fold_1093_ = lean_uint64_xor(v___x_1090_, v___x_1092_);
v___x_1094_ = 16ULL;
v___x_1095_ = lean_uint64_shift_right(v_fold_1093_, v___x_1094_);
v___x_1096_ = lean_uint64_xor(v_fold_1093_, v___x_1095_);
v___x_1097_ = lean_uint64_to_usize(v___x_1096_);
v___x_1098_ = lean_usize_of_nat(v___x_1089_);
v___x_1099_ = ((size_t)1ULL);
v___x_1100_ = lean_usize_sub(v___x_1098_, v___x_1099_);
v___x_1101_ = lean_usize_land(v___x_1097_, v___x_1100_);
v_bkt_1102_ = lean_array_uget_borrowed(v_buckets_1085_, v___x_1101_);
v___x_1103_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg(v_a_1082_, v_bkt_1102_);
if (v___x_1103_ == 0)
{
lean_object* v___x_1104_; lean_object* v_size_x27_1105_; lean_object* v___x_1106_; lean_object* v_buckets_x27_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; uint8_t v___x_1113_; 
v___x_1104_ = lean_unsigned_to_nat(1u);
v_size_x27_1105_ = lean_nat_add(v_size_1084_, v___x_1104_);
lean_dec(v_size_1084_);
lean_inc(v_bkt_1102_);
v___x_1106_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1106_, 0, v_a_1082_);
lean_ctor_set(v___x_1106_, 1, v_b_1083_);
lean_ctor_set(v___x_1106_, 2, v_bkt_1102_);
v_buckets_x27_1107_ = lean_array_uset(v_buckets_1085_, v___x_1101_, v___x_1106_);
v___x_1108_ = lean_unsigned_to_nat(4u);
v___x_1109_ = lean_nat_mul(v_size_x27_1105_, v___x_1108_);
v___x_1110_ = lean_unsigned_to_nat(3u);
v___x_1111_ = lean_nat_div(v___x_1109_, v___x_1110_);
lean_dec(v___x_1109_);
v___x_1112_ = lean_array_get_size(v_buckets_x27_1107_);
v___x_1113_ = lean_nat_dec_le(v___x_1111_, v___x_1112_);
lean_dec(v___x_1111_);
if (v___x_1113_ == 0)
{
lean_object* v_val_1114_; lean_object* v___x_1116_; 
v_val_1114_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21___redArg(v_buckets_x27_1107_);
if (v_isShared_1088_ == 0)
{
lean_ctor_set(v___x_1087_, 1, v_val_1114_);
lean_ctor_set(v___x_1087_, 0, v_size_x27_1105_);
v___x_1116_ = v___x_1087_;
goto v_reusejp_1115_;
}
else
{
lean_object* v_reuseFailAlloc_1117_; 
v_reuseFailAlloc_1117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1117_, 0, v_size_x27_1105_);
lean_ctor_set(v_reuseFailAlloc_1117_, 1, v_val_1114_);
v___x_1116_ = v_reuseFailAlloc_1117_;
goto v_reusejp_1115_;
}
v_reusejp_1115_:
{
return v___x_1116_;
}
}
else
{
lean_object* v___x_1119_; 
if (v_isShared_1088_ == 0)
{
lean_ctor_set(v___x_1087_, 1, v_buckets_x27_1107_);
lean_ctor_set(v___x_1087_, 0, v_size_x27_1105_);
v___x_1119_ = v___x_1087_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v_size_x27_1105_);
lean_ctor_set(v_reuseFailAlloc_1120_, 1, v_buckets_x27_1107_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
return v___x_1119_;
}
}
}
else
{
lean_object* v___x_1121_; lean_object* v_buckets_x27_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1126_; 
lean_inc(v_bkt_1102_);
v___x_1121_ = lean_box(0);
v_buckets_x27_1122_ = lean_array_uset(v_buckets_1085_, v___x_1101_, v___x_1121_);
v___x_1123_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22___redArg(v_a_1082_, v_b_1083_, v_bkt_1102_);
v___x_1124_ = lean_array_uset(v_buckets_x27_1122_, v___x_1101_, v___x_1123_);
if (v_isShared_1088_ == 0)
{
lean_ctor_set(v___x_1087_, 1, v___x_1124_);
v___x_1126_ = v___x_1087_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_size_1084_);
lean_ctor_set(v_reuseFailAlloc_1127_, 1, v___x_1124_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
return v___x_1126_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15(lean_object* v_opts_1129_, lean_object* v_opt_1130_){
_start:
{
lean_object* v_name_1131_; lean_object* v_defValue_1132_; lean_object* v_map_1133_; lean_object* v___x_1134_; 
v_name_1131_ = lean_ctor_get(v_opt_1130_, 0);
v_defValue_1132_ = lean_ctor_get(v_opt_1130_, 1);
v_map_1133_ = lean_ctor_get(v_opts_1129_, 0);
v___x_1134_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1133_, v_name_1131_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_inc(v_defValue_1132_);
return v_defValue_1132_;
}
else
{
lean_object* v_val_1135_; 
v_val_1135_ = lean_ctor_get(v___x_1134_, 0);
lean_inc(v_val_1135_);
lean_dec_ref_known(v___x_1134_, 1);
if (lean_obj_tag(v_val_1135_) == 3)
{
lean_object* v_v_1136_; 
v_v_1136_ = lean_ctor_get(v_val_1135_, 0);
lean_inc(v_v_1136_);
lean_dec_ref_known(v_val_1135_, 1);
return v_v_1136_;
}
else
{
lean_dec(v_val_1135_);
lean_inc(v_defValue_1132_);
return v_defValue_1132_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15___boxed(lean_object* v_opts_1137_, lean_object* v_opt_1138_){
_start:
{
lean_object* v_res_1139_; 
v_res_1139_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15(v_opts_1137_, v_opt_1138_);
lean_dec_ref(v_opt_1138_);
lean_dec_ref(v_opts_1137_);
return v_res_1139_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14(lean_object* v_e_1140_){
_start:
{
if (lean_obj_tag(v_e_1140_) == 0)
{
uint8_t v___x_1141_; 
v___x_1141_ = 2;
return v___x_1141_;
}
else
{
lean_object* v_a_1142_; uint8_t v___x_1143_; 
v_a_1142_ = lean_ctor_get(v_e_1140_, 0);
v___x_1143_ = lean_unbox(v_a_1142_);
if (v___x_1143_ == 0)
{
uint8_t v___x_1144_; 
v___x_1144_ = 1;
return v___x_1144_;
}
else
{
uint8_t v___x_1145_; 
v___x_1145_ = 0;
return v___x_1145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14___boxed(lean_object* v_e_1146_){
_start:
{
uint8_t v_res_1147_; lean_object* v_r_1148_; 
v_res_1147_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14(v_e_1146_);
lean_dec_ref(v_e_1146_);
v_r_1148_ = lean_box(v_res_1147_);
return v_r_1148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(lean_object* v_x_1149_){
_start:
{
if (lean_obj_tag(v_x_1149_) == 0)
{
lean_object* v_a_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1158_; 
v_a_1151_ = lean_ctor_get(v_x_1149_, 0);
v_isSharedCheck_1158_ = !lean_is_exclusive(v_x_1149_);
if (v_isSharedCheck_1158_ == 0)
{
v___x_1153_ = v_x_1149_;
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_a_1151_);
lean_dec(v_x_1149_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1156_; 
if (v_isShared_1154_ == 0)
{
lean_ctor_set_tag(v___x_1153_, 1);
v___x_1156_ = v___x_1153_;
goto v_reusejp_1155_;
}
else
{
lean_object* v_reuseFailAlloc_1157_; 
v_reuseFailAlloc_1157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1157_, 0, v_a_1151_);
v___x_1156_ = v_reuseFailAlloc_1157_;
goto v_reusejp_1155_;
}
v_reusejp_1155_:
{
return v___x_1156_;
}
}
}
else
{
lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1166_; 
v_a_1159_ = lean_ctor_get(v_x_1149_, 0);
v_isSharedCheck_1166_ = !lean_is_exclusive(v_x_1149_);
if (v_isSharedCheck_1166_ == 0)
{
v___x_1161_ = v_x_1149_;
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v_x_1149_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
lean_object* v___x_1164_; 
if (v_isShared_1162_ == 0)
{
lean_ctor_set_tag(v___x_1161_, 0);
v___x_1164_ = v___x_1161_;
goto v_reusejp_1163_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v_a_1159_);
v___x_1164_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1163_;
}
v_reusejp_1163_:
{
return v___x_1164_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg___boxed(lean_object* v_x_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(v_x_1167_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16(size_t v_sz_1170_, size_t v_i_1171_, lean_object* v_bs_1172_){
_start:
{
uint8_t v___x_1173_; 
v___x_1173_ = lean_usize_dec_lt(v_i_1171_, v_sz_1170_);
if (v___x_1173_ == 0)
{
return v_bs_1172_;
}
else
{
lean_object* v_v_1174_; lean_object* v_msg_1175_; lean_object* v___x_1176_; lean_object* v_bs_x27_1177_; size_t v___x_1178_; size_t v___x_1179_; lean_object* v___x_1180_; 
v_v_1174_ = lean_array_uget_borrowed(v_bs_1172_, v_i_1171_);
v_msg_1175_ = lean_ctor_get(v_v_1174_, 1);
lean_inc_ref(v_msg_1175_);
v___x_1176_ = lean_unsigned_to_nat(0u);
v_bs_x27_1177_ = lean_array_uset(v_bs_1172_, v_i_1171_, v___x_1176_);
v___x_1178_ = ((size_t)1ULL);
v___x_1179_ = lean_usize_add(v_i_1171_, v___x_1178_);
v___x_1180_ = lean_array_uset(v_bs_x27_1177_, v_i_1171_, v_msg_1175_);
v_i_1171_ = v___x_1179_;
v_bs_1172_ = v___x_1180_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16___boxed(lean_object* v_sz_1182_, lean_object* v_i_1183_, lean_object* v_bs_1184_){
_start:
{
size_t v_sz_boxed_1185_; size_t v_i_boxed_1186_; lean_object* v_res_1187_; 
v_sz_boxed_1185_ = lean_unbox_usize(v_sz_1182_);
lean_dec(v_sz_1182_);
v_i_boxed_1186_ = lean_unbox_usize(v_i_1183_);
lean_dec(v_i_1183_);
v_res_1187_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16(v_sz_boxed_1185_, v_i_boxed_1186_, v_bs_1184_);
return v_res_1187_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg(lean_object* v_oldTraces_1188_, lean_object* v_data_1189_, lean_object* v_ref_1190_, lean_object* v_msg_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_){
_start:
{
lean_object* v_fileName_1197_; lean_object* v_fileMap_1198_; lean_object* v_options_1199_; lean_object* v_currRecDepth_1200_; lean_object* v_maxRecDepth_1201_; lean_object* v_ref_1202_; lean_object* v_currNamespace_1203_; lean_object* v_openDecls_1204_; lean_object* v_initHeartbeats_1205_; lean_object* v_maxHeartbeats_1206_; lean_object* v_quotContext_1207_; lean_object* v_currMacroScope_1208_; uint8_t v_diag_1209_; lean_object* v_cancelTk_x3f_1210_; uint8_t v_suppressElabErrors_1211_; lean_object* v_inheritedTraceOptions_1212_; lean_object* v___x_1213_; lean_object* v_traceState_1214_; lean_object* v_traces_1215_; lean_object* v_ref_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; size_t v_sz_1219_; size_t v___x_1220_; lean_object* v___x_1221_; lean_object* v_msg_1222_; lean_object* v___x_1223_; lean_object* v_a_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1261_; 
v_fileName_1197_ = lean_ctor_get(v___y_1194_, 0);
v_fileMap_1198_ = lean_ctor_get(v___y_1194_, 1);
v_options_1199_ = lean_ctor_get(v___y_1194_, 2);
v_currRecDepth_1200_ = lean_ctor_get(v___y_1194_, 3);
v_maxRecDepth_1201_ = lean_ctor_get(v___y_1194_, 4);
v_ref_1202_ = lean_ctor_get(v___y_1194_, 5);
v_currNamespace_1203_ = lean_ctor_get(v___y_1194_, 6);
v_openDecls_1204_ = lean_ctor_get(v___y_1194_, 7);
v_initHeartbeats_1205_ = lean_ctor_get(v___y_1194_, 8);
v_maxHeartbeats_1206_ = lean_ctor_get(v___y_1194_, 9);
v_quotContext_1207_ = lean_ctor_get(v___y_1194_, 10);
v_currMacroScope_1208_ = lean_ctor_get(v___y_1194_, 11);
v_diag_1209_ = lean_ctor_get_uint8(v___y_1194_, sizeof(void*)*14);
v_cancelTk_x3f_1210_ = lean_ctor_get(v___y_1194_, 12);
v_suppressElabErrors_1211_ = lean_ctor_get_uint8(v___y_1194_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1212_ = lean_ctor_get(v___y_1194_, 13);
v___x_1213_ = lean_st_ref_get(v___y_1195_);
v_traceState_1214_ = lean_ctor_get(v___x_1213_, 4);
lean_inc_ref(v_traceState_1214_);
lean_dec(v___x_1213_);
v_traces_1215_ = lean_ctor_get(v_traceState_1214_, 0);
lean_inc_ref(v_traces_1215_);
lean_dec_ref(v_traceState_1214_);
v_ref_1216_ = l_Lean_replaceRef(v_ref_1190_, v_ref_1202_);
lean_inc_ref(v_inheritedTraceOptions_1212_);
lean_inc(v_cancelTk_x3f_1210_);
lean_inc(v_currMacroScope_1208_);
lean_inc(v_quotContext_1207_);
lean_inc(v_maxHeartbeats_1206_);
lean_inc(v_initHeartbeats_1205_);
lean_inc(v_openDecls_1204_);
lean_inc(v_currNamespace_1203_);
lean_inc(v_maxRecDepth_1201_);
lean_inc(v_currRecDepth_1200_);
lean_inc_ref(v_options_1199_);
lean_inc_ref(v_fileMap_1198_);
lean_inc_ref(v_fileName_1197_);
v___x_1217_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1217_, 0, v_fileName_1197_);
lean_ctor_set(v___x_1217_, 1, v_fileMap_1198_);
lean_ctor_set(v___x_1217_, 2, v_options_1199_);
lean_ctor_set(v___x_1217_, 3, v_currRecDepth_1200_);
lean_ctor_set(v___x_1217_, 4, v_maxRecDepth_1201_);
lean_ctor_set(v___x_1217_, 5, v_ref_1216_);
lean_ctor_set(v___x_1217_, 6, v_currNamespace_1203_);
lean_ctor_set(v___x_1217_, 7, v_openDecls_1204_);
lean_ctor_set(v___x_1217_, 8, v_initHeartbeats_1205_);
lean_ctor_set(v___x_1217_, 9, v_maxHeartbeats_1206_);
lean_ctor_set(v___x_1217_, 10, v_quotContext_1207_);
lean_ctor_set(v___x_1217_, 11, v_currMacroScope_1208_);
lean_ctor_set(v___x_1217_, 12, v_cancelTk_x3f_1210_);
lean_ctor_set(v___x_1217_, 13, v_inheritedTraceOptions_1212_);
lean_ctor_set_uint8(v___x_1217_, sizeof(void*)*14, v_diag_1209_);
lean_ctor_set_uint8(v___x_1217_, sizeof(void*)*14 + 1, v_suppressElabErrors_1211_);
v___x_1218_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1215_);
lean_dec_ref(v_traces_1215_);
v_sz_1219_ = lean_array_size(v___x_1218_);
v___x_1220_ = ((size_t)0ULL);
v___x_1221_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12_spec__16(v_sz_1219_, v___x_1220_, v___x_1218_);
v_msg_1222_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1222_, 0, v_data_1189_);
lean_ctor_set(v_msg_1222_, 1, v_msg_1191_);
lean_ctor_set(v_msg_1222_, 2, v___x_1221_);
v___x_1223_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0(v_msg_1222_, v___y_1192_, v___y_1193_, v___x_1217_, v___y_1195_);
lean_dec_ref_known(v___x_1217_, 14);
v_a_1224_ = lean_ctor_get(v___x_1223_, 0);
v_isSharedCheck_1261_ = !lean_is_exclusive(v___x_1223_);
if (v_isSharedCheck_1261_ == 0)
{
v___x_1226_ = v___x_1223_;
v_isShared_1227_ = v_isSharedCheck_1261_;
goto v_resetjp_1225_;
}
else
{
lean_inc(v_a_1224_);
lean_dec(v___x_1223_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1261_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1228_; lean_object* v_traceState_1229_; lean_object* v_env_1230_; lean_object* v_nextMacroScope_1231_; lean_object* v_ngen_1232_; lean_object* v_auxDeclNGen_1233_; lean_object* v_cache_1234_; lean_object* v_messages_1235_; lean_object* v_infoState_1236_; lean_object* v_snapshotTasks_1237_; lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1260_; 
v___x_1228_ = lean_st_ref_take(v___y_1195_);
v_traceState_1229_ = lean_ctor_get(v___x_1228_, 4);
v_env_1230_ = lean_ctor_get(v___x_1228_, 0);
v_nextMacroScope_1231_ = lean_ctor_get(v___x_1228_, 1);
v_ngen_1232_ = lean_ctor_get(v___x_1228_, 2);
v_auxDeclNGen_1233_ = lean_ctor_get(v___x_1228_, 3);
v_cache_1234_ = lean_ctor_get(v___x_1228_, 5);
v_messages_1235_ = lean_ctor_get(v___x_1228_, 6);
v_infoState_1236_ = lean_ctor_get(v___x_1228_, 7);
v_snapshotTasks_1237_ = lean_ctor_get(v___x_1228_, 8);
v_isSharedCheck_1260_ = !lean_is_exclusive(v___x_1228_);
if (v_isSharedCheck_1260_ == 0)
{
v___x_1239_ = v___x_1228_;
v_isShared_1240_ = v_isSharedCheck_1260_;
goto v_resetjp_1238_;
}
else
{
lean_inc(v_snapshotTasks_1237_);
lean_inc(v_infoState_1236_);
lean_inc(v_messages_1235_);
lean_inc(v_cache_1234_);
lean_inc(v_traceState_1229_);
lean_inc(v_auxDeclNGen_1233_);
lean_inc(v_ngen_1232_);
lean_inc(v_nextMacroScope_1231_);
lean_inc(v_env_1230_);
lean_dec(v___x_1228_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1260_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
uint64_t v_tid_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1258_; 
v_tid_1241_ = lean_ctor_get_uint64(v_traceState_1229_, sizeof(void*)*1);
v_isSharedCheck_1258_ = !lean_is_exclusive(v_traceState_1229_);
if (v_isSharedCheck_1258_ == 0)
{
lean_object* v_unused_1259_; 
v_unused_1259_ = lean_ctor_get(v_traceState_1229_, 0);
lean_dec(v_unused_1259_);
v___x_1243_ = v_traceState_1229_;
v_isShared_1244_ = v_isSharedCheck_1258_;
goto v_resetjp_1242_;
}
else
{
lean_dec(v_traceState_1229_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1258_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1248_; 
v___x_1245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1245_, 0, v_ref_1190_);
lean_ctor_set(v___x_1245_, 1, v_a_1224_);
v___x_1246_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1188_, v___x_1245_);
if (v_isShared_1244_ == 0)
{
lean_ctor_set(v___x_1243_, 0, v___x_1246_);
v___x_1248_ = v___x_1243_;
goto v_reusejp_1247_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v___x_1246_);
lean_ctor_set_uint64(v_reuseFailAlloc_1257_, sizeof(void*)*1, v_tid_1241_);
v___x_1248_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1247_;
}
v_reusejp_1247_:
{
lean_object* v___x_1250_; 
if (v_isShared_1240_ == 0)
{
lean_ctor_set(v___x_1239_, 4, v___x_1248_);
v___x_1250_ = v___x_1239_;
goto v_reusejp_1249_;
}
else
{
lean_object* v_reuseFailAlloc_1256_; 
v_reuseFailAlloc_1256_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1256_, 0, v_env_1230_);
lean_ctor_set(v_reuseFailAlloc_1256_, 1, v_nextMacroScope_1231_);
lean_ctor_set(v_reuseFailAlloc_1256_, 2, v_ngen_1232_);
lean_ctor_set(v_reuseFailAlloc_1256_, 3, v_auxDeclNGen_1233_);
lean_ctor_set(v_reuseFailAlloc_1256_, 4, v___x_1248_);
lean_ctor_set(v_reuseFailAlloc_1256_, 5, v_cache_1234_);
lean_ctor_set(v_reuseFailAlloc_1256_, 6, v_messages_1235_);
lean_ctor_set(v_reuseFailAlloc_1256_, 7, v_infoState_1236_);
lean_ctor_set(v_reuseFailAlloc_1256_, 8, v_snapshotTasks_1237_);
v___x_1250_ = v_reuseFailAlloc_1256_;
goto v_reusejp_1249_;
}
v_reusejp_1249_:
{
lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1254_; 
v___x_1251_ = lean_st_ref_set(v___y_1195_, v___x_1250_);
v___x_1252_ = lean_box(0);
if (v_isShared_1227_ == 0)
{
lean_ctor_set(v___x_1226_, 0, v___x_1252_);
v___x_1254_ = v___x_1226_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v___x_1252_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg___boxed(lean_object* v_oldTraces_1262_, lean_object* v_data_1263_, lean_object* v_ref_1264_, lean_object* v_msg_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_){
_start:
{
lean_object* v_res_1271_; 
v_res_1271_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg(v_oldTraces_1262_, v_data_1263_, v_ref_1264_, v_msg_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
return v_res_1271_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(lean_object* v_opts_1272_, lean_object* v_opt_1273_){
_start:
{
lean_object* v_name_1274_; lean_object* v_defValue_1275_; lean_object* v_map_1276_; lean_object* v___x_1277_; 
v_name_1274_ = lean_ctor_get(v_opt_1273_, 0);
v_defValue_1275_ = lean_ctor_get(v_opt_1273_, 1);
v_map_1276_ = lean_ctor_get(v_opts_1272_, 0);
v___x_1277_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1276_, v_name_1274_);
if (lean_obj_tag(v___x_1277_) == 0)
{
uint8_t v___x_1278_; 
v___x_1278_ = lean_unbox(v_defValue_1275_);
return v___x_1278_;
}
else
{
lean_object* v_val_1279_; 
v_val_1279_ = lean_ctor_get(v___x_1277_, 0);
lean_inc(v_val_1279_);
lean_dec_ref_known(v___x_1277_, 1);
if (lean_obj_tag(v_val_1279_) == 1)
{
uint8_t v_v_1280_; 
v_v_1280_ = lean_ctor_get_uint8(v_val_1279_, 0);
lean_dec_ref_known(v_val_1279_, 0);
return v_v_1280_;
}
else
{
uint8_t v___x_1281_; 
lean_dec(v_val_1279_);
v___x_1281_ = lean_unbox(v_defValue_1275_);
return v___x_1281_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2___boxed(lean_object* v_opts_1282_, lean_object* v_opt_1283_){
_start:
{
uint8_t v_res_1284_; lean_object* v_r_1285_; 
v_res_1284_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(v_opts_1282_, v_opt_1283_);
lean_dec_ref(v_opt_1283_);
lean_dec_ref(v_opts_1282_);
v_r_1285_ = lean_box(v_res_1284_);
return v_r_1285_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1286_; double v___x_1287_; 
v___x_1286_ = lean_unsigned_to_nat(0u);
v___x_1287_ = lean_float_of_nat(v___x_1286_);
return v___x_1287_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1288_; double v___x_1289_; 
v___x_1288_ = lean_unsigned_to_nat(1000u);
v___x_1289_ = lean_float_of_nat(v___x_1288_);
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3(lean_object* v_cls_1290_, uint8_t v_collapsed_1291_, lean_object* v_tag_1292_, lean_object* v_opts_1293_, uint8_t v_clsEnabled_1294_, lean_object* v_oldTraces_1295_, lean_object* v_ref_1296_, lean_object* v_msg_1297_, lean_object* v_resStartStop_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_){
_start:
{
lean_object* v_fst_1306_; lean_object* v_snd_1307_; lean_object* v_data_1309_; lean_object* v_fst_1320_; lean_object* v_snd_1321_; lean_object* v___x_1322_; uint8_t v___x_1323_; uint8_t v___y_1334_; double v___y_1365_; 
v_fst_1306_ = lean_ctor_get(v_resStartStop_1298_, 0);
lean_inc(v_fst_1306_);
v_snd_1307_ = lean_ctor_get(v_resStartStop_1298_, 1);
lean_inc(v_snd_1307_);
lean_dec_ref(v_resStartStop_1298_);
v_fst_1320_ = lean_ctor_get(v_snd_1307_, 0);
lean_inc(v_fst_1320_);
v_snd_1321_ = lean_ctor_get(v_snd_1307_, 1);
lean_inc(v_snd_1321_);
lean_dec(v_snd_1307_);
v___x_1322_ = l_Lean_trace_profiler;
v___x_1323_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(v_opts_1293_, v___x_1322_);
if (v___x_1323_ == 0)
{
v___y_1334_ = v___x_1323_;
goto v___jp_1333_;
}
else
{
lean_object* v___x_1370_; uint8_t v___x_1371_; 
v___x_1370_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1371_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(v_opts_1293_, v___x_1370_);
if (v___x_1371_ == 0)
{
lean_object* v___x_1372_; lean_object* v___x_1373_; double v___x_1374_; double v___x_1375_; double v___x_1376_; 
v___x_1372_ = l_Lean_trace_profiler_threshold;
v___x_1373_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15(v_opts_1293_, v___x_1372_);
v___x_1374_ = lean_float_of_nat(v___x_1373_);
v___x_1375_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__1);
v___x_1376_ = lean_float_div(v___x_1374_, v___x_1375_);
v___y_1365_ = v___x_1376_;
goto v___jp_1364_;
}
else
{
lean_object* v___x_1377_; lean_object* v___x_1378_; double v___x_1379_; 
v___x_1377_ = l_Lean_trace_profiler_threshold;
v___x_1378_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__15(v_opts_1293_, v___x_1377_);
v___x_1379_ = lean_float_of_nat(v___x_1378_);
v___y_1365_ = v___x_1379_;
goto v___jp_1364_;
}
}
v___jp_1308_:
{
lean_object* v___x_1310_; 
v___x_1310_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg(v_oldTraces_1295_, v_data_1309_, v_ref_1296_, v_msg_1297_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1310_) == 0)
{
lean_object* v___x_1311_; 
lean_dec_ref_known(v___x_1310_, 1);
v___x_1311_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(v_fst_1306_);
return v___x_1311_;
}
else
{
lean_object* v_a_1312_; lean_object* v___x_1314_; uint8_t v_isShared_1315_; uint8_t v_isSharedCheck_1319_; 
lean_dec(v_fst_1306_);
v_a_1312_ = lean_ctor_get(v___x_1310_, 0);
v_isSharedCheck_1319_ = !lean_is_exclusive(v___x_1310_);
if (v_isSharedCheck_1319_ == 0)
{
v___x_1314_ = v___x_1310_;
v_isShared_1315_ = v_isSharedCheck_1319_;
goto v_resetjp_1313_;
}
else
{
lean_inc(v_a_1312_);
lean_dec(v___x_1310_);
v___x_1314_ = lean_box(0);
v_isShared_1315_ = v_isSharedCheck_1319_;
goto v_resetjp_1313_;
}
v_resetjp_1313_:
{
lean_object* v___x_1317_; 
if (v_isShared_1315_ == 0)
{
v___x_1317_ = v___x_1314_;
goto v_reusejp_1316_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v_a_1312_);
v___x_1317_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1316_;
}
v_reusejp_1316_:
{
return v___x_1317_;
}
}
}
}
v___jp_1324_:
{
uint8_t v_result_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; double v___x_1328_; lean_object* v_data_1329_; 
v_result_1325_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__14(v_fst_1306_);
v___x_1326_ = lean_box(v_result_1325_);
v___x_1327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
v___x_1328_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___closed__0);
lean_inc_ref(v_tag_1292_);
lean_inc_ref(v___x_1327_);
lean_inc(v_cls_1290_);
v_data_1329_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1329_, 0, v_cls_1290_);
lean_ctor_set(v_data_1329_, 1, v___x_1327_);
lean_ctor_set(v_data_1329_, 2, v_tag_1292_);
lean_ctor_set_float(v_data_1329_, sizeof(void*)*3, v___x_1328_);
lean_ctor_set_float(v_data_1329_, sizeof(void*)*3 + 8, v___x_1328_);
lean_ctor_set_uint8(v_data_1329_, sizeof(void*)*3 + 16, v_collapsed_1291_);
if (v___x_1323_ == 0)
{
lean_dec_ref_known(v___x_1327_, 1);
lean_dec(v_snd_1321_);
lean_dec(v_fst_1320_);
lean_dec_ref(v_tag_1292_);
lean_dec(v_cls_1290_);
v_data_1309_ = v_data_1329_;
goto v___jp_1308_;
}
else
{
lean_object* v_data_1330_; double v___x_1331_; double v___x_1332_; 
lean_dec_ref_known(v_data_1329_, 3);
v_data_1330_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1330_, 0, v_cls_1290_);
lean_ctor_set(v_data_1330_, 1, v___x_1327_);
lean_ctor_set(v_data_1330_, 2, v_tag_1292_);
v___x_1331_ = lean_unbox_float(v_fst_1320_);
lean_dec(v_fst_1320_);
lean_ctor_set_float(v_data_1330_, sizeof(void*)*3, v___x_1331_);
v___x_1332_ = lean_unbox_float(v_snd_1321_);
lean_dec(v_snd_1321_);
lean_ctor_set_float(v_data_1330_, sizeof(void*)*3 + 8, v___x_1332_);
lean_ctor_set_uint8(v_data_1330_, sizeof(void*)*3 + 16, v_collapsed_1291_);
v_data_1309_ = v_data_1330_;
goto v___jp_1308_;
}
}
v___jp_1333_:
{
if (v_clsEnabled_1294_ == 0)
{
if (v___y_1334_ == 0)
{
lean_object* v___x_1335_; lean_object* v_traceState_1336_; lean_object* v_env_1337_; lean_object* v_nextMacroScope_1338_; lean_object* v_ngen_1339_; lean_object* v_auxDeclNGen_1340_; lean_object* v_cache_1341_; lean_object* v_messages_1342_; lean_object* v_infoState_1343_; lean_object* v_snapshotTasks_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1363_; 
lean_dec(v_snd_1321_);
lean_dec(v_fst_1320_);
lean_dec_ref(v_msg_1297_);
lean_dec(v_ref_1296_);
lean_dec_ref(v_tag_1292_);
lean_dec(v_cls_1290_);
v___x_1335_ = lean_st_ref_take(v___y_1304_);
v_traceState_1336_ = lean_ctor_get(v___x_1335_, 4);
v_env_1337_ = lean_ctor_get(v___x_1335_, 0);
v_nextMacroScope_1338_ = lean_ctor_get(v___x_1335_, 1);
v_ngen_1339_ = lean_ctor_get(v___x_1335_, 2);
v_auxDeclNGen_1340_ = lean_ctor_get(v___x_1335_, 3);
v_cache_1341_ = lean_ctor_get(v___x_1335_, 5);
v_messages_1342_ = lean_ctor_get(v___x_1335_, 6);
v_infoState_1343_ = lean_ctor_get(v___x_1335_, 7);
v_snapshotTasks_1344_ = lean_ctor_get(v___x_1335_, 8);
v_isSharedCheck_1363_ = !lean_is_exclusive(v___x_1335_);
if (v_isSharedCheck_1363_ == 0)
{
v___x_1346_ = v___x_1335_;
v_isShared_1347_ = v_isSharedCheck_1363_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_snapshotTasks_1344_);
lean_inc(v_infoState_1343_);
lean_inc(v_messages_1342_);
lean_inc(v_cache_1341_);
lean_inc(v_traceState_1336_);
lean_inc(v_auxDeclNGen_1340_);
lean_inc(v_ngen_1339_);
lean_inc(v_nextMacroScope_1338_);
lean_inc(v_env_1337_);
lean_dec(v___x_1335_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1363_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
uint64_t v_tid_1348_; lean_object* v_traces_1349_; lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1362_; 
v_tid_1348_ = lean_ctor_get_uint64(v_traceState_1336_, sizeof(void*)*1);
v_traces_1349_ = lean_ctor_get(v_traceState_1336_, 0);
v_isSharedCheck_1362_ = !lean_is_exclusive(v_traceState_1336_);
if (v_isSharedCheck_1362_ == 0)
{
v___x_1351_ = v_traceState_1336_;
v_isShared_1352_ = v_isSharedCheck_1362_;
goto v_resetjp_1350_;
}
else
{
lean_inc(v_traces_1349_);
lean_dec(v_traceState_1336_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1362_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1353_; lean_object* v___x_1355_; 
v___x_1353_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1295_, v_traces_1349_);
lean_dec_ref(v_traces_1349_);
if (v_isShared_1352_ == 0)
{
lean_ctor_set(v___x_1351_, 0, v___x_1353_);
v___x_1355_ = v___x_1351_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1361_; 
v_reuseFailAlloc_1361_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1361_, 0, v___x_1353_);
lean_ctor_set_uint64(v_reuseFailAlloc_1361_, sizeof(void*)*1, v_tid_1348_);
v___x_1355_ = v_reuseFailAlloc_1361_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
lean_object* v___x_1357_; 
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 4, v___x_1355_);
v___x_1357_ = v___x_1346_;
goto v_reusejp_1356_;
}
else
{
lean_object* v_reuseFailAlloc_1360_; 
v_reuseFailAlloc_1360_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1360_, 0, v_env_1337_);
lean_ctor_set(v_reuseFailAlloc_1360_, 1, v_nextMacroScope_1338_);
lean_ctor_set(v_reuseFailAlloc_1360_, 2, v_ngen_1339_);
lean_ctor_set(v_reuseFailAlloc_1360_, 3, v_auxDeclNGen_1340_);
lean_ctor_set(v_reuseFailAlloc_1360_, 4, v___x_1355_);
lean_ctor_set(v_reuseFailAlloc_1360_, 5, v_cache_1341_);
lean_ctor_set(v_reuseFailAlloc_1360_, 6, v_messages_1342_);
lean_ctor_set(v_reuseFailAlloc_1360_, 7, v_infoState_1343_);
lean_ctor_set(v_reuseFailAlloc_1360_, 8, v_snapshotTasks_1344_);
v___x_1357_ = v_reuseFailAlloc_1360_;
goto v_reusejp_1356_;
}
v_reusejp_1356_:
{
lean_object* v___x_1358_; lean_object* v___x_1359_; 
v___x_1358_ = lean_st_ref_set(v___y_1304_, v___x_1357_);
v___x_1359_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(v_fst_1306_);
return v___x_1359_;
}
}
}
}
}
else
{
goto v___jp_1324_;
}
}
else
{
goto v___jp_1324_;
}
}
v___jp_1364_:
{
double v___x_1366_; double v___x_1367_; double v___x_1368_; uint8_t v___x_1369_; 
v___x_1366_ = lean_unbox_float(v_snd_1321_);
v___x_1367_ = lean_unbox_float(v_fst_1320_);
v___x_1368_ = lean_float_sub(v___x_1366_, v___x_1367_);
v___x_1369_ = lean_float_decLt(v___y_1365_, v___x_1368_);
v___y_1334_ = v___x_1369_;
goto v___jp_1333_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3___boxed(lean_object* v_cls_1380_, lean_object* v_collapsed_1381_, lean_object* v_tag_1382_, lean_object* v_opts_1383_, lean_object* v_clsEnabled_1384_, lean_object* v_oldTraces_1385_, lean_object* v_ref_1386_, lean_object* v_msg_1387_, lean_object* v_resStartStop_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
uint8_t v_collapsed_boxed_1396_; uint8_t v_clsEnabled_boxed_1397_; lean_object* v_res_1398_; 
v_collapsed_boxed_1396_ = lean_unbox(v_collapsed_1381_);
v_clsEnabled_boxed_1397_ = lean_unbox(v_clsEnabled_1384_);
v_res_1398_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3(v_cls_1380_, v_collapsed_boxed_1396_, v_tag_1382_, v_opts_1383_, v_clsEnabled_boxed_1397_, v_oldTraces_1385_, v_ref_1386_, v_msg_1387_, v_resStartStop_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v___y_1390_);
lean_dec_ref(v___y_1389_);
lean_dec_ref(v_opts_1383_);
return v_res_1398_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg(lean_object* v_a_1399_, lean_object* v_x_1400_){
_start:
{
if (lean_obj_tag(v_x_1400_) == 0)
{
uint8_t v___x_1401_; 
v___x_1401_ = 0;
return v___x_1401_;
}
else
{
lean_object* v_key_1402_; lean_object* v_tail_1403_; uint8_t v___x_1404_; 
v_key_1402_ = lean_ctor_get(v_x_1400_, 0);
v_tail_1403_ = lean_ctor_get(v_x_1400_, 2);
v___x_1404_ = l_Lean_instBEqFVarId_beq(v_key_1402_, v_a_1399_);
if (v___x_1404_ == 0)
{
v_x_1400_ = v_tail_1403_;
goto _start;
}
else
{
return v___x_1404_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg___boxed(lean_object* v_a_1406_, lean_object* v_x_1407_){
_start:
{
uint8_t v_res_1408_; lean_object* v_r_1409_; 
v_res_1408_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg(v_a_1406_, v_x_1407_);
lean_dec(v_x_1407_);
lean_dec(v_a_1406_);
v_r_1409_ = lean_box(v_res_1408_);
return v_r_1409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8___redArg(lean_object* v_a_1410_, lean_object* v_b_1411_, lean_object* v_x_1412_){
_start:
{
if (lean_obj_tag(v_x_1412_) == 0)
{
lean_dec(v_b_1411_);
lean_dec(v_a_1410_);
return v_x_1412_;
}
else
{
lean_object* v_key_1413_; lean_object* v_value_1414_; lean_object* v_tail_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1427_; 
v_key_1413_ = lean_ctor_get(v_x_1412_, 0);
v_value_1414_ = lean_ctor_get(v_x_1412_, 1);
v_tail_1415_ = lean_ctor_get(v_x_1412_, 2);
v_isSharedCheck_1427_ = !lean_is_exclusive(v_x_1412_);
if (v_isSharedCheck_1427_ == 0)
{
v___x_1417_ = v_x_1412_;
v_isShared_1418_ = v_isSharedCheck_1427_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_tail_1415_);
lean_inc(v_value_1414_);
lean_inc(v_key_1413_);
lean_dec(v_x_1412_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1427_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
uint8_t v___x_1419_; 
v___x_1419_ = l_Lean_instBEqFVarId_beq(v_key_1413_, v_a_1410_);
if (v___x_1419_ == 0)
{
lean_object* v___x_1420_; lean_object* v___x_1422_; 
v___x_1420_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8___redArg(v_a_1410_, v_b_1411_, v_tail_1415_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set(v___x_1417_, 2, v___x_1420_);
v___x_1422_ = v___x_1417_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v_key_1413_);
lean_ctor_set(v_reuseFailAlloc_1423_, 1, v_value_1414_);
lean_ctor_set(v_reuseFailAlloc_1423_, 2, v___x_1420_);
v___x_1422_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
return v___x_1422_;
}
}
else
{
lean_object* v___x_1425_; 
lean_dec(v_value_1414_);
lean_dec(v_key_1413_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set(v___x_1417_, 1, v_b_1411_);
lean_ctor_set(v___x_1417_, 0, v_a_1410_);
v___x_1425_ = v___x_1417_;
goto v_reusejp_1424_;
}
else
{
lean_object* v_reuseFailAlloc_1426_; 
v_reuseFailAlloc_1426_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1426_, 0, v_a_1410_);
lean_ctor_set(v_reuseFailAlloc_1426_, 1, v_b_1411_);
lean_ctor_set(v_reuseFailAlloc_1426_, 2, v_tail_1415_);
v___x_1425_ = v_reuseFailAlloc_1426_;
goto v_reusejp_1424_;
}
v_reusejp_1424_:
{
return v___x_1425_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19___redArg(lean_object* v_x_1428_, lean_object* v_x_1429_){
_start:
{
if (lean_obj_tag(v_x_1429_) == 0)
{
return v_x_1428_;
}
else
{
lean_object* v_key_1430_; lean_object* v_value_1431_; lean_object* v_tail_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1455_; 
v_key_1430_ = lean_ctor_get(v_x_1429_, 0);
v_value_1431_ = lean_ctor_get(v_x_1429_, 1);
v_tail_1432_ = lean_ctor_get(v_x_1429_, 2);
v_isSharedCheck_1455_ = !lean_is_exclusive(v_x_1429_);
if (v_isSharedCheck_1455_ == 0)
{
v___x_1434_ = v_x_1429_;
v_isShared_1435_ = v_isSharedCheck_1455_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_tail_1432_);
lean_inc(v_value_1431_);
lean_inc(v_key_1430_);
lean_dec(v_x_1429_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1455_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1436_; uint64_t v___x_1437_; uint64_t v___x_1438_; uint64_t v___x_1439_; uint64_t v_fold_1440_; uint64_t v___x_1441_; uint64_t v___x_1442_; uint64_t v___x_1443_; size_t v___x_1444_; size_t v___x_1445_; size_t v___x_1446_; size_t v___x_1447_; size_t v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1451_; 
v___x_1436_ = lean_array_get_size(v_x_1428_);
v___x_1437_ = l_Lean_instHashableFVarId_hash(v_key_1430_);
v___x_1438_ = 32ULL;
v___x_1439_ = lean_uint64_shift_right(v___x_1437_, v___x_1438_);
v_fold_1440_ = lean_uint64_xor(v___x_1437_, v___x_1439_);
v___x_1441_ = 16ULL;
v___x_1442_ = lean_uint64_shift_right(v_fold_1440_, v___x_1441_);
v___x_1443_ = lean_uint64_xor(v_fold_1440_, v___x_1442_);
v___x_1444_ = lean_uint64_to_usize(v___x_1443_);
v___x_1445_ = lean_usize_of_nat(v___x_1436_);
v___x_1446_ = ((size_t)1ULL);
v___x_1447_ = lean_usize_sub(v___x_1445_, v___x_1446_);
v___x_1448_ = lean_usize_land(v___x_1444_, v___x_1447_);
v___x_1449_ = lean_array_uget_borrowed(v_x_1428_, v___x_1448_);
lean_inc(v___x_1449_);
if (v_isShared_1435_ == 0)
{
lean_ctor_set(v___x_1434_, 2, v___x_1449_);
v___x_1451_ = v___x_1434_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1454_; 
v_reuseFailAlloc_1454_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1454_, 0, v_key_1430_);
lean_ctor_set(v_reuseFailAlloc_1454_, 1, v_value_1431_);
lean_ctor_set(v_reuseFailAlloc_1454_, 2, v___x_1449_);
v___x_1451_ = v_reuseFailAlloc_1454_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
lean_object* v___x_1452_; 
v___x_1452_ = lean_array_uset(v_x_1428_, v___x_1448_, v___x_1451_);
v_x_1428_ = v___x_1452_;
v_x_1429_ = v_tail_1432_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12___redArg(lean_object* v_i_1456_, lean_object* v_source_1457_, lean_object* v_target_1458_){
_start:
{
lean_object* v___x_1459_; uint8_t v___x_1460_; 
v___x_1459_ = lean_array_get_size(v_source_1457_);
v___x_1460_ = lean_nat_dec_lt(v_i_1456_, v___x_1459_);
if (v___x_1460_ == 0)
{
lean_dec_ref(v_source_1457_);
lean_dec(v_i_1456_);
return v_target_1458_;
}
else
{
lean_object* v_es_1461_; lean_object* v___x_1462_; lean_object* v_source_1463_; lean_object* v_target_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; 
v_es_1461_ = lean_array_fget(v_source_1457_, v_i_1456_);
v___x_1462_ = lean_box(0);
v_source_1463_ = lean_array_fset(v_source_1457_, v_i_1456_, v___x_1462_);
v_target_1464_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19___redArg(v_target_1458_, v_es_1461_);
v___x_1465_ = lean_unsigned_to_nat(1u);
v___x_1466_ = lean_nat_add(v_i_1456_, v___x_1465_);
lean_dec(v_i_1456_);
v_i_1456_ = v___x_1466_;
v_source_1457_ = v_source_1463_;
v_target_1458_ = v_target_1464_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7___redArg(lean_object* v_data_1468_){
_start:
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v_nbuckets_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
v___x_1469_ = lean_array_get_size(v_data_1468_);
v___x_1470_ = lean_unsigned_to_nat(2u);
v_nbuckets_1471_ = lean_nat_mul(v___x_1469_, v___x_1470_);
v___x_1472_ = lean_unsigned_to_nat(0u);
v___x_1473_ = lean_box(0);
v___x_1474_ = lean_mk_array(v_nbuckets_1471_, v___x_1473_);
v___x_1475_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12___redArg(v___x_1472_, v_data_1468_, v___x_1474_);
return v___x_1475_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0___redArg(lean_object* v_m_1476_, lean_object* v_a_1477_, lean_object* v_b_1478_){
_start:
{
lean_object* v_size_1479_; lean_object* v_buckets_1480_; lean_object* v___x_1482_; uint8_t v_isShared_1483_; uint8_t v_isSharedCheck_1523_; 
v_size_1479_ = lean_ctor_get(v_m_1476_, 0);
v_buckets_1480_ = lean_ctor_get(v_m_1476_, 1);
v_isSharedCheck_1523_ = !lean_is_exclusive(v_m_1476_);
if (v_isSharedCheck_1523_ == 0)
{
v___x_1482_ = v_m_1476_;
v_isShared_1483_ = v_isSharedCheck_1523_;
goto v_resetjp_1481_;
}
else
{
lean_inc(v_buckets_1480_);
lean_inc(v_size_1479_);
lean_dec(v_m_1476_);
v___x_1482_ = lean_box(0);
v_isShared_1483_ = v_isSharedCheck_1523_;
goto v_resetjp_1481_;
}
v_resetjp_1481_:
{
lean_object* v___x_1484_; uint64_t v___x_1485_; uint64_t v___x_1486_; uint64_t v___x_1487_; uint64_t v_fold_1488_; uint64_t v___x_1489_; uint64_t v___x_1490_; uint64_t v___x_1491_; size_t v___x_1492_; size_t v___x_1493_; size_t v___x_1494_; size_t v___x_1495_; size_t v___x_1496_; lean_object* v_bkt_1497_; uint8_t v___x_1498_; 
v___x_1484_ = lean_array_get_size(v_buckets_1480_);
v___x_1485_ = l_Lean_instHashableFVarId_hash(v_a_1477_);
v___x_1486_ = 32ULL;
v___x_1487_ = lean_uint64_shift_right(v___x_1485_, v___x_1486_);
v_fold_1488_ = lean_uint64_xor(v___x_1485_, v___x_1487_);
v___x_1489_ = 16ULL;
v___x_1490_ = lean_uint64_shift_right(v_fold_1488_, v___x_1489_);
v___x_1491_ = lean_uint64_xor(v_fold_1488_, v___x_1490_);
v___x_1492_ = lean_uint64_to_usize(v___x_1491_);
v___x_1493_ = lean_usize_of_nat(v___x_1484_);
v___x_1494_ = ((size_t)1ULL);
v___x_1495_ = lean_usize_sub(v___x_1493_, v___x_1494_);
v___x_1496_ = lean_usize_land(v___x_1492_, v___x_1495_);
v_bkt_1497_ = lean_array_uget_borrowed(v_buckets_1480_, v___x_1496_);
v___x_1498_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg(v_a_1477_, v_bkt_1497_);
if (v___x_1498_ == 0)
{
lean_object* v___x_1499_; lean_object* v_size_x27_1500_; lean_object* v___x_1501_; lean_object* v_buckets_x27_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; uint8_t v___x_1508_; 
v___x_1499_ = lean_unsigned_to_nat(1u);
v_size_x27_1500_ = lean_nat_add(v_size_1479_, v___x_1499_);
lean_dec(v_size_1479_);
lean_inc(v_bkt_1497_);
v___x_1501_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1501_, 0, v_a_1477_);
lean_ctor_set(v___x_1501_, 1, v_b_1478_);
lean_ctor_set(v___x_1501_, 2, v_bkt_1497_);
v_buckets_x27_1502_ = lean_array_uset(v_buckets_1480_, v___x_1496_, v___x_1501_);
v___x_1503_ = lean_unsigned_to_nat(4u);
v___x_1504_ = lean_nat_mul(v_size_x27_1500_, v___x_1503_);
v___x_1505_ = lean_unsigned_to_nat(3u);
v___x_1506_ = lean_nat_div(v___x_1504_, v___x_1505_);
lean_dec(v___x_1504_);
v___x_1507_ = lean_array_get_size(v_buckets_x27_1502_);
v___x_1508_ = lean_nat_dec_le(v___x_1506_, v___x_1507_);
lean_dec(v___x_1506_);
if (v___x_1508_ == 0)
{
lean_object* v_val_1509_; lean_object* v___x_1511_; 
v_val_1509_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7___redArg(v_buckets_x27_1502_);
if (v_isShared_1483_ == 0)
{
lean_ctor_set(v___x_1482_, 1, v_val_1509_);
lean_ctor_set(v___x_1482_, 0, v_size_x27_1500_);
v___x_1511_ = v___x_1482_;
goto v_reusejp_1510_;
}
else
{
lean_object* v_reuseFailAlloc_1512_; 
v_reuseFailAlloc_1512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1512_, 0, v_size_x27_1500_);
lean_ctor_set(v_reuseFailAlloc_1512_, 1, v_val_1509_);
v___x_1511_ = v_reuseFailAlloc_1512_;
goto v_reusejp_1510_;
}
v_reusejp_1510_:
{
return v___x_1511_;
}
}
else
{
lean_object* v___x_1514_; 
if (v_isShared_1483_ == 0)
{
lean_ctor_set(v___x_1482_, 1, v_buckets_x27_1502_);
lean_ctor_set(v___x_1482_, 0, v_size_x27_1500_);
v___x_1514_ = v___x_1482_;
goto v_reusejp_1513_;
}
else
{
lean_object* v_reuseFailAlloc_1515_; 
v_reuseFailAlloc_1515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1515_, 0, v_size_x27_1500_);
lean_ctor_set(v_reuseFailAlloc_1515_, 1, v_buckets_x27_1502_);
v___x_1514_ = v_reuseFailAlloc_1515_;
goto v_reusejp_1513_;
}
v_reusejp_1513_:
{
return v___x_1514_;
}
}
}
else
{
lean_object* v___x_1516_; lean_object* v_buckets_x27_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1521_; 
lean_inc(v_bkt_1497_);
v___x_1516_ = lean_box(0);
v_buckets_x27_1517_ = lean_array_uset(v_buckets_1480_, v___x_1496_, v___x_1516_);
v___x_1518_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8___redArg(v_a_1477_, v_b_1478_, v_bkt_1497_);
v___x_1519_ = lean_array_uset(v_buckets_x27_1517_, v___x_1496_, v___x_1518_);
if (v_isShared_1483_ == 0)
{
lean_ctor_set(v___x_1482_, 1, v___x_1519_);
v___x_1521_ = v___x_1482_;
goto v_reusejp_1520_;
}
else
{
lean_object* v_reuseFailAlloc_1522_; 
v_reuseFailAlloc_1522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1522_, 0, v_size_1479_);
lean_ctor_set(v_reuseFailAlloc_1522_, 1, v___x_1519_);
v___x_1521_ = v_reuseFailAlloc_1522_;
goto v_reusejp_1520_;
}
v_reusejp_1520_:
{
return v___x_1521_;
}
}
}
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; 
v___x_1524_ = lean_unsigned_to_nat(32u);
v___x_1525_ = lean_mk_empty_array_with_capacity(v___x_1524_);
v___x_1526_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1526_, 0, v___x_1525_);
return v___x_1526_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1(void){
_start:
{
size_t v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; 
v___x_1527_ = ((size_t)5ULL);
v___x_1528_ = lean_unsigned_to_nat(0u);
v___x_1529_ = lean_unsigned_to_nat(32u);
v___x_1530_ = lean_mk_empty_array_with_capacity(v___x_1529_);
v___x_1531_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__0);
v___x_1532_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1532_, 0, v___x_1531_);
lean_ctor_set(v___x_1532_, 1, v___x_1530_);
lean_ctor_set(v___x_1532_, 2, v___x_1528_);
lean_ctor_set(v___x_1532_, 3, v___x_1528_);
lean_ctor_set_usize(v___x_1532_, 4, v___x_1527_);
return v___x_1532_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg(lean_object* v___y_1533_){
_start:
{
lean_object* v___x_1535_; lean_object* v_traceState_1536_; lean_object* v_traces_1537_; lean_object* v___x_1538_; lean_object* v_traceState_1539_; lean_object* v_env_1540_; lean_object* v_nextMacroScope_1541_; lean_object* v_ngen_1542_; lean_object* v_auxDeclNGen_1543_; lean_object* v_cache_1544_; lean_object* v_messages_1545_; lean_object* v_infoState_1546_; lean_object* v_snapshotTasks_1547_; lean_object* v___x_1549_; uint8_t v_isShared_1550_; uint8_t v_isSharedCheck_1566_; 
v___x_1535_ = lean_st_ref_get(v___y_1533_);
v_traceState_1536_ = lean_ctor_get(v___x_1535_, 4);
lean_inc_ref(v_traceState_1536_);
lean_dec(v___x_1535_);
v_traces_1537_ = lean_ctor_get(v_traceState_1536_, 0);
lean_inc_ref(v_traces_1537_);
lean_dec_ref(v_traceState_1536_);
v___x_1538_ = lean_st_ref_take(v___y_1533_);
v_traceState_1539_ = lean_ctor_get(v___x_1538_, 4);
v_env_1540_ = lean_ctor_get(v___x_1538_, 0);
v_nextMacroScope_1541_ = lean_ctor_get(v___x_1538_, 1);
v_ngen_1542_ = lean_ctor_get(v___x_1538_, 2);
v_auxDeclNGen_1543_ = lean_ctor_get(v___x_1538_, 3);
v_cache_1544_ = lean_ctor_get(v___x_1538_, 5);
v_messages_1545_ = lean_ctor_get(v___x_1538_, 6);
v_infoState_1546_ = lean_ctor_get(v___x_1538_, 7);
v_snapshotTasks_1547_ = lean_ctor_get(v___x_1538_, 8);
v_isSharedCheck_1566_ = !lean_is_exclusive(v___x_1538_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1549_ = v___x_1538_;
v_isShared_1550_ = v_isSharedCheck_1566_;
goto v_resetjp_1548_;
}
else
{
lean_inc(v_snapshotTasks_1547_);
lean_inc(v_infoState_1546_);
lean_inc(v_messages_1545_);
lean_inc(v_cache_1544_);
lean_inc(v_traceState_1539_);
lean_inc(v_auxDeclNGen_1543_);
lean_inc(v_ngen_1542_);
lean_inc(v_nextMacroScope_1541_);
lean_inc(v_env_1540_);
lean_dec(v___x_1538_);
v___x_1549_ = lean_box(0);
v_isShared_1550_ = v_isSharedCheck_1566_;
goto v_resetjp_1548_;
}
v_resetjp_1548_:
{
uint64_t v_tid_1551_; lean_object* v___x_1553_; uint8_t v_isShared_1554_; uint8_t v_isSharedCheck_1564_; 
v_tid_1551_ = lean_ctor_get_uint64(v_traceState_1539_, sizeof(void*)*1);
v_isSharedCheck_1564_ = !lean_is_exclusive(v_traceState_1539_);
if (v_isSharedCheck_1564_ == 0)
{
lean_object* v_unused_1565_; 
v_unused_1565_ = lean_ctor_get(v_traceState_1539_, 0);
lean_dec(v_unused_1565_);
v___x_1553_ = v_traceState_1539_;
v_isShared_1554_ = v_isSharedCheck_1564_;
goto v_resetjp_1552_;
}
else
{
lean_dec(v_traceState_1539_);
v___x_1553_ = lean_box(0);
v_isShared_1554_ = v_isSharedCheck_1564_;
goto v_resetjp_1552_;
}
v_resetjp_1552_:
{
lean_object* v___x_1555_; lean_object* v___x_1557_; 
v___x_1555_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___closed__1);
if (v_isShared_1554_ == 0)
{
lean_ctor_set(v___x_1553_, 0, v___x_1555_);
v___x_1557_ = v___x_1553_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1563_; 
v_reuseFailAlloc_1563_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1563_, 0, v___x_1555_);
lean_ctor_set_uint64(v_reuseFailAlloc_1563_, sizeof(void*)*1, v_tid_1551_);
v___x_1557_ = v_reuseFailAlloc_1563_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
lean_object* v___x_1559_; 
if (v_isShared_1550_ == 0)
{
lean_ctor_set(v___x_1549_, 4, v___x_1557_);
v___x_1559_ = v___x_1549_;
goto v_reusejp_1558_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_env_1540_);
lean_ctor_set(v_reuseFailAlloc_1562_, 1, v_nextMacroScope_1541_);
lean_ctor_set(v_reuseFailAlloc_1562_, 2, v_ngen_1542_);
lean_ctor_set(v_reuseFailAlloc_1562_, 3, v_auxDeclNGen_1543_);
lean_ctor_set(v_reuseFailAlloc_1562_, 4, v___x_1557_);
lean_ctor_set(v_reuseFailAlloc_1562_, 5, v_cache_1544_);
lean_ctor_set(v_reuseFailAlloc_1562_, 6, v_messages_1545_);
lean_ctor_set(v_reuseFailAlloc_1562_, 7, v_infoState_1546_);
lean_ctor_set(v_reuseFailAlloc_1562_, 8, v_snapshotTasks_1547_);
v___x_1559_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1558_;
}
v_reusejp_1558_:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1560_ = lean_st_ref_set(v___y_1533_, v___x_1559_);
v___x_1561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1561_, 0, v_traces_1537_);
return v___x_1561_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg___boxed(lean_object* v___y_1567_, lean_object* v___y_1568_){
_start:
{
lean_object* v_res_1569_; 
v_res_1569_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg(v___y_1567_);
lean_dec(v___y_1567_);
return v_res_1569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___lam__0(lean_object* v_x1_1570_, lean_object* v_x2_1571_){
_start:
{
uint8_t v___x_1572_; 
v___x_1572_ = l_Lean_LocalDecl_isImplementationDetail(v_x2_1571_);
if (v___x_1572_ == 0)
{
lean_object* v___x_1573_; 
v___x_1573_ = lean_array_push(v_x1_1570_, v_x2_1571_);
return v___x_1573_;
}
else
{
lean_dec_ref(v_x2_1571_);
return v_x1_1570_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg(lean_object* v_a_1574_, lean_object* v_x_1575_){
_start:
{
if (lean_obj_tag(v_x_1575_) == 0)
{
lean_object* v___x_1576_; 
v___x_1576_ = lean_box(0);
return v___x_1576_;
}
else
{
lean_object* v_key_1577_; lean_object* v_value_1578_; lean_object* v_tail_1579_; uint8_t v___x_1580_; 
v_key_1577_ = lean_ctor_get(v_x_1575_, 0);
v_value_1578_ = lean_ctor_get(v_x_1575_, 1);
v_tail_1579_ = lean_ctor_get(v_x_1575_, 2);
v___x_1580_ = l_Lean_instBEqMVarId_beq(v_key_1577_, v_a_1574_);
if (v___x_1580_ == 0)
{
v_x_1575_ = v_tail_1579_;
goto _start;
}
else
{
lean_object* v___x_1582_; 
lean_inc(v_value_1578_);
v___x_1582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1582_, 0, v_value_1578_);
return v___x_1582_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg___boxed(lean_object* v_a_1583_, lean_object* v_x_1584_){
_start:
{
lean_object* v_res_1585_; 
v_res_1585_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg(v_a_1583_, v_x_1584_);
lean_dec(v_x_1584_);
lean_dec(v_a_1583_);
return v_res_1585_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(lean_object* v_m_1586_, lean_object* v_a_1587_){
_start:
{
lean_object* v_buckets_1588_; lean_object* v___x_1589_; uint64_t v___x_1590_; uint64_t v___x_1591_; uint64_t v___x_1592_; uint64_t v_fold_1593_; uint64_t v___x_1594_; uint64_t v___x_1595_; uint64_t v___x_1596_; size_t v___x_1597_; size_t v___x_1598_; size_t v___x_1599_; size_t v___x_1600_; size_t v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; 
v_buckets_1588_ = lean_ctor_get(v_m_1586_, 1);
v___x_1589_ = lean_array_get_size(v_buckets_1588_);
v___x_1590_ = l_Lean_instHashableMVarId_hash(v_a_1587_);
v___x_1591_ = 32ULL;
v___x_1592_ = lean_uint64_shift_right(v___x_1590_, v___x_1591_);
v_fold_1593_ = lean_uint64_xor(v___x_1590_, v___x_1592_);
v___x_1594_ = 16ULL;
v___x_1595_ = lean_uint64_shift_right(v_fold_1593_, v___x_1594_);
v___x_1596_ = lean_uint64_xor(v_fold_1593_, v___x_1595_);
v___x_1597_ = lean_uint64_to_usize(v___x_1596_);
v___x_1598_ = lean_usize_of_nat(v___x_1589_);
v___x_1599_ = ((size_t)1ULL);
v___x_1600_ = lean_usize_sub(v___x_1598_, v___x_1599_);
v___x_1601_ = lean_usize_land(v___x_1597_, v___x_1600_);
v___x_1602_ = lean_array_uget_borrowed(v_buckets_1588_, v___x_1601_);
v___x_1603_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg(v_a_1587_, v___x_1602_);
return v___x_1603_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg___boxed(lean_object* v_m_1604_, lean_object* v_a_1605_){
_start:
{
lean_object* v_res_1606_; 
v_res_1606_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(v_m_1604_, v_a_1605_);
lean_dec(v_a_1605_);
lean_dec_ref(v_m_1604_);
return v_res_1606_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11(lean_object* v_msg_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_){
_start:
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___f_1621_; lean_object* v___x_1622_; lean_object* v___f_1623_; lean_object* v___x_1624_; lean_object* v_toApplicative_1625_; lean_object* v_toFunctor_1626_; lean_object* v_toSeq_1627_; lean_object* v_toSeqLeft_1628_; lean_object* v_toSeqRight_1629_; lean_object* v___f_1630_; lean_object* v___f_1631_; lean_object* v___f_1632_; lean_object* v___f_1633_; lean_object* v___x_1634_; lean_object* v___f_1635_; lean_object* v___f_1636_; lean_object* v___f_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v_toApplicative_1641_; lean_object* v___x_1643_; uint8_t v_isShared_1644_; uint8_t v_isSharedCheck_1677_; 
v___x_1619_ = lean_box(0);
v___x_1620_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0));
v___f_1621_ = lean_alloc_closure((void*)(l_instBEqProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_1621_, 0, v___x_1620_);
lean_closure_set(v___f_1621_, 1, v___x_1620_);
v___x_1622_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1));
v___f_1623_ = lean_alloc_closure((void*)(l_instHashableProd___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1623_, 0, v___x_1622_);
lean_closure_set(v___f_1623_, 1, v___x_1622_);
v___x_1624_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_1625_ = lean_ctor_get(v___x_1624_, 0);
v_toFunctor_1626_ = lean_ctor_get(v_toApplicative_1625_, 0);
v_toSeq_1627_ = lean_ctor_get(v_toApplicative_1625_, 2);
v_toSeqLeft_1628_ = lean_ctor_get(v_toApplicative_1625_, 3);
v_toSeqRight_1629_ = lean_ctor_get(v_toApplicative_1625_, 4);
v___f_1630_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_1631_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_1626_, 2);
v___f_1632_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1632_, 0, v_toFunctor_1626_);
v___f_1633_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1633_, 0, v_toFunctor_1626_);
v___x_1634_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1634_, 0, v___f_1632_);
lean_ctor_set(v___x_1634_, 1, v___f_1633_);
lean_inc(v_toSeqRight_1629_);
v___f_1635_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1635_, 0, v_toSeqRight_1629_);
lean_inc(v_toSeqLeft_1628_);
v___f_1636_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1636_, 0, v_toSeqLeft_1628_);
lean_inc(v_toSeq_1627_);
v___f_1637_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1637_, 0, v_toSeq_1627_);
v___x_1638_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1634_);
lean_ctor_set(v___x_1638_, 1, v___f_1630_);
lean_ctor_set(v___x_1638_, 2, v___f_1637_);
lean_ctor_set(v___x_1638_, 3, v___f_1636_);
lean_ctor_set(v___x_1638_, 4, v___f_1635_);
v___x_1639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1639_, 0, v___x_1638_);
lean_ctor_set(v___x_1639_, 1, v___f_1631_);
v___x_1640_ = l_StateRefT_x27_instMonad___redArg(v___x_1639_);
v_toApplicative_1641_ = lean_ctor_get(v___x_1640_, 0);
v_isSharedCheck_1677_ = !lean_is_exclusive(v___x_1640_);
if (v_isSharedCheck_1677_ == 0)
{
lean_object* v_unused_1678_; 
v_unused_1678_ = lean_ctor_get(v___x_1640_, 1);
lean_dec(v_unused_1678_);
v___x_1643_ = v___x_1640_;
v_isShared_1644_ = v_isSharedCheck_1677_;
goto v_resetjp_1642_;
}
else
{
lean_inc(v_toApplicative_1641_);
lean_dec(v___x_1640_);
v___x_1643_ = lean_box(0);
v_isShared_1644_ = v_isSharedCheck_1677_;
goto v_resetjp_1642_;
}
v_resetjp_1642_:
{
lean_object* v_toFunctor_1645_; lean_object* v_toSeq_1646_; lean_object* v_toSeqLeft_1647_; lean_object* v_toSeqRight_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1675_; 
v_toFunctor_1645_ = lean_ctor_get(v_toApplicative_1641_, 0);
v_toSeq_1646_ = lean_ctor_get(v_toApplicative_1641_, 2);
v_toSeqLeft_1647_ = lean_ctor_get(v_toApplicative_1641_, 3);
v_toSeqRight_1648_ = lean_ctor_get(v_toApplicative_1641_, 4);
v_isSharedCheck_1675_ = !lean_is_exclusive(v_toApplicative_1641_);
if (v_isSharedCheck_1675_ == 0)
{
lean_object* v_unused_1676_; 
v_unused_1676_ = lean_ctor_get(v_toApplicative_1641_, 1);
lean_dec(v_unused_1676_);
v___x_1650_ = v_toApplicative_1641_;
v_isShared_1651_ = v_isSharedCheck_1675_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_toSeqRight_1648_);
lean_inc(v_toSeqLeft_1647_);
lean_inc(v_toSeq_1646_);
lean_inc(v_toFunctor_1645_);
lean_dec(v_toApplicative_1641_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1675_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v___f_1652_; lean_object* v___f_1653_; lean_object* v___f_1654_; lean_object* v___f_1655_; lean_object* v___x_1656_; lean_object* v___f_1657_; lean_object* v___f_1658_; lean_object* v___f_1659_; lean_object* v___x_1661_; 
v___f_1652_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_1653_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_1645_);
v___f_1654_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1654_, 0, v_toFunctor_1645_);
v___f_1655_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1655_, 0, v_toFunctor_1645_);
v___x_1656_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1656_, 0, v___f_1654_);
lean_ctor_set(v___x_1656_, 1, v___f_1655_);
v___f_1657_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1657_, 0, v_toSeqRight_1648_);
v___f_1658_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1658_, 0, v_toSeqLeft_1647_);
v___f_1659_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1659_, 0, v_toSeq_1646_);
if (v_isShared_1651_ == 0)
{
lean_ctor_set(v___x_1650_, 4, v___f_1657_);
lean_ctor_set(v___x_1650_, 3, v___f_1658_);
lean_ctor_set(v___x_1650_, 2, v___f_1659_);
lean_ctor_set(v___x_1650_, 1, v___f_1652_);
lean_ctor_set(v___x_1650_, 0, v___x_1656_);
v___x_1661_ = v___x_1650_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1674_; 
v_reuseFailAlloc_1674_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1674_, 0, v___x_1656_);
lean_ctor_set(v_reuseFailAlloc_1674_, 1, v___f_1652_);
lean_ctor_set(v_reuseFailAlloc_1674_, 2, v___f_1659_);
lean_ctor_set(v_reuseFailAlloc_1674_, 3, v___f_1658_);
lean_ctor_set(v_reuseFailAlloc_1674_, 4, v___f_1657_);
v___x_1661_ = v_reuseFailAlloc_1674_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
lean_object* v___x_1663_; 
if (v_isShared_1644_ == 0)
{
lean_ctor_set(v___x_1643_, 1, v___f_1653_);
lean_ctor_set(v___x_1643_, 0, v___x_1661_);
v___x_1663_ = v___x_1643_;
goto v_reusejp_1662_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v___x_1661_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v___f_1653_);
v___x_1663_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1662_;
}
v_reusejp_1662_:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; uint8_t v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_252518__overap_1671_; lean_object* v___x_1672_; 
v___x_1664_ = l_StateRefT_x27_instMonad___redArg(v___x_1663_);
v___x_1665_ = l_ReaderT_instMonad___redArg(v___x_1664_);
v___x_1666_ = l_ReaderT_instMonad___redArg(v___x_1665_);
v___x_1667_ = l_Lean_MonadCacheT_instMonad___redArg(v___x_1619_, v___f_1621_, v___f_1623_, v___x_1666_);
v___x_1668_ = 0;
v___x_1669_ = lean_box(v___x_1668_);
v___x_1670_ = l_instInhabitedOfMonad___redArg(v___x_1667_, v___x_1669_);
v___x_252518__overap_1671_ = lean_panic_fn_borrowed(v___x_1670_, v_msg_1609_);
lean_dec(v___x_1670_);
lean_inc(v___y_1617_);
lean_inc_ref(v___y_1616_);
lean_inc(v___y_1615_);
lean_inc_ref(v___y_1614_);
lean_inc(v___y_1613_);
lean_inc_ref(v___y_1612_);
lean_inc_ref(v___y_1611_);
lean_inc(v___y_1610_);
v___x_1672_ = lean_apply_9(v___x_252518__overap_1671_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_, v___y_1615_, v___y_1616_, v___y_1617_, lean_box(0));
return v___x_1672_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___boxed(lean_object* v_msg_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_){
_start:
{
lean_object* v_res_1689_; 
v_res_1689_ = lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11(v_msg_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_, v___y_1684_, v___y_1685_, v___y_1686_, v___y_1687_);
lean_dec(v___y_1687_);
lean_dec_ref(v___y_1686_);
lean_dec(v___y_1685_);
lean_dec_ref(v___y_1684_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec_ref(v___y_1681_);
lean_dec(v___y_1680_);
return v_res_1689_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1694_; lean_object* v___x_1695_; 
v___x_1694_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__2));
v___x_1695_ = l_Lean_stringToMessageData(v___x_1694_);
return v___x_1695_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1697_; lean_object* v___x_1698_; 
v___x_1697_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__4));
v___x_1698_ = l_Lean_stringToMessageData(v___x_1697_);
return v___x_1698_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1(void){
_start:
{
lean_object* v___x_1699_; lean_object* v___f_1700_; 
v___x_1699_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_1700_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1700_, 0, v___x_1699_);
return v___f_1700_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0(void){
_start:
{
lean_object* v___x_1701_; lean_object* v___f_1702_; 
v___x_1701_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_1702_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1702_, 0, v___x_1701_);
return v___f_1702_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2(void){
_start:
{
lean_object* v___f_1703_; lean_object* v___f_1704_; lean_object* v___x_1705_; 
v___f_1703_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__1);
v___f_1704_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__0);
v___x_1705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1705_, 0, v___f_1704_);
lean_ctor_set(v___x_1705_, 1, v___f_1703_);
return v___x_1705_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4(void){
_start:
{
lean_object* v___x_1706_; lean_object* v___f_1707_; 
v___x_1706_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2);
v___f_1707_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1707_, 0, v___x_1706_);
return v___f_1707_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3(void){
_start:
{
lean_object* v___x_1708_; lean_object* v___f_1709_; 
v___x_1708_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__2);
v___f_1709_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1709_, 0, v___x_1708_);
return v___f_1709_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5(void){
_start:
{
lean_object* v___f_1710_; lean_object* v___f_1711_; lean_object* v___x_1712_; 
v___f_1710_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__4);
v___f_1711_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__3);
v___x_1712_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1712_, 0, v___f_1711_);
lean_ctor_set(v___x_1712_, 1, v___f_1710_);
return v___x_1712_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7(void){
_start:
{
lean_object* v___x_1713_; lean_object* v___f_1714_; 
v___x_1713_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5);
v___f_1714_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1714_, 0, v___x_1713_);
return v___f_1714_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6(void){
_start:
{
lean_object* v___x_1715_; lean_object* v___f_1716_; 
v___x_1715_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__5);
v___f_1716_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1716_, 0, v___x_1715_);
return v___f_1716_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8(void){
_start:
{
lean_object* v___f_1717_; lean_object* v___f_1718_; lean_object* v___x_1719_; 
v___f_1717_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__7);
v___f_1718_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__6);
v___x_1719_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1719_, 0, v___f_1718_);
lean_ctor_set(v___x_1719_, 1, v___f_1717_);
return v___x_1719_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10(void){
_start:
{
lean_object* v___x_1720_; lean_object* v___f_1721_; 
v___x_1720_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8);
v___f_1721_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1721_, 0, v___x_1720_);
return v___f_1721_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9(void){
_start:
{
lean_object* v___x_1722_; lean_object* v___f_1723_; 
v___x_1722_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__8);
v___f_1723_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1723_, 0, v___x_1722_);
return v___f_1723_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11(void){
_start:
{
lean_object* v___f_1724_; lean_object* v___f_1725_; lean_object* v___x_1726_; 
v___f_1724_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__10);
v___f_1725_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__9);
v___x_1726_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1726_, 0, v___f_1725_);
lean_ctor_set(v___x_1726_, 1, v___f_1724_);
return v___x_1726_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13(void){
_start:
{
lean_object* v___x_1727_; lean_object* v___f_1728_; 
v___x_1727_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11);
v___f_1728_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1728_, 0, v___x_1727_);
return v___f_1728_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12(void){
_start:
{
lean_object* v___x_1729_; lean_object* v___f_1730_; 
v___x_1729_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11);
v___f_1730_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1730_, 0, v___x_1729_);
return v___f_1730_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14(void){
_start:
{
lean_object* v___f_1731_; lean_object* v___f_1732_; lean_object* v___x_1733_; 
v___f_1731_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__13);
v___f_1732_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__12);
v___x_1733_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1733_, 0, v___f_1732_);
lean_ctor_set(v___x_1733_, 1, v___f_1731_);
return v___x_1733_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8(void){
_start:
{
lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; 
v___x_1736_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_1737_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
v___x_1738_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__7));
v___x_1739_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1738_, v___x_1737_, v___x_1736_);
return v___x_1739_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9(void){
_start:
{
lean_object* v___x_1742_; lean_object* v___f_1743_; lean_object* v___f_1744_; lean_object* v___x_1745_; 
v___x_1742_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__8);
v___f_1743_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___f_1744_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6));
v___x_1745_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1744_, v___f_1743_, v___x_1742_);
return v___x_1745_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10(void){
_start:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; 
v___x_1746_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__9);
v___x_1747_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
v___x_1748_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__7));
v___x_1749_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1748_, v___x_1747_, v___x_1746_);
return v___x_1749_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11(void){
_start:
{
lean_object* v___x_1750_; lean_object* v___f_1751_; lean_object* v___f_1752_; lean_object* v___x_1753_; 
v___x_1750_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__10);
v___f_1751_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___f_1752_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6));
v___x_1753_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1752_, v___f_1751_, v___x_1750_);
return v___x_1753_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15(void){
_start:
{
lean_object* v___x_1754_; lean_object* v___f_1755_; lean_object* v___f_1756_; lean_object* v___x_1757_; 
v___x_1754_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11);
v___f_1755_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___f_1756_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__6));
v___x_1757_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1756_, v___f_1755_, v___x_1754_);
return v___x_1757_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12(void){
_start:
{
lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___f_1760_; 
v___x_1758_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
v___x_1759_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_1760_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1760_, 0, v___x_1759_);
lean_closure_set(v___f_1760_, 1, v___x_1758_);
return v___f_1760_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13(void){
_start:
{
lean_object* v___f_1761_; lean_object* v___f_1762_; lean_object* v___f_1763_; 
v___f_1761_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___f_1762_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__12);
v___f_1763_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1763_, 0, v___f_1762_);
lean_closure_set(v___f_1763_, 1, v___f_1761_);
return v___f_1763_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16(void){
_start:
{
lean_object* v___f_1764_; lean_object* v___f_1765_; lean_object* v___f_1766_; 
v___f_1764_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___f_1765_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13);
v___f_1766_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1766_, 0, v___f_1765_);
lean_closure_set(v___f_1766_, 1, v___f_1764_);
return v___f_1766_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1(void){
_start:
{
lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; 
v___x_1767_ = l_Lean_Core_instMonadTraceCoreM;
v___x_1768_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
v___x_1769_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1768_, v___x_1767_);
return v___x_1769_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2(void){
_start:
{
lean_object* v___x_1770_; lean_object* v___f_1771_; lean_object* v___x_1772_; 
v___x_1770_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__1);
v___f_1771_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___x_1772_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1771_, v___x_1770_);
return v___x_1772_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3(void){
_start:
{
lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1773_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__2);
v___x_1774_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
v___x_1775_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1774_, v___x_1773_);
return v___x_1775_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4(void){
_start:
{
lean_object* v___x_1776_; lean_object* v___f_1777_; lean_object* v___x_1778_; 
v___x_1776_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__3);
v___f_1777_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___x_1778_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1777_, v___x_1776_);
return v___x_1778_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17(void){
_start:
{
lean_object* v___x_1779_; lean_object* v___f_1780_; lean_object* v___x_1781_; 
v___x_1779_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4);
v___f_1780_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___x_1781_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1780_, v___x_1779_);
return v___x_1781_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18(void){
_start:
{
lean_object* v___x_1782_; 
v___x_1782_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_1782_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19(void){
_start:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1783_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__18);
v___x_1784_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1783_);
return v___x_1784_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20(void){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; 
v___x_1785_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__19);
v___x_1786_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1785_);
return v___x_1786_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21(void){
_start:
{
lean_object* v___x_1787_; lean_object* v___x_1788_; 
v___x_1787_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__20);
v___x_1788_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1787_);
return v___x_1788_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22(void){
_start:
{
lean_object* v___x_1789_; lean_object* v___x_1790_; 
v___x_1789_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__21);
v___x_1790_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1789_);
return v___x_1790_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23(void){
_start:
{
lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1791_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__22);
v___x_1792_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1791_);
return v___x_1792_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24(void){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; 
v___x_1793_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__23);
v___x_1794_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1793_);
return v___x_1794_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25(void){
_start:
{
lean_object* v___x_1795_; lean_object* v___x_1796_; 
v___x_1795_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24);
v___x_1796_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1795_);
return v___x_1796_;
}
}
static double _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1(void){
_start:
{
lean_object* v___x_1799_; double v___x_1800_; 
v___x_1799_ = lean_unsigned_to_nat(1000000000u);
v___x_1800_ = lean_float_of_nat(v___x_1799_);
return v___x_1800_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32(void){
_start:
{
lean_object* v___x_1808_; lean_object* v___x_1809_; 
v___x_1808_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__31));
v___x_1809_ = l_Lean_stringToMessageData(v___x_1808_);
return v___x_1809_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5(void){
_start:
{
lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; 
v___x_1810_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_1811_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1));
v___x_1812_ = l_Lean_Name_append(v___x_1811_, v___x_1810_);
return v___x_1812_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(lean_object* v_e_u2081_1813_, lean_object* v_e_u2082_1814_, lean_object* v_a_1815_, lean_object* v_a_1816_, lean_object* v_a_1817_, lean_object* v_a_1818_, lean_object* v_a_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_, lean_object* v_a_1822_){
_start:
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___f_1826_; lean_object* v___x_1827_; lean_object* v___f_1828_; lean_object* v___x_1829_; lean_object* v_toApplicative_1830_; lean_object* v_toFunctor_1831_; lean_object* v_toSeq_1832_; lean_object* v_toSeqLeft_1833_; lean_object* v_toSeqRight_1834_; lean_object* v___f_1835_; lean_object* v___f_1836_; lean_object* v___f_1837_; lean_object* v___f_1838_; lean_object* v___x_1839_; lean_object* v___f_1840_; lean_object* v___f_1841_; lean_object* v___f_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v_toApplicative_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_2114_; 
v___x_1824_ = lean_box(0);
v___x_1825_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0));
v___f_1826_ = lean_alloc_closure((void*)(l_instBEqProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_1826_, 0, v___x_1825_);
lean_closure_set(v___f_1826_, 1, v___x_1825_);
v___x_1827_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1));
v___f_1828_ = lean_alloc_closure((void*)(l_instHashableProd___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1828_, 0, v___x_1827_);
lean_closure_set(v___f_1828_, 1, v___x_1827_);
v___x_1829_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_1830_ = lean_ctor_get(v___x_1829_, 0);
v_toFunctor_1831_ = lean_ctor_get(v_toApplicative_1830_, 0);
v_toSeq_1832_ = lean_ctor_get(v_toApplicative_1830_, 2);
v_toSeqLeft_1833_ = lean_ctor_get(v_toApplicative_1830_, 3);
v_toSeqRight_1834_ = lean_ctor_get(v_toApplicative_1830_, 4);
v___f_1835_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_1836_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_1831_, 2);
v___f_1837_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1837_, 0, v_toFunctor_1831_);
v___f_1838_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1838_, 0, v_toFunctor_1831_);
v___x_1839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1839_, 0, v___f_1837_);
lean_ctor_set(v___x_1839_, 1, v___f_1838_);
lean_inc(v_toSeqRight_1834_);
v___f_1840_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1840_, 0, v_toSeqRight_1834_);
lean_inc(v_toSeqLeft_1833_);
v___f_1841_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1841_, 0, v_toSeqLeft_1833_);
lean_inc(v_toSeq_1832_);
v___f_1842_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1842_, 0, v_toSeq_1832_);
v___x_1843_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1843_, 0, v___x_1839_);
lean_ctor_set(v___x_1843_, 1, v___f_1835_);
lean_ctor_set(v___x_1843_, 2, v___f_1842_);
lean_ctor_set(v___x_1843_, 3, v___f_1841_);
lean_ctor_set(v___x_1843_, 4, v___f_1840_);
v___x_1844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1844_, 0, v___x_1843_);
lean_ctor_set(v___x_1844_, 1, v___f_1836_);
v___x_1845_ = l_StateRefT_x27_instMonad___redArg(v___x_1844_);
v_toApplicative_1846_ = lean_ctor_get(v___x_1845_, 0);
v_isSharedCheck_2114_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_2114_ == 0)
{
lean_object* v_unused_2115_; 
v_unused_2115_ = lean_ctor_get(v___x_1845_, 1);
lean_dec(v_unused_2115_);
v___x_1848_ = v___x_1845_;
v_isShared_1849_ = v_isSharedCheck_2114_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_toApplicative_1846_);
lean_dec(v___x_1845_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_2114_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v_toFunctor_1850_; lean_object* v_toSeq_1851_; lean_object* v_toSeqLeft_1852_; lean_object* v_toSeqRight_1853_; lean_object* v___x_1855_; uint8_t v_isShared_1856_; uint8_t v_isSharedCheck_2112_; 
v_toFunctor_1850_ = lean_ctor_get(v_toApplicative_1846_, 0);
v_toSeq_1851_ = lean_ctor_get(v_toApplicative_1846_, 2);
v_toSeqLeft_1852_ = lean_ctor_get(v_toApplicative_1846_, 3);
v_toSeqRight_1853_ = lean_ctor_get(v_toApplicative_1846_, 4);
v_isSharedCheck_2112_ = !lean_is_exclusive(v_toApplicative_1846_);
if (v_isSharedCheck_2112_ == 0)
{
lean_object* v_unused_2113_; 
v_unused_2113_ = lean_ctor_get(v_toApplicative_1846_, 1);
lean_dec(v_unused_2113_);
v___x_1855_ = v_toApplicative_1846_;
v_isShared_1856_ = v_isSharedCheck_2112_;
goto v_resetjp_1854_;
}
else
{
lean_inc(v_toSeqRight_1853_);
lean_inc(v_toSeqLeft_1852_);
lean_inc(v_toSeq_1851_);
lean_inc(v_toFunctor_1850_);
lean_dec(v_toApplicative_1846_);
v___x_1855_ = lean_box(0);
v_isShared_1856_ = v_isSharedCheck_2112_;
goto v_resetjp_1854_;
}
v_resetjp_1854_:
{
lean_object* v___f_1857_; lean_object* v___f_1858_; lean_object* v___f_1859_; lean_object* v___f_1860_; lean_object* v___x_1861_; lean_object* v___f_1862_; lean_object* v___f_1863_; lean_object* v___f_1864_; lean_object* v___x_1866_; 
v___f_1857_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_1858_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_1850_);
v___f_1859_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1859_, 0, v_toFunctor_1850_);
v___f_1860_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1860_, 0, v_toFunctor_1850_);
v___x_1861_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1861_, 0, v___f_1859_);
lean_ctor_set(v___x_1861_, 1, v___f_1860_);
v___f_1862_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1862_, 0, v_toSeqRight_1853_);
v___f_1863_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1863_, 0, v_toSeqLeft_1852_);
v___f_1864_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1864_, 0, v_toSeq_1851_);
if (v_isShared_1856_ == 0)
{
lean_ctor_set(v___x_1855_, 4, v___f_1862_);
lean_ctor_set(v___x_1855_, 3, v___f_1863_);
lean_ctor_set(v___x_1855_, 2, v___f_1864_);
lean_ctor_set(v___x_1855_, 1, v___f_1857_);
lean_ctor_set(v___x_1855_, 0, v___x_1861_);
v___x_1866_ = v___x_1855_;
goto v_reusejp_1865_;
}
else
{
lean_object* v_reuseFailAlloc_2111_; 
v_reuseFailAlloc_2111_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2111_, 0, v___x_1861_);
lean_ctor_set(v_reuseFailAlloc_2111_, 1, v___f_1857_);
lean_ctor_set(v_reuseFailAlloc_2111_, 2, v___f_1864_);
lean_ctor_set(v_reuseFailAlloc_2111_, 3, v___f_1863_);
lean_ctor_set(v_reuseFailAlloc_2111_, 4, v___f_1862_);
v___x_1866_ = v_reuseFailAlloc_2111_;
goto v_reusejp_1865_;
}
v_reusejp_1865_:
{
lean_object* v___x_1868_; 
if (v_isShared_1849_ == 0)
{
lean_ctor_set(v___x_1848_, 1, v___f_1858_);
lean_ctor_set(v___x_1848_, 0, v___x_1866_);
v___x_1868_ = v___x_1848_;
goto v_reusejp_1867_;
}
else
{
lean_object* v_reuseFailAlloc_2110_; 
v_reuseFailAlloc_2110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2110_, 0, v___x_1866_);
lean_ctor_set(v_reuseFailAlloc_2110_, 1, v___f_1858_);
v___x_1868_ = v_reuseFailAlloc_2110_;
goto v_reusejp_1867_;
}
v_reusejp_1867_:
{
lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v_toMonadRef_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___f_1879_; lean_object* v___f_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v_fileName_1887_; lean_object* v_fileMap_1888_; lean_object* v_options_1889_; lean_object* v_currRecDepth_1890_; lean_object* v_maxRecDepth_1891_; lean_object* v_ref_1892_; lean_object* v_currNamespace_1893_; lean_object* v_openDecls_1894_; lean_object* v_initHeartbeats_1895_; lean_object* v_maxHeartbeats_1896_; lean_object* v_quotContext_1897_; lean_object* v_currMacroScope_1898_; uint8_t v_diag_1899_; lean_object* v_cancelTk_x3f_1900_; uint8_t v_suppressElabErrors_1901_; lean_object* v_inheritedTraceOptions_1902_; lean_object* v___f_1903_; lean_object* v_cls_1904_; size_t v___x_1905_; size_t v___x_1906_; uint8_t v___x_1907_; uint8_t v___x_1908_; lean_object* v___x_1909_; lean_object* v___y_1911_; uint8_t v___y_1912_; lean_object* v___y_1913_; lean_object* v___y_1914_; lean_object* v___y_1915_; lean_object* v_a_1916_; lean_object* v___y_1927_; uint8_t v___y_1928_; lean_object* v___y_1929_; lean_object* v___y_1930_; lean_object* v___y_1931_; lean_object* v_a_1932_; uint8_t v___y_1946_; lean_object* v___y_1947_; lean_object* v___y_1948_; lean_object* v___y_1949_; lean_object* v_a_1950_; lean_object* v___y_2048_; uint8_t v___y_2049_; lean_object* v___y_2050_; lean_object* v___y_2051_; lean_object* v___y_2052_; uint8_t v___y_2063_; lean_object* v___y_2064_; lean_object* v___y_2065_; lean_object* v___x_2105_; uint8_t v___x_2106_; 
v___x_1869_ = l_StateRefT_x27_instMonad___redArg(v___x_1868_);
v___x_1870_ = l_ReaderT_instMonad___redArg(v___x_1869_);
v___x_1871_ = l_ReaderT_instMonad___redArg(v___x_1870_);
lean_inc_ref_n(v___f_1828_, 5);
lean_inc_ref_n(v___f_1826_, 5);
v___x_1872_ = l_Lean_MonadCacheT_instMonad___redArg(v___x_1824_, v___f_1826_, v___f_1828_, v___x_1871_);
v___x_1873_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__14);
v___x_1874_ = l_Lean_MonadCacheT_instMonadExceptOf___redArg(v___x_1824_, v___f_1826_, v___f_1828_, v___x_1873_);
v___x_1875_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__15);
v_toMonadRef_1876_ = lean_ctor_get(v___x_1875_, 0);
lean_inc_ref(v_toMonadRef_1876_);
v___x_1877_ = l_Lean_MonadCacheT_instMonadRef___redArg(v___x_1824_, v___f_1826_, v___f_1828_, v_toMonadRef_1876_);
v___x_1878_ = lean_alloc_closure((void*)(l_Lean_MonadCacheT_instMonadLift___aux__1___boxed), 10, 7);
lean_closure_set(v___x_1878_, 0, lean_box(0));
lean_closure_set(v___x_1878_, 1, lean_box(0));
lean_closure_set(v___x_1878_, 2, lean_box(0));
lean_closure_set(v___x_1878_, 3, lean_box(0));
lean_closure_set(v___x_1878_, 4, v___x_1824_);
lean_closure_set(v___x_1878_, 5, v___f_1826_);
lean_closure_set(v___x_1878_, 6, v___f_1828_);
v___f_1879_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__16);
lean_inc_ref(v___x_1878_);
v___f_1880_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1880_, 0, v___f_1879_);
lean_closure_set(v___f_1880_, 1, v___x_1878_);
lean_inc_ref(v___x_1872_);
lean_inc_ref(v___f_1880_);
v___x_1881_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_1880_, v___x_1872_);
lean_inc_ref(v___x_1877_);
v___x_1882_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1882_, 0, v___x_1874_);
lean_ctor_set(v___x_1882_, 1, v___x_1877_);
lean_ctor_set(v___x_1882_, 2, v___x_1881_);
v___x_1883_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__17);
v___x_1884_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1878_, v___x_1883_);
v___x_1885_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__25);
v___x_1886_ = l_Lean_MonadCacheT_instMonadExceptOf___redArg(v___x_1824_, v___f_1826_, v___f_1828_, v___x_1885_);
v_fileName_1887_ = lean_ctor_get(v_a_1821_, 0);
v_fileMap_1888_ = lean_ctor_get(v_a_1821_, 1);
v_options_1889_ = lean_ctor_get(v_a_1821_, 2);
v_currRecDepth_1890_ = lean_ctor_get(v_a_1821_, 3);
v_maxRecDepth_1891_ = lean_ctor_get(v_a_1821_, 4);
v_ref_1892_ = lean_ctor_get(v_a_1821_, 5);
v_currNamespace_1893_ = lean_ctor_get(v_a_1821_, 6);
v_openDecls_1894_ = lean_ctor_get(v_a_1821_, 7);
v_initHeartbeats_1895_ = lean_ctor_get(v_a_1821_, 8);
v_maxHeartbeats_1896_ = lean_ctor_get(v_a_1821_, 9);
v_quotContext_1897_ = lean_ctor_get(v_a_1821_, 10);
v_currMacroScope_1898_ = lean_ctor_get(v_a_1821_, 11);
v_diag_1899_ = lean_ctor_get_uint8(v_a_1821_, sizeof(void*)*14);
v_cancelTk_x3f_1900_ = lean_ctor_get(v_a_1821_, 12);
v_suppressElabErrors_1901_ = lean_ctor_get_uint8(v_a_1821_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1902_ = lean_ctor_get(v_a_1821_, 13);
v___f_1903_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__26));
v_cls_1904_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_1905_ = lean_ptr_addr(v_e_u2081_1813_);
v___x_1906_ = lean_ptr_addr(v_e_u2082_1814_);
v___x_1907_ = lean_usize_dec_eq(v___x_1905_, v___x_1906_);
v___x_1908_ = 1;
v___x_1909_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0));
v___x_2105_ = lean_unsigned_to_nat(0u);
v___x_2106_ = lean_nat_dec_eq(v_maxRecDepth_1891_, v___x_2105_);
if (v___x_2106_ == 0)
{
uint8_t v___x_2107_; 
v___x_2107_ = lean_nat_dec_eq(v_currRecDepth_1890_, v_maxRecDepth_1891_);
if (v___x_2107_ == 0)
{
lean_dec_ref_known(v___x_1882_, 3);
goto v___jp_2092_;
}
else
{
lean_object* v___x_257704__overap_2108_; lean_object* v___x_2109_; 
lean_dec_ref(v___x_1886_);
lean_dec_ref(v___x_1884_);
lean_dec_ref(v___f_1880_);
lean_dec_ref(v___x_1877_);
lean_dec_ref(v___x_1872_);
lean_dec_ref(v___f_1828_);
lean_dec_ref(v___f_1826_);
lean_dec_ref(v_e_u2082_1814_);
lean_dec_ref(v_e_u2081_1813_);
lean_inc(v_ref_1892_);
v___x_257704__overap_2108_ = l_Lean_throwMaxRecDepthAt___redArg(v___x_1882_, v_ref_1892_);
lean_inc(v_a_1822_);
lean_inc_ref(v_a_1821_);
lean_inc(v_a_1820_);
lean_inc_ref(v_a_1819_);
lean_inc(v_a_1818_);
lean_inc_ref(v_a_1817_);
lean_inc_ref(v_a_1816_);
lean_inc(v_a_1815_);
v___x_2109_ = lean_apply_9(v___x_257704__overap_2108_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v_a_1821_, v_a_1822_, lean_box(0));
return v___x_2109_;
}
}
else
{
lean_dec_ref_known(v___x_1882_, 3);
goto v___jp_2092_;
}
v___jp_1910_:
{
lean_object* v___x_1917_; double v___x_1918_; double v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_257786__overap_1924_; lean_object* v___x_1925_; 
v___x_1917_ = lean_io_get_num_heartbeats();
v___x_1918_ = lean_float_of_nat(v___y_1913_);
v___x_1919_ = lean_float_of_nat(v___x_1917_);
v___x_1920_ = lean_box_float(v___x_1918_);
v___x_1921_ = lean_box_float(v___x_1919_);
v___x_1922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1922_, 0, v___x_1920_);
lean_ctor_set(v___x_1922_, 1, v___x_1921_);
v___x_1923_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1923_, 0, v_a_1916_);
lean_ctor_set(v___x_1923_, 1, v___x_1922_);
lean_inc(v_ref_1892_);
v___x_257786__overap_1924_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_1872_, v___x_1884_, lean_box(0), v___x_1877_, v___f_1880_, v___x_1886_, v___f_1903_, v_cls_1904_, v___x_1908_, v___x_1909_, v_options_1889_, v___y_1912_, v___y_1911_, v_ref_1892_, v___y_1914_, v___x_1923_);
lean_inc(v_a_1822_);
lean_inc(v_a_1820_);
lean_inc_ref(v_a_1819_);
lean_inc(v_a_1818_);
lean_inc_ref(v_a_1817_);
lean_inc_ref(v_a_1816_);
lean_inc(v_a_1815_);
v___x_1925_ = lean_apply_9(v___x_257786__overap_1924_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___y_1915_, v_a_1822_, lean_box(0));
return v___x_1925_;
}
v___jp_1926_:
{
lean_object* v___x_1933_; double v___x_1934_; double v___x_1935_; double v___x_1936_; double v___x_1937_; double v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_257765__overap_1943_; lean_object* v___x_1944_; 
v___x_1933_ = lean_io_mono_nanos_now();
v___x_1934_ = lean_float_of_nat(v___y_1931_);
v___x_1935_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_1936_ = lean_float_div(v___x_1934_, v___x_1935_);
v___x_1937_ = lean_float_of_nat(v___x_1933_);
v___x_1938_ = lean_float_div(v___x_1937_, v___x_1935_);
v___x_1939_ = lean_box_float(v___x_1936_);
v___x_1940_ = lean_box_float(v___x_1938_);
v___x_1941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1941_, 0, v___x_1939_);
lean_ctor_set(v___x_1941_, 1, v___x_1940_);
v___x_1942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1942_, 0, v_a_1932_);
lean_ctor_set(v___x_1942_, 1, v___x_1941_);
lean_inc(v_ref_1892_);
v___x_257765__overap_1943_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_1872_, v___x_1884_, lean_box(0), v___x_1877_, v___f_1880_, v___x_1886_, v___f_1903_, v_cls_1904_, v___x_1908_, v___x_1909_, v_options_1889_, v___y_1928_, v___y_1927_, v_ref_1892_, v___y_1929_, v___x_1942_);
lean_inc(v_a_1822_);
lean_inc(v_a_1820_);
lean_inc_ref(v_a_1819_);
lean_inc(v_a_1818_);
lean_inc_ref(v_a_1817_);
lean_inc_ref(v_a_1816_);
lean_inc(v_a_1815_);
v___x_1944_ = lean_apply_9(v___x_257765__overap_1943_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___y_1930_, v_a_1822_, lean_box(0));
return v___x_1944_;
}
v___jp_1945_:
{
lean_object* v_toApplicative_1951_; lean_object* v_toFunctor_1952_; lean_object* v_toSeq_1953_; lean_object* v_toSeqLeft_1954_; lean_object* v_toSeqRight_1955_; lean_object* v___f_1956_; lean_object* v___f_1957_; lean_object* v___x_1958_; lean_object* v___f_1959_; lean_object* v___f_1960_; lean_object* v___f_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v_toApplicative_1965_; lean_object* v___x_1967_; uint8_t v_isShared_1968_; uint8_t v_isSharedCheck_2045_; 
v_toApplicative_1951_ = lean_ctor_get(v___x_1829_, 0);
v_toFunctor_1952_ = lean_ctor_get(v_toApplicative_1951_, 0);
v_toSeq_1953_ = lean_ctor_get(v_toApplicative_1951_, 2);
v_toSeqLeft_1954_ = lean_ctor_get(v_toApplicative_1951_, 3);
v_toSeqRight_1955_ = lean_ctor_get(v_toApplicative_1951_, 4);
lean_inc_ref_n(v_toFunctor_1952_, 2);
v___f_1956_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1956_, 0, v_toFunctor_1952_);
v___f_1957_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1957_, 0, v_toFunctor_1952_);
v___x_1958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1958_, 0, v___f_1956_);
lean_ctor_set(v___x_1958_, 1, v___f_1957_);
lean_inc(v_toSeqRight_1955_);
v___f_1959_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1959_, 0, v_toSeqRight_1955_);
lean_inc(v_toSeqLeft_1954_);
v___f_1960_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1960_, 0, v_toSeqLeft_1954_);
lean_inc(v_toSeq_1953_);
v___f_1961_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1961_, 0, v_toSeq_1953_);
v___x_1962_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1962_, 0, v___x_1958_);
lean_ctor_set(v___x_1962_, 1, v___f_1835_);
lean_ctor_set(v___x_1962_, 2, v___f_1961_);
lean_ctor_set(v___x_1962_, 3, v___f_1960_);
lean_ctor_set(v___x_1962_, 4, v___f_1959_);
v___x_1963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1963_, 0, v___x_1962_);
lean_ctor_set(v___x_1963_, 1, v___f_1836_);
v___x_1964_ = l_StateRefT_x27_instMonad___redArg(v___x_1963_);
v_toApplicative_1965_ = lean_ctor_get(v___x_1964_, 0);
v_isSharedCheck_2045_ = !lean_is_exclusive(v___x_1964_);
if (v_isSharedCheck_2045_ == 0)
{
lean_object* v_unused_2046_; 
v_unused_2046_ = lean_ctor_get(v___x_1964_, 1);
lean_dec(v_unused_2046_);
v___x_1967_ = v___x_1964_;
v_isShared_1968_ = v_isSharedCheck_2045_;
goto v_resetjp_1966_;
}
else
{
lean_inc(v_toApplicative_1965_);
lean_dec(v___x_1964_);
v___x_1967_ = lean_box(0);
v_isShared_1968_ = v_isSharedCheck_2045_;
goto v_resetjp_1966_;
}
v_resetjp_1966_:
{
lean_object* v_toFunctor_1969_; lean_object* v_toSeq_1970_; lean_object* v_toSeqLeft_1971_; lean_object* v_toSeqRight_1972_; lean_object* v___x_1974_; uint8_t v_isShared_1975_; uint8_t v_isSharedCheck_2043_; 
v_toFunctor_1969_ = lean_ctor_get(v_toApplicative_1965_, 0);
v_toSeq_1970_ = lean_ctor_get(v_toApplicative_1965_, 2);
v_toSeqLeft_1971_ = lean_ctor_get(v_toApplicative_1965_, 3);
v_toSeqRight_1972_ = lean_ctor_get(v_toApplicative_1965_, 4);
v_isSharedCheck_2043_ = !lean_is_exclusive(v_toApplicative_1965_);
if (v_isSharedCheck_2043_ == 0)
{
lean_object* v_unused_2044_; 
v_unused_2044_ = lean_ctor_get(v_toApplicative_1965_, 1);
lean_dec(v_unused_2044_);
v___x_1974_ = v_toApplicative_1965_;
v_isShared_1975_ = v_isSharedCheck_2043_;
goto v_resetjp_1973_;
}
else
{
lean_inc(v_toSeqRight_1972_);
lean_inc(v_toSeqLeft_1971_);
lean_inc(v_toSeq_1970_);
lean_inc(v_toFunctor_1969_);
lean_dec(v_toApplicative_1965_);
v___x_1974_ = lean_box(0);
v_isShared_1975_ = v_isSharedCheck_2043_;
goto v_resetjp_1973_;
}
v_resetjp_1973_:
{
lean_object* v___f_1976_; lean_object* v___f_1977_; lean_object* v___x_1978_; lean_object* v___f_1979_; lean_object* v___f_1980_; lean_object* v___f_1981_; lean_object* v___x_1983_; 
lean_inc_ref(v_toFunctor_1969_);
v___f_1976_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1976_, 0, v_toFunctor_1969_);
v___f_1977_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1977_, 0, v_toFunctor_1969_);
v___x_1978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1978_, 0, v___f_1976_);
lean_ctor_set(v___x_1978_, 1, v___f_1977_);
v___f_1979_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1979_, 0, v_toSeqRight_1972_);
v___f_1980_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1980_, 0, v_toSeqLeft_1971_);
v___f_1981_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1981_, 0, v_toSeq_1970_);
if (v_isShared_1975_ == 0)
{
lean_ctor_set(v___x_1974_, 4, v___f_1979_);
lean_ctor_set(v___x_1974_, 3, v___f_1980_);
lean_ctor_set(v___x_1974_, 2, v___f_1981_);
lean_ctor_set(v___x_1974_, 1, v___f_1857_);
lean_ctor_set(v___x_1974_, 0, v___x_1978_);
v___x_1983_ = v___x_1974_;
goto v_reusejp_1982_;
}
else
{
lean_object* v_reuseFailAlloc_2042_; 
v_reuseFailAlloc_2042_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2042_, 0, v___x_1978_);
lean_ctor_set(v_reuseFailAlloc_2042_, 1, v___f_1857_);
lean_ctor_set(v_reuseFailAlloc_2042_, 2, v___f_1981_);
lean_ctor_set(v_reuseFailAlloc_2042_, 3, v___f_1980_);
lean_ctor_set(v_reuseFailAlloc_2042_, 4, v___f_1979_);
v___x_1983_ = v_reuseFailAlloc_2042_;
goto v_reusejp_1982_;
}
v_reusejp_1982_:
{
lean_object* v___x_1985_; 
if (v_isShared_1968_ == 0)
{
lean_ctor_set(v___x_1967_, 1, v___f_1858_);
lean_ctor_set(v___x_1967_, 0, v___x_1983_);
v___x_1985_ = v___x_1967_;
goto v_reusejp_1984_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_1983_);
lean_ctor_set(v_reuseFailAlloc_2041_, 1, v___f_1858_);
v___x_1985_ = v_reuseFailAlloc_2041_;
goto v_reusejp_1984_;
}
v_reusejp_1984_:
{
lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___f_1988_; lean_object* v___x_1989_; lean_object* v___x_257742__overap_1990_; lean_object* v___x_1991_; 
v___x_1986_ = l_Lean_Meta_instMonadEnvMetaM;
v___x_1987_ = l_Lean_Meta_instMonadMCtxMetaM;
v___f_1988_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27));
v___x_1989_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30));
v___x_257742__overap_1990_ = l_Lean_addMessageContextFull___redArg(v___x_1985_, v___x_1986_, v___x_1987_, v___f_1988_, v___x_1989_, v_a_1950_);
lean_inc(v_a_1822_);
lean_inc(v_a_1820_);
lean_inc_ref(v_a_1819_);
v___x_1991_ = lean_apply_5(v___x_257742__overap_1990_, v_a_1819_, v_a_1820_, v___y_1949_, v_a_1822_, lean_box(0));
if (lean_obj_tag(v___x_1991_) == 0)
{
lean_object* v_a_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; uint8_t v___x_1996_; 
v_a_1992_ = lean_ctor_get(v___x_1991_, 0);
lean_inc(v_a_1992_);
lean_dec_ref_known(v___x_1991_, 1);
v___x_1993_ = l_Lean_KVMap_instValueBool;
v___x_1994_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1995_ = l_Lean_Option_get___redArg(v___x_1993_, v_options_1889_, v___x_1994_);
v___x_1996_ = lean_unbox(v___x_1995_);
lean_dec(v___x_1995_);
if (v___x_1996_ == 0)
{
lean_object* v___x_1997_; lean_object* v___x_1998_; 
v___x_1997_ = lean_io_mono_nanos_now();
lean_inc_ref(v___f_1880_);
lean_inc_ref(v___x_1877_);
lean_inc_ref(v___x_1884_);
lean_inc_ref(v___x_1872_);
v___x_1998_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(v___x_1907_, v_e_u2081_1813_, v_e_u2082_1814_, v___f_1826_, v___f_1828_, v___x_1908_, v_cls_1904_, v___x_1872_, v___x_1884_, v___x_1877_, v___f_1880_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___y_1948_, v_a_1822_);
if (lean_obj_tag(v___x_1998_) == 0)
{
lean_object* v_a_1999_; lean_object* v___x_2001_; uint8_t v_isShared_2002_; uint8_t v_isSharedCheck_2006_; 
v_a_1999_ = lean_ctor_get(v___x_1998_, 0);
v_isSharedCheck_2006_ = !lean_is_exclusive(v___x_1998_);
if (v_isSharedCheck_2006_ == 0)
{
v___x_2001_ = v___x_1998_;
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
else
{
lean_inc(v_a_1999_);
lean_dec(v___x_1998_);
v___x_2001_ = lean_box(0);
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
v_resetjp_2000_:
{
lean_object* v___x_2004_; 
if (v_isShared_2002_ == 0)
{
lean_ctor_set_tag(v___x_2001_, 1);
v___x_2004_ = v___x_2001_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2005_; 
v_reuseFailAlloc_2005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2005_, 0, v_a_1999_);
v___x_2004_ = v_reuseFailAlloc_2005_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
v___y_1927_ = v___y_1947_;
v___y_1928_ = v___y_1946_;
v___y_1929_ = v_a_1992_;
v___y_1930_ = v___y_1948_;
v___y_1931_ = v___x_1997_;
v_a_1932_ = v___x_2004_;
goto v___jp_1926_;
}
}
}
else
{
lean_object* v_a_2007_; lean_object* v___x_2009_; uint8_t v_isShared_2010_; uint8_t v_isSharedCheck_2014_; 
v_a_2007_ = lean_ctor_get(v___x_1998_, 0);
v_isSharedCheck_2014_ = !lean_is_exclusive(v___x_1998_);
if (v_isSharedCheck_2014_ == 0)
{
v___x_2009_ = v___x_1998_;
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
else
{
lean_inc(v_a_2007_);
lean_dec(v___x_1998_);
v___x_2009_ = lean_box(0);
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
v_resetjp_2008_:
{
lean_object* v___x_2012_; 
if (v_isShared_2010_ == 0)
{
lean_ctor_set_tag(v___x_2009_, 0);
v___x_2012_ = v___x_2009_;
goto v_reusejp_2011_;
}
else
{
lean_object* v_reuseFailAlloc_2013_; 
v_reuseFailAlloc_2013_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2013_, 0, v_a_2007_);
v___x_2012_ = v_reuseFailAlloc_2013_;
goto v_reusejp_2011_;
}
v_reusejp_2011_:
{
v___y_1927_ = v___y_1947_;
v___y_1928_ = v___y_1946_;
v___y_1929_ = v_a_1992_;
v___y_1930_ = v___y_1948_;
v___y_1931_ = v___x_1997_;
v_a_1932_ = v___x_2012_;
goto v___jp_1926_;
}
}
}
}
else
{
lean_object* v___x_2015_; lean_object* v___x_2016_; 
v___x_2015_ = lean_io_get_num_heartbeats();
lean_inc_ref(v___f_1880_);
lean_inc_ref(v___x_1877_);
lean_inc_ref(v___x_1884_);
lean_inc_ref(v___x_1872_);
v___x_2016_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(v___x_1907_, v_e_u2081_1813_, v_e_u2082_1814_, v___f_1826_, v___f_1828_, v___x_1908_, v_cls_1904_, v___x_1872_, v___x_1884_, v___x_1877_, v___f_1880_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___y_1948_, v_a_1822_);
if (lean_obj_tag(v___x_2016_) == 0)
{
lean_object* v_a_2017_; lean_object* v___x_2019_; uint8_t v_isShared_2020_; uint8_t v_isSharedCheck_2024_; 
v_a_2017_ = lean_ctor_get(v___x_2016_, 0);
v_isSharedCheck_2024_ = !lean_is_exclusive(v___x_2016_);
if (v_isSharedCheck_2024_ == 0)
{
v___x_2019_ = v___x_2016_;
v_isShared_2020_ = v_isSharedCheck_2024_;
goto v_resetjp_2018_;
}
else
{
lean_inc(v_a_2017_);
lean_dec(v___x_2016_);
v___x_2019_ = lean_box(0);
v_isShared_2020_ = v_isSharedCheck_2024_;
goto v_resetjp_2018_;
}
v_resetjp_2018_:
{
lean_object* v___x_2022_; 
if (v_isShared_2020_ == 0)
{
lean_ctor_set_tag(v___x_2019_, 1);
v___x_2022_ = v___x_2019_;
goto v_reusejp_2021_;
}
else
{
lean_object* v_reuseFailAlloc_2023_; 
v_reuseFailAlloc_2023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2023_, 0, v_a_2017_);
v___x_2022_ = v_reuseFailAlloc_2023_;
goto v_reusejp_2021_;
}
v_reusejp_2021_:
{
v___y_1911_ = v___y_1947_;
v___y_1912_ = v___y_1946_;
v___y_1913_ = v___x_2015_;
v___y_1914_ = v_a_1992_;
v___y_1915_ = v___y_1948_;
v_a_1916_ = v___x_2022_;
goto v___jp_1910_;
}
}
}
else
{
lean_object* v_a_2025_; lean_object* v___x_2027_; uint8_t v_isShared_2028_; uint8_t v_isSharedCheck_2032_; 
v_a_2025_ = lean_ctor_get(v___x_2016_, 0);
v_isSharedCheck_2032_ = !lean_is_exclusive(v___x_2016_);
if (v_isSharedCheck_2032_ == 0)
{
v___x_2027_ = v___x_2016_;
v_isShared_2028_ = v_isSharedCheck_2032_;
goto v_resetjp_2026_;
}
else
{
lean_inc(v_a_2025_);
lean_dec(v___x_2016_);
v___x_2027_ = lean_box(0);
v_isShared_2028_ = v_isSharedCheck_2032_;
goto v_resetjp_2026_;
}
v_resetjp_2026_:
{
lean_object* v___x_2030_; 
if (v_isShared_2028_ == 0)
{
lean_ctor_set_tag(v___x_2027_, 0);
v___x_2030_ = v___x_2027_;
goto v_reusejp_2029_;
}
else
{
lean_object* v_reuseFailAlloc_2031_; 
v_reuseFailAlloc_2031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2031_, 0, v_a_2025_);
v___x_2030_ = v_reuseFailAlloc_2031_;
goto v_reusejp_2029_;
}
v_reusejp_2029_:
{
v___y_1911_ = v___y_1947_;
v___y_1912_ = v___y_1946_;
v___y_1913_ = v___x_2015_;
v___y_1914_ = v_a_1992_;
v___y_1915_ = v___y_1948_;
v_a_1916_ = v___x_2030_;
goto v___jp_1910_;
}
}
}
}
}
else
{
lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2040_; 
lean_dec_ref(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec_ref(v___x_1886_);
lean_dec_ref(v___x_1884_);
lean_dec_ref(v___f_1880_);
lean_dec_ref(v___x_1877_);
lean_dec_ref(v___x_1872_);
lean_dec_ref(v___f_1828_);
lean_dec_ref(v___f_1826_);
lean_dec_ref(v_e_u2082_1814_);
lean_dec_ref(v_e_u2081_1813_);
v_a_2033_ = lean_ctor_get(v___x_1991_, 0);
v_isSharedCheck_2040_ = !lean_is_exclusive(v___x_1991_);
if (v_isSharedCheck_2040_ == 0)
{
v___x_2035_ = v___x_1991_;
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_1991_);
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
}
}
}
}
v___jp_2047_:
{
if (lean_obj_tag(v___y_2052_) == 0)
{
lean_object* v_a_2053_; 
v_a_2053_ = lean_ctor_get(v___y_2052_, 0);
lean_inc(v_a_2053_);
lean_dec_ref_known(v___y_2052_, 1);
v___y_1946_ = v___y_2049_;
v___y_1947_ = v___y_2048_;
v___y_1948_ = v___y_2050_;
v___y_1949_ = v___y_2051_;
v_a_1950_ = v_a_2053_;
goto v___jp_1945_;
}
else
{
lean_object* v_a_2054_; lean_object* v___x_2056_; uint8_t v_isShared_2057_; uint8_t v_isSharedCheck_2061_; 
lean_dec_ref(v___y_2051_);
lean_dec_ref(v___y_2050_);
lean_dec_ref(v___y_2048_);
lean_dec_ref(v___x_1886_);
lean_dec_ref(v___x_1884_);
lean_dec_ref(v___f_1880_);
lean_dec_ref(v___x_1877_);
lean_dec_ref(v___x_1872_);
lean_dec_ref(v___f_1828_);
lean_dec_ref(v___f_1826_);
lean_dec_ref(v_e_u2082_1814_);
lean_dec_ref(v_e_u2081_1813_);
v_a_2054_ = lean_ctor_get(v___y_2052_, 0);
v_isSharedCheck_2061_ = !lean_is_exclusive(v___y_2052_);
if (v_isSharedCheck_2061_ == 0)
{
v___x_2056_ = v___y_2052_;
v_isShared_2057_ = v_isSharedCheck_2061_;
goto v_resetjp_2055_;
}
else
{
lean_inc(v_a_2054_);
lean_dec(v___y_2052_);
v___x_2056_ = lean_box(0);
v_isShared_2057_ = v_isSharedCheck_2061_;
goto v_resetjp_2055_;
}
v_resetjp_2055_:
{
lean_object* v___x_2059_; 
if (v_isShared_2057_ == 0)
{
v___x_2059_ = v___x_2056_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2060_; 
v_reuseFailAlloc_2060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2060_, 0, v_a_2054_);
v___x_2059_ = v_reuseFailAlloc_2060_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
return v___x_2059_;
}
}
}
}
v___jp_2062_:
{
lean_object* v___x_257706__overap_2066_; lean_object* v___x_2067_; 
lean_inc_ref(v___x_1884_);
lean_inc_ref(v___x_1872_);
v___x_257706__overap_2066_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1872_, v___x_1884_);
lean_inc(v_a_1822_);
lean_inc_ref(v___y_2065_);
lean_inc(v_a_1820_);
lean_inc_ref(v_a_1819_);
lean_inc(v_a_1818_);
lean_inc_ref(v_a_1817_);
lean_inc_ref(v_a_1816_);
lean_inc(v_a_1815_);
v___x_2067_ = lean_apply_9(v___x_257706__overap_2066_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___y_2065_, v_a_1822_, lean_box(0));
if (lean_obj_tag(v___x_2067_) == 0)
{
lean_object* v_a_2068_; lean_object* v_mctx_u2081_2069_; lean_object* v_mctx_u2082_2070_; lean_object* v_lctx_u2081_2071_; lean_object* v_localInstances_u2081_2072_; lean_object* v_lctx_u2082_2073_; lean_object* v_localInstances_u2082_2074_; lean_object* v_ref_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; 
v_a_2068_ = lean_ctor_get(v___x_2067_, 0);
lean_inc(v_a_2068_);
lean_dec_ref_known(v___x_2067_, 1);
v_mctx_u2081_2069_ = lean_ctor_get(v_a_1817_, 1);
v_mctx_u2082_2070_ = lean_ctor_get(v_a_1817_, 2);
v_lctx_u2081_2071_ = lean_ctor_get(v_a_1816_, 0);
v_localInstances_u2081_2072_ = lean_ctor_get(v_a_1816_, 1);
v_lctx_u2082_2073_ = lean_ctor_get(v_a_1816_, 2);
v_localInstances_u2082_2074_ = lean_ctor_get(v_a_1816_, 3);
v_ref_2075_ = l_Lean_replaceRef(v_ref_1892_, v_ref_1892_);
lean_inc_ref(v_inheritedTraceOptions_1902_);
lean_inc(v_cancelTk_x3f_1900_);
lean_inc(v_currMacroScope_1898_);
lean_inc(v_quotContext_1897_);
lean_inc(v_maxHeartbeats_1896_);
lean_inc(v_initHeartbeats_1895_);
lean_inc(v_openDecls_1894_);
lean_inc(v_currNamespace_1893_);
lean_inc(v_maxRecDepth_1891_);
lean_inc_ref(v_options_1889_);
lean_inc_ref(v_fileMap_1888_);
lean_inc_ref(v_fileName_1887_);
v___x_2076_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2076_, 0, v_fileName_1887_);
lean_ctor_set(v___x_2076_, 1, v_fileMap_1888_);
lean_ctor_set(v___x_2076_, 2, v_options_1889_);
lean_ctor_set(v___x_2076_, 3, v___y_2064_);
lean_ctor_set(v___x_2076_, 4, v_maxRecDepth_1891_);
lean_ctor_set(v___x_2076_, 5, v_ref_2075_);
lean_ctor_set(v___x_2076_, 6, v_currNamespace_1893_);
lean_ctor_set(v___x_2076_, 7, v_openDecls_1894_);
lean_ctor_set(v___x_2076_, 8, v_initHeartbeats_1895_);
lean_ctor_set(v___x_2076_, 9, v_maxHeartbeats_1896_);
lean_ctor_set(v___x_2076_, 10, v_quotContext_1897_);
lean_ctor_set(v___x_2076_, 11, v_currMacroScope_1898_);
lean_ctor_set(v___x_2076_, 12, v_cancelTk_x3f_1900_);
lean_ctor_set(v___x_2076_, 13, v_inheritedTraceOptions_1902_);
lean_ctor_set_uint8(v___x_2076_, sizeof(void*)*14, v_diag_1899_);
lean_ctor_set_uint8(v___x_2076_, sizeof(void*)*14 + 1, v_suppressElabErrors_1901_);
lean_inc_ref(v_e_u2081_1813_);
lean_inc_ref(v_localInstances_u2081_2072_);
lean_inc_ref(v_lctx_u2081_2071_);
lean_inc_ref(v_mctx_u2081_2069_);
v___x_2077_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr(v_mctx_u2081_2069_, v_lctx_u2081_2071_, v_localInstances_u2081_2072_, v_e_u2081_1813_, v_a_1819_, v_a_1820_, v___x_2076_, v_a_1822_);
if (lean_obj_tag(v___x_2077_) == 0)
{
lean_object* v_a_2078_; lean_object* v___x_2079_; 
v_a_2078_ = lean_ctor_get(v___x_2077_, 0);
lean_inc(v_a_2078_);
lean_dec_ref_known(v___x_2077_, 1);
lean_inc_ref(v_e_u2082_1814_);
lean_inc_ref(v_localInstances_u2082_2074_);
lean_inc_ref(v_lctx_u2082_2073_);
lean_inc_ref(v_mctx_u2082_2070_);
v___x_2079_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr(v_mctx_u2082_2070_, v_lctx_u2082_2073_, v_localInstances_u2082_2074_, v_e_u2082_1814_, v_a_1819_, v_a_1820_, v___x_2076_, v_a_1822_);
if (lean_obj_tag(v___x_2079_) == 0)
{
lean_object* v_a_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; 
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
lean_inc(v_a_2080_);
lean_dec_ref_known(v___x_2079_, 1);
v___x_2081_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__32);
v___x_2082_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2082_, 0, v_a_2078_);
lean_ctor_set(v___x_2082_, 1, v___x_2081_);
v___x_2083_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2083_, 0, v___x_2082_);
lean_ctor_set(v___x_2083_, 1, v_a_2080_);
v___y_1946_ = v___y_2063_;
v___y_1947_ = v_a_2068_;
v___y_1948_ = v___y_2065_;
v___y_1949_ = v___x_2076_;
v_a_1950_ = v___x_2083_;
goto v___jp_1945_;
}
else
{
lean_dec(v_a_2078_);
v___y_2048_ = v_a_2068_;
v___y_2049_ = v___y_2063_;
v___y_2050_ = v___y_2065_;
v___y_2051_ = v___x_2076_;
v___y_2052_ = v___x_2079_;
goto v___jp_2047_;
}
}
else
{
v___y_2048_ = v_a_2068_;
v___y_2049_ = v___y_2063_;
v___y_2050_ = v___y_2065_;
v___y_2051_ = v___x_2076_;
v___y_2052_ = v___x_2077_;
goto v___jp_2047_;
}
}
else
{
lean_object* v_a_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2091_; 
lean_dec_ref(v___y_2065_);
lean_dec(v___y_2064_);
lean_dec_ref(v___x_1886_);
lean_dec_ref(v___x_1884_);
lean_dec_ref(v___f_1880_);
lean_dec_ref(v___x_1877_);
lean_dec_ref(v___x_1872_);
lean_dec_ref(v___f_1828_);
lean_dec_ref(v___f_1826_);
lean_dec_ref(v_e_u2082_1814_);
lean_dec_ref(v_e_u2081_1813_);
v_a_2084_ = lean_ctor_get(v___x_2067_, 0);
v_isSharedCheck_2091_ = !lean_is_exclusive(v___x_2067_);
if (v_isSharedCheck_2091_ == 0)
{
v___x_2086_ = v___x_2067_;
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_a_2084_);
lean_dec(v___x_2067_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v___x_2089_; 
if (v_isShared_2087_ == 0)
{
v___x_2089_ = v___x_2086_;
goto v_reusejp_2088_;
}
else
{
lean_object* v_reuseFailAlloc_2090_; 
v_reuseFailAlloc_2090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2090_, 0, v_a_2084_);
v___x_2089_ = v_reuseFailAlloc_2090_;
goto v_reusejp_2088_;
}
v_reusejp_2088_:
{
return v___x_2089_;
}
}
}
}
v___jp_2092_:
{
uint8_t v_hasTrace_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; 
v_hasTrace_2093_ = lean_ctor_get_uint8(v_options_1889_, sizeof(void*)*1);
v___x_2094_ = lean_unsigned_to_nat(1u);
v___x_2095_ = lean_nat_add(v_currRecDepth_1890_, v___x_2094_);
lean_inc_ref(v_inheritedTraceOptions_1902_);
lean_inc(v_cancelTk_x3f_1900_);
lean_inc(v_currMacroScope_1898_);
lean_inc(v_quotContext_1897_);
lean_inc(v_maxHeartbeats_1896_);
lean_inc(v_initHeartbeats_1895_);
lean_inc(v_openDecls_1894_);
lean_inc(v_currNamespace_1893_);
lean_inc(v_ref_1892_);
lean_inc(v_maxRecDepth_1891_);
lean_inc(v___x_2095_);
lean_inc_ref(v_options_1889_);
lean_inc_ref(v_fileMap_1888_);
lean_inc_ref(v_fileName_1887_);
v___x_2096_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2096_, 0, v_fileName_1887_);
lean_ctor_set(v___x_2096_, 1, v_fileMap_1888_);
lean_ctor_set(v___x_2096_, 2, v_options_1889_);
lean_ctor_set(v___x_2096_, 3, v___x_2095_);
lean_ctor_set(v___x_2096_, 4, v_maxRecDepth_1891_);
lean_ctor_set(v___x_2096_, 5, v_ref_1892_);
lean_ctor_set(v___x_2096_, 6, v_currNamespace_1893_);
lean_ctor_set(v___x_2096_, 7, v_openDecls_1894_);
lean_ctor_set(v___x_2096_, 8, v_initHeartbeats_1895_);
lean_ctor_set(v___x_2096_, 9, v_maxHeartbeats_1896_);
lean_ctor_set(v___x_2096_, 10, v_quotContext_1897_);
lean_ctor_set(v___x_2096_, 11, v_currMacroScope_1898_);
lean_ctor_set(v___x_2096_, 12, v_cancelTk_x3f_1900_);
lean_ctor_set(v___x_2096_, 13, v_inheritedTraceOptions_1902_);
lean_ctor_set_uint8(v___x_2096_, sizeof(void*)*14, v_diag_1899_);
lean_ctor_set_uint8(v___x_2096_, sizeof(void*)*14 + 1, v_suppressElabErrors_1901_);
if (v_hasTrace_2093_ == 0)
{
lean_object* v___x_2097_; 
lean_dec(v___x_2095_);
lean_dec_ref(v___x_1886_);
v___x_2097_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(v___x_1907_, v_e_u2081_1813_, v_e_u2082_1814_, v___f_1826_, v___f_1828_, v___x_1908_, v_cls_1904_, v___x_1872_, v___x_1884_, v___x_1877_, v___f_1880_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___x_2096_, v_a_1822_);
lean_dec_ref_known(v___x_2096_, 14);
return v___x_2097_;
}
else
{
lean_object* v___x_2098_; uint8_t v___x_2099_; 
v___x_2098_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_2099_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1902_, v_options_1889_, v___x_2098_);
if (v___x_2099_ == 0)
{
lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; uint8_t v___x_2103_; 
v___x_2100_ = l_Lean_KVMap_instValueBool;
v___x_2101_ = l_Lean_trace_profiler;
v___x_2102_ = l_Lean_Option_get___redArg(v___x_2100_, v_options_1889_, v___x_2101_);
v___x_2103_ = lean_unbox(v___x_2102_);
lean_dec(v___x_2102_);
if (v___x_2103_ == 0)
{
lean_object* v___x_2104_; 
lean_dec(v___x_2095_);
lean_dec_ref(v___x_1886_);
v___x_2104_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(v___x_1907_, v_e_u2081_1813_, v_e_u2082_1814_, v___f_1826_, v___f_1828_, v___x_1908_, v_cls_1904_, v___x_1872_, v___x_1884_, v___x_1877_, v___f_1880_, v_a_1815_, v_a_1816_, v_a_1817_, v_a_1818_, v_a_1819_, v_a_1820_, v___x_2096_, v_a_1822_);
lean_dec_ref_known(v___x_2096_, 14);
return v___x_2104_;
}
else
{
v___y_2063_ = v___x_2099_;
v___y_2064_ = v___x_2095_;
v___y_2065_ = v___x_2096_;
goto v___jp_2062_;
}
}
else
{
v___y_2063_ = v___x_2099_;
v___y_2064_ = v___x_2095_;
v___y_2065_ = v___x_2096_;
goto v___jp_2062_;
}
}
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3(void){
_start:
{
lean_object* v___x_2119_; lean_object* v___x_2120_; 
v___x_2119_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__2));
v___x_2120_ = l_Lean_stringToMessageData(v___x_2119_);
return v___x_2120_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0(void){
_start:
{
lean_object* v___x_2121_; 
v___x_2121_ = l_instMonadControlReaderT(lean_box(0), lean_box(0));
return v___x_2121_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1(void){
_start:
{
lean_object* v___x_2122_; 
v___x_2122_ = l_instMonadControlStateRefT_x27(lean_box(0), lean_box(0), lean_box(0));
return v___x_2122_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4(void){
_start:
{
lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; 
v___x_2123_ = lean_box(0);
v___x_2124_ = lean_unsigned_to_nat(16u);
v___x_2125_ = lean_mk_array(v___x_2124_, v___x_2123_);
return v___x_2125_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5(void){
_start:
{
lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; 
v___x_2126_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__4);
v___x_2127_ = lean_unsigned_to_nat(0u);
v___x_2128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2128_, 0, v___x_2127_);
lean_ctor_set(v___x_2128_, 1, v___x_2126_);
return v___x_2128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(lean_object* v_e_u2081_2129_, lean_object* v_e_u2082_2130_, lean_object* v_a_2131_, lean_object* v_a_2132_, lean_object* v_a_2133_, lean_object* v_a_2134_, lean_object* v_a_2135_, lean_object* v_a_2136_, lean_object* v_a_2137_){
_start:
{
lean_object* v___x_2139_; lean_object* v_toApplicative_2140_; lean_object* v_toFunctor_2141_; lean_object* v_toSeq_2142_; lean_object* v_toSeqLeft_2143_; lean_object* v_toSeqRight_2144_; lean_object* v___f_2145_; lean_object* v___f_2146_; lean_object* v___f_2147_; lean_object* v___f_2148_; lean_object* v___x_2149_; lean_object* v___f_2150_; lean_object* v___f_2151_; lean_object* v___f_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v_toApplicative_2156_; lean_object* v___x_2158_; uint8_t v_isShared_2159_; uint8_t v_isSharedCheck_2263_; 
v___x_2139_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_2140_ = lean_ctor_get(v___x_2139_, 0);
v_toFunctor_2141_ = lean_ctor_get(v_toApplicative_2140_, 0);
v_toSeq_2142_ = lean_ctor_get(v_toApplicative_2140_, 2);
v_toSeqLeft_2143_ = lean_ctor_get(v_toApplicative_2140_, 3);
v_toSeqRight_2144_ = lean_ctor_get(v_toApplicative_2140_, 4);
v___f_2145_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_2146_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_2141_, 2);
v___f_2147_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2147_, 0, v_toFunctor_2141_);
v___f_2148_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2148_, 0, v_toFunctor_2141_);
v___x_2149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2149_, 0, v___f_2147_);
lean_ctor_set(v___x_2149_, 1, v___f_2148_);
lean_inc(v_toSeqRight_2144_);
v___f_2150_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2150_, 0, v_toSeqRight_2144_);
lean_inc(v_toSeqLeft_2143_);
v___f_2151_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2151_, 0, v_toSeqLeft_2143_);
lean_inc(v_toSeq_2142_);
v___f_2152_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2152_, 0, v_toSeq_2142_);
v___x_2153_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2153_, 0, v___x_2149_);
lean_ctor_set(v___x_2153_, 1, v___f_2145_);
lean_ctor_set(v___x_2153_, 2, v___f_2152_);
lean_ctor_set(v___x_2153_, 3, v___f_2151_);
lean_ctor_set(v___x_2153_, 4, v___f_2150_);
v___x_2154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2154_, 0, v___x_2153_);
lean_ctor_set(v___x_2154_, 1, v___f_2146_);
v___x_2155_ = l_StateRefT_x27_instMonad___redArg(v___x_2154_);
v_toApplicative_2156_ = lean_ctor_get(v___x_2155_, 0);
v_isSharedCheck_2263_ = !lean_is_exclusive(v___x_2155_);
if (v_isSharedCheck_2263_ == 0)
{
lean_object* v_unused_2264_; 
v_unused_2264_ = lean_ctor_get(v___x_2155_, 1);
lean_dec(v_unused_2264_);
v___x_2158_ = v___x_2155_;
v_isShared_2159_ = v_isSharedCheck_2263_;
goto v_resetjp_2157_;
}
else
{
lean_inc(v_toApplicative_2156_);
lean_dec(v___x_2155_);
v___x_2158_ = lean_box(0);
v_isShared_2159_ = v_isSharedCheck_2263_;
goto v_resetjp_2157_;
}
v_resetjp_2157_:
{
lean_object* v_toFunctor_2160_; lean_object* v_toSeq_2161_; lean_object* v_toSeqLeft_2162_; lean_object* v_toSeqRight_2163_; lean_object* v___x_2165_; uint8_t v_isShared_2166_; uint8_t v_isSharedCheck_2261_; 
v_toFunctor_2160_ = lean_ctor_get(v_toApplicative_2156_, 0);
v_toSeq_2161_ = lean_ctor_get(v_toApplicative_2156_, 2);
v_toSeqLeft_2162_ = lean_ctor_get(v_toApplicative_2156_, 3);
v_toSeqRight_2163_ = lean_ctor_get(v_toApplicative_2156_, 4);
v_isSharedCheck_2261_ = !lean_is_exclusive(v_toApplicative_2156_);
if (v_isSharedCheck_2261_ == 0)
{
lean_object* v_unused_2262_; 
v_unused_2262_ = lean_ctor_get(v_toApplicative_2156_, 1);
lean_dec(v_unused_2262_);
v___x_2165_ = v_toApplicative_2156_;
v_isShared_2166_ = v_isSharedCheck_2261_;
goto v_resetjp_2164_;
}
else
{
lean_inc(v_toSeqRight_2163_);
lean_inc(v_toSeqLeft_2162_);
lean_inc(v_toSeq_2161_);
lean_inc(v_toFunctor_2160_);
lean_dec(v_toApplicative_2156_);
v___x_2165_ = lean_box(0);
v_isShared_2166_ = v_isSharedCheck_2261_;
goto v_resetjp_2164_;
}
v_resetjp_2164_:
{
lean_object* v___f_2167_; lean_object* v___f_2168_; lean_object* v___f_2169_; lean_object* v___f_2170_; lean_object* v___x_2171_; lean_object* v___f_2172_; lean_object* v___f_2173_; lean_object* v___f_2174_; lean_object* v___x_2176_; 
v___f_2167_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_2168_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_2160_);
v___f_2169_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2169_, 0, v_toFunctor_2160_);
v___f_2170_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2170_, 0, v_toFunctor_2160_);
v___x_2171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2171_, 0, v___f_2169_);
lean_ctor_set(v___x_2171_, 1, v___f_2170_);
v___f_2172_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2172_, 0, v_toSeqRight_2163_);
v___f_2173_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2173_, 0, v_toSeqLeft_2162_);
v___f_2174_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2174_, 0, v_toSeq_2161_);
if (v_isShared_2166_ == 0)
{
lean_ctor_set(v___x_2165_, 4, v___f_2172_);
lean_ctor_set(v___x_2165_, 3, v___f_2173_);
lean_ctor_set(v___x_2165_, 2, v___f_2174_);
lean_ctor_set(v___x_2165_, 1, v___f_2167_);
lean_ctor_set(v___x_2165_, 0, v___x_2171_);
v___x_2176_ = v___x_2165_;
goto v_reusejp_2175_;
}
else
{
lean_object* v_reuseFailAlloc_2260_; 
v_reuseFailAlloc_2260_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2260_, 0, v___x_2171_);
lean_ctor_set(v_reuseFailAlloc_2260_, 1, v___f_2167_);
lean_ctor_set(v_reuseFailAlloc_2260_, 2, v___f_2174_);
lean_ctor_set(v_reuseFailAlloc_2260_, 3, v___f_2173_);
lean_ctor_set(v_reuseFailAlloc_2260_, 4, v___f_2172_);
v___x_2176_ = v_reuseFailAlloc_2260_;
goto v_reusejp_2175_;
}
v_reusejp_2175_:
{
lean_object* v___x_2178_; 
if (v_isShared_2159_ == 0)
{
lean_ctor_set(v___x_2158_, 1, v___f_2168_);
lean_ctor_set(v___x_2158_, 0, v___x_2176_);
v___x_2178_ = v___x_2158_;
goto v_reusejp_2177_;
}
else
{
lean_object* v_reuseFailAlloc_2259_; 
v_reuseFailAlloc_2259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2259_, 0, v___x_2176_);
lean_ctor_set(v_reuseFailAlloc_2259_, 1, v___f_2168_);
v___x_2178_ = v_reuseFailAlloc_2259_;
goto v_reusejp_2177_;
}
v_reusejp_2177_:
{
lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v_toApplicative_2184_; lean_object* v_toFunctor_2185_; lean_object* v_toSeq_2186_; lean_object* v_toSeqLeft_2187_; lean_object* v_toSeqRight_2188_; lean_object* v___f_2189_; lean_object* v___f_2190_; lean_object* v___x_2191_; lean_object* v___f_2192_; lean_object* v___f_2193_; lean_object* v___f_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___f_2200_; lean_object* v___f_2201_; lean_object* v___x_2202_; lean_object* v___f_2203_; lean_object* v___f_2204_; lean_object* v___x_2205_; lean_object* v___f_2206_; lean_object* v___f_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v_getMCtx_2210_; lean_object* v_modifyMCtx_2211_; lean_object* v___f_2212_; lean_object* v___x_2213_; lean_object* v___f_2214_; lean_object* v___x_2215_; lean_object* v___f_2216_; lean_object* v___x_2217_; lean_object* v___f_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v_mctx_u2081_2221_; lean_object* v_mctx_u2082_2222_; lean_object* v___x_2223_; lean_object* v___x_27338__overap_2224_; lean_object* v___x_2225_; 
v___x_2179_ = l_StateRefT_x27_instMonad___redArg(v___x_2178_);
v___x_2180_ = l_ReaderT_instMonad___redArg(v___x_2179_);
v___x_2181_ = l_ReaderT_instMonad___redArg(v___x_2180_);
v___x_2182_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__0);
v___x_2183_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__1);
v_toApplicative_2184_ = lean_ctor_get(v___x_2139_, 0);
v_toFunctor_2185_ = lean_ctor_get(v_toApplicative_2184_, 0);
v_toSeq_2186_ = lean_ctor_get(v_toApplicative_2184_, 2);
v_toSeqLeft_2187_ = lean_ctor_get(v_toApplicative_2184_, 3);
v_toSeqRight_2188_ = lean_ctor_get(v_toApplicative_2184_, 4);
lean_inc_ref_n(v_toFunctor_2185_, 2);
v___f_2189_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2189_, 0, v_toFunctor_2185_);
v___f_2190_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2190_, 0, v_toFunctor_2185_);
v___x_2191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2191_, 0, v___f_2189_);
lean_ctor_set(v___x_2191_, 1, v___f_2190_);
lean_inc(v_toSeqRight_2188_);
v___f_2192_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2192_, 0, v_toSeqRight_2188_);
lean_inc(v_toSeqLeft_2187_);
v___f_2193_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2193_, 0, v_toSeqLeft_2187_);
lean_inc(v_toSeq_2186_);
v___f_2194_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2194_, 0, v_toSeq_2186_);
v___x_2195_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2195_, 0, v___x_2191_);
lean_ctor_set(v___x_2195_, 1, v___f_2145_);
lean_ctor_set(v___x_2195_, 2, v___f_2194_);
lean_ctor_set(v___x_2195_, 3, v___f_2193_);
lean_ctor_set(v___x_2195_, 4, v___f_2192_);
v___x_2196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2196_, 0, v___x_2195_);
lean_ctor_set(v___x_2196_, 1, v___f_2146_);
v___x_2197_ = l_StateRefT_x27_instMonad___redArg(v___x_2196_);
v___x_2198_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_2198_, 0, lean_box(0));
lean_closure_set(v___x_2198_, 1, lean_box(0));
lean_closure_set(v___x_2198_, 2, v___x_2197_);
v___x_2199_ = l_instMonadControlTOfPure___redArg(v___x_2198_);
lean_inc_ref(v___x_2199_);
v___f_2200_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2200_, 0, v___x_2183_);
lean_closure_set(v___f_2200_, 1, v___x_2199_);
v___f_2201_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2201_, 0, v___x_2183_);
lean_closure_set(v___f_2201_, 1, v___x_2199_);
v___x_2202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2202_, 0, v___f_2200_);
lean_ctor_set(v___x_2202_, 1, v___f_2201_);
lean_inc_ref(v___x_2202_);
v___f_2203_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2203_, 0, v___x_2182_);
lean_closure_set(v___f_2203_, 1, v___x_2202_);
v___f_2204_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2204_, 0, v___x_2182_);
lean_closure_set(v___f_2204_, 1, v___x_2202_);
v___x_2205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2205_, 0, v___f_2203_);
lean_ctor_set(v___x_2205_, 1, v___f_2204_);
lean_inc_ref(v___x_2205_);
v___f_2206_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2206_, 0, v___x_2182_);
lean_closure_set(v___f_2206_, 1, v___x_2205_);
v___f_2207_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2207_, 0, v___x_2182_);
lean_closure_set(v___f_2207_, 1, v___x_2205_);
v___x_2208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2208_, 0, v___f_2206_);
lean_ctor_set(v___x_2208_, 1, v___f_2207_);
v___x_2209_ = l_Lean_Meta_instMonadMCtxMetaM;
v_getMCtx_2210_ = lean_ctor_get(v___x_2209_, 0);
v_modifyMCtx_2211_ = lean_ctor_get(v___x_2209_, 1);
v___f_2212_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__2));
v___x_2213_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__3));
lean_inc(v_modifyMCtx_2211_);
v___f_2214_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2214_, 0, v_modifyMCtx_2211_);
lean_closure_set(v___f_2214_, 1, v___x_2213_);
lean_inc(v_getMCtx_2210_);
v___x_2215_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2215_, 0, lean_box(0));
lean_closure_set(v___x_2215_, 1, lean_box(0));
lean_closure_set(v___x_2215_, 2, lean_box(0));
lean_closure_set(v___x_2215_, 3, lean_box(0));
lean_closure_set(v___x_2215_, 4, v_getMCtx_2210_);
v___f_2216_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2216_, 0, v___f_2214_);
lean_closure_set(v___f_2216_, 1, v___f_2212_);
v___x_2217_ = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(v___x_2217_, 0, lean_box(0));
lean_closure_set(v___x_2217_, 1, v___x_2215_);
v___f_2218_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2218_, 0, v___f_2216_);
lean_closure_set(v___f_2218_, 1, v___f_2212_);
v___x_2219_ = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(v___x_2219_, 0, lean_box(0));
lean_closure_set(v___x_2219_, 1, v___x_2217_);
v___x_2220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2220_, 0, v___x_2219_);
lean_ctor_set(v___x_2220_, 1, v___f_2218_);
v_mctx_u2081_2221_ = lean_ctor_get(v_a_2132_, 1);
v_mctx_u2082_2222_ = lean_ctor_get(v_a_2132_, 2);
lean_inc_ref(v___x_2220_);
lean_inc_ref_n(v___x_2181_, 2);
v___x_2223_ = l_Lean_instantiateMVars___redArg(v___x_2181_, v___x_2220_, v_e_u2081_2129_);
lean_inc_ref(v_mctx_u2081_2221_);
lean_inc_ref(v___x_2208_);
v___x_27338__overap_2224_ = l_Lean_Meta_withMCtx___redArg(v___x_2208_, v___x_2181_, v_mctx_u2081_2221_, v___x_2223_);
lean_inc(v_a_2137_);
lean_inc_ref(v_a_2136_);
lean_inc(v_a_2135_);
lean_inc_ref(v_a_2134_);
lean_inc(v_a_2133_);
lean_inc_ref(v_a_2132_);
lean_inc_ref(v_a_2131_);
v___x_2225_ = lean_apply_8(v___x_27338__overap_2224_, v_a_2131_, v_a_2132_, v_a_2133_, v_a_2134_, v_a_2135_, v_a_2136_, v_a_2137_, lean_box(0));
if (lean_obj_tag(v___x_2225_) == 0)
{
lean_object* v_a_2226_; lean_object* v___x_2227_; lean_object* v___x_27343__overap_2228_; lean_object* v___x_2229_; 
v_a_2226_ = lean_ctor_get(v___x_2225_, 0);
lean_inc(v_a_2226_);
lean_dec_ref_known(v___x_2225_, 1);
lean_inc_ref(v___x_2181_);
v___x_2227_ = l_Lean_instantiateMVars___redArg(v___x_2181_, v___x_2220_, v_e_u2082_2130_);
lean_inc_ref(v_mctx_u2082_2222_);
v___x_27343__overap_2228_ = l_Lean_Meta_withMCtx___redArg(v___x_2208_, v___x_2181_, v_mctx_u2082_2222_, v___x_2227_);
lean_inc(v_a_2137_);
lean_inc_ref(v_a_2136_);
lean_inc(v_a_2135_);
lean_inc_ref(v_a_2134_);
lean_inc(v_a_2133_);
lean_inc_ref(v_a_2132_);
lean_inc_ref(v_a_2131_);
v___x_2229_ = lean_apply_8(v___x_27343__overap_2228_, v_a_2131_, v_a_2132_, v_a_2133_, v_a_2134_, v_a_2135_, v_a_2136_, v_a_2137_, lean_box(0));
if (lean_obj_tag(v___x_2229_) == 0)
{
lean_object* v_a_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; 
v_a_2230_ = lean_ctor_get(v___x_2229_, 0);
lean_inc(v_a_2230_);
lean_dec_ref_known(v___x_2229_, 1);
v___x_2231_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___closed__5);
v___x_2232_ = lean_st_mk_ref(v___x_2231_);
v___x_2233_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_a_2226_, v_a_2230_, v___x_2232_, v_a_2131_, v_a_2132_, v_a_2133_, v_a_2134_, v_a_2135_, v_a_2136_, v_a_2137_);
if (lean_obj_tag(v___x_2233_) == 0)
{
lean_object* v_a_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2242_; 
v_a_2234_ = lean_ctor_get(v___x_2233_, 0);
v_isSharedCheck_2242_ = !lean_is_exclusive(v___x_2233_);
if (v_isSharedCheck_2242_ == 0)
{
v___x_2236_ = v___x_2233_;
v_isShared_2237_ = v_isSharedCheck_2242_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_a_2234_);
lean_dec(v___x_2233_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2242_;
goto v_resetjp_2235_;
}
v_resetjp_2235_:
{
lean_object* v___x_2238_; lean_object* v___x_2240_; 
v___x_2238_ = lean_st_ref_get(v___x_2232_);
lean_dec(v___x_2232_);
lean_dec(v___x_2238_);
if (v_isShared_2237_ == 0)
{
v___x_2240_ = v___x_2236_;
goto v_reusejp_2239_;
}
else
{
lean_object* v_reuseFailAlloc_2241_; 
v_reuseFailAlloc_2241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2241_, 0, v_a_2234_);
v___x_2240_ = v_reuseFailAlloc_2241_;
goto v_reusejp_2239_;
}
v_reusejp_2239_:
{
return v___x_2240_;
}
}
}
else
{
lean_dec(v___x_2232_);
return v___x_2233_;
}
}
else
{
lean_object* v_a_2243_; lean_object* v___x_2245_; uint8_t v_isShared_2246_; uint8_t v_isSharedCheck_2250_; 
lean_dec(v_a_2226_);
v_a_2243_ = lean_ctor_get(v___x_2229_, 0);
v_isSharedCheck_2250_ = !lean_is_exclusive(v___x_2229_);
if (v_isSharedCheck_2250_ == 0)
{
v___x_2245_ = v___x_2229_;
v_isShared_2246_ = v_isSharedCheck_2250_;
goto v_resetjp_2244_;
}
else
{
lean_inc(v_a_2243_);
lean_dec(v___x_2229_);
v___x_2245_ = lean_box(0);
v_isShared_2246_ = v_isSharedCheck_2250_;
goto v_resetjp_2244_;
}
v_resetjp_2244_:
{
lean_object* v___x_2248_; 
if (v_isShared_2246_ == 0)
{
v___x_2248_ = v___x_2245_;
goto v_reusejp_2247_;
}
else
{
lean_object* v_reuseFailAlloc_2249_; 
v_reuseFailAlloc_2249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2249_, 0, v_a_2243_);
v___x_2248_ = v_reuseFailAlloc_2249_;
goto v_reusejp_2247_;
}
v_reusejp_2247_:
{
return v___x_2248_;
}
}
}
}
else
{
lean_object* v_a_2251_; lean_object* v___x_2253_; uint8_t v_isShared_2254_; uint8_t v_isSharedCheck_2258_; 
lean_dec_ref_known(v___x_2220_, 2);
lean_dec_ref_known(v___x_2208_, 2);
lean_dec_ref(v___x_2181_);
lean_dec_ref(v_e_u2082_2130_);
v_a_2251_ = lean_ctor_get(v___x_2225_, 0);
v_isSharedCheck_2258_ = !lean_is_exclusive(v___x_2225_);
if (v_isSharedCheck_2258_ == 0)
{
v___x_2253_ = v___x_2225_;
v_isShared_2254_ = v_isSharedCheck_2258_;
goto v_resetjp_2252_;
}
else
{
lean_inc(v_a_2251_);
lean_dec(v___x_2225_);
v___x_2253_ = lean_box(0);
v_isShared_2254_ = v_isSharedCheck_2258_;
goto v_resetjp_2252_;
}
v_resetjp_2252_:
{
lean_object* v___x_2256_; 
if (v_isShared_2254_ == 0)
{
v___x_2256_ = v___x_2253_;
goto v_reusejp_2255_;
}
else
{
lean_object* v_reuseFailAlloc_2257_; 
v_reuseFailAlloc_2257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2257_, 0, v_a_2251_);
v___x_2256_ = v_reuseFailAlloc_2257_;
goto v_reusejp_2255_;
}
v_reusejp_2255_:
{
return v___x_2256_;
}
}
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5(void){
_start:
{
lean_object* v___x_2266_; lean_object* v___x_2267_; 
v___x_2266_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__4));
v___x_2267_ = l_Lean_stringToMessageData(v___x_2266_);
return v___x_2267_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9(void){
_start:
{
lean_object* v___x_2271_; lean_object* v___x_2272_; 
v___x_2271_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__8));
v___x_2272_ = l_Lean_stringToMessageData(v___x_2271_);
return v___x_2272_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11(void){
_start:
{
lean_object* v___x_2274_; lean_object* v___x_2275_; 
v___x_2274_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__10));
v___x_2275_ = l_Lean_stringToMessageData(v___x_2274_);
return v___x_2275_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13(void){
_start:
{
lean_object* v___x_2277_; lean_object* v___x_2278_; 
v___x_2277_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__12));
v___x_2278_ = l_Lean_stringToMessageData(v___x_2277_);
return v___x_2278_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15(void){
_start:
{
lean_object* v___x_2281_; lean_object* v___x_2282_; 
v___x_2281_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__14));
v___x_2282_ = l_Lean_stringToMessageData(v___x_2281_);
return v___x_2282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(lean_object* v_ldecl_u2081_2283_, lean_object* v_ldecl_u2082_2284_, lean_object* v_a_2285_, lean_object* v_a_2286_, lean_object* v_a_2287_, lean_object* v_a_2288_, lean_object* v_a_2289_, lean_object* v_a_2290_, lean_object* v_a_2291_){
_start:
{
uint8_t v___x_2293_; uint8_t v___x_2294_; uint8_t v___x_2295_; 
v___x_2293_ = 0;
v___x_2294_ = l_Lean_LocalDecl_isLet(v_ldecl_u2081_2283_, v___x_2293_);
v___x_2295_ = l_Lean_LocalDecl_isLet(v_ldecl_u2082_2284_, v___x_2293_);
if (v___x_2294_ == 0)
{
if (v___x_2295_ == 0)
{
lean_object* v___x_2296_; lean_object* v___x_2297_; uint8_t v___y_2318_; uint8_t v___x_2321_; uint8_t v___x_2322_; 
v___x_2296_ = l_Lean_LocalDecl_userName(v_ldecl_u2081_2283_);
v___x_2297_ = l_Lean_LocalDecl_userName(v_ldecl_u2082_2284_);
v___x_2321_ = l_Lean_Name_hasMacroScopes(v___x_2296_);
v___x_2322_ = l_Lean_Name_hasMacroScopes(v___x_2297_);
if (v___x_2321_ == 0)
{
if (v___x_2322_ == 0)
{
goto v___jp_2298_;
}
else
{
v___y_2318_ = v___x_2321_;
goto v___jp_2317_;
}
}
else
{
v___y_2318_ = v___x_2322_;
goto v___jp_2317_;
}
v___jp_2298_:
{
lean_object* v___x_2299_; lean_object* v___x_2300_; uint8_t v___x_2301_; 
v___x_2299_ = l_Lean_Name_eraseMacroScopes(v___x_2296_);
lean_dec(v___x_2296_);
v___x_2300_ = l_Lean_Name_eraseMacroScopes(v___x_2297_);
lean_dec(v___x_2297_);
v___x_2301_ = lean_name_eq(v___x_2299_, v___x_2300_);
lean_dec(v___x_2300_);
lean_dec(v___x_2299_);
if (v___x_2301_ == 0)
{
lean_object* v___x_2302_; lean_object* v___x_2303_; 
v___x_2302_ = lean_box(v___x_2293_);
v___x_2303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2302_);
return v___x_2303_;
}
else
{
uint8_t v___x_2304_; uint8_t v___x_2305_; uint8_t v___x_2306_; 
v___x_2304_ = l_Lean_LocalDecl_binderInfo(v_ldecl_u2081_2283_);
v___x_2305_ = l_Lean_LocalDecl_binderInfo(v_ldecl_u2082_2284_);
v___x_2306_ = l_Lean_instBEqBinderInfo_beq(v___x_2304_, v___x_2305_);
if (v___x_2306_ == 0)
{
lean_object* v___x_2307_; lean_object* v___x_2308_; 
v___x_2307_ = lean_box(v___x_2293_);
v___x_2308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2308_, 0, v___x_2307_);
return v___x_2308_;
}
else
{
uint8_t v___x_2309_; uint8_t v___x_2310_; uint8_t v___x_2311_; 
v___x_2309_ = l_Lean_LocalDecl_kind(v_ldecl_u2081_2283_);
v___x_2310_ = l_Lean_LocalDecl_kind(v_ldecl_u2082_2284_);
v___x_2311_ = l_Lean_instDecidableEqLocalDeclKind(v___x_2309_, v___x_2310_);
if (v___x_2311_ == 0)
{
lean_object* v___x_2312_; lean_object* v___x_2313_; 
v___x_2312_ = lean_box(v___x_2311_);
v___x_2313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2313_, 0, v___x_2312_);
return v___x_2313_;
}
else
{
lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; 
v___x_2314_ = l_Lean_LocalDecl_type(v_ldecl_u2081_2283_);
v___x_2315_ = l_Lean_LocalDecl_type(v_ldecl_u2082_2284_);
v___x_2316_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___x_2314_, v___x_2315_, v_a_2285_, v_a_2286_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_);
return v___x_2316_;
}
}
}
}
v___jp_2317_:
{
if (v___y_2318_ == 0)
{
lean_object* v___x_2319_; lean_object* v___x_2320_; 
lean_dec(v___x_2297_);
lean_dec(v___x_2296_);
v___x_2319_ = lean_box(v___x_2293_);
v___x_2320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2320_, 0, v___x_2319_);
return v___x_2320_;
}
else
{
goto v___jp_2298_;
}
}
}
else
{
lean_object* v___x_2323_; lean_object* v___x_2324_; 
v___x_2323_ = lean_box(v___x_2293_);
v___x_2324_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2324_, 0, v___x_2323_);
return v___x_2324_;
}
}
else
{
if (v___x_2295_ == 1)
{
lean_object* v___x_2325_; lean_object* v___x_2326_; uint8_t v___y_2328_; uint8_t v___x_2349_; uint8_t v___x_2350_; 
v___x_2325_ = l_Lean_LocalDecl_userName(v_ldecl_u2081_2283_);
v___x_2326_ = l_Lean_LocalDecl_userName(v_ldecl_u2082_2284_);
v___x_2349_ = l_Lean_Name_hasMacroScopes(v___x_2325_);
v___x_2350_ = l_Lean_Name_hasMacroScopes(v___x_2326_);
if (v___x_2349_ == 0)
{
if (v___x_2350_ == 0)
{
v___y_2328_ = v___x_2295_;
goto v___jp_2327_;
}
else
{
v___y_2328_ = v___x_2349_;
goto v___jp_2327_;
}
}
else
{
v___y_2328_ = v___x_2350_;
goto v___jp_2327_;
}
v___jp_2327_:
{
if (v___y_2328_ == 0)
{
lean_object* v___x_2329_; lean_object* v___x_2330_; 
lean_dec(v___x_2326_);
lean_dec(v___x_2325_);
v___x_2329_ = lean_box(v___x_2293_);
v___x_2330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2330_, 0, v___x_2329_);
return v___x_2330_;
}
else
{
lean_object* v___x_2331_; lean_object* v___x_2332_; uint8_t v___x_2333_; 
v___x_2331_ = l_Lean_Name_eraseMacroScopes(v___x_2325_);
lean_dec(v___x_2325_);
v___x_2332_ = l_Lean_Name_eraseMacroScopes(v___x_2326_);
lean_dec(v___x_2326_);
v___x_2333_ = lean_name_eq(v___x_2331_, v___x_2332_);
lean_dec(v___x_2332_);
lean_dec(v___x_2331_);
if (v___x_2333_ == 0)
{
lean_object* v___x_2334_; lean_object* v___x_2335_; 
v___x_2334_ = lean_box(v___x_2293_);
v___x_2335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2335_, 0, v___x_2334_);
return v___x_2335_;
}
else
{
uint8_t v___x_2336_; uint8_t v___x_2337_; uint8_t v___x_2338_; 
v___x_2336_ = l_Lean_LocalDecl_kind(v_ldecl_u2081_2283_);
v___x_2337_ = l_Lean_LocalDecl_kind(v_ldecl_u2082_2284_);
v___x_2338_ = l_Lean_instDecidableEqLocalDeclKind(v___x_2336_, v___x_2337_);
if (v___x_2338_ == 0)
{
lean_object* v___x_2339_; lean_object* v___x_2340_; 
v___x_2339_ = lean_box(v___x_2338_);
v___x_2340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2340_, 0, v___x_2339_);
return v___x_2340_;
}
else
{
lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; 
v___x_2341_ = l_Lean_LocalDecl_type(v_ldecl_u2081_2283_);
v___x_2342_ = l_Lean_LocalDecl_type(v_ldecl_u2082_2284_);
v___x_2343_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___x_2341_, v___x_2342_, v_a_2285_, v_a_2286_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_);
if (lean_obj_tag(v___x_2343_) == 0)
{
lean_object* v_a_2344_; uint8_t v___x_2345_; 
v_a_2344_ = lean_ctor_get(v___x_2343_, 0);
lean_inc(v_a_2344_);
v___x_2345_ = lean_unbox(v_a_2344_);
lean_dec(v_a_2344_);
if (v___x_2345_ == 0)
{
return v___x_2343_;
}
else
{
lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; 
lean_dec_ref_known(v___x_2343_, 1);
v___x_2346_ = l_Lean_LocalDecl_value(v_ldecl_u2081_2283_, v___x_2293_);
v___x_2347_ = l_Lean_LocalDecl_value(v_ldecl_u2082_2284_, v___x_2293_);
v___x_2348_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___x_2346_, v___x_2347_, v_a_2285_, v_a_2286_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_);
return v___x_2348_;
}
}
else
{
return v___x_2343_;
}
}
}
}
}
}
else
{
lean_object* v___x_2351_; lean_object* v___x_2352_; 
v___x_2351_ = lean_box(v___x_2293_);
v___x_2352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2352_, 0, v___x_2351_);
return v___x_2352_;
}
}
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_2354_; lean_object* v___x_2355_; 
v___x_2354_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__2));
v___x_2355_ = l_Lean_stringToMessageData(v___x_2354_);
return v___x_2355_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5(void){
_start:
{
lean_object* v___x_2357_; lean_object* v___x_2358_; 
v___x_2357_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__4));
v___x_2358_ = l_Lean_stringToMessageData(v___x_2357_);
return v___x_2358_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg(lean_object* v_decls_u2081_2359_, lean_object* v_decls_u2082_2360_, lean_object* v_i_2361_, lean_object* v_gctx_2362_, lean_object* v_a_2363_, lean_object* v_a_2364_, lean_object* v_a_2365_, lean_object* v_a_2366_, lean_object* v_a_2367_, lean_object* v_a_2368_){
_start:
{
lean_object* v___x_2370_; uint8_t v___x_2371_; 
v___x_2370_ = lean_array_get_size(v_decls_u2081_2359_);
v___x_2371_ = lean_nat_dec_lt(v_i_2361_, v___x_2370_);
if (v___x_2371_ == 0)
{
lean_object* v___x_2372_; lean_object* v___x_2373_; 
lean_dec(v_i_2361_);
v___x_2372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2372_, 0, v_gctx_2362_);
v___x_2373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2373_, 0, v___x_2372_);
return v___x_2373_;
}
else
{
lean_object* v_options_2374_; lean_object* v_fileName_2375_; lean_object* v_fileMap_2376_; lean_object* v_currRecDepth_2377_; lean_object* v_maxRecDepth_2378_; lean_object* v_ref_2379_; lean_object* v_currNamespace_2380_; lean_object* v_openDecls_2381_; lean_object* v_initHeartbeats_2382_; lean_object* v_maxHeartbeats_2383_; lean_object* v_quotContext_2384_; lean_object* v_currMacroScope_2385_; uint8_t v_diag_2386_; lean_object* v_cancelTk_x3f_2387_; uint8_t v_suppressElabErrors_2388_; lean_object* v_inheritedTraceOptions_2389_; uint8_t v_hasTrace_2390_; lean_object* v_ldecl_u2081_2391_; lean_object* v_ldecl_u2082_2392_; lean_object* v___y_2394_; 
v_options_2374_ = lean_ctor_get(v_a_2367_, 2);
v_fileName_2375_ = lean_ctor_get(v_a_2367_, 0);
v_fileMap_2376_ = lean_ctor_get(v_a_2367_, 1);
v_currRecDepth_2377_ = lean_ctor_get(v_a_2367_, 3);
v_maxRecDepth_2378_ = lean_ctor_get(v_a_2367_, 4);
v_ref_2379_ = lean_ctor_get(v_a_2367_, 5);
v_currNamespace_2380_ = lean_ctor_get(v_a_2367_, 6);
v_openDecls_2381_ = lean_ctor_get(v_a_2367_, 7);
v_initHeartbeats_2382_ = lean_ctor_get(v_a_2367_, 8);
v_maxHeartbeats_2383_ = lean_ctor_get(v_a_2367_, 9);
v_quotContext_2384_ = lean_ctor_get(v_a_2367_, 10);
v_currMacroScope_2385_ = lean_ctor_get(v_a_2367_, 11);
v_diag_2386_ = lean_ctor_get_uint8(v_a_2367_, sizeof(void*)*14);
v_cancelTk_x3f_2387_ = lean_ctor_get(v_a_2367_, 12);
v_suppressElabErrors_2388_ = lean_ctor_get_uint8(v_a_2367_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2389_ = lean_ctor_get(v_a_2367_, 13);
v_hasTrace_2390_ = lean_ctor_get_uint8(v_options_2374_, sizeof(void*)*1);
v_ldecl_u2081_2391_ = lean_array_fget_borrowed(v_decls_u2081_2359_, v_i_2361_);
v_ldecl_u2082_2392_ = lean_array_fget_borrowed(v_decls_u2082_2360_, v_i_2361_);
if (v_hasTrace_2390_ == 0)
{
lean_object* v___x_2431_; 
v___x_2431_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(v_ldecl_u2081_2391_, v_ldecl_u2082_2392_, v_gctx_2362_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
v___y_2394_ = v___x_2431_;
goto v___jp_2393_;
}
else
{
lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; uint8_t v___x_2435_; lean_object* v___y_2437_; lean_object* v___y_2438_; lean_object* v___y_2439_; lean_object* v_a_2440_; lean_object* v___y_2453_; lean_object* v___y_2454_; lean_object* v___y_2455_; lean_object* v_a_2456_; 
v___x_2432_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_2433_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0));
v___x_2434_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_2435_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2389_, v_options_2374_, v___x_2434_);
if (v___x_2435_ == 0)
{
lean_object* v___x_2535_; uint8_t v___x_2536_; 
v___x_2535_ = l_Lean_trace_profiler;
v___x_2536_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(v_options_2374_, v___x_2535_);
if (v___x_2536_ == 0)
{
lean_object* v___x_2537_; 
v___x_2537_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(v_ldecl_u2081_2391_, v_ldecl_u2082_2392_, v_gctx_2362_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
v___y_2394_ = v___x_2537_;
goto v___jp_2393_;
}
else
{
goto v___jp_2465_;
}
}
else
{
goto v___jp_2465_;
}
v___jp_2436_:
{
lean_object* v___x_2441_; double v___x_2442_; double v___x_2443_; double v___x_2444_; double v___x_2445_; double v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; 
v___x_2441_ = lean_io_mono_nanos_now();
v___x_2442_ = lean_float_of_nat(v___y_2439_);
v___x_2443_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_2444_ = lean_float_div(v___x_2442_, v___x_2443_);
v___x_2445_ = lean_float_of_nat(v___x_2441_);
v___x_2446_ = lean_float_div(v___x_2445_, v___x_2443_);
v___x_2447_ = lean_box_float(v___x_2444_);
v___x_2448_ = lean_box_float(v___x_2446_);
v___x_2449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2449_, 0, v___x_2447_);
lean_ctor_set(v___x_2449_, 1, v___x_2448_);
v___x_2450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2450_, 0, v_a_2440_);
lean_ctor_set(v___x_2450_, 1, v___x_2449_);
lean_inc(v_ref_2379_);
v___x_2451_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3(v___x_2432_, v_hasTrace_2390_, v___x_2433_, v_options_2374_, v___x_2435_, v___y_2438_, v_ref_2379_, v___y_2437_, v___x_2450_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
v___y_2394_ = v___x_2451_;
goto v___jp_2393_;
}
v___jp_2452_:
{
lean_object* v___x_2457_; double v___x_2458_; double v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; 
v___x_2457_ = lean_io_get_num_heartbeats();
v___x_2458_ = lean_float_of_nat(v___y_2455_);
v___x_2459_ = lean_float_of_nat(v___x_2457_);
v___x_2460_ = lean_box_float(v___x_2458_);
v___x_2461_ = lean_box_float(v___x_2459_);
v___x_2462_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2462_, 0, v___x_2460_);
lean_ctor_set(v___x_2462_, 1, v___x_2461_);
v___x_2463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2463_, 0, v_a_2456_);
lean_ctor_set(v___x_2463_, 1, v___x_2462_);
lean_inc(v_ref_2379_);
v___x_2464_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3(v___x_2432_, v_hasTrace_2390_, v___x_2433_, v_options_2374_, v___x_2435_, v___y_2454_, v_ref_2379_, v___y_2453_, v___x_2463_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
v___y_2394_ = v___x_2464_;
goto v___jp_2393_;
}
v___jp_2465_:
{
lean_object* v___x_2466_; 
v___x_2466_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg(v_a_2368_);
if (lean_obj_tag(v___x_2466_) == 0)
{
lean_object* v_a_2467_; lean_object* v_ref_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; 
v_a_2467_ = lean_ctor_get(v___x_2466_, 0);
lean_inc(v_a_2467_);
lean_dec_ref_known(v___x_2466_, 1);
v_ref_2468_ = l_Lean_replaceRef(v_ref_2379_, v_ref_2379_);
lean_inc_ref(v_inheritedTraceOptions_2389_);
lean_inc(v_cancelTk_x3f_2387_);
lean_inc(v_currMacroScope_2385_);
lean_inc(v_quotContext_2384_);
lean_inc(v_maxHeartbeats_2383_);
lean_inc(v_initHeartbeats_2382_);
lean_inc(v_openDecls_2381_);
lean_inc(v_currNamespace_2380_);
lean_inc(v_maxRecDepth_2378_);
lean_inc(v_currRecDepth_2377_);
lean_inc_ref(v_options_2374_);
lean_inc_ref(v_fileMap_2376_);
lean_inc_ref(v_fileName_2375_);
v___x_2469_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2469_, 0, v_fileName_2375_);
lean_ctor_set(v___x_2469_, 1, v_fileMap_2376_);
lean_ctor_set(v___x_2469_, 2, v_options_2374_);
lean_ctor_set(v___x_2469_, 3, v_currRecDepth_2377_);
lean_ctor_set(v___x_2469_, 4, v_maxRecDepth_2378_);
lean_ctor_set(v___x_2469_, 5, v_ref_2468_);
lean_ctor_set(v___x_2469_, 6, v_currNamespace_2380_);
lean_ctor_set(v___x_2469_, 7, v_openDecls_2381_);
lean_ctor_set(v___x_2469_, 8, v_initHeartbeats_2382_);
lean_ctor_set(v___x_2469_, 9, v_maxHeartbeats_2383_);
lean_ctor_set(v___x_2469_, 10, v_quotContext_2384_);
lean_ctor_set(v___x_2469_, 11, v_currMacroScope_2385_);
lean_ctor_set(v___x_2469_, 12, v_cancelTk_x3f_2387_);
lean_ctor_set(v___x_2469_, 13, v_inheritedTraceOptions_2389_);
lean_ctor_set_uint8(v___x_2469_, sizeof(void*)*14, v_diag_2386_);
lean_ctor_set_uint8(v___x_2469_, sizeof(void*)*14 + 1, v_suppressElabErrors_2388_);
v___x_2470_ = lean_obj_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__3);
v___x_2471_ = l_Lean_LocalDecl_userName(v_ldecl_u2081_2391_);
v___x_2472_ = l_Lean_MessageData_ofName(v___x_2471_);
v___x_2473_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2473_, 0, v___x_2470_);
lean_ctor_set(v___x_2473_, 1, v___x_2472_);
v___x_2474_ = lean_obj_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5);
v___x_2475_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2475_, 0, v___x_2473_);
lean_ctor_set(v___x_2475_, 1, v___x_2474_);
v___x_2476_ = l_Lean_LocalDecl_userName(v_ldecl_u2082_2392_);
v___x_2477_ = l_Lean_MessageData_ofName(v___x_2476_);
v___x_2478_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2478_, 0, v___x_2475_);
lean_ctor_set(v___x_2478_, 1, v___x_2477_);
v___x_2479_ = lp_aesop_Lean_addMessageContextFull___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082_printExpr_spec__0(v___x_2478_, v_a_2365_, v_a_2366_, v___x_2469_, v_a_2368_);
lean_dec_ref_known(v___x_2469_, 14);
if (lean_obj_tag(v___x_2479_) == 0)
{
lean_object* v_a_2480_; lean_object* v___x_2481_; uint8_t v___x_2482_; 
v_a_2480_ = lean_ctor_get(v___x_2479_, 0);
lean_inc(v_a_2480_);
lean_dec_ref_known(v___x_2479_, 1);
v___x_2481_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2482_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__2(v_options_2374_, v___x_2481_);
if (v___x_2482_ == 0)
{
lean_object* v___x_2483_; lean_object* v___x_2484_; 
v___x_2483_ = lean_io_mono_nanos_now();
v___x_2484_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(v_ldecl_u2081_2391_, v_ldecl_u2082_2392_, v_gctx_2362_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
if (lean_obj_tag(v___x_2484_) == 0)
{
lean_object* v_a_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2492_; 
v_a_2485_ = lean_ctor_get(v___x_2484_, 0);
v_isSharedCheck_2492_ = !lean_is_exclusive(v___x_2484_);
if (v_isSharedCheck_2492_ == 0)
{
v___x_2487_ = v___x_2484_;
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_a_2485_);
lean_dec(v___x_2484_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
lean_object* v___x_2490_; 
if (v_isShared_2488_ == 0)
{
lean_ctor_set_tag(v___x_2487_, 1);
v___x_2490_ = v___x_2487_;
goto v_reusejp_2489_;
}
else
{
lean_object* v_reuseFailAlloc_2491_; 
v_reuseFailAlloc_2491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2491_, 0, v_a_2485_);
v___x_2490_ = v_reuseFailAlloc_2491_;
goto v_reusejp_2489_;
}
v_reusejp_2489_:
{
v___y_2437_ = v_a_2480_;
v___y_2438_ = v_a_2467_;
v___y_2439_ = v___x_2483_;
v_a_2440_ = v___x_2490_;
goto v___jp_2436_;
}
}
}
else
{
lean_object* v_a_2493_; lean_object* v___x_2495_; uint8_t v_isShared_2496_; uint8_t v_isSharedCheck_2500_; 
v_a_2493_ = lean_ctor_get(v___x_2484_, 0);
v_isSharedCheck_2500_ = !lean_is_exclusive(v___x_2484_);
if (v_isSharedCheck_2500_ == 0)
{
v___x_2495_ = v___x_2484_;
v_isShared_2496_ = v_isSharedCheck_2500_;
goto v_resetjp_2494_;
}
else
{
lean_inc(v_a_2493_);
lean_dec(v___x_2484_);
v___x_2495_ = lean_box(0);
v_isShared_2496_ = v_isSharedCheck_2500_;
goto v_resetjp_2494_;
}
v_resetjp_2494_:
{
lean_object* v___x_2498_; 
if (v_isShared_2496_ == 0)
{
lean_ctor_set_tag(v___x_2495_, 0);
v___x_2498_ = v___x_2495_;
goto v_reusejp_2497_;
}
else
{
lean_object* v_reuseFailAlloc_2499_; 
v_reuseFailAlloc_2499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2499_, 0, v_a_2493_);
v___x_2498_ = v_reuseFailAlloc_2499_;
goto v_reusejp_2497_;
}
v_reusejp_2497_:
{
v___y_2437_ = v_a_2480_;
v___y_2438_ = v_a_2467_;
v___y_2439_ = v___x_2483_;
v_a_2440_ = v___x_2498_;
goto v___jp_2436_;
}
}
}
}
else
{
lean_object* v___x_2501_; lean_object* v___x_2502_; 
v___x_2501_ = lean_io_get_num_heartbeats();
v___x_2502_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(v_ldecl_u2081_2391_, v_ldecl_u2082_2392_, v_gctx_2362_, v_a_2363_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_);
if (lean_obj_tag(v___x_2502_) == 0)
{
lean_object* v_a_2503_; lean_object* v___x_2505_; uint8_t v_isShared_2506_; uint8_t v_isSharedCheck_2510_; 
v_a_2503_ = lean_ctor_get(v___x_2502_, 0);
v_isSharedCheck_2510_ = !lean_is_exclusive(v___x_2502_);
if (v_isSharedCheck_2510_ == 0)
{
v___x_2505_ = v___x_2502_;
v_isShared_2506_ = v_isSharedCheck_2510_;
goto v_resetjp_2504_;
}
else
{
lean_inc(v_a_2503_);
lean_dec(v___x_2502_);
v___x_2505_ = lean_box(0);
v_isShared_2506_ = v_isSharedCheck_2510_;
goto v_resetjp_2504_;
}
v_resetjp_2504_:
{
lean_object* v___x_2508_; 
if (v_isShared_2506_ == 0)
{
lean_ctor_set_tag(v___x_2505_, 1);
v___x_2508_ = v___x_2505_;
goto v_reusejp_2507_;
}
else
{
lean_object* v_reuseFailAlloc_2509_; 
v_reuseFailAlloc_2509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2509_, 0, v_a_2503_);
v___x_2508_ = v_reuseFailAlloc_2509_;
goto v_reusejp_2507_;
}
v_reusejp_2507_:
{
v___y_2453_ = v_a_2480_;
v___y_2454_ = v_a_2467_;
v___y_2455_ = v___x_2501_;
v_a_2456_ = v___x_2508_;
goto v___jp_2452_;
}
}
}
else
{
lean_object* v_a_2511_; lean_object* v___x_2513_; uint8_t v_isShared_2514_; uint8_t v_isSharedCheck_2518_; 
v_a_2511_ = lean_ctor_get(v___x_2502_, 0);
v_isSharedCheck_2518_ = !lean_is_exclusive(v___x_2502_);
if (v_isSharedCheck_2518_ == 0)
{
v___x_2513_ = v___x_2502_;
v_isShared_2514_ = v_isSharedCheck_2518_;
goto v_resetjp_2512_;
}
else
{
lean_inc(v_a_2511_);
lean_dec(v___x_2502_);
v___x_2513_ = lean_box(0);
v_isShared_2514_ = v_isSharedCheck_2518_;
goto v_resetjp_2512_;
}
v_resetjp_2512_:
{
lean_object* v___x_2516_; 
if (v_isShared_2514_ == 0)
{
lean_ctor_set_tag(v___x_2513_, 0);
v___x_2516_ = v___x_2513_;
goto v_reusejp_2515_;
}
else
{
lean_object* v_reuseFailAlloc_2517_; 
v_reuseFailAlloc_2517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2517_, 0, v_a_2511_);
v___x_2516_ = v_reuseFailAlloc_2517_;
goto v_reusejp_2515_;
}
v_reusejp_2515_:
{
v___y_2453_ = v_a_2480_;
v___y_2454_ = v_a_2467_;
v___y_2455_ = v___x_2501_;
v_a_2456_ = v___x_2516_;
goto v___jp_2452_;
}
}
}
}
}
else
{
lean_object* v_a_2519_; lean_object* v___x_2521_; uint8_t v_isShared_2522_; uint8_t v_isSharedCheck_2526_; 
lean_dec(v_a_2467_);
lean_dec_ref(v_gctx_2362_);
lean_dec(v_i_2361_);
v_a_2519_ = lean_ctor_get(v___x_2479_, 0);
v_isSharedCheck_2526_ = !lean_is_exclusive(v___x_2479_);
if (v_isSharedCheck_2526_ == 0)
{
v___x_2521_ = v___x_2479_;
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
else
{
lean_inc(v_a_2519_);
lean_dec(v___x_2479_);
v___x_2521_ = lean_box(0);
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
v_resetjp_2520_:
{
lean_object* v___x_2524_; 
if (v_isShared_2522_ == 0)
{
v___x_2524_ = v___x_2521_;
goto v_reusejp_2523_;
}
else
{
lean_object* v_reuseFailAlloc_2525_; 
v_reuseFailAlloc_2525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2525_, 0, v_a_2519_);
v___x_2524_ = v_reuseFailAlloc_2525_;
goto v_reusejp_2523_;
}
v_reusejp_2523_:
{
return v___x_2524_;
}
}
}
}
else
{
lean_object* v_a_2527_; lean_object* v___x_2529_; uint8_t v_isShared_2530_; uint8_t v_isSharedCheck_2534_; 
lean_dec_ref(v_gctx_2362_);
lean_dec(v_i_2361_);
v_a_2527_ = lean_ctor_get(v___x_2466_, 0);
v_isSharedCheck_2534_ = !lean_is_exclusive(v___x_2466_);
if (v_isSharedCheck_2534_ == 0)
{
v___x_2529_ = v___x_2466_;
v_isShared_2530_ = v_isSharedCheck_2534_;
goto v_resetjp_2528_;
}
else
{
lean_inc(v_a_2527_);
lean_dec(v___x_2466_);
v___x_2529_ = lean_box(0);
v_isShared_2530_ = v_isSharedCheck_2534_;
goto v_resetjp_2528_;
}
v_resetjp_2528_:
{
lean_object* v___x_2532_; 
if (v_isShared_2530_ == 0)
{
v___x_2532_ = v___x_2529_;
goto v_reusejp_2531_;
}
else
{
lean_object* v_reuseFailAlloc_2533_; 
v_reuseFailAlloc_2533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2533_, 0, v_a_2527_);
v___x_2532_ = v_reuseFailAlloc_2533_;
goto v_reusejp_2531_;
}
v_reusejp_2531_:
{
return v___x_2532_;
}
}
}
}
}
v___jp_2393_:
{
if (lean_obj_tag(v___y_2394_) == 0)
{
lean_object* v_a_2395_; lean_object* v___x_2397_; uint8_t v_isShared_2398_; uint8_t v_isSharedCheck_2422_; 
v_a_2395_ = lean_ctor_get(v___y_2394_, 0);
v_isSharedCheck_2422_ = !lean_is_exclusive(v___y_2394_);
if (v_isSharedCheck_2422_ == 0)
{
v___x_2397_ = v___y_2394_;
v_isShared_2398_ = v_isSharedCheck_2422_;
goto v_resetjp_2396_;
}
else
{
lean_inc(v_a_2395_);
lean_dec(v___y_2394_);
v___x_2397_ = lean_box(0);
v_isShared_2398_ = v_isSharedCheck_2422_;
goto v_resetjp_2396_;
}
v_resetjp_2396_:
{
uint8_t v___x_2399_; 
v___x_2399_ = lean_unbox(v_a_2395_);
lean_dec(v_a_2395_);
if (v___x_2399_ == 0)
{
lean_object* v___x_2400_; lean_object* v___x_2402_; 
lean_dec_ref(v_gctx_2362_);
lean_dec(v_i_2361_);
v___x_2400_ = lean_box(0);
if (v_isShared_2398_ == 0)
{
lean_ctor_set(v___x_2397_, 0, v___x_2400_);
v___x_2402_ = v___x_2397_;
goto v_reusejp_2401_;
}
else
{
lean_object* v_reuseFailAlloc_2403_; 
v_reuseFailAlloc_2403_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2403_, 0, v___x_2400_);
v___x_2402_ = v_reuseFailAlloc_2403_;
goto v_reusejp_2401_;
}
v_reusejp_2401_:
{
return v___x_2402_;
}
}
else
{
lean_object* v_lctx_u2081_2404_; lean_object* v_localInstances_u2081_2405_; lean_object* v_lctx_u2082_2406_; lean_object* v_localInstances_u2082_2407_; lean_object* v_equalFVarIds_2408_; lean_object* v___x_2410_; uint8_t v_isShared_2411_; uint8_t v_isSharedCheck_2421_; 
lean_del_object(v___x_2397_);
v_lctx_u2081_2404_ = lean_ctor_get(v_gctx_2362_, 0);
v_localInstances_u2081_2405_ = lean_ctor_get(v_gctx_2362_, 1);
v_lctx_u2082_2406_ = lean_ctor_get(v_gctx_2362_, 2);
v_localInstances_u2082_2407_ = lean_ctor_get(v_gctx_2362_, 3);
v_equalFVarIds_2408_ = lean_ctor_get(v_gctx_2362_, 4);
v_isSharedCheck_2421_ = !lean_is_exclusive(v_gctx_2362_);
if (v_isSharedCheck_2421_ == 0)
{
v___x_2410_ = v_gctx_2362_;
v_isShared_2411_ = v_isSharedCheck_2421_;
goto v_resetjp_2409_;
}
else
{
lean_inc(v_equalFVarIds_2408_);
lean_inc(v_localInstances_u2082_2407_);
lean_inc(v_lctx_u2082_2406_);
lean_inc(v_localInstances_u2081_2405_);
lean_inc(v_lctx_u2081_2404_);
lean_dec(v_gctx_2362_);
v___x_2410_ = lean_box(0);
v_isShared_2411_ = v_isSharedCheck_2421_;
goto v_resetjp_2409_;
}
v_resetjp_2409_:
{
lean_object* v___x_2412_; lean_object* v___x_2413_; lean_object* v___x_2414_; lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2418_; 
v___x_2412_ = l_Lean_LocalDecl_fvarId(v_ldecl_u2081_2391_);
v___x_2413_ = l_Lean_LocalDecl_fvarId(v_ldecl_u2082_2392_);
v___x_2414_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0___redArg(v_equalFVarIds_2408_, v___x_2412_, v___x_2413_);
v___x_2415_ = lean_unsigned_to_nat(1u);
v___x_2416_ = lean_nat_add(v_i_2361_, v___x_2415_);
lean_dec(v_i_2361_);
if (v_isShared_2411_ == 0)
{
lean_ctor_set(v___x_2410_, 4, v___x_2414_);
v___x_2418_ = v___x_2410_;
goto v_reusejp_2417_;
}
else
{
lean_object* v_reuseFailAlloc_2420_; 
v_reuseFailAlloc_2420_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2420_, 0, v_lctx_u2081_2404_);
lean_ctor_set(v_reuseFailAlloc_2420_, 1, v_localInstances_u2081_2405_);
lean_ctor_set(v_reuseFailAlloc_2420_, 2, v_lctx_u2082_2406_);
lean_ctor_set(v_reuseFailAlloc_2420_, 3, v_localInstances_u2082_2407_);
lean_ctor_set(v_reuseFailAlloc_2420_, 4, v___x_2414_);
v___x_2418_ = v_reuseFailAlloc_2420_;
goto v_reusejp_2417_;
}
v_reusejp_2417_:
{
v_i_2361_ = v___x_2416_;
v_gctx_2362_ = v___x_2418_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_2423_; lean_object* v___x_2425_; uint8_t v_isShared_2426_; uint8_t v_isSharedCheck_2430_; 
lean_dec_ref(v_gctx_2362_);
lean_dec(v_i_2361_);
v_a_2423_ = lean_ctor_get(v___y_2394_, 0);
v_isSharedCheck_2430_ = !lean_is_exclusive(v___y_2394_);
if (v_isSharedCheck_2430_ == 0)
{
v___x_2425_ = v___y_2394_;
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
else
{
lean_inc(v_a_2423_);
lean_dec(v___y_2394_);
v___x_2425_ = lean_box(0);
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
v_resetjp_2424_:
{
lean_object* v___x_2428_; 
if (v_isShared_2426_ == 0)
{
v___x_2428_ = v___x_2425_;
goto v_reusejp_2427_;
}
else
{
lean_object* v_reuseFailAlloc_2429_; 
v_reuseFailAlloc_2429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2429_, 0, v_a_2423_);
v___x_2428_ = v_reuseFailAlloc_2429_;
goto v_reusejp_2427_;
}
v_reusejp_2427_:
{
return v___x_2428_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(lean_object* v_lctx_u2081_2538_, lean_object* v_lctx_u2082_2539_, lean_object* v_localInstances_u2081_2540_, lean_object* v_localInstances_u2082_2541_, lean_object* v_a_2542_, lean_object* v_a_2543_, lean_object* v_a_2544_, lean_object* v_a_2545_, lean_object* v_a_2546_, lean_object* v_a_2547_){
_start:
{
lean_object* v___x_2552_; lean_object* v_toApplicative_2553_; lean_object* v_toFunctor_2554_; lean_object* v_toSeq_2555_; lean_object* v_toSeqLeft_2556_; lean_object* v_toSeqRight_2557_; lean_object* v___f_2558_; lean_object* v___f_2559_; lean_object* v___f_2560_; lean_object* v___f_2561_; lean_object* v___x_2562_; lean_object* v___f_2563_; lean_object* v___f_2564_; lean_object* v___f_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v_toApplicative_2569_; lean_object* v___x_2571_; uint8_t v_isShared_2572_; uint8_t v_isSharedCheck_2634_; 
v___x_2552_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_2553_ = lean_ctor_get(v___x_2552_, 0);
v_toFunctor_2554_ = lean_ctor_get(v_toApplicative_2553_, 0);
v_toSeq_2555_ = lean_ctor_get(v_toApplicative_2553_, 2);
v_toSeqLeft_2556_ = lean_ctor_get(v_toApplicative_2553_, 3);
v_toSeqRight_2557_ = lean_ctor_get(v_toApplicative_2553_, 4);
v___f_2558_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_2559_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_2554_, 2);
v___f_2560_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2560_, 0, v_toFunctor_2554_);
v___f_2561_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2561_, 0, v_toFunctor_2554_);
v___x_2562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2562_, 0, v___f_2560_);
lean_ctor_set(v___x_2562_, 1, v___f_2561_);
lean_inc(v_toSeqRight_2557_);
v___f_2563_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2563_, 0, v_toSeqRight_2557_);
lean_inc(v_toSeqLeft_2556_);
v___f_2564_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2564_, 0, v_toSeqLeft_2556_);
lean_inc(v_toSeq_2555_);
v___f_2565_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2565_, 0, v_toSeq_2555_);
v___x_2566_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2566_, 0, v___x_2562_);
lean_ctor_set(v___x_2566_, 1, v___f_2558_);
lean_ctor_set(v___x_2566_, 2, v___f_2565_);
lean_ctor_set(v___x_2566_, 3, v___f_2564_);
lean_ctor_set(v___x_2566_, 4, v___f_2563_);
v___x_2567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2567_, 0, v___x_2566_);
lean_ctor_set(v___x_2567_, 1, v___f_2559_);
v___x_2568_ = l_StateRefT_x27_instMonad___redArg(v___x_2567_);
v_toApplicative_2569_ = lean_ctor_get(v___x_2568_, 0);
v_isSharedCheck_2634_ = !lean_is_exclusive(v___x_2568_);
if (v_isSharedCheck_2634_ == 0)
{
lean_object* v_unused_2635_; 
v_unused_2635_ = lean_ctor_get(v___x_2568_, 1);
lean_dec(v_unused_2635_);
v___x_2571_ = v___x_2568_;
v_isShared_2572_ = v_isSharedCheck_2634_;
goto v_resetjp_2570_;
}
else
{
lean_inc(v_toApplicative_2569_);
lean_dec(v___x_2568_);
v___x_2571_ = lean_box(0);
v_isShared_2572_ = v_isSharedCheck_2634_;
goto v_resetjp_2570_;
}
v___jp_2549_:
{
lean_object* v___x_2550_; lean_object* v___x_2551_; 
v___x_2550_ = lean_box(0);
v___x_2551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2551_, 0, v___x_2550_);
return v___x_2551_;
}
v_resetjp_2570_:
{
lean_object* v_toFunctor_2573_; lean_object* v_toSeq_2574_; lean_object* v_toSeqLeft_2575_; lean_object* v_toSeqRight_2576_; lean_object* v___x_2578_; uint8_t v_isShared_2579_; uint8_t v_isSharedCheck_2632_; 
v_toFunctor_2573_ = lean_ctor_get(v_toApplicative_2569_, 0);
v_toSeq_2574_ = lean_ctor_get(v_toApplicative_2569_, 2);
v_toSeqLeft_2575_ = lean_ctor_get(v_toApplicative_2569_, 3);
v_toSeqRight_2576_ = lean_ctor_get(v_toApplicative_2569_, 4);
v_isSharedCheck_2632_ = !lean_is_exclusive(v_toApplicative_2569_);
if (v_isSharedCheck_2632_ == 0)
{
lean_object* v_unused_2633_; 
v_unused_2633_ = lean_ctor_get(v_toApplicative_2569_, 1);
lean_dec(v_unused_2633_);
v___x_2578_ = v_toApplicative_2569_;
v_isShared_2579_ = v_isSharedCheck_2632_;
goto v_resetjp_2577_;
}
else
{
lean_inc(v_toSeqRight_2576_);
lean_inc(v_toSeqLeft_2575_);
lean_inc(v_toSeq_2574_);
lean_inc(v_toFunctor_2573_);
lean_dec(v_toApplicative_2569_);
v___x_2578_ = lean_box(0);
v_isShared_2579_ = v_isSharedCheck_2632_;
goto v_resetjp_2577_;
}
v_resetjp_2577_:
{
lean_object* v___f_2580_; lean_object* v___f_2581_; lean_object* v___f_2582_; lean_object* v___f_2583_; lean_object* v___f_2584_; lean_object* v___x_2585_; lean_object* v___f_2586_; lean_object* v___f_2587_; lean_object* v___f_2588_; lean_object* v___x_2590_; 
v___f_2580_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__0));
v___f_2581_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_2582_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_2573_);
v___f_2583_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2583_, 0, v_toFunctor_2573_);
v___f_2584_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2584_, 0, v_toFunctor_2573_);
v___x_2585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2585_, 0, v___f_2583_);
lean_ctor_set(v___x_2585_, 1, v___f_2584_);
v___f_2586_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2586_, 0, v_toSeqRight_2576_);
v___f_2587_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2587_, 0, v_toSeqLeft_2575_);
v___f_2588_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2588_, 0, v_toSeq_2574_);
if (v_isShared_2579_ == 0)
{
lean_ctor_set(v___x_2578_, 4, v___f_2586_);
lean_ctor_set(v___x_2578_, 3, v___f_2587_);
lean_ctor_set(v___x_2578_, 2, v___f_2588_);
lean_ctor_set(v___x_2578_, 1, v___f_2581_);
lean_ctor_set(v___x_2578_, 0, v___x_2585_);
v___x_2590_ = v___x_2578_;
goto v_reusejp_2589_;
}
else
{
lean_object* v_reuseFailAlloc_2631_; 
v_reuseFailAlloc_2631_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2631_, 0, v___x_2585_);
lean_ctor_set(v_reuseFailAlloc_2631_, 1, v___f_2581_);
lean_ctor_set(v_reuseFailAlloc_2631_, 2, v___f_2588_);
lean_ctor_set(v_reuseFailAlloc_2631_, 3, v___f_2587_);
lean_ctor_set(v_reuseFailAlloc_2631_, 4, v___f_2586_);
v___x_2590_ = v_reuseFailAlloc_2631_;
goto v_reusejp_2589_;
}
v_reusejp_2589_:
{
lean_object* v___x_2592_; 
if (v_isShared_2572_ == 0)
{
lean_ctor_set(v___x_2571_, 1, v___f_2582_);
lean_ctor_set(v___x_2571_, 0, v___x_2590_);
v___x_2592_ = v___x_2571_;
goto v_reusejp_2591_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v___x_2590_);
lean_ctor_set(v_reuseFailAlloc_2630_, 1, v___f_2582_);
v___x_2592_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2591_;
}
v_reusejp_2591_:
{
lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; uint8_t v___x_2605_; 
v___x_2593_ = l_StateRefT_x27_instMonad___redArg(v___x_2592_);
v___x_2594_ = l_ReaderT_instMonad___redArg(v___x_2593_);
lean_inc_ref_n(v_lctx_u2081_2538_, 2);
v___x_2595_ = lean_local_ctx_num_indices(v_lctx_u2081_2538_);
v___x_2596_ = lean_mk_empty_array_with_capacity(v___x_2595_);
lean_dec(v___x_2595_);
v___x_2597_ = lean_unsigned_to_nat(0u);
v___x_2598_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_lctxDecls___redArg___closed__10));
v___x_2599_ = l_Lean_LocalContext_foldlM___redArg(v___x_2598_, v_lctx_u2081_2538_, v___f_2580_, v___x_2596_, v___x_2597_);
lean_inc_ref_n(v_lctx_u2082_2539_, 2);
v___x_2600_ = lean_local_ctx_num_indices(v_lctx_u2082_2539_);
v___x_2601_ = lean_mk_empty_array_with_capacity(v___x_2600_);
lean_dec(v___x_2600_);
v___x_2602_ = l_Lean_LocalContext_foldlM___redArg(v___x_2598_, v_lctx_u2082_2539_, v___f_2580_, v___x_2601_, v___x_2597_);
v___x_2603_ = lean_array_get_size(v___x_2599_);
v___x_2604_ = lean_array_get_size(v___x_2602_);
v___x_2605_ = lean_nat_dec_eq(v___x_2603_, v___x_2604_);
if (v___x_2605_ == 0)
{
lean_object* v___x_2606_; lean_object* v_options_2607_; uint8_t v_hasTrace_2608_; 
lean_dec(v___x_2602_);
lean_dec(v___x_2599_);
lean_dec_ref(v_localInstances_u2082_2541_);
lean_dec_ref(v_localInstances_u2081_2540_);
lean_dec_ref(v_lctx_u2082_2539_);
lean_dec_ref(v_lctx_u2081_2538_);
v___x_2606_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4);
v_options_2607_ = lean_ctor_get(v_a_2546_, 2);
v_hasTrace_2608_ = lean_ctor_get_uint8(v_options_2607_, sizeof(void*)*1);
if (v_hasTrace_2608_ == 0)
{
lean_dec_ref(v___x_2594_);
goto v___jp_2549_;
}
else
{
lean_object* v_inheritedTraceOptions_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; uint8_t v___x_2612_; 
v_inheritedTraceOptions_2609_ = lean_ctor_get(v_a_2546_, 13);
v___x_2610_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
v___x_2611_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_2612_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2609_, v_options_2607_, v___x_2611_);
if (v___x_2612_ == 0)
{
lean_dec_ref(v___x_2594_);
goto v___jp_2549_;
}
else
{
lean_object* v___x_2613_; lean_object* v_toMonadRef_2614_; lean_object* v___f_2615_; lean_object* v___x_2616_; lean_object* v___x_249845__overap_2617_; lean_object* v___x_2618_; 
v___x_2613_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11);
v_toMonadRef_2614_ = lean_ctor_get(v___x_2613_, 0);
v___f_2615_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13);
v___x_2616_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__15);
lean_inc_ref(v_toMonadRef_2614_);
v___x_249845__overap_2617_ = l_Lean_addTrace___redArg(v___x_2594_, v___x_2606_, v_toMonadRef_2614_, v___f_2615_, v___x_2610_, v___x_2616_);
lean_inc(v_a_2547_);
lean_inc_ref(v_a_2546_);
lean_inc(v_a_2545_);
lean_inc_ref(v_a_2544_);
lean_inc(v_a_2543_);
lean_inc_ref(v_a_2542_);
v___x_2618_ = lean_apply_7(v___x_249845__overap_2617_, v_a_2542_, v_a_2543_, v_a_2544_, v_a_2545_, v_a_2546_, v_a_2547_, lean_box(0));
if (lean_obj_tag(v___x_2618_) == 0)
{
lean_dec_ref_known(v___x_2618_, 1);
goto v___jp_2549_;
}
else
{
lean_object* v_a_2619_; lean_object* v___x_2621_; uint8_t v_isShared_2622_; uint8_t v_isSharedCheck_2626_; 
v_a_2619_ = lean_ctor_get(v___x_2618_, 0);
v_isSharedCheck_2626_ = !lean_is_exclusive(v___x_2618_);
if (v_isSharedCheck_2626_ == 0)
{
v___x_2621_ = v___x_2618_;
v_isShared_2622_ = v_isSharedCheck_2626_;
goto v_resetjp_2620_;
}
else
{
lean_inc(v_a_2619_);
lean_dec(v___x_2618_);
v___x_2621_ = lean_box(0);
v_isShared_2622_ = v_isSharedCheck_2626_;
goto v_resetjp_2620_;
}
v_resetjp_2620_:
{
lean_object* v___x_2624_; 
if (v_isShared_2622_ == 0)
{
v___x_2624_ = v___x_2621_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2625_; 
v_reuseFailAlloc_2625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2625_, 0, v_a_2619_);
v___x_2624_ = v_reuseFailAlloc_2625_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
return v___x_2624_;
}
}
}
}
}
}
else
{
lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; 
lean_dec_ref(v___x_2594_);
v___x_2627_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1, &lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1);
v___x_2628_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2628_, 0, v_lctx_u2081_2538_);
lean_ctor_set(v___x_2628_, 1, v_localInstances_u2081_2540_);
lean_ctor_set(v___x_2628_, 2, v_lctx_u2082_2539_);
lean_ctor_set(v___x_2628_, 3, v_localInstances_u2082_2541_);
lean_ctor_set(v___x_2628_, 4, v___x_2627_);
v___x_2629_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg(v___x_2599_, v___x_2602_, v___x_2597_, v___x_2628_, v_a_2542_, v_a_2543_, v_a_2544_, v_a_2545_, v_a_2546_, v_a_2547_);
lean_dec(v___x_2602_);
lean_dec(v___x_2599_);
return v___x_2629_;
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15(void){
_start:
{
lean_object* v___x_2637_; lean_object* v___x_2638_; 
v___x_2637_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__14));
v___x_2638_ = l_Lean_stringToMessageData(v___x_2637_);
return v___x_2638_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17(void){
_start:
{
lean_object* v___x_2640_; lean_object* v___x_2641_; 
v___x_2640_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__16));
v___x_2641_ = l_Lean_stringToMessageData(v___x_2640_);
return v___x_2641_;
}
}
static lean_object* _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19(void){
_start:
{
lean_object* v___x_2643_; lean_object* v___x_2644_; 
v___x_2643_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__18));
v___x_2644_ = l_Lean_stringToMessageData(v___x_2643_);
return v___x_2644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(lean_object* v_mvarId_u2081_2645_, lean_object* v_mvarId_u2082_2646_, lean_object* v_a_2647_, lean_object* v_a_2648_, lean_object* v_a_2649_, lean_object* v_a_2650_, lean_object* v_a_2651_, lean_object* v_a_2652_){
_start:
{
lean_object* v___x_2654_; lean_object* v_toApplicative_2655_; lean_object* v_toFunctor_2656_; lean_object* v_toSeq_2657_; lean_object* v_toSeqLeft_2658_; lean_object* v_toSeqRight_2659_; lean_object* v___f_2660_; lean_object* v___f_2661_; lean_object* v___f_2662_; lean_object* v___f_2663_; lean_object* v___x_2664_; lean_object* v___f_2665_; lean_object* v___f_2666_; lean_object* v___f_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v_toApplicative_2671_; lean_object* v___x_2673_; uint8_t v_isShared_2674_; uint8_t v_isSharedCheck_3895_; 
v___x_2654_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_2655_ = lean_ctor_get(v___x_2654_, 0);
v_toFunctor_2656_ = lean_ctor_get(v_toApplicative_2655_, 0);
v_toSeq_2657_ = lean_ctor_get(v_toApplicative_2655_, 2);
v_toSeqLeft_2658_ = lean_ctor_get(v_toApplicative_2655_, 3);
v_toSeqRight_2659_ = lean_ctor_get(v_toApplicative_2655_, 4);
v___f_2660_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_2661_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_2656_, 2);
v___f_2662_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2662_, 0, v_toFunctor_2656_);
v___f_2663_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2663_, 0, v_toFunctor_2656_);
v___x_2664_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2664_, 0, v___f_2662_);
lean_ctor_set(v___x_2664_, 1, v___f_2663_);
lean_inc(v_toSeqRight_2659_);
v___f_2665_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2665_, 0, v_toSeqRight_2659_);
lean_inc(v_toSeqLeft_2658_);
v___f_2666_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2666_, 0, v_toSeqLeft_2658_);
lean_inc(v_toSeq_2657_);
v___f_2667_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2667_, 0, v_toSeq_2657_);
v___x_2668_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2668_, 0, v___x_2664_);
lean_ctor_set(v___x_2668_, 1, v___f_2660_);
lean_ctor_set(v___x_2668_, 2, v___f_2667_);
lean_ctor_set(v___x_2668_, 3, v___f_2666_);
lean_ctor_set(v___x_2668_, 4, v___f_2665_);
v___x_2669_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2669_, 0, v___x_2668_);
lean_ctor_set(v___x_2669_, 1, v___f_2661_);
v___x_2670_ = l_StateRefT_x27_instMonad___redArg(v___x_2669_);
v_toApplicative_2671_ = lean_ctor_get(v___x_2670_, 0);
v_isSharedCheck_3895_ = !lean_is_exclusive(v___x_2670_);
if (v_isSharedCheck_3895_ == 0)
{
lean_object* v_unused_3896_; 
v_unused_3896_ = lean_ctor_get(v___x_2670_, 1);
lean_dec(v_unused_3896_);
v___x_2673_ = v___x_2670_;
v_isShared_2674_ = v_isSharedCheck_3895_;
goto v_resetjp_2672_;
}
else
{
lean_inc(v_toApplicative_2671_);
lean_dec(v___x_2670_);
v___x_2673_ = lean_box(0);
v_isShared_2674_ = v_isSharedCheck_3895_;
goto v_resetjp_2672_;
}
v_resetjp_2672_:
{
lean_object* v_toFunctor_2675_; lean_object* v_toSeq_2676_; lean_object* v_toSeqLeft_2677_; lean_object* v_toSeqRight_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_3893_; 
v_toFunctor_2675_ = lean_ctor_get(v_toApplicative_2671_, 0);
v_toSeq_2676_ = lean_ctor_get(v_toApplicative_2671_, 2);
v_toSeqLeft_2677_ = lean_ctor_get(v_toApplicative_2671_, 3);
v_toSeqRight_2678_ = lean_ctor_get(v_toApplicative_2671_, 4);
v_isSharedCheck_3893_ = !lean_is_exclusive(v_toApplicative_2671_);
if (v_isSharedCheck_3893_ == 0)
{
lean_object* v_unused_3894_; 
v_unused_3894_ = lean_ctor_get(v_toApplicative_2671_, 1);
lean_dec(v_unused_3894_);
v___x_2680_ = v_toApplicative_2671_;
v_isShared_2681_ = v_isSharedCheck_3893_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_toSeqRight_2678_);
lean_inc(v_toSeqLeft_2677_);
lean_inc(v_toSeq_2676_);
lean_inc(v_toFunctor_2675_);
lean_dec(v_toApplicative_2671_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_3893_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___f_2682_; lean_object* v___f_2683_; lean_object* v___f_2684_; lean_object* v___f_2685_; lean_object* v___x_2686_; lean_object* v___f_2687_; lean_object* v___f_2688_; lean_object* v___f_2689_; lean_object* v___x_2691_; 
v___f_2682_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_2683_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_2675_);
v___f_2684_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2684_, 0, v_toFunctor_2675_);
v___f_2685_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2685_, 0, v_toFunctor_2675_);
v___x_2686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2686_, 0, v___f_2684_);
lean_ctor_set(v___x_2686_, 1, v___f_2685_);
v___f_2687_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2687_, 0, v_toSeqRight_2678_);
v___f_2688_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2688_, 0, v_toSeqLeft_2677_);
v___f_2689_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2689_, 0, v_toSeq_2676_);
if (v_isShared_2681_ == 0)
{
lean_ctor_set(v___x_2680_, 4, v___f_2687_);
lean_ctor_set(v___x_2680_, 3, v___f_2688_);
lean_ctor_set(v___x_2680_, 2, v___f_2689_);
lean_ctor_set(v___x_2680_, 1, v___f_2682_);
lean_ctor_set(v___x_2680_, 0, v___x_2686_);
v___x_2691_ = v___x_2680_;
goto v_reusejp_2690_;
}
else
{
lean_object* v_reuseFailAlloc_3892_; 
v_reuseFailAlloc_3892_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3892_, 0, v___x_2686_);
lean_ctor_set(v_reuseFailAlloc_3892_, 1, v___f_2682_);
lean_ctor_set(v_reuseFailAlloc_3892_, 2, v___f_2689_);
lean_ctor_set(v_reuseFailAlloc_3892_, 3, v___f_2688_);
lean_ctor_set(v_reuseFailAlloc_3892_, 4, v___f_2687_);
v___x_2691_ = v_reuseFailAlloc_3892_;
goto v_reusejp_2690_;
}
v_reusejp_2690_:
{
lean_object* v___x_2693_; 
if (v_isShared_2674_ == 0)
{
lean_ctor_set(v___x_2673_, 1, v___f_2683_);
lean_ctor_set(v___x_2673_, 0, v___x_2691_);
v___x_2693_ = v___x_2673_;
goto v_reusejp_2692_;
}
else
{
lean_object* v_reuseFailAlloc_3891_; 
v_reuseFailAlloc_3891_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3891_, 0, v___x_2691_);
lean_ctor_set(v_reuseFailAlloc_3891_, 1, v___f_2683_);
v___x_2693_ = v_reuseFailAlloc_3891_;
goto v_reusejp_2692_;
}
v_reusejp_2692_:
{
lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v_toMonadRef_2698_; lean_object* v___f_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v_options_2704_; lean_object* v_fileName_2705_; lean_object* v_fileMap_2706_; lean_object* v_currRecDepth_2707_; lean_object* v_maxRecDepth_2708_; lean_object* v_ref_2709_; lean_object* v_currNamespace_2710_; lean_object* v_openDecls_2711_; lean_object* v_initHeartbeats_2712_; lean_object* v_maxHeartbeats_2713_; lean_object* v_quotContext_2714_; lean_object* v_currMacroScope_2715_; uint8_t v_diag_2716_; lean_object* v_cancelTk_x3f_2717_; uint8_t v_suppressElabErrors_2718_; lean_object* v_inheritedTraceOptions_2719_; uint8_t v_hasTrace_2720_; lean_object* v___f_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; uint8_t v_____do__lift_2725_; lean_object* v___y_2726_; lean_object* v_cls_2745_; lean_object* v___y_2747_; uint8_t v___y_2748_; lean_object* v___y_2749_; lean_object* v___y_2750_; lean_object* v___y_2751_; lean_object* v___y_2752_; lean_object* v___y_2753_; lean_object* v___y_2754_; lean_object* v___y_2755_; lean_object* v___y_2778_; lean_object* v___y_2779_; lean_object* v___y_2780_; lean_object* v___y_2781_; lean_object* v___y_2782_; lean_object* v___y_2783_; lean_object* v___y_2784_; lean_object* v___y_2785_; uint8_t v___y_2786_; lean_object* v___y_2787_; lean_object* v___y_2788_; lean_object* v___y_2789_; lean_object* v___y_2790_; uint8_t v___y_2791_; lean_object* v_a_2792_; lean_object* v___y_2806_; lean_object* v___y_2807_; lean_object* v___y_2808_; lean_object* v___y_2809_; lean_object* v___y_2810_; lean_object* v___y_2811_; lean_object* v___y_2812_; lean_object* v___y_2813_; uint8_t v___y_2814_; lean_object* v___y_2815_; lean_object* v___y_2816_; lean_object* v___y_2817_; uint8_t v___y_2818_; lean_object* v___y_2819_; uint8_t v_a_2820_; lean_object* v___y_2824_; lean_object* v___y_2825_; lean_object* v___y_2826_; lean_object* v___y_2827_; lean_object* v___y_2828_; lean_object* v___y_2829_; lean_object* v___y_2830_; lean_object* v___y_2831_; uint8_t v___y_2832_; lean_object* v___y_2833_; lean_object* v___y_2834_; lean_object* v___y_2835_; lean_object* v___y_2836_; uint8_t v___y_2837_; lean_object* v_a_2838_; lean_object* v___y_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; lean_object* v___y_2852_; lean_object* v___y_2853_; lean_object* v___y_2854_; lean_object* v___y_2855_; lean_object* v___y_2856_; uint8_t v___y_2857_; lean_object* v___y_2858_; lean_object* v___y_2859_; lean_object* v___y_2860_; uint8_t v___y_2861_; lean_object* v___y_2862_; uint8_t v_a_2863_; uint8_t v___y_2867_; lean_object* v___y_2868_; uint8_t v___y_2869_; lean_object* v___y_2870_; lean_object* v___y_2871_; lean_object* v___y_2872_; lean_object* v___y_2873_; uint8_t v___y_2874_; lean_object* v___y_2875_; lean_object* v___y_2876_; lean_object* v___y_2877_; lean_object* v___y_2878_; lean_object* v___y_2879_; lean_object* v___y_2880_; lean_object* v___y_2881_; lean_object* v___y_2882_; lean_object* v___y_2883_; lean_object* v___y_2884_; lean_object* v___y_2885_; lean_object* v___y_2886_; lean_object* v___y_2887_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v___y_2890_; lean_object* v___y_2891_; uint8_t v___y_2892_; lean_object* v___y_2893_; lean_object* v___y_2894_; lean_object* v_____do__lift_3029_; lean_object* v___y_3030_; lean_object* v___y_3031_; lean_object* v___y_3032_; lean_object* v___y_3033_; lean_object* v___y_3034_; lean_object* v___y_3035_; 
v___x_2694_ = l_StateRefT_x27_instMonad___redArg(v___x_2693_);
v___x_2695_ = l_ReaderT_instMonad___redArg(v___x_2694_);
v___x_2696_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__4);
v___x_2697_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__11);
v_toMonadRef_2698_ = lean_ctor_get(v___x_2697_, 0);
v___f_2699_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__13);
v___x_2700_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__24);
v___x_2701_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__11);
lean_inc_ref(v___x_2695_);
v___x_2702_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_2699_, v___x_2695_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_2703_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2703_, 0, v___x_2701_);
lean_ctor_set(v___x_2703_, 1, v_toMonadRef_2698_);
lean_ctor_set(v___x_2703_, 2, v___x_2702_);
v_options_2704_ = lean_ctor_get(v_a_2651_, 2);
v_fileName_2705_ = lean_ctor_get(v_a_2651_, 0);
v_fileMap_2706_ = lean_ctor_get(v_a_2651_, 1);
v_currRecDepth_2707_ = lean_ctor_get(v_a_2651_, 3);
v_maxRecDepth_2708_ = lean_ctor_get(v_a_2651_, 4);
v_ref_2709_ = lean_ctor_get(v_a_2651_, 5);
v_currNamespace_2710_ = lean_ctor_get(v_a_2651_, 6);
v_openDecls_2711_ = lean_ctor_get(v_a_2651_, 7);
v_initHeartbeats_2712_ = lean_ctor_get(v_a_2651_, 8);
v_maxHeartbeats_2713_ = lean_ctor_get(v_a_2651_, 9);
v_quotContext_2714_ = lean_ctor_get(v_a_2651_, 10);
v_currMacroScope_2715_ = lean_ctor_get(v_a_2651_, 11);
v_diag_2716_ = lean_ctor_get_uint8(v_a_2651_, sizeof(void*)*14);
v_cancelTk_x3f_2717_ = lean_ctor_get(v_a_2651_, 12);
v_suppressElabErrors_2718_ = lean_ctor_get_uint8(v_a_2651_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2719_ = lean_ctor_get(v_a_2651_, 13);
v_hasTrace_2720_ = lean_ctor_get_uint8(v_options_2704_, sizeof(void*)*1);
v___f_2721_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__26));
v___x_2722_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__0));
v___x_2723_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__1));
v_cls_2745_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn___closed__3_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_));
if (v_hasTrace_2720_ == 0)
{
lean_object* v___x_3222_; 
v___x_3222_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_2645_, v_mvarId_u2082_2646_, v_a_2647_);
if (lean_obj_tag(v___x_3222_) == 0)
{
lean_object* v_a_3223_; 
v_a_3223_ = lean_ctor_get(v___x_3222_, 0);
lean_inc(v_a_3223_);
lean_dec_ref_known(v___x_3222_, 1);
v_____do__lift_3029_ = v_a_3223_;
v___y_3030_ = v_a_2647_;
v___y_3031_ = v_a_2648_;
v___y_3032_ = v_a_2649_;
v___y_3033_ = v_a_2650_;
v___y_3034_ = v_a_2651_;
v___y_3035_ = v_a_2652_;
goto v___jp_3028_;
}
else
{
lean_object* v_a_3224_; lean_object* v___x_3226_; uint8_t v_isShared_3227_; uint8_t v_isSharedCheck_3231_; 
lean_dec_ref_known(v___x_2703_, 3);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3224_ = lean_ctor_get(v___x_3222_, 0);
v_isSharedCheck_3231_ = !lean_is_exclusive(v___x_3222_);
if (v_isSharedCheck_3231_ == 0)
{
v___x_3226_ = v___x_3222_;
v_isShared_3227_ = v_isSharedCheck_3231_;
goto v_resetjp_3225_;
}
else
{
lean_inc(v_a_3224_);
lean_dec(v___x_3222_);
v___x_3226_ = lean_box(0);
v_isShared_3227_ = v_isSharedCheck_3231_;
goto v_resetjp_3225_;
}
v_resetjp_3225_:
{
lean_object* v___x_3229_; 
if (v_isShared_3227_ == 0)
{
v___x_3229_ = v___x_3226_;
goto v_reusejp_3228_;
}
else
{
lean_object* v_reuseFailAlloc_3230_; 
v_reuseFailAlloc_3230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3230_, 0, v_a_3224_);
v___x_3229_ = v_reuseFailAlloc_3230_;
goto v_reusejp_3228_;
}
v_reusejp_3228_:
{
return v___x_3229_;
}
}
}
}
else
{
lean_object* v___x_3232_; lean_object* v___x_3233_; uint8_t v___x_3234_; lean_object* v___y_3236_; lean_object* v___y_3237_; lean_object* v___y_3238_; lean_object* v_a_3239_; lean_object* v___y_3253_; lean_object* v___y_3254_; lean_object* v___y_3255_; lean_object* v_a_3256_; lean_object* v___y_3259_; lean_object* v___y_3260_; lean_object* v___y_3261_; uint8_t v_a_3262_; lean_object* v___y_3266_; lean_object* v___y_3267_; lean_object* v___y_3268_; lean_object* v___y_3269_; uint8_t v___y_3270_; lean_object* v___y_3271_; lean_object* v___y_3278_; lean_object* v___y_3279_; lean_object* v___y_3280_; lean_object* v___y_3281_; lean_object* v___y_3286_; lean_object* v___y_3287_; lean_object* v___y_3288_; uint8_t v___y_3289_; lean_object* v___y_3290_; lean_object* v___y_3291_; lean_object* v___y_3292_; lean_object* v_a_3293_; lean_object* v___y_3304_; uint8_t v___y_3305_; lean_object* v___y_3306_; lean_object* v___y_3307_; lean_object* v___y_3308_; lean_object* v___y_3309_; lean_object* v___y_3310_; uint8_t v_a_3311_; lean_object* v___y_3315_; lean_object* v___y_3316_; lean_object* v___y_3317_; lean_object* v___y_3318_; uint8_t v___y_3319_; lean_object* v___y_3320_; lean_object* v___y_3321_; lean_object* v_a_3322_; lean_object* v___y_3336_; lean_object* v___y_3337_; uint8_t v___y_3338_; lean_object* v___y_3339_; lean_object* v___y_3340_; lean_object* v___y_3341_; lean_object* v___y_3342_; uint8_t v_a_3343_; uint8_t v___y_3347_; lean_object* v___y_3348_; lean_object* v___y_3349_; lean_object* v___y_3350_; lean_object* v___y_3351_; lean_object* v___y_3352_; lean_object* v___y_3353_; lean_object* v___y_3354_; lean_object* v___y_3355_; uint8_t v___y_3356_; lean_object* v___y_3357_; lean_object* v___y_3358_; lean_object* v___y_3359_; lean_object* v___y_3360_; lean_object* v___y_3428_; lean_object* v___y_3429_; lean_object* v___y_3430_; lean_object* v_a_3431_; lean_object* v___y_3442_; lean_object* v___y_3443_; lean_object* v___y_3444_; lean_object* v_a_3445_; lean_object* v___y_3448_; lean_object* v___y_3449_; lean_object* v___y_3450_; uint8_t v_a_3451_; lean_object* v___y_3455_; lean_object* v___y_3456_; lean_object* v___y_3457_; lean_object* v___y_3458_; uint8_t v___y_3459_; lean_object* v___y_3460_; lean_object* v___y_3467_; lean_object* v___y_3468_; lean_object* v___y_3469_; lean_object* v___y_3470_; uint8_t v___y_3475_; lean_object* v___y_3476_; uint8_t v___y_3477_; lean_object* v___y_3478_; lean_object* v___y_3479_; lean_object* v___y_3480_; lean_object* v___y_3481_; lean_object* v___y_3482_; lean_object* v_a_3483_; uint8_t v___y_3494_; uint8_t v___y_3495_; lean_object* v___y_3496_; lean_object* v___y_3497_; lean_object* v___y_3498_; lean_object* v___y_3499_; lean_object* v___y_3500_; lean_object* v___y_3501_; uint8_t v_a_3502_; uint8_t v___y_3506_; lean_object* v___y_3507_; uint8_t v___y_3508_; lean_object* v___y_3509_; lean_object* v___y_3510_; lean_object* v___y_3511_; lean_object* v___y_3512_; lean_object* v___y_3513_; lean_object* v_a_3514_; uint8_t v___y_3528_; uint8_t v___y_3529_; lean_object* v___y_3530_; lean_object* v___y_3531_; lean_object* v___y_3532_; lean_object* v___y_3533_; lean_object* v___y_3534_; lean_object* v___y_3535_; uint8_t v_a_3536_; uint8_t v___y_3540_; uint8_t v___y_3541_; lean_object* v___y_3542_; lean_object* v___y_3543_; lean_object* v___y_3544_; lean_object* v___y_3545_; lean_object* v___y_3546_; lean_object* v___y_3547_; lean_object* v___y_3548_; lean_object* v___y_3549_; lean_object* v___y_3550_; lean_object* v___y_3551_; lean_object* v___y_3552_; lean_object* v___y_3553_; 
v___x_3232_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0));
v___x_3233_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_3234_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2719_, v_options_2704_, v___x_3233_);
if (v___x_3234_ == 0)
{
lean_object* v___x_3877_; lean_object* v___x_3878_; lean_object* v___x_3879_; uint8_t v___x_3880_; 
v___x_3877_ = l_Lean_KVMap_instValueBool;
v___x_3878_ = l_Lean_trace_profiler;
v___x_3879_ = l_Lean_Option_get___redArg(v___x_3877_, v_options_2704_, v___x_3878_);
v___x_3880_ = lean_unbox(v___x_3879_);
lean_dec(v___x_3879_);
if (v___x_3880_ == 0)
{
lean_object* v___x_3881_; 
v___x_3881_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_2645_, v_mvarId_u2082_2646_, v_a_2647_);
if (lean_obj_tag(v___x_3881_) == 0)
{
lean_object* v_a_3882_; 
v_a_3882_ = lean_ctor_get(v___x_3881_, 0);
lean_inc(v_a_3882_);
lean_dec_ref_known(v___x_3881_, 1);
v_____do__lift_3029_ = v_a_3882_;
v___y_3030_ = v_a_2647_;
v___y_3031_ = v_a_2648_;
v___y_3032_ = v_a_2649_;
v___y_3033_ = v_a_2650_;
v___y_3034_ = v_a_2651_;
v___y_3035_ = v_a_2652_;
goto v___jp_3028_;
}
else
{
lean_object* v_a_3883_; lean_object* v___x_3885_; uint8_t v_isShared_3886_; uint8_t v_isSharedCheck_3890_; 
lean_dec_ref_known(v___x_2703_, 3);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3883_ = lean_ctor_get(v___x_3881_, 0);
v_isSharedCheck_3890_ = !lean_is_exclusive(v___x_3881_);
if (v_isSharedCheck_3890_ == 0)
{
v___x_3885_ = v___x_3881_;
v_isShared_3886_ = v_isSharedCheck_3890_;
goto v_resetjp_3884_;
}
else
{
lean_inc(v_a_3883_);
lean_dec(v___x_3881_);
v___x_3885_ = lean_box(0);
v_isShared_3886_ = v_isSharedCheck_3890_;
goto v_resetjp_3884_;
}
v_resetjp_3884_:
{
lean_object* v___x_3888_; 
if (v_isShared_3886_ == 0)
{
v___x_3888_ = v___x_3885_;
goto v_reusejp_3887_;
}
else
{
lean_object* v_reuseFailAlloc_3889_; 
v_reuseFailAlloc_3889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3889_, 0, v_a_3883_);
v___x_3888_ = v_reuseFailAlloc_3889_;
goto v_reusejp_3887_;
}
v_reusejp_3887_:
{
return v___x_3888_;
}
}
}
}
else
{
goto v___jp_3621_;
}
}
else
{
goto v___jp_3621_;
}
v___jp_3235_:
{
lean_object* v___x_3240_; double v___x_3241_; double v___x_3242_; double v___x_3243_; double v___x_3244_; double v___x_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; lean_object* v___x_3248_; lean_object* v___x_3249_; lean_object* v___x_258306__overap_3250_; lean_object* v___x_3251_; 
v___x_3240_ = lean_io_mono_nanos_now();
v___x_3241_ = lean_float_of_nat(v___y_3237_);
v___x_3242_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_3243_ = lean_float_div(v___x_3241_, v___x_3242_);
v___x_3244_ = lean_float_of_nat(v___x_3240_);
v___x_3245_ = lean_float_div(v___x_3244_, v___x_3242_);
v___x_3246_ = lean_box_float(v___x_3243_);
v___x_3247_ = lean_box_float(v___x_3245_);
v___x_3248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3248_, 0, v___x_3246_);
lean_ctor_set(v___x_3248_, 1, v___x_3247_);
v___x_3249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3249_, 0, v_a_3239_);
lean_ctor_set(v___x_3249_, 1, v___x_3248_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_258306__overap_3250_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v_hasTrace_2720_, v___x_3232_, v_options_2704_, v___x_3234_, v___y_3236_, v_ref_2709_, v___y_3238_, v___x_3249_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3251_ = lean_apply_7(v___x_258306__overap_3250_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
return v___x_3251_;
}
v___jp_3252_:
{
lean_object* v___x_3257_; 
v___x_3257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3257_, 0, v_a_3256_);
v___y_3236_ = v___y_3254_;
v___y_3237_ = v___y_3253_;
v___y_3238_ = v___y_3255_;
v_a_3239_ = v___x_3257_;
goto v___jp_3235_;
}
v___jp_3258_:
{
lean_object* v___x_3263_; lean_object* v___x_3264_; 
v___x_3263_ = lean_box(v_a_3262_);
v___x_3264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3264_, 0, v___x_3263_);
v___y_3236_ = v___y_3260_;
v___y_3237_ = v___y_3259_;
v___y_3238_ = v___y_3261_;
v_a_3239_ = v___x_3264_;
goto v___jp_3235_;
}
v___jp_3265_:
{
lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_258335__overap_3274_; lean_object* v___x_3275_; 
lean_inc_ref(v___y_3271_);
v___x_3272_ = l_Lean_stringToMessageData(v___y_3271_);
lean_inc_ref(v___y_3266_);
v___x_3273_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3273_, 0, v___y_3266_);
lean_ctor_set(v___x_3273_, 1, v___x_3272_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258335__overap_3274_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3273_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3275_ = lean_apply_7(v___x_258335__overap_3274_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3275_) == 0)
{
lean_dec_ref_known(v___x_3275_, 1);
v___y_3259_ = v___y_3268_;
v___y_3260_ = v___y_3267_;
v___y_3261_ = v___y_3269_;
v_a_3262_ = v___y_3270_;
goto v___jp_3258_;
}
else
{
lean_object* v_a_3276_; 
v_a_3276_ = lean_ctor_get(v___x_3275_, 0);
lean_inc(v_a_3276_);
lean_dec_ref_known(v___x_3275_, 1);
v___y_3253_ = v___y_3268_;
v___y_3254_ = v___y_3267_;
v___y_3255_ = v___y_3269_;
v_a_3256_ = v_a_3276_;
goto v___jp_3252_;
}
}
v___jp_3277_:
{
if (lean_obj_tag(v___y_3281_) == 0)
{
lean_object* v_a_3282_; uint8_t v___x_3283_; 
v_a_3282_ = lean_ctor_get(v___y_3281_, 0);
lean_inc(v_a_3282_);
lean_dec_ref_known(v___y_3281_, 1);
v___x_3283_ = lean_unbox(v_a_3282_);
lean_dec(v_a_3282_);
v___y_3259_ = v___y_3279_;
v___y_3260_ = v___y_3278_;
v___y_3261_ = v___y_3280_;
v_a_3262_ = v___x_3283_;
goto v___jp_3258_;
}
else
{
lean_object* v_a_3284_; 
v_a_3284_ = lean_ctor_get(v___y_3281_, 0);
lean_inc(v_a_3284_);
lean_dec_ref_known(v___y_3281_, 1);
v___y_3253_ = v___y_3279_;
v___y_3254_ = v___y_3278_;
v___y_3255_ = v___y_3280_;
v_a_3256_ = v_a_3284_;
goto v___jp_3252_;
}
}
v___jp_3285_:
{
lean_object* v___x_3294_; double v___x_3295_; double v___x_3296_; lean_object* v___x_3297_; lean_object* v___x_3298_; lean_object* v___x_3299_; lean_object* v___x_3300_; lean_object* v___x_258837__overap_3301_; lean_object* v___x_3302_; 
v___x_3294_ = lean_io_get_num_heartbeats();
v___x_3295_ = lean_float_of_nat(v___y_3291_);
v___x_3296_ = lean_float_of_nat(v___x_3294_);
v___x_3297_ = lean_box_float(v___x_3295_);
v___x_3298_ = lean_box_float(v___x_3296_);
v___x_3299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3299_, 0, v___x_3297_);
lean_ctor_set(v___x_3299_, 1, v___x_3298_);
v___x_3300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3300_, 0, v_a_3293_);
lean_ctor_set(v___x_3300_, 1, v___x_3299_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258837__overap_3301_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v_hasTrace_2720_, v___x_3232_, v_options_2704_, v___y_3289_, v___y_3292_, v_ref_2709_, v___y_3286_, v___x_3300_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3302_ = lean_apply_7(v___x_258837__overap_3301_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3278_ = v___y_3288_;
v___y_3279_ = v___y_3287_;
v___y_3280_ = v___y_3290_;
v___y_3281_ = v___x_3302_;
goto v___jp_3277_;
}
v___jp_3303_:
{
lean_object* v___x_3312_; lean_object* v___x_3313_; 
v___x_3312_ = lean_box(v_a_3311_);
v___x_3313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3313_, 0, v___x_3312_);
v___y_3286_ = v___y_3304_;
v___y_3287_ = v___y_3307_;
v___y_3288_ = v___y_3306_;
v___y_3289_ = v___y_3305_;
v___y_3290_ = v___y_3308_;
v___y_3291_ = v___y_3309_;
v___y_3292_ = v___y_3310_;
v_a_3293_ = v___x_3313_;
goto v___jp_3285_;
}
v___jp_3314_:
{
lean_object* v___x_3323_; double v___x_3324_; double v___x_3325_; double v___x_3326_; double v___x_3327_; double v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_258797__overap_3333_; lean_object* v___x_3334_; 
v___x_3323_ = lean_io_mono_nanos_now();
v___x_3324_ = lean_float_of_nat(v___y_3316_);
v___x_3325_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_3326_ = lean_float_div(v___x_3324_, v___x_3325_);
v___x_3327_ = lean_float_of_nat(v___x_3323_);
v___x_3328_ = lean_float_div(v___x_3327_, v___x_3325_);
v___x_3329_ = lean_box_float(v___x_3326_);
v___x_3330_ = lean_box_float(v___x_3328_);
v___x_3331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3331_, 0, v___x_3329_);
lean_ctor_set(v___x_3331_, 1, v___x_3330_);
v___x_3332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3332_, 0, v_a_3322_);
lean_ctor_set(v___x_3332_, 1, v___x_3331_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258797__overap_3333_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v_hasTrace_2720_, v___x_3232_, v_options_2704_, v___y_3319_, v___y_3321_, v_ref_2709_, v___y_3315_, v___x_3332_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3334_ = lean_apply_7(v___x_258797__overap_3333_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3278_ = v___y_3318_;
v___y_3279_ = v___y_3317_;
v___y_3280_ = v___y_3320_;
v___y_3281_ = v___x_3334_;
goto v___jp_3277_;
}
v___jp_3335_:
{
lean_object* v___x_3344_; lean_object* v___x_3345_; 
v___x_3344_ = lean_box(v_a_3343_);
v___x_3345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3345_, 0, v___x_3344_);
v___y_3315_ = v___y_3336_;
v___y_3316_ = v___y_3337_;
v___y_3317_ = v___y_3340_;
v___y_3318_ = v___y_3339_;
v___y_3319_ = v___y_3338_;
v___y_3320_ = v___y_3341_;
v___y_3321_ = v___y_3342_;
v_a_3322_ = v___x_3345_;
goto v___jp_3314_;
}
v___jp_3346_:
{
lean_object* v___x_258772__overap_3361_; lean_object* v___x_3362_; 
lean_inc_ref(v___x_2695_);
v___x_258772__overap_3361_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2695_, v___x_2696_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3362_ = lean_apply_7(v___x_258772__overap_3361_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3362_) == 0)
{
lean_object* v_a_3363_; lean_object* v___x_3364_; lean_object* v___x_258777__overap_3365_; lean_object* v___x_3366_; 
v_a_3363_ = lean_ctor_get(v___x_3362_, 0);
lean_inc(v_a_3363_);
lean_dec_ref_known(v___x_3362_, 1);
v___x_3364_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3);
v___x_258777__overap_3365_ = l_Lean_addMessageContextFull___redArg(v___y_3348_, v___y_3351_, v___y_3354_, v___y_3352_, v___y_3360_, v___x_3364_);
lean_inc(v_a_2652_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
v___x_3366_ = lean_apply_5(v___x_258777__overap_3365_, v_a_2649_, v_a_2650_, v___y_3358_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3366_) == 0)
{
if (v___y_3347_ == 0)
{
lean_object* v_a_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; 
v_a_3367_ = lean_ctor_get(v___x_3366_, 0);
lean_inc(v_a_3367_);
lean_dec_ref_known(v___x_3366_, 1);
v___x_3368_ = lean_io_mono_nanos_now();
v___x_3369_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_3359_, v___y_3350_, v___y_3353_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec_ref(v___y_3353_);
if (lean_obj_tag(v___x_3369_) == 0)
{
lean_object* v_a_3370_; uint8_t v___x_3371_; 
v_a_3370_ = lean_ctor_get(v___x_3369_, 0);
lean_inc(v_a_3370_);
lean_dec_ref_known(v___x_3369_, 1);
v___x_3371_ = lean_unbox(v_a_3370_);
lean_dec(v_a_3370_);
if (v___x_3371_ == 0)
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___y_3336_ = v_a_3367_;
v___y_3337_ = v___x_3368_;
v___y_3338_ = v___y_3356_;
v___y_3339_ = v___y_3355_;
v___y_3340_ = v___y_3349_;
v___y_3341_ = v___y_3357_;
v___y_3342_ = v_a_3363_;
v_a_3343_ = v___y_3347_;
goto v___jp_3335_;
}
else
{
lean_object* v___x_3372_; lean_object* v_equalMVarIds_3373_; lean_object* v_equalLMVarIds_3374_; lean_object* v_leftUnassignedMVarValues_3375_; lean_object* v_rightUnassignedMVarValues_3376_; lean_object* v___x_3378_; uint8_t v_isShared_3379_; uint8_t v_isSharedCheck_3385_; 
v___x_3372_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3373_ = lean_ctor_get(v___x_3372_, 0);
v_equalLMVarIds_3374_ = lean_ctor_get(v___x_3372_, 1);
v_leftUnassignedMVarValues_3375_ = lean_ctor_get(v___x_3372_, 2);
v_rightUnassignedMVarValues_3376_ = lean_ctor_get(v___x_3372_, 3);
v_isSharedCheck_3385_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3385_ == 0)
{
v___x_3378_ = v___x_3372_;
v_isShared_3379_ = v_isSharedCheck_3385_;
goto v_resetjp_3377_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3376_);
lean_inc(v_leftUnassignedMVarValues_3375_);
lean_inc(v_equalLMVarIds_3374_);
lean_inc(v_equalMVarIds_3373_);
lean_dec(v___x_3372_);
v___x_3378_ = lean_box(0);
v_isShared_3379_ = v_isSharedCheck_3385_;
goto v_resetjp_3377_;
}
v_resetjp_3377_:
{
lean_object* v___x_3380_; lean_object* v___x_3382_; 
v___x_3380_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3373_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3379_ == 0)
{
lean_ctor_set(v___x_3378_, 0, v___x_3380_);
v___x_3382_ = v___x_3378_;
goto v_reusejp_3381_;
}
else
{
lean_object* v_reuseFailAlloc_3384_; 
v_reuseFailAlloc_3384_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3384_, 0, v___x_3380_);
lean_ctor_set(v_reuseFailAlloc_3384_, 1, v_equalLMVarIds_3374_);
lean_ctor_set(v_reuseFailAlloc_3384_, 2, v_leftUnassignedMVarValues_3375_);
lean_ctor_set(v_reuseFailAlloc_3384_, 3, v_rightUnassignedMVarValues_3376_);
v___x_3382_ = v_reuseFailAlloc_3384_;
goto v_reusejp_3381_;
}
v_reusejp_3381_:
{
lean_object* v___x_3383_; 
v___x_3383_ = lean_st_ref_set(v_a_2648_, v___x_3382_);
v___y_3336_ = v_a_3367_;
v___y_3337_ = v___x_3368_;
v___y_3338_ = v___y_3356_;
v___y_3339_ = v___y_3355_;
v___y_3340_ = v___y_3349_;
v___y_3341_ = v___y_3357_;
v___y_3342_ = v_a_3363_;
v_a_3343_ = v_hasTrace_2720_;
goto v___jp_3335_;
}
}
}
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3369_) == 0)
{
lean_object* v_a_3386_; uint8_t v___x_3387_; 
v_a_3386_ = lean_ctor_get(v___x_3369_, 0);
lean_inc(v_a_3386_);
lean_dec_ref_known(v___x_3369_, 1);
v___x_3387_ = lean_unbox(v_a_3386_);
lean_dec(v_a_3386_);
v___y_3336_ = v_a_3367_;
v___y_3337_ = v___x_3368_;
v___y_3338_ = v___y_3356_;
v___y_3339_ = v___y_3355_;
v___y_3340_ = v___y_3349_;
v___y_3341_ = v___y_3357_;
v___y_3342_ = v_a_3363_;
v_a_3343_ = v___x_3387_;
goto v___jp_3335_;
}
else
{
lean_object* v_a_3388_; lean_object* v___x_3390_; uint8_t v_isShared_3391_; uint8_t v_isSharedCheck_3395_; 
v_a_3388_ = lean_ctor_get(v___x_3369_, 0);
v_isSharedCheck_3395_ = !lean_is_exclusive(v___x_3369_);
if (v_isSharedCheck_3395_ == 0)
{
v___x_3390_ = v___x_3369_;
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
else
{
lean_inc(v_a_3388_);
lean_dec(v___x_3369_);
v___x_3390_ = lean_box(0);
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
v_resetjp_3389_:
{
lean_object* v___x_3393_; 
if (v_isShared_3391_ == 0)
{
lean_ctor_set_tag(v___x_3390_, 0);
v___x_3393_ = v___x_3390_;
goto v_reusejp_3392_;
}
else
{
lean_object* v_reuseFailAlloc_3394_; 
v_reuseFailAlloc_3394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3394_, 0, v_a_3388_);
v___x_3393_ = v_reuseFailAlloc_3394_;
goto v_reusejp_3392_;
}
v_reusejp_3392_:
{
v___y_3315_ = v_a_3367_;
v___y_3316_ = v___x_3368_;
v___y_3317_ = v___y_3349_;
v___y_3318_ = v___y_3355_;
v___y_3319_ = v___y_3356_;
v___y_3320_ = v___y_3357_;
v___y_3321_ = v_a_3363_;
v_a_3322_ = v___x_3393_;
goto v___jp_3314_;
}
}
}
}
}
else
{
lean_object* v_a_3396_; lean_object* v___x_3397_; lean_object* v___x_3398_; 
v_a_3396_ = lean_ctor_get(v___x_3366_, 0);
lean_inc(v_a_3396_);
lean_dec_ref_known(v___x_3366_, 1);
v___x_3397_ = lean_io_get_num_heartbeats();
v___x_3398_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_3359_, v___y_3350_, v___y_3353_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec_ref(v___y_3353_);
if (lean_obj_tag(v___x_3398_) == 0)
{
lean_object* v_a_3399_; uint8_t v___x_3400_; 
v_a_3399_ = lean_ctor_get(v___x_3398_, 0);
lean_inc(v_a_3399_);
lean_dec_ref_known(v___x_3398_, 1);
v___x_3400_ = lean_unbox(v_a_3399_);
lean_dec(v_a_3399_);
if (v___x_3400_ == 0)
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___y_3304_ = v_a_3396_;
v___y_3305_ = v___y_3356_;
v___y_3306_ = v___y_3355_;
v___y_3307_ = v___y_3349_;
v___y_3308_ = v___y_3357_;
v___y_3309_ = v___x_3397_;
v___y_3310_ = v_a_3363_;
v_a_3311_ = v___y_3347_;
goto v___jp_3303_;
}
else
{
lean_object* v___x_3401_; lean_object* v_equalMVarIds_3402_; lean_object* v_equalLMVarIds_3403_; lean_object* v_leftUnassignedMVarValues_3404_; lean_object* v_rightUnassignedMVarValues_3405_; lean_object* v___x_3407_; uint8_t v_isShared_3408_; uint8_t v_isSharedCheck_3414_; 
v___x_3401_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3402_ = lean_ctor_get(v___x_3401_, 0);
v_equalLMVarIds_3403_ = lean_ctor_get(v___x_3401_, 1);
v_leftUnassignedMVarValues_3404_ = lean_ctor_get(v___x_3401_, 2);
v_rightUnassignedMVarValues_3405_ = lean_ctor_get(v___x_3401_, 3);
v_isSharedCheck_3414_ = !lean_is_exclusive(v___x_3401_);
if (v_isSharedCheck_3414_ == 0)
{
v___x_3407_ = v___x_3401_;
v_isShared_3408_ = v_isSharedCheck_3414_;
goto v_resetjp_3406_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3405_);
lean_inc(v_leftUnassignedMVarValues_3404_);
lean_inc(v_equalLMVarIds_3403_);
lean_inc(v_equalMVarIds_3402_);
lean_dec(v___x_3401_);
v___x_3407_ = lean_box(0);
v_isShared_3408_ = v_isSharedCheck_3414_;
goto v_resetjp_3406_;
}
v_resetjp_3406_:
{
lean_object* v___x_3409_; lean_object* v___x_3411_; 
v___x_3409_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3402_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3408_ == 0)
{
lean_ctor_set(v___x_3407_, 0, v___x_3409_);
v___x_3411_ = v___x_3407_;
goto v_reusejp_3410_;
}
else
{
lean_object* v_reuseFailAlloc_3413_; 
v_reuseFailAlloc_3413_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3413_, 0, v___x_3409_);
lean_ctor_set(v_reuseFailAlloc_3413_, 1, v_equalLMVarIds_3403_);
lean_ctor_set(v_reuseFailAlloc_3413_, 2, v_leftUnassignedMVarValues_3404_);
lean_ctor_set(v_reuseFailAlloc_3413_, 3, v_rightUnassignedMVarValues_3405_);
v___x_3411_ = v_reuseFailAlloc_3413_;
goto v_reusejp_3410_;
}
v_reusejp_3410_:
{
lean_object* v___x_3412_; 
v___x_3412_ = lean_st_ref_set(v_a_2648_, v___x_3411_);
v___y_3304_ = v_a_3396_;
v___y_3305_ = v___y_3356_;
v___y_3306_ = v___y_3355_;
v___y_3307_ = v___y_3349_;
v___y_3308_ = v___y_3357_;
v___y_3309_ = v___x_3397_;
v___y_3310_ = v_a_3363_;
v_a_3311_ = v___y_3347_;
goto v___jp_3303_;
}
}
}
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3398_) == 0)
{
lean_object* v_a_3415_; uint8_t v___x_3416_; 
v_a_3415_ = lean_ctor_get(v___x_3398_, 0);
lean_inc(v_a_3415_);
lean_dec_ref_known(v___x_3398_, 1);
v___x_3416_ = lean_unbox(v_a_3415_);
lean_dec(v_a_3415_);
v___y_3304_ = v_a_3396_;
v___y_3305_ = v___y_3356_;
v___y_3306_ = v___y_3355_;
v___y_3307_ = v___y_3349_;
v___y_3308_ = v___y_3357_;
v___y_3309_ = v___x_3397_;
v___y_3310_ = v_a_3363_;
v_a_3311_ = v___x_3416_;
goto v___jp_3303_;
}
else
{
lean_object* v_a_3417_; lean_object* v___x_3419_; uint8_t v_isShared_3420_; uint8_t v_isSharedCheck_3424_; 
v_a_3417_ = lean_ctor_get(v___x_3398_, 0);
v_isSharedCheck_3424_ = !lean_is_exclusive(v___x_3398_);
if (v_isSharedCheck_3424_ == 0)
{
v___x_3419_ = v___x_3398_;
v_isShared_3420_ = v_isSharedCheck_3424_;
goto v_resetjp_3418_;
}
else
{
lean_inc(v_a_3417_);
lean_dec(v___x_3398_);
v___x_3419_ = lean_box(0);
v_isShared_3420_ = v_isSharedCheck_3424_;
goto v_resetjp_3418_;
}
v_resetjp_3418_:
{
lean_object* v___x_3422_; 
if (v_isShared_3420_ == 0)
{
lean_ctor_set_tag(v___x_3419_, 0);
v___x_3422_ = v___x_3419_;
goto v_reusejp_3421_;
}
else
{
lean_object* v_reuseFailAlloc_3423_; 
v_reuseFailAlloc_3423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3423_, 0, v_a_3417_);
v___x_3422_ = v_reuseFailAlloc_3423_;
goto v_reusejp_3421_;
}
v_reusejp_3421_:
{
v___y_3286_ = v_a_3396_;
v___y_3287_ = v___y_3349_;
v___y_3288_ = v___y_3355_;
v___y_3289_ = v___y_3356_;
v___y_3290_ = v___y_3357_;
v___y_3291_ = v___x_3397_;
v___y_3292_ = v_a_3363_;
v_a_3293_ = v___x_3422_;
goto v___jp_3285_;
}
}
}
}
}
}
else
{
lean_object* v_a_3425_; 
lean_dec(v_a_3363_);
lean_dec_ref(v___y_3359_);
lean_dec_ref(v___y_3353_);
lean_dec_ref(v___y_3350_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3425_ = lean_ctor_get(v___x_3366_, 0);
lean_inc(v_a_3425_);
lean_dec_ref_known(v___x_3366_, 1);
v___y_3253_ = v___y_3349_;
v___y_3254_ = v___y_3355_;
v___y_3255_ = v___y_3357_;
v_a_3256_ = v_a_3425_;
goto v___jp_3252_;
}
}
else
{
lean_object* v_a_3426_; 
lean_dec(v___y_3360_);
lean_dec_ref(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec_ref(v___y_3354_);
lean_dec_ref(v___y_3353_);
lean_dec_ref(v___y_3352_);
lean_dec_ref(v___y_3351_);
lean_dec_ref(v___y_3350_);
lean_dec_ref(v___y_3348_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3426_ = lean_ctor_get(v___x_3362_, 0);
lean_inc(v_a_3426_);
lean_dec_ref_known(v___x_3362_, 1);
v___y_3253_ = v___y_3349_;
v___y_3254_ = v___y_3355_;
v___y_3255_ = v___y_3357_;
v_a_3256_ = v_a_3426_;
goto v___jp_3252_;
}
}
v___jp_3427_:
{
lean_object* v___x_3432_; double v___x_3433_; double v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; lean_object* v___x_258537__overap_3439_; lean_object* v___x_3440_; 
v___x_3432_ = lean_io_get_num_heartbeats();
v___x_3433_ = lean_float_of_nat(v___y_3428_);
v___x_3434_ = lean_float_of_nat(v___x_3432_);
v___x_3435_ = lean_box_float(v___x_3433_);
v___x_3436_ = lean_box_float(v___x_3434_);
v___x_3437_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3437_, 0, v___x_3435_);
lean_ctor_set(v___x_3437_, 1, v___x_3436_);
v___x_3438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3438_, 0, v_a_3431_);
lean_ctor_set(v___x_3438_, 1, v___x_3437_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_258537__overap_3439_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v_hasTrace_2720_, v___x_3232_, v_options_2704_, v___x_3234_, v___y_3429_, v_ref_2709_, v___y_3430_, v___x_3438_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3440_ = lean_apply_7(v___x_258537__overap_3439_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
return v___x_3440_;
}
v___jp_3441_:
{
lean_object* v___x_3446_; 
v___x_3446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3446_, 0, v_a_3445_);
v___y_3428_ = v___y_3442_;
v___y_3429_ = v___y_3443_;
v___y_3430_ = v___y_3444_;
v_a_3431_ = v___x_3446_;
goto v___jp_3427_;
}
v___jp_3447_:
{
lean_object* v___x_3452_; lean_object* v___x_3453_; 
v___x_3452_ = lean_box(v_a_3451_);
v___x_3453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3453_, 0, v___x_3452_);
v___y_3428_ = v___y_3448_;
v___y_3429_ = v___y_3449_;
v___y_3430_ = v___y_3450_;
v_a_3431_ = v___x_3453_;
goto v___jp_3427_;
}
v___jp_3454_:
{
lean_object* v___x_3461_; lean_object* v___x_3462_; lean_object* v___x_258566__overap_3463_; lean_object* v___x_3464_; 
lean_inc_ref(v___y_3460_);
v___x_3461_ = l_Lean_stringToMessageData(v___y_3460_);
lean_inc_ref(v___y_3456_);
v___x_3462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3462_, 0, v___y_3456_);
lean_ctor_set(v___x_3462_, 1, v___x_3461_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258566__overap_3463_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3462_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3464_ = lean_apply_7(v___x_258566__overap_3463_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3464_) == 0)
{
lean_dec_ref_known(v___x_3464_, 1);
v___y_3448_ = v___y_3455_;
v___y_3449_ = v___y_3457_;
v___y_3450_ = v___y_3458_;
v_a_3451_ = v___y_3459_;
goto v___jp_3447_;
}
else
{
lean_object* v_a_3465_; 
v_a_3465_ = lean_ctor_get(v___x_3464_, 0);
lean_inc(v_a_3465_);
lean_dec_ref_known(v___x_3464_, 1);
v___y_3442_ = v___y_3455_;
v___y_3443_ = v___y_3457_;
v___y_3444_ = v___y_3458_;
v_a_3445_ = v_a_3465_;
goto v___jp_3441_;
}
}
v___jp_3466_:
{
if (lean_obj_tag(v___y_3470_) == 0)
{
lean_object* v_a_3471_; uint8_t v___x_3472_; 
v_a_3471_ = lean_ctor_get(v___y_3470_, 0);
lean_inc(v_a_3471_);
lean_dec_ref_known(v___y_3470_, 1);
v___x_3472_ = lean_unbox(v_a_3471_);
lean_dec(v_a_3471_);
v___y_3448_ = v___y_3467_;
v___y_3449_ = v___y_3468_;
v___y_3450_ = v___y_3469_;
v_a_3451_ = v___x_3472_;
goto v___jp_3447_;
}
else
{
lean_object* v_a_3473_; 
v_a_3473_ = lean_ctor_get(v___y_3470_, 0);
lean_inc(v_a_3473_);
lean_dec_ref_known(v___y_3470_, 1);
v___y_3442_ = v___y_3467_;
v___y_3443_ = v___y_3468_;
v___y_3444_ = v___y_3469_;
v_a_3445_ = v_a_3473_;
goto v___jp_3441_;
}
}
v___jp_3474_:
{
lean_object* v___x_3484_; double v___x_3485_; double v___x_3486_; lean_object* v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v___x_3490_; lean_object* v___x_258942__overap_3491_; lean_object* v___x_3492_; 
v___x_3484_ = lean_io_get_num_heartbeats();
v___x_3485_ = lean_float_of_nat(v___y_3478_);
v___x_3486_ = lean_float_of_nat(v___x_3484_);
v___x_3487_ = lean_box_float(v___x_3485_);
v___x_3488_ = lean_box_float(v___x_3486_);
v___x_3489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3489_, 0, v___x_3487_);
lean_ctor_set(v___x_3489_, 1, v___x_3488_);
v___x_3490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3490_, 0, v_a_3483_);
lean_ctor_set(v___x_3490_, 1, v___x_3489_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258942__overap_3491_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v___y_3477_, v___x_3232_, v_options_2704_, v___y_3475_, v___y_3482_, v_ref_2709_, v___y_3481_, v___x_3490_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3492_ = lean_apply_7(v___x_258942__overap_3491_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3467_ = v___y_3476_;
v___y_3468_ = v___y_3479_;
v___y_3469_ = v___y_3480_;
v___y_3470_ = v___x_3492_;
goto v___jp_3466_;
}
v___jp_3493_:
{
lean_object* v___x_3503_; lean_object* v___x_3504_; 
v___x_3503_ = lean_box(v_a_3502_);
v___x_3504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3504_, 0, v___x_3503_);
v___y_3475_ = v___y_3494_;
v___y_3476_ = v___y_3496_;
v___y_3477_ = v___y_3495_;
v___y_3478_ = v___y_3497_;
v___y_3479_ = v___y_3498_;
v___y_3480_ = v___y_3499_;
v___y_3481_ = v___y_3501_;
v___y_3482_ = v___y_3500_;
v_a_3483_ = v___x_3504_;
goto v___jp_3474_;
}
v___jp_3505_:
{
lean_object* v___x_3515_; double v___x_3516_; double v___x_3517_; double v___x_3518_; double v___x_3519_; double v___x_3520_; lean_object* v___x_3521_; lean_object* v___x_3522_; lean_object* v___x_3523_; lean_object* v___x_3524_; lean_object* v___x_258902__overap_3525_; lean_object* v___x_3526_; 
v___x_3515_ = lean_io_mono_nanos_now();
v___x_3516_ = lean_float_of_nat(v___y_3513_);
v___x_3517_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_3518_ = lean_float_div(v___x_3516_, v___x_3517_);
v___x_3519_ = lean_float_of_nat(v___x_3515_);
v___x_3520_ = lean_float_div(v___x_3519_, v___x_3517_);
v___x_3521_ = lean_box_float(v___x_3518_);
v___x_3522_ = lean_box_float(v___x_3520_);
v___x_3523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3523_, 0, v___x_3521_);
lean_ctor_set(v___x_3523_, 1, v___x_3522_);
v___x_3524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3524_, 0, v_a_3514_);
lean_ctor_set(v___x_3524_, 1, v___x_3523_);
lean_inc(v_ref_2709_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258902__overap_3525_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v___y_3508_, v___x_3232_, v_options_2704_, v___y_3506_, v___y_3512_, v_ref_2709_, v___y_3511_, v___x_3524_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3526_ = lean_apply_7(v___x_258902__overap_3525_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3467_ = v___y_3507_;
v___y_3468_ = v___y_3509_;
v___y_3469_ = v___y_3510_;
v___y_3470_ = v___x_3526_;
goto v___jp_3466_;
}
v___jp_3527_:
{
lean_object* v___x_3537_; lean_object* v___x_3538_; 
v___x_3537_ = lean_box(v_a_3536_);
v___x_3538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3538_, 0, v___x_3537_);
v___y_3506_ = v___y_3528_;
v___y_3507_ = v___y_3530_;
v___y_3508_ = v___y_3529_;
v___y_3509_ = v___y_3531_;
v___y_3510_ = v___y_3532_;
v___y_3511_ = v___y_3534_;
v___y_3512_ = v___y_3533_;
v___y_3513_ = v___y_3535_;
v_a_3514_ = v___x_3538_;
goto v___jp_3505_;
}
v___jp_3539_:
{
lean_object* v___x_258877__overap_3554_; lean_object* v___x_3555_; 
lean_inc_ref(v___x_2695_);
v___x_258877__overap_3554_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2695_, v___x_2696_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3555_ = lean_apply_7(v___x_258877__overap_3554_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3555_) == 0)
{
lean_object* v_a_3556_; lean_object* v___x_3557_; lean_object* v___x_258882__overap_3558_; lean_object* v___x_3559_; 
v_a_3556_ = lean_ctor_get(v___x_3555_, 0);
lean_inc(v_a_3556_);
lean_dec_ref_known(v___x_3555_, 1);
v___x_3557_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3);
v___x_258882__overap_3558_ = l_Lean_addMessageContextFull___redArg(v___y_3543_, v___y_3544_, v___y_3547_, v___y_3546_, v___y_3552_, v___x_3557_);
lean_inc(v_a_2652_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
v___x_3559_ = lean_apply_5(v___x_258882__overap_3558_, v_a_2649_, v_a_2650_, v___y_3550_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3559_) == 0)
{
if (v___y_3541_ == 0)
{
lean_object* v_a_3560_; lean_object* v___x_3561_; lean_object* v___x_3562_; 
v_a_3560_ = lean_ctor_get(v___x_3559_, 0);
lean_inc(v_a_3560_);
lean_dec_ref_known(v___x_3559_, 1);
v___x_3561_ = lean_io_mono_nanos_now();
v___x_3562_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_3545_, v___y_3553_, v___y_3551_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec_ref(v___y_3551_);
if (lean_obj_tag(v___x_3562_) == 0)
{
lean_object* v_a_3563_; uint8_t v___x_3564_; 
v_a_3563_ = lean_ctor_get(v___x_3562_, 0);
lean_inc(v_a_3563_);
lean_dec_ref_known(v___x_3562_, 1);
v___x_3564_ = lean_unbox(v_a_3563_);
lean_dec(v_a_3563_);
if (v___x_3564_ == 0)
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___y_3528_ = v___y_3540_;
v___y_3529_ = v___y_3541_;
v___y_3530_ = v___y_3542_;
v___y_3531_ = v___y_3548_;
v___y_3532_ = v___y_3549_;
v___y_3533_ = v_a_3556_;
v___y_3534_ = v_a_3560_;
v___y_3535_ = v___x_3561_;
v_a_3536_ = v___y_3541_;
goto v___jp_3527_;
}
else
{
lean_object* v___x_3565_; lean_object* v_equalMVarIds_3566_; lean_object* v_equalLMVarIds_3567_; lean_object* v_leftUnassignedMVarValues_3568_; lean_object* v_rightUnassignedMVarValues_3569_; lean_object* v___x_3571_; uint8_t v_isShared_3572_; uint8_t v_isSharedCheck_3578_; 
v___x_3565_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3566_ = lean_ctor_get(v___x_3565_, 0);
v_equalLMVarIds_3567_ = lean_ctor_get(v___x_3565_, 1);
v_leftUnassignedMVarValues_3568_ = lean_ctor_get(v___x_3565_, 2);
v_rightUnassignedMVarValues_3569_ = lean_ctor_get(v___x_3565_, 3);
v_isSharedCheck_3578_ = !lean_is_exclusive(v___x_3565_);
if (v_isSharedCheck_3578_ == 0)
{
v___x_3571_ = v___x_3565_;
v_isShared_3572_ = v_isSharedCheck_3578_;
goto v_resetjp_3570_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3569_);
lean_inc(v_leftUnassignedMVarValues_3568_);
lean_inc(v_equalLMVarIds_3567_);
lean_inc(v_equalMVarIds_3566_);
lean_dec(v___x_3565_);
v___x_3571_ = lean_box(0);
v_isShared_3572_ = v_isSharedCheck_3578_;
goto v_resetjp_3570_;
}
v_resetjp_3570_:
{
lean_object* v___x_3573_; lean_object* v___x_3575_; 
v___x_3573_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3566_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3572_ == 0)
{
lean_ctor_set(v___x_3571_, 0, v___x_3573_);
v___x_3575_ = v___x_3571_;
goto v_reusejp_3574_;
}
else
{
lean_object* v_reuseFailAlloc_3577_; 
v_reuseFailAlloc_3577_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3577_, 0, v___x_3573_);
lean_ctor_set(v_reuseFailAlloc_3577_, 1, v_equalLMVarIds_3567_);
lean_ctor_set(v_reuseFailAlloc_3577_, 2, v_leftUnassignedMVarValues_3568_);
lean_ctor_set(v_reuseFailAlloc_3577_, 3, v_rightUnassignedMVarValues_3569_);
v___x_3575_ = v_reuseFailAlloc_3577_;
goto v_reusejp_3574_;
}
v_reusejp_3574_:
{
lean_object* v___x_3576_; 
v___x_3576_ = lean_st_ref_set(v_a_2648_, v___x_3575_);
v___y_3528_ = v___y_3540_;
v___y_3529_ = v___y_3541_;
v___y_3530_ = v___y_3542_;
v___y_3531_ = v___y_3548_;
v___y_3532_ = v___y_3549_;
v___y_3533_ = v_a_3556_;
v___y_3534_ = v_a_3560_;
v___y_3535_ = v___x_3561_;
v_a_3536_ = v_hasTrace_2720_;
goto v___jp_3527_;
}
}
}
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3562_) == 0)
{
lean_object* v_a_3579_; uint8_t v___x_3580_; 
v_a_3579_ = lean_ctor_get(v___x_3562_, 0);
lean_inc(v_a_3579_);
lean_dec_ref_known(v___x_3562_, 1);
v___x_3580_ = lean_unbox(v_a_3579_);
lean_dec(v_a_3579_);
v___y_3528_ = v___y_3540_;
v___y_3529_ = v___y_3541_;
v___y_3530_ = v___y_3542_;
v___y_3531_ = v___y_3548_;
v___y_3532_ = v___y_3549_;
v___y_3533_ = v_a_3556_;
v___y_3534_ = v_a_3560_;
v___y_3535_ = v___x_3561_;
v_a_3536_ = v___x_3580_;
goto v___jp_3527_;
}
else
{
lean_object* v_a_3581_; lean_object* v___x_3583_; uint8_t v_isShared_3584_; uint8_t v_isSharedCheck_3588_; 
v_a_3581_ = lean_ctor_get(v___x_3562_, 0);
v_isSharedCheck_3588_ = !lean_is_exclusive(v___x_3562_);
if (v_isSharedCheck_3588_ == 0)
{
v___x_3583_ = v___x_3562_;
v_isShared_3584_ = v_isSharedCheck_3588_;
goto v_resetjp_3582_;
}
else
{
lean_inc(v_a_3581_);
lean_dec(v___x_3562_);
v___x_3583_ = lean_box(0);
v_isShared_3584_ = v_isSharedCheck_3588_;
goto v_resetjp_3582_;
}
v_resetjp_3582_:
{
lean_object* v___x_3586_; 
if (v_isShared_3584_ == 0)
{
lean_ctor_set_tag(v___x_3583_, 0);
v___x_3586_ = v___x_3583_;
goto v_reusejp_3585_;
}
else
{
lean_object* v_reuseFailAlloc_3587_; 
v_reuseFailAlloc_3587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3587_, 0, v_a_3581_);
v___x_3586_ = v_reuseFailAlloc_3587_;
goto v_reusejp_3585_;
}
v_reusejp_3585_:
{
v___y_3506_ = v___y_3540_;
v___y_3507_ = v___y_3542_;
v___y_3508_ = v___y_3541_;
v___y_3509_ = v___y_3548_;
v___y_3510_ = v___y_3549_;
v___y_3511_ = v_a_3560_;
v___y_3512_ = v_a_3556_;
v___y_3513_ = v___x_3561_;
v_a_3514_ = v___x_3586_;
goto v___jp_3505_;
}
}
}
}
}
else
{
lean_object* v_a_3589_; lean_object* v___x_3590_; lean_object* v___x_3591_; 
v_a_3589_ = lean_ctor_get(v___x_3559_, 0);
lean_inc(v_a_3589_);
lean_dec_ref_known(v___x_3559_, 1);
v___x_3590_ = lean_io_get_num_heartbeats();
v___x_3591_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_3545_, v___y_3553_, v___y_3551_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec_ref(v___y_3551_);
if (lean_obj_tag(v___x_3591_) == 0)
{
lean_object* v_a_3592_; uint8_t v___x_3593_; 
v_a_3592_ = lean_ctor_get(v___x_3591_, 0);
lean_inc(v_a_3592_);
lean_dec_ref_known(v___x_3591_, 1);
v___x_3593_ = lean_unbox(v_a_3592_);
if (v___x_3593_ == 0)
{
uint8_t v___x_3594_; 
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3594_ = lean_unbox(v_a_3592_);
lean_dec(v_a_3592_);
v___y_3494_ = v___y_3540_;
v___y_3495_ = v___y_3541_;
v___y_3496_ = v___y_3542_;
v___y_3497_ = v___x_3590_;
v___y_3498_ = v___y_3548_;
v___y_3499_ = v___y_3549_;
v___y_3500_ = v_a_3556_;
v___y_3501_ = v_a_3589_;
v_a_3502_ = v___x_3594_;
goto v___jp_3493_;
}
else
{
lean_object* v___x_3595_; lean_object* v_equalMVarIds_3596_; lean_object* v_equalLMVarIds_3597_; lean_object* v_leftUnassignedMVarValues_3598_; lean_object* v_rightUnassignedMVarValues_3599_; lean_object* v___x_3601_; uint8_t v_isShared_3602_; uint8_t v_isSharedCheck_3608_; 
lean_dec(v_a_3592_);
v___x_3595_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3596_ = lean_ctor_get(v___x_3595_, 0);
v_equalLMVarIds_3597_ = lean_ctor_get(v___x_3595_, 1);
v_leftUnassignedMVarValues_3598_ = lean_ctor_get(v___x_3595_, 2);
v_rightUnassignedMVarValues_3599_ = lean_ctor_get(v___x_3595_, 3);
v_isSharedCheck_3608_ = !lean_is_exclusive(v___x_3595_);
if (v_isSharedCheck_3608_ == 0)
{
v___x_3601_ = v___x_3595_;
v_isShared_3602_ = v_isSharedCheck_3608_;
goto v_resetjp_3600_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3599_);
lean_inc(v_leftUnassignedMVarValues_3598_);
lean_inc(v_equalLMVarIds_3597_);
lean_inc(v_equalMVarIds_3596_);
lean_dec(v___x_3595_);
v___x_3601_ = lean_box(0);
v_isShared_3602_ = v_isSharedCheck_3608_;
goto v_resetjp_3600_;
}
v_resetjp_3600_:
{
lean_object* v___x_3603_; lean_object* v___x_3605_; 
v___x_3603_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3596_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3602_ == 0)
{
lean_ctor_set(v___x_3601_, 0, v___x_3603_);
v___x_3605_ = v___x_3601_;
goto v_reusejp_3604_;
}
else
{
lean_object* v_reuseFailAlloc_3607_; 
v_reuseFailAlloc_3607_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3607_, 0, v___x_3603_);
lean_ctor_set(v_reuseFailAlloc_3607_, 1, v_equalLMVarIds_3597_);
lean_ctor_set(v_reuseFailAlloc_3607_, 2, v_leftUnassignedMVarValues_3598_);
lean_ctor_set(v_reuseFailAlloc_3607_, 3, v_rightUnassignedMVarValues_3599_);
v___x_3605_ = v_reuseFailAlloc_3607_;
goto v_reusejp_3604_;
}
v_reusejp_3604_:
{
lean_object* v___x_3606_; 
v___x_3606_ = lean_st_ref_set(v_a_2648_, v___x_3605_);
v___y_3494_ = v___y_3540_;
v___y_3495_ = v___y_3541_;
v___y_3496_ = v___y_3542_;
v___y_3497_ = v___x_3590_;
v___y_3498_ = v___y_3548_;
v___y_3499_ = v___y_3549_;
v___y_3500_ = v_a_3556_;
v___y_3501_ = v_a_3589_;
v_a_3502_ = v___y_3541_;
goto v___jp_3493_;
}
}
}
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3591_) == 0)
{
lean_object* v_a_3609_; uint8_t v___x_3610_; 
v_a_3609_ = lean_ctor_get(v___x_3591_, 0);
lean_inc(v_a_3609_);
lean_dec_ref_known(v___x_3591_, 1);
v___x_3610_ = lean_unbox(v_a_3609_);
lean_dec(v_a_3609_);
v___y_3494_ = v___y_3540_;
v___y_3495_ = v___y_3541_;
v___y_3496_ = v___y_3542_;
v___y_3497_ = v___x_3590_;
v___y_3498_ = v___y_3548_;
v___y_3499_ = v___y_3549_;
v___y_3500_ = v_a_3556_;
v___y_3501_ = v_a_3589_;
v_a_3502_ = v___x_3610_;
goto v___jp_3493_;
}
else
{
lean_object* v_a_3611_; lean_object* v___x_3613_; uint8_t v_isShared_3614_; uint8_t v_isSharedCheck_3618_; 
v_a_3611_ = lean_ctor_get(v___x_3591_, 0);
v_isSharedCheck_3618_ = !lean_is_exclusive(v___x_3591_);
if (v_isSharedCheck_3618_ == 0)
{
v___x_3613_ = v___x_3591_;
v_isShared_3614_ = v_isSharedCheck_3618_;
goto v_resetjp_3612_;
}
else
{
lean_inc(v_a_3611_);
lean_dec(v___x_3591_);
v___x_3613_ = lean_box(0);
v_isShared_3614_ = v_isSharedCheck_3618_;
goto v_resetjp_3612_;
}
v_resetjp_3612_:
{
lean_object* v___x_3616_; 
if (v_isShared_3614_ == 0)
{
lean_ctor_set_tag(v___x_3613_, 0);
v___x_3616_ = v___x_3613_;
goto v_reusejp_3615_;
}
else
{
lean_object* v_reuseFailAlloc_3617_; 
v_reuseFailAlloc_3617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3617_, 0, v_a_3611_);
v___x_3616_ = v_reuseFailAlloc_3617_;
goto v_reusejp_3615_;
}
v_reusejp_3615_:
{
v___y_3475_ = v___y_3540_;
v___y_3476_ = v___y_3542_;
v___y_3477_ = v___y_3541_;
v___y_3478_ = v___x_3590_;
v___y_3479_ = v___y_3548_;
v___y_3480_ = v___y_3549_;
v___y_3481_ = v_a_3589_;
v___y_3482_ = v_a_3556_;
v_a_3483_ = v___x_3616_;
goto v___jp_3474_;
}
}
}
}
}
}
else
{
lean_object* v_a_3619_; 
lean_dec(v_a_3556_);
lean_dec_ref(v___y_3553_);
lean_dec_ref(v___y_3551_);
lean_dec_ref(v___y_3545_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3619_ = lean_ctor_get(v___x_3559_, 0);
lean_inc(v_a_3619_);
lean_dec_ref_known(v___x_3559_, 1);
v___y_3442_ = v___y_3542_;
v___y_3443_ = v___y_3548_;
v___y_3444_ = v___y_3549_;
v_a_3445_ = v_a_3619_;
goto v___jp_3441_;
}
}
else
{
lean_object* v_a_3620_; 
lean_dec_ref(v___y_3553_);
lean_dec(v___y_3552_);
lean_dec_ref(v___y_3551_);
lean_dec_ref(v___y_3550_);
lean_dec_ref(v___y_3547_);
lean_dec_ref(v___y_3546_);
lean_dec_ref(v___y_3545_);
lean_dec_ref(v___y_3544_);
lean_dec_ref(v___y_3543_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3620_ = lean_ctor_get(v___x_3555_, 0);
lean_inc(v_a_3620_);
lean_dec_ref_known(v___x_3555_, 1);
v___y_3442_ = v___y_3542_;
v___y_3443_ = v___y_3548_;
v___y_3444_ = v___y_3549_;
v_a_3445_ = v_a_3620_;
goto v___jp_3441_;
}
}
v___jp_3621_:
{
lean_object* v___x_258180__overap_3622_; lean_object* v___x_3623_; 
lean_inc_ref(v___x_2695_);
v___x_258180__overap_3622_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2695_, v___x_2696_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3623_ = lean_apply_7(v___x_258180__overap_3622_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3623_) == 0)
{
lean_object* v_toApplicative_3624_; lean_object* v_a_3625_; lean_object* v_toFunctor_3626_; lean_object* v_toSeq_3627_; lean_object* v_toSeqLeft_3628_; lean_object* v_toSeqRight_3629_; lean_object* v___f_3630_; lean_object* v___f_3631_; lean_object* v___x_3632_; lean_object* v___f_3633_; lean_object* v___f_3634_; lean_object* v___f_3635_; lean_object* v___x_3636_; lean_object* v___x_3637_; lean_object* v___x_3638_; lean_object* v_toApplicative_3639_; lean_object* v___x_3641_; uint8_t v_isShared_3642_; uint8_t v_isSharedCheck_3867_; 
v_toApplicative_3624_ = lean_ctor_get(v___x_2654_, 0);
v_a_3625_ = lean_ctor_get(v___x_3623_, 0);
lean_inc(v_a_3625_);
lean_dec_ref_known(v___x_3623_, 1);
v_toFunctor_3626_ = lean_ctor_get(v_toApplicative_3624_, 0);
v_toSeq_3627_ = lean_ctor_get(v_toApplicative_3624_, 2);
v_toSeqLeft_3628_ = lean_ctor_get(v_toApplicative_3624_, 3);
v_toSeqRight_3629_ = lean_ctor_get(v_toApplicative_3624_, 4);
lean_inc_ref_n(v_toFunctor_3626_, 2);
v___f_3630_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3630_, 0, v_toFunctor_3626_);
v___f_3631_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3631_, 0, v_toFunctor_3626_);
v___x_3632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3632_, 0, v___f_3630_);
lean_ctor_set(v___x_3632_, 1, v___f_3631_);
lean_inc(v_toSeqRight_3629_);
v___f_3633_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3633_, 0, v_toSeqRight_3629_);
lean_inc(v_toSeqLeft_3628_);
v___f_3634_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3634_, 0, v_toSeqLeft_3628_);
lean_inc(v_toSeq_3627_);
v___f_3635_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3635_, 0, v_toSeq_3627_);
v___x_3636_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3636_, 0, v___x_3632_);
lean_ctor_set(v___x_3636_, 1, v___f_2660_);
lean_ctor_set(v___x_3636_, 2, v___f_3635_);
lean_ctor_set(v___x_3636_, 3, v___f_3634_);
lean_ctor_set(v___x_3636_, 4, v___f_3633_);
v___x_3637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3637_, 0, v___x_3636_);
lean_ctor_set(v___x_3637_, 1, v___f_2661_);
v___x_3638_ = l_StateRefT_x27_instMonad___redArg(v___x_3637_);
v_toApplicative_3639_ = lean_ctor_get(v___x_3638_, 0);
v_isSharedCheck_3867_ = !lean_is_exclusive(v___x_3638_);
if (v_isSharedCheck_3867_ == 0)
{
lean_object* v_unused_3868_; 
v_unused_3868_ = lean_ctor_get(v___x_3638_, 1);
lean_dec(v_unused_3868_);
v___x_3641_ = v___x_3638_;
v_isShared_3642_ = v_isSharedCheck_3867_;
goto v_resetjp_3640_;
}
else
{
lean_inc(v_toApplicative_3639_);
lean_dec(v___x_3638_);
v___x_3641_ = lean_box(0);
v_isShared_3642_ = v_isSharedCheck_3867_;
goto v_resetjp_3640_;
}
v_resetjp_3640_:
{
lean_object* v_toFunctor_3643_; lean_object* v_toSeq_3644_; lean_object* v_toSeqLeft_3645_; lean_object* v_toSeqRight_3646_; lean_object* v___x_3648_; uint8_t v_isShared_3649_; uint8_t v_isSharedCheck_3865_; 
v_toFunctor_3643_ = lean_ctor_get(v_toApplicative_3639_, 0);
v_toSeq_3644_ = lean_ctor_get(v_toApplicative_3639_, 2);
v_toSeqLeft_3645_ = lean_ctor_get(v_toApplicative_3639_, 3);
v_toSeqRight_3646_ = lean_ctor_get(v_toApplicative_3639_, 4);
v_isSharedCheck_3865_ = !lean_is_exclusive(v_toApplicative_3639_);
if (v_isSharedCheck_3865_ == 0)
{
lean_object* v_unused_3866_; 
v_unused_3866_ = lean_ctor_get(v_toApplicative_3639_, 1);
lean_dec(v_unused_3866_);
v___x_3648_ = v_toApplicative_3639_;
v_isShared_3649_ = v_isSharedCheck_3865_;
goto v_resetjp_3647_;
}
else
{
lean_inc(v_toSeqRight_3646_);
lean_inc(v_toSeqLeft_3645_);
lean_inc(v_toSeq_3644_);
lean_inc(v_toFunctor_3643_);
lean_dec(v_toApplicative_3639_);
v___x_3648_ = lean_box(0);
v_isShared_3649_ = v_isSharedCheck_3865_;
goto v_resetjp_3647_;
}
v_resetjp_3647_:
{
lean_object* v_ref_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_3654_; lean_object* v___x_3655_; lean_object* v___x_3656_; lean_object* v___x_3657_; lean_object* v___x_3658_; lean_object* v___f_3659_; lean_object* v___f_3660_; lean_object* v___x_3661_; lean_object* v___f_3662_; lean_object* v___f_3663_; lean_object* v___f_3664_; lean_object* v___x_3666_; 
v_ref_3650_ = l_Lean_replaceRef(v_ref_2709_, v_ref_2709_);
lean_inc_ref(v_inheritedTraceOptions_2719_);
lean_inc(v_cancelTk_x3f_2717_);
lean_inc(v_currMacroScope_2715_);
lean_inc(v_quotContext_2714_);
lean_inc(v_maxHeartbeats_2713_);
lean_inc(v_initHeartbeats_2712_);
lean_inc(v_openDecls_2711_);
lean_inc(v_currNamespace_2710_);
lean_inc(v_maxRecDepth_2708_);
lean_inc(v_currRecDepth_2707_);
lean_inc_ref(v_options_2704_);
lean_inc_ref(v_fileMap_2706_);
lean_inc_ref(v_fileName_2705_);
v___x_3651_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3651_, 0, v_fileName_2705_);
lean_ctor_set(v___x_3651_, 1, v_fileMap_2706_);
lean_ctor_set(v___x_3651_, 2, v_options_2704_);
lean_ctor_set(v___x_3651_, 3, v_currRecDepth_2707_);
lean_ctor_set(v___x_3651_, 4, v_maxRecDepth_2708_);
lean_ctor_set(v___x_3651_, 5, v_ref_3650_);
lean_ctor_set(v___x_3651_, 6, v_currNamespace_2710_);
lean_ctor_set(v___x_3651_, 7, v_openDecls_2711_);
lean_ctor_set(v___x_3651_, 8, v_initHeartbeats_2712_);
lean_ctor_set(v___x_3651_, 9, v_maxHeartbeats_2713_);
lean_ctor_set(v___x_3651_, 10, v_quotContext_2714_);
lean_ctor_set(v___x_3651_, 11, v_currMacroScope_2715_);
lean_ctor_set(v___x_3651_, 12, v_cancelTk_x3f_2717_);
lean_ctor_set(v___x_3651_, 13, v_inheritedTraceOptions_2719_);
lean_ctor_set_uint8(v___x_3651_, sizeof(void*)*14, v_diag_2716_);
lean_ctor_set_uint8(v___x_3651_, sizeof(void*)*14 + 1, v_suppressElabErrors_2718_);
v___x_3652_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__19);
lean_inc(v_mvarId_u2081_2645_);
v___x_3653_ = l_Lean_MessageData_ofName(v_mvarId_u2081_2645_);
lean_inc_ref(v___x_3653_);
v___x_3654_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3654_, 0, v___x_3652_);
lean_ctor_set(v___x_3654_, 1, v___x_3653_);
v___x_3655_ = lean_obj_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__5);
v___x_3656_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3656_, 0, v___x_3654_);
lean_ctor_set(v___x_3656_, 1, v___x_3655_);
lean_inc(v_mvarId_u2082_2646_);
v___x_3657_ = l_Lean_MessageData_ofName(v_mvarId_u2082_2646_);
lean_inc_ref(v___x_3657_);
v___x_3658_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3658_, 0, v___x_3656_);
lean_ctor_set(v___x_3658_, 1, v___x_3657_);
lean_inc_ref(v_toFunctor_3643_);
v___f_3659_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3659_, 0, v_toFunctor_3643_);
v___f_3660_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3660_, 0, v_toFunctor_3643_);
v___x_3661_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3661_, 0, v___f_3659_);
lean_ctor_set(v___x_3661_, 1, v___f_3660_);
v___f_3662_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3662_, 0, v_toSeqRight_3646_);
v___f_3663_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3663_, 0, v_toSeqLeft_3645_);
v___f_3664_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3664_, 0, v_toSeq_3644_);
if (v_isShared_3649_ == 0)
{
lean_ctor_set(v___x_3648_, 4, v___f_3662_);
lean_ctor_set(v___x_3648_, 3, v___f_3663_);
lean_ctor_set(v___x_3648_, 2, v___f_3664_);
lean_ctor_set(v___x_3648_, 1, v___f_2682_);
lean_ctor_set(v___x_3648_, 0, v___x_3661_);
v___x_3666_ = v___x_3648_;
goto v_reusejp_3665_;
}
else
{
lean_object* v_reuseFailAlloc_3864_; 
v_reuseFailAlloc_3864_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3864_, 0, v___x_3661_);
lean_ctor_set(v_reuseFailAlloc_3864_, 1, v___f_2682_);
lean_ctor_set(v_reuseFailAlloc_3864_, 2, v___f_3664_);
lean_ctor_set(v_reuseFailAlloc_3864_, 3, v___f_3663_);
lean_ctor_set(v_reuseFailAlloc_3864_, 4, v___f_3662_);
v___x_3666_ = v_reuseFailAlloc_3864_;
goto v_reusejp_3665_;
}
v_reusejp_3665_:
{
lean_object* v___x_3668_; 
if (v_isShared_3642_ == 0)
{
lean_ctor_set(v___x_3641_, 1, v___f_2683_);
lean_ctor_set(v___x_3641_, 0, v___x_3666_);
v___x_3668_ = v___x_3641_;
goto v_reusejp_3667_;
}
else
{
lean_object* v_reuseFailAlloc_3863_; 
v_reuseFailAlloc_3863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3863_, 0, v___x_3666_);
lean_ctor_set(v_reuseFailAlloc_3863_, 1, v___f_2683_);
v___x_3668_ = v_reuseFailAlloc_3863_;
goto v_reusejp_3667_;
}
v_reusejp_3667_:
{
lean_object* v___x_3669_; lean_object* v___x_3670_; lean_object* v___f_3671_; lean_object* v___x_3672_; lean_object* v___x_258237__overap_3673_; lean_object* v___x_3674_; 
v___x_3669_ = l_Lean_Meta_instMonadEnvMetaM;
v___x_3670_ = l_Lean_Meta_instMonadMCtxMetaM;
v___f_3671_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27));
v___x_3672_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30));
lean_inc_ref(v___x_3668_);
v___x_258237__overap_3673_ = l_Lean_addMessageContextFull___redArg(v___x_3668_, v___x_3669_, v___x_3670_, v___f_3671_, v___x_3672_, v___x_3658_);
lean_inc(v_a_2652_);
lean_inc_ref(v___x_3651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
v___x_3674_ = lean_apply_5(v___x_258237__overap_3673_, v_a_2649_, v_a_2650_, v___x_3651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3674_) == 0)
{
lean_object* v_a_3675_; lean_object* v___x_3676_; lean_object* v___x_3677_; lean_object* v___x_3678_; uint8_t v___x_3679_; 
v_a_3675_ = lean_ctor_get(v___x_3674_, 0);
lean_inc(v_a_3675_);
lean_dec_ref_known(v___x_3674_, 1);
v___x_3676_ = l_Lean_KVMap_instValueBool;
v___x_3677_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3678_ = l_Lean_Option_get___redArg(v___x_3676_, v_options_2704_, v___x_3677_);
v___x_3679_ = lean_unbox(v___x_3678_);
if (v___x_3679_ == 0)
{
lean_object* v___x_3680_; lean_object* v___x_3681_; 
v___x_3680_ = lean_io_mono_nanos_now();
v___x_3681_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_2645_, v_mvarId_u2082_2646_, v_a_2647_);
if (lean_obj_tag(v___x_3681_) == 0)
{
lean_object* v_a_3682_; 
v_a_3682_ = lean_ctor_get(v___x_3681_, 0);
lean_inc(v_a_3682_);
lean_dec_ref_known(v___x_3681_, 1);
if (lean_obj_tag(v_a_3682_) == 1)
{
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref(v___x_3653_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (v___x_3234_ == 0)
{
lean_object* v_val_3683_; uint8_t v___x_3684_; 
v_val_3683_ = lean_ctor_get(v_a_3682_, 0);
lean_inc(v_val_3683_);
lean_dec_ref_known(v_a_3682_, 1);
v___x_3684_ = lean_unbox(v_val_3683_);
lean_dec(v_val_3683_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v___x_3684_;
goto v___jp_3258_;
}
else
{
lean_object* v_val_3685_; lean_object* v___x_3686_; uint8_t v___x_3687_; 
v_val_3685_ = lean_ctor_get(v_a_3682_, 0);
lean_inc(v_val_3685_);
lean_dec_ref_known(v_a_3682_, 1);
v___x_3686_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5);
v___x_3687_ = lean_unbox(v_val_3685_);
if (v___x_3687_ == 0)
{
lean_object* v___x_3688_; uint8_t v___x_3689_; 
v___x_3688_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6));
v___x_3689_ = lean_unbox(v_val_3685_);
lean_dec(v_val_3685_);
v___y_3266_ = v___x_3686_;
v___y_3267_ = v_a_3625_;
v___y_3268_ = v___x_3680_;
v___y_3269_ = v_a_3675_;
v___y_3270_ = v___x_3689_;
v___y_3271_ = v___x_3688_;
goto v___jp_3265_;
}
else
{
lean_object* v___x_3690_; uint8_t v___x_3691_; 
v___x_3690_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7));
v___x_3691_ = lean_unbox(v_val_3685_);
lean_dec(v_val_3685_);
v___y_3266_ = v___x_3686_;
v___y_3267_ = v_a_3625_;
v___y_3268_ = v___x_3680_;
v___y_3269_ = v_a_3675_;
v___y_3270_ = v___x_3691_;
v___y_3271_ = v___x_3690_;
goto v___jp_3265_;
}
}
}
else
{
lean_object* v___x_3692_; lean_object* v_equalMVarIds_3693_; lean_object* v___x_3694_; 
lean_dec(v_a_3682_);
v___x_3692_ = lean_st_ref_get(v_a_2648_);
v_equalMVarIds_3693_ = lean_ctor_get(v___x_3692_, 0);
lean_inc_ref(v_equalMVarIds_3693_);
lean_dec(v___x_3692_);
lean_inc(v_mvarId_u2081_2645_);
v___x_3694_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3693_, v_mvarId_u2081_2645_);
lean_dec_ref(v_equalMVarIds_3693_);
if (lean_obj_tag(v___x_3694_) == 1)
{
lean_object* v_val_3695_; uint8_t v___x_3696_; 
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2081_2645_);
v_val_3695_ = lean_ctor_get(v___x_3694_, 0);
lean_inc(v_val_3695_);
lean_dec_ref_known(v___x_3694_, 1);
v___x_3696_ = l_Lean_instBEqMVarId_beq(v_mvarId_u2082_2646_, v_val_3695_);
lean_dec(v_mvarId_u2082_2646_);
if (v___x_3696_ == 0)
{
if (v___x_3234_ == 0)
{
uint8_t v___x_3697_; 
lean_dec(v_val_3695_);
lean_dec_ref(v___x_3653_);
v___x_3697_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v___x_3697_;
goto v___jp_3258_;
}
else
{
lean_object* v___x_3698_; lean_object* v___x_3699_; lean_object* v___x_3700_; lean_object* v___x_3701_; lean_object* v___x_3702_; lean_object* v___x_3703_; lean_object* v___x_258361__overap_3704_; lean_object* v___x_3705_; 
v___x_3698_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9);
v___x_3699_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3699_, 0, v___x_3698_);
lean_ctor_set(v___x_3699_, 1, v___x_3653_);
v___x_3700_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11);
v___x_3701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3701_, 0, v___x_3699_);
lean_ctor_set(v___x_3701_, 1, v___x_3700_);
v___x_3702_ = l_Lean_MessageData_ofName(v_val_3695_);
v___x_3703_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3703_, 0, v___x_3701_);
lean_ctor_set(v___x_3703_, 1, v___x_3702_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258361__overap_3704_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3703_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3705_ = lean_apply_7(v___x_258361__overap_3704_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3705_) == 0)
{
uint8_t v___x_3706_; 
lean_dec_ref_known(v___x_3705_, 1);
v___x_3706_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v___x_3706_;
goto v___jp_3258_;
}
else
{
lean_object* v_a_3707_; 
lean_dec(v___x_3678_);
v_a_3707_ = lean_ctor_get(v___x_3705_, 0);
lean_inc(v_a_3707_);
lean_dec_ref_known(v___x_3705_, 1);
v___y_3253_ = v___x_3680_;
v___y_3254_ = v_a_3625_;
v___y_3255_ = v_a_3675_;
v_a_3256_ = v_a_3707_;
goto v___jp_3252_;
}
}
}
else
{
lean_dec(v_val_3695_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3653_);
if (v___x_3234_ == 0)
{
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v_hasTrace_2720_;
goto v___jp_3258_;
}
else
{
lean_object* v___x_3708_; lean_object* v___x_258372__overap_3709_; lean_object* v___x_3710_; 
v___x_3708_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258372__overap_3709_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3708_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3710_ = lean_apply_7(v___x_258372__overap_3709_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3710_) == 0)
{
lean_dec_ref_known(v___x_3710_, 1);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v_hasTrace_2720_;
goto v___jp_3258_;
}
else
{
lean_object* v_a_3711_; 
v_a_3711_ = lean_ctor_get(v___x_3710_, 0);
lean_inc(v_a_3711_);
lean_dec_ref_known(v___x_3710_, 1);
v___y_3253_ = v___x_3680_;
v___y_3254_ = v_a_3625_;
v___y_3255_ = v_a_3675_;
v_a_3256_ = v_a_3711_;
goto v___jp_3252_;
}
}
}
}
else
{
lean_object* v_mctx_u2081_3712_; lean_object* v_mctx_u2082_3713_; lean_object* v_decls_3714_; lean_object* v___x_3715_; 
lean_dec(v___x_3694_);
v_mctx_u2081_3712_ = lean_ctor_get(v_a_2647_, 1);
v_mctx_u2082_3713_ = lean_ctor_get(v_a_2647_, 2);
v_decls_3714_ = lean_ctor_get(v_mctx_u2081_3712_, 5);
lean_inc(v_mvarId_u2081_2645_);
v___x_3715_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3714_, v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3715_) == 1)
{
lean_object* v_val_3716_; lean_object* v_decls_3717_; lean_object* v___x_3718_; 
lean_dec_ref(v___x_3653_);
v_val_3716_ = lean_ctor_get(v___x_3715_, 0);
lean_inc(v_val_3716_);
lean_dec_ref_known(v___x_3715_, 1);
v_decls_3717_ = lean_ctor_get(v_mctx_u2082_3713_, 5);
lean_inc(v_mvarId_u2082_2646_);
v___x_3718_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3717_, v_mvarId_u2082_2646_);
if (lean_obj_tag(v___x_3718_) == 1)
{
lean_object* v_val_3719_; lean_object* v_lctx_3720_; lean_object* v_type_3721_; lean_object* v_lctx_3722_; lean_object* v_type_3723_; lean_object* v_localInstances_3724_; lean_object* v___x_3725_; 
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_2703_, 3);
v_val_3719_ = lean_ctor_get(v___x_3718_, 0);
lean_inc(v_val_3719_);
lean_dec_ref_known(v___x_3718_, 1);
v_lctx_3720_ = lean_ctor_get(v_val_3716_, 1);
lean_inc_ref(v_lctx_3720_);
v_type_3721_ = lean_ctor_get(v_val_3716_, 2);
lean_inc_ref(v_type_3721_);
lean_dec(v_val_3716_);
v_lctx_3722_ = lean_ctor_get(v_val_3719_, 1);
lean_inc_ref(v_lctx_3722_);
v_type_3723_ = lean_ctor_get(v_val_3719_, 2);
lean_inc_ref(v_type_3723_);
v_localInstances_3724_ = lean_ctor_get(v_val_3719_, 4);
lean_inc_ref_n(v_localInstances_3724_, 2);
lean_dec(v_val_3719_);
v___x_3725_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(v_lctx_3720_, v_lctx_3722_, v_localInstances_3724_, v_localInstances_3724_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
if (lean_obj_tag(v___x_3725_) == 0)
{
lean_object* v_a_3726_; 
v_a_3726_ = lean_ctor_get(v___x_3725_, 0);
lean_inc(v_a_3726_);
lean_dec_ref_known(v___x_3725_, 1);
if (lean_obj_tag(v_a_3726_) == 1)
{
if (v___x_3234_ == 0)
{
lean_object* v_val_3727_; lean_object* v___x_3728_; lean_object* v___x_3729_; uint8_t v___x_3730_; 
v_val_3727_ = lean_ctor_get(v_a_3726_, 0);
lean_inc(v_val_3727_);
lean_dec_ref_known(v_a_3726_, 1);
v___x_3728_ = l_Lean_trace_profiler;
v___x_3729_ = l_Lean_Option_get___redArg(v___x_3676_, v_options_2704_, v___x_3728_);
v___x_3730_ = lean_unbox(v___x_3729_);
lean_dec(v___x_3729_);
if (v___x_3730_ == 0)
{
lean_object* v___x_3731_; 
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
v___x_3731_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v_type_3721_, v_type_3723_, v_val_3727_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec(v_val_3727_);
if (lean_obj_tag(v___x_3731_) == 0)
{
lean_object* v_a_3732_; uint8_t v___x_3733_; 
v_a_3732_ = lean_ctor_get(v___x_3731_, 0);
lean_inc(v_a_3732_);
lean_dec_ref_known(v___x_3731_, 1);
v___x_3733_ = lean_unbox(v_a_3732_);
lean_dec(v_a_3732_);
if (v___x_3733_ == 0)
{
uint8_t v___x_3734_; 
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3734_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v___x_3734_;
goto v___jp_3258_;
}
else
{
lean_object* v___x_3735_; lean_object* v_equalMVarIds_3736_; lean_object* v_equalLMVarIds_3737_; lean_object* v_leftUnassignedMVarValues_3738_; lean_object* v_rightUnassignedMVarValues_3739_; lean_object* v___x_3741_; uint8_t v_isShared_3742_; uint8_t v_isSharedCheck_3748_; 
lean_dec(v___x_3678_);
v___x_3735_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3736_ = lean_ctor_get(v___x_3735_, 0);
v_equalLMVarIds_3737_ = lean_ctor_get(v___x_3735_, 1);
v_leftUnassignedMVarValues_3738_ = lean_ctor_get(v___x_3735_, 2);
v_rightUnassignedMVarValues_3739_ = lean_ctor_get(v___x_3735_, 3);
v_isSharedCheck_3748_ = !lean_is_exclusive(v___x_3735_);
if (v_isSharedCheck_3748_ == 0)
{
v___x_3741_ = v___x_3735_;
v_isShared_3742_ = v_isSharedCheck_3748_;
goto v_resetjp_3740_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3739_);
lean_inc(v_leftUnassignedMVarValues_3738_);
lean_inc(v_equalLMVarIds_3737_);
lean_inc(v_equalMVarIds_3736_);
lean_dec(v___x_3735_);
v___x_3741_ = lean_box(0);
v_isShared_3742_ = v_isSharedCheck_3748_;
goto v_resetjp_3740_;
}
v_resetjp_3740_:
{
lean_object* v___x_3743_; lean_object* v___x_3745_; 
v___x_3743_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3736_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3742_ == 0)
{
lean_ctor_set(v___x_3741_, 0, v___x_3743_);
v___x_3745_ = v___x_3741_;
goto v_reusejp_3744_;
}
else
{
lean_object* v_reuseFailAlloc_3747_; 
v_reuseFailAlloc_3747_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3747_, 0, v___x_3743_);
lean_ctor_set(v_reuseFailAlloc_3747_, 1, v_equalLMVarIds_3737_);
lean_ctor_set(v_reuseFailAlloc_3747_, 2, v_leftUnassignedMVarValues_3738_);
lean_ctor_set(v_reuseFailAlloc_3747_, 3, v_rightUnassignedMVarValues_3739_);
v___x_3745_ = v_reuseFailAlloc_3747_;
goto v_reusejp_3744_;
}
v_reusejp_3744_:
{
lean_object* v___x_3746_; 
v___x_3746_ = lean_st_ref_set(v_a_2648_, v___x_3745_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v_hasTrace_2720_;
goto v___jp_3258_;
}
}
}
}
else
{
lean_dec(v___x_3678_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___y_3278_ = v_a_3625_;
v___y_3279_ = v___x_3680_;
v___y_3280_ = v_a_3675_;
v___y_3281_ = v___x_3731_;
goto v___jp_3277_;
}
}
else
{
uint8_t v___x_3749_; 
v___x_3749_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3347_ = v___x_3749_;
v___y_3348_ = v___x_3668_;
v___y_3349_ = v___x_3680_;
v___y_3350_ = v_type_3723_;
v___y_3351_ = v___x_3669_;
v___y_3352_ = v___f_3671_;
v___y_3353_ = v_val_3727_;
v___y_3354_ = v___x_3670_;
v___y_3355_ = v_a_3625_;
v___y_3356_ = v___x_3234_;
v___y_3357_ = v_a_3675_;
v___y_3358_ = v___x_3651_;
v___y_3359_ = v_type_3721_;
v___y_3360_ = v___x_3672_;
goto v___jp_3346_;
}
}
else
{
lean_object* v_val_3750_; uint8_t v___x_3751_; 
v_val_3750_ = lean_ctor_get(v_a_3726_, 0);
lean_inc(v_val_3750_);
lean_dec_ref_known(v_a_3726_, 1);
v___x_3751_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3347_ = v___x_3751_;
v___y_3348_ = v___x_3668_;
v___y_3349_ = v___x_3680_;
v___y_3350_ = v_type_3723_;
v___y_3351_ = v___x_3669_;
v___y_3352_ = v___f_3671_;
v___y_3353_ = v_val_3750_;
v___y_3354_ = v___x_3670_;
v___y_3355_ = v_a_3625_;
v___y_3356_ = v___x_3234_;
v___y_3357_ = v_a_3675_;
v___y_3358_ = v___x_3651_;
v___y_3359_ = v_type_3721_;
v___y_3360_ = v___x_3672_;
goto v___jp_3346_;
}
}
else
{
uint8_t v___x_3752_; 
lean_dec(v_a_3726_);
lean_dec_ref(v_type_3723_);
lean_dec_ref(v_type_3721_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3752_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3259_ = v___x_3680_;
v___y_3260_ = v_a_3625_;
v___y_3261_ = v_a_3675_;
v_a_3262_ = v___x_3752_;
goto v___jp_3258_;
}
}
else
{
lean_object* v_a_3753_; 
lean_dec_ref(v_type_3723_);
lean_dec_ref(v_type_3721_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3753_ = lean_ctor_get(v___x_3725_, 0);
lean_inc(v_a_3753_);
lean_dec_ref_known(v___x_3725_, 1);
v___y_3253_ = v___x_3680_;
v___y_3254_ = v_a_3625_;
v___y_3255_ = v_a_3675_;
v_a_3256_ = v_a_3753_;
goto v___jp_3252_;
}
}
else
{
lean_object* v___x_3754_; lean_object* v___x_3755_; lean_object* v___x_3756_; lean_object* v___x_3757_; lean_object* v___x_258514__overap_3758_; lean_object* v___x_3759_; 
lean_dec(v___x_3718_);
lean_dec(v_val_3716_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3754_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3755_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3755_, 0, v___x_3754_);
lean_ctor_set(v___x_3755_, 1, v___x_3657_);
v___x_3756_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3757_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3757_, 0, v___x_3755_);
lean_ctor_set(v___x_3757_, 1, v___x_3756_);
lean_inc_ref(v___x_2695_);
v___x_258514__overap_3758_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3757_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3759_ = lean_apply_7(v___x_258514__overap_3758_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3278_ = v_a_3625_;
v___y_3279_ = v___x_3680_;
v___y_3280_ = v_a_3675_;
v___y_3281_ = v___x_3759_;
goto v___jp_3277_;
}
}
else
{
lean_object* v___x_3760_; lean_object* v___x_3761_; lean_object* v___x_3762_; lean_object* v___x_3763_; lean_object* v___x_258521__overap_3764_; lean_object* v___x_3765_; 
lean_dec(v___x_3715_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3760_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3761_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3761_, 0, v___x_3760_);
lean_ctor_set(v___x_3761_, 1, v___x_3653_);
v___x_3762_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3763_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3763_, 0, v___x_3761_);
lean_ctor_set(v___x_3763_, 1, v___x_3762_);
lean_inc_ref(v___x_2695_);
v___x_258521__overap_3764_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3763_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3765_ = lean_apply_7(v___x_258521__overap_3764_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3278_ = v_a_3625_;
v___y_3279_ = v___x_3680_;
v___y_3280_ = v_a_3675_;
v___y_3281_ = v___x_3765_;
goto v___jp_3277_;
}
}
}
}
else
{
lean_object* v_a_3766_; 
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref(v___x_3653_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3766_ = lean_ctor_get(v___x_3681_, 0);
lean_inc(v_a_3766_);
lean_dec_ref_known(v___x_3681_, 1);
v___y_3253_ = v___x_3680_;
v___y_3254_ = v_a_3625_;
v___y_3255_ = v_a_3675_;
v_a_3256_ = v_a_3766_;
goto v___jp_3252_;
}
}
else
{
lean_object* v___x_3767_; lean_object* v___x_3768_; 
v___x_3767_ = lean_io_get_num_heartbeats();
v___x_3768_ = lp_aesop_Aesop_EqualUpToIds_equalCommonMVars_x3f___redArg(v_mvarId_u2081_2645_, v_mvarId_u2082_2646_, v_a_2647_);
if (lean_obj_tag(v___x_3768_) == 0)
{
lean_object* v_a_3769_; 
v_a_3769_ = lean_ctor_get(v___x_3768_, 0);
lean_inc(v_a_3769_);
lean_dec_ref_known(v___x_3768_, 1);
if (lean_obj_tag(v_a_3769_) == 1)
{
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref(v___x_3653_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (v___x_3234_ == 0)
{
lean_object* v_val_3770_; uint8_t v___x_3771_; 
v_val_3770_ = lean_ctor_get(v_a_3769_, 0);
lean_inc(v_val_3770_);
lean_dec_ref_known(v_a_3769_, 1);
v___x_3771_ = lean_unbox(v_val_3770_);
lean_dec(v_val_3770_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3771_;
goto v___jp_3447_;
}
else
{
lean_object* v_val_3772_; lean_object* v___x_3773_; uint8_t v___x_3774_; 
v_val_3772_ = lean_ctor_get(v_a_3769_, 0);
lean_inc(v_val_3772_);
lean_dec_ref_known(v_a_3769_, 1);
v___x_3773_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5);
v___x_3774_ = lean_unbox(v_val_3772_);
if (v___x_3774_ == 0)
{
lean_object* v___x_3775_; uint8_t v___x_3776_; 
v___x_3775_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6));
v___x_3776_ = lean_unbox(v_val_3772_);
lean_dec(v_val_3772_);
v___y_3455_ = v___x_3767_;
v___y_3456_ = v___x_3773_;
v___y_3457_ = v_a_3625_;
v___y_3458_ = v_a_3675_;
v___y_3459_ = v___x_3776_;
v___y_3460_ = v___x_3775_;
goto v___jp_3454_;
}
else
{
lean_object* v___x_3777_; uint8_t v___x_3778_; 
v___x_3777_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7));
v___x_3778_ = lean_unbox(v_val_3772_);
lean_dec(v_val_3772_);
v___y_3455_ = v___x_3767_;
v___y_3456_ = v___x_3773_;
v___y_3457_ = v_a_3625_;
v___y_3458_ = v_a_3675_;
v___y_3459_ = v___x_3778_;
v___y_3460_ = v___x_3777_;
goto v___jp_3454_;
}
}
}
else
{
lean_object* v___x_3779_; lean_object* v_equalMVarIds_3780_; lean_object* v___x_3781_; 
lean_dec(v_a_3769_);
v___x_3779_ = lean_st_ref_get(v_a_2648_);
v_equalMVarIds_3780_ = lean_ctor_get(v___x_3779_, 0);
lean_inc_ref(v_equalMVarIds_3780_);
lean_dec(v___x_3779_);
lean_inc(v_mvarId_u2081_2645_);
v___x_3781_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3780_, v_mvarId_u2081_2645_);
lean_dec_ref(v_equalMVarIds_3780_);
if (lean_obj_tag(v___x_3781_) == 1)
{
lean_object* v_val_3782_; uint8_t v___x_3783_; 
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2081_2645_);
v_val_3782_ = lean_ctor_get(v___x_3781_, 0);
lean_inc(v_val_3782_);
lean_dec_ref_known(v___x_3781_, 1);
v___x_3783_ = l_Lean_instBEqMVarId_beq(v_mvarId_u2082_2646_, v_val_3782_);
lean_dec(v_mvarId_u2082_2646_);
if (v___x_3783_ == 0)
{
lean_dec(v___x_3678_);
if (v___x_3234_ == 0)
{
lean_dec(v_val_3782_);
lean_dec_ref(v___x_3653_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3783_;
goto v___jp_3447_;
}
else
{
lean_object* v___x_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___x_3789_; lean_object* v___x_258592__overap_3790_; lean_object* v___x_3791_; 
v___x_3784_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9);
v___x_3785_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3785_, 0, v___x_3784_);
lean_ctor_set(v___x_3785_, 1, v___x_3653_);
v___x_3786_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11);
v___x_3787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3787_, 0, v___x_3785_);
lean_ctor_set(v___x_3787_, 1, v___x_3786_);
v___x_3788_ = l_Lean_MessageData_ofName(v_val_3782_);
v___x_3789_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3789_, 0, v___x_3787_);
lean_ctor_set(v___x_3789_, 1, v___x_3788_);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258592__overap_3790_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3789_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3791_ = lean_apply_7(v___x_258592__overap_3790_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3791_) == 0)
{
lean_dec_ref_known(v___x_3791_, 1);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3783_;
goto v___jp_3447_;
}
else
{
lean_object* v_a_3792_; 
v_a_3792_ = lean_ctor_get(v___x_3791_, 0);
lean_inc(v_a_3792_);
lean_dec_ref_known(v___x_3791_, 1);
v___y_3442_ = v___x_3767_;
v___y_3443_ = v_a_3625_;
v___y_3444_ = v_a_3675_;
v_a_3445_ = v_a_3792_;
goto v___jp_3441_;
}
}
}
else
{
lean_dec(v_val_3782_);
lean_dec_ref(v___x_3653_);
if (v___x_3234_ == 0)
{
uint8_t v___x_3793_; 
v___x_3793_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3793_;
goto v___jp_3447_;
}
else
{
lean_object* v___x_3794_; lean_object* v___x_258603__overap_3795_; lean_object* v___x_3796_; 
v___x_3794_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13);
lean_inc_ref(v_toMonadRef_2698_);
lean_inc_ref(v___x_2695_);
v___x_258603__overap_3795_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3794_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3796_ = lean_apply_7(v___x_258603__overap_3795_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
if (lean_obj_tag(v___x_3796_) == 0)
{
uint8_t v___x_3797_; 
lean_dec_ref_known(v___x_3796_, 1);
v___x_3797_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3797_;
goto v___jp_3447_;
}
else
{
lean_object* v_a_3798_; 
lean_dec(v___x_3678_);
v_a_3798_ = lean_ctor_get(v___x_3796_, 0);
lean_inc(v_a_3798_);
lean_dec_ref_known(v___x_3796_, 1);
v___y_3442_ = v___x_3767_;
v___y_3443_ = v_a_3625_;
v___y_3444_ = v_a_3675_;
v_a_3445_ = v_a_3798_;
goto v___jp_3441_;
}
}
}
}
else
{
lean_object* v_mctx_u2081_3799_; lean_object* v_mctx_u2082_3800_; lean_object* v_decls_3801_; lean_object* v___x_3802_; 
lean_dec(v___x_3781_);
v_mctx_u2081_3799_ = lean_ctor_get(v_a_2647_, 1);
v_mctx_u2082_3800_ = lean_ctor_get(v_a_2647_, 2);
v_decls_3801_ = lean_ctor_get(v_mctx_u2081_3799_, 5);
lean_inc(v_mvarId_u2081_2645_);
v___x_3802_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3801_, v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3802_) == 1)
{
lean_object* v_val_3803_; lean_object* v_decls_3804_; lean_object* v___x_3805_; 
lean_dec_ref(v___x_3653_);
v_val_3803_ = lean_ctor_get(v___x_3802_, 0);
lean_inc(v_val_3803_);
lean_dec_ref_known(v___x_3802_, 1);
v_decls_3804_ = lean_ctor_get(v_mctx_u2082_3800_, 5);
lean_inc(v_mvarId_u2082_2646_);
v___x_3805_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3804_, v_mvarId_u2082_2646_);
if (lean_obj_tag(v___x_3805_) == 1)
{
lean_object* v_val_3806_; lean_object* v_lctx_3807_; lean_object* v_type_3808_; lean_object* v_lctx_3809_; lean_object* v_type_3810_; lean_object* v_localInstances_3811_; lean_object* v___x_3812_; 
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_2703_, 3);
v_val_3806_ = lean_ctor_get(v___x_3805_, 0);
lean_inc(v_val_3806_);
lean_dec_ref_known(v___x_3805_, 1);
v_lctx_3807_ = lean_ctor_get(v_val_3803_, 1);
lean_inc_ref(v_lctx_3807_);
v_type_3808_ = lean_ctor_get(v_val_3803_, 2);
lean_inc_ref(v_type_3808_);
lean_dec(v_val_3803_);
v_lctx_3809_ = lean_ctor_get(v_val_3806_, 1);
lean_inc_ref(v_lctx_3809_);
v_type_3810_ = lean_ctor_get(v_val_3806_, 2);
lean_inc_ref(v_type_3810_);
v_localInstances_3811_ = lean_ctor_get(v_val_3806_, 4);
lean_inc_ref_n(v_localInstances_3811_, 2);
lean_dec(v_val_3806_);
v___x_3812_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(v_lctx_3807_, v_lctx_3809_, v_localInstances_3811_, v_localInstances_3811_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
if (lean_obj_tag(v___x_3812_) == 0)
{
lean_object* v_a_3813_; 
v_a_3813_ = lean_ctor_get(v___x_3812_, 0);
lean_inc(v_a_3813_);
lean_dec_ref_known(v___x_3812_, 1);
if (lean_obj_tag(v_a_3813_) == 1)
{
if (v___x_3234_ == 0)
{
lean_object* v_val_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; uint8_t v___x_3817_; 
v_val_3814_ = lean_ctor_get(v_a_3813_, 0);
lean_inc(v_val_3814_);
lean_dec_ref_known(v_a_3813_, 1);
v___x_3815_ = l_Lean_trace_profiler;
v___x_3816_ = l_Lean_Option_get___redArg(v___x_3676_, v_options_2704_, v___x_3815_);
v___x_3817_ = lean_unbox(v___x_3816_);
lean_dec(v___x_3816_);
if (v___x_3817_ == 0)
{
lean_object* v___x_3818_; 
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
v___x_3818_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v_type_3808_, v_type_3810_, v_val_3814_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_);
lean_dec(v_val_3814_);
if (lean_obj_tag(v___x_3818_) == 0)
{
lean_object* v_a_3819_; uint8_t v___x_3820_; 
v_a_3819_ = lean_ctor_get(v___x_3818_, 0);
lean_inc(v_a_3819_);
lean_dec_ref_known(v___x_3818_, 1);
v___x_3820_ = lean_unbox(v_a_3819_);
if (v___x_3820_ == 0)
{
uint8_t v___x_3821_; 
lean_dec(v___x_3678_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3821_ = lean_unbox(v_a_3819_);
lean_dec(v_a_3819_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3821_;
goto v___jp_3447_;
}
else
{
lean_object* v___x_3822_; lean_object* v_equalMVarIds_3823_; lean_object* v_equalLMVarIds_3824_; lean_object* v_leftUnassignedMVarValues_3825_; lean_object* v_rightUnassignedMVarValues_3826_; lean_object* v___x_3828_; uint8_t v_isShared_3829_; uint8_t v_isSharedCheck_3836_; 
lean_dec(v_a_3819_);
v___x_3822_ = lean_st_ref_take(v_a_2648_);
v_equalMVarIds_3823_ = lean_ctor_get(v___x_3822_, 0);
v_equalLMVarIds_3824_ = lean_ctor_get(v___x_3822_, 1);
v_leftUnassignedMVarValues_3825_ = lean_ctor_get(v___x_3822_, 2);
v_rightUnassignedMVarValues_3826_ = lean_ctor_get(v___x_3822_, 3);
v_isSharedCheck_3836_ = !lean_is_exclusive(v___x_3822_);
if (v_isSharedCheck_3836_ == 0)
{
v___x_3828_ = v___x_3822_;
v_isShared_3829_ = v_isSharedCheck_3836_;
goto v_resetjp_3827_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3826_);
lean_inc(v_leftUnassignedMVarValues_3825_);
lean_inc(v_equalLMVarIds_3824_);
lean_inc(v_equalMVarIds_3823_);
lean_dec(v___x_3822_);
v___x_3828_ = lean_box(0);
v_isShared_3829_ = v_isSharedCheck_3836_;
goto v_resetjp_3827_;
}
v_resetjp_3827_:
{
lean_object* v___x_3830_; lean_object* v___x_3832_; 
v___x_3830_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3823_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_3829_ == 0)
{
lean_ctor_set(v___x_3828_, 0, v___x_3830_);
v___x_3832_ = v___x_3828_;
goto v_reusejp_3831_;
}
else
{
lean_object* v_reuseFailAlloc_3835_; 
v_reuseFailAlloc_3835_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3835_, 0, v___x_3830_);
lean_ctor_set(v_reuseFailAlloc_3835_, 1, v_equalLMVarIds_3824_);
lean_ctor_set(v_reuseFailAlloc_3835_, 2, v_leftUnassignedMVarValues_3825_);
lean_ctor_set(v_reuseFailAlloc_3835_, 3, v_rightUnassignedMVarValues_3826_);
v___x_3832_ = v_reuseFailAlloc_3835_;
goto v_reusejp_3831_;
}
v_reusejp_3831_:
{
lean_object* v___x_3833_; uint8_t v___x_3834_; 
v___x_3833_ = lean_st_ref_set(v_a_2648_, v___x_3832_);
v___x_3834_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3834_;
goto v___jp_3447_;
}
}
}
}
else
{
lean_dec(v___x_3678_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___y_3467_ = v___x_3767_;
v___y_3468_ = v_a_3625_;
v___y_3469_ = v_a_3675_;
v___y_3470_ = v___x_3818_;
goto v___jp_3466_;
}
}
else
{
uint8_t v___x_3837_; 
v___x_3837_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3540_ = v___x_3234_;
v___y_3541_ = v___x_3837_;
v___y_3542_ = v___x_3767_;
v___y_3543_ = v___x_3668_;
v___y_3544_ = v___x_3669_;
v___y_3545_ = v_type_3808_;
v___y_3546_ = v___f_3671_;
v___y_3547_ = v___x_3670_;
v___y_3548_ = v_a_3625_;
v___y_3549_ = v_a_3675_;
v___y_3550_ = v___x_3651_;
v___y_3551_ = v_val_3814_;
v___y_3552_ = v___x_3672_;
v___y_3553_ = v_type_3810_;
goto v___jp_3539_;
}
}
else
{
lean_object* v_val_3838_; uint8_t v___x_3839_; 
v_val_3838_ = lean_ctor_get(v_a_3813_, 0);
lean_inc(v_val_3838_);
lean_dec_ref_known(v_a_3813_, 1);
v___x_3839_ = lean_unbox(v___x_3678_);
lean_dec(v___x_3678_);
v___y_3540_ = v___x_3234_;
v___y_3541_ = v___x_3839_;
v___y_3542_ = v___x_3767_;
v___y_3543_ = v___x_3668_;
v___y_3544_ = v___x_3669_;
v___y_3545_ = v_type_3808_;
v___y_3546_ = v___f_3671_;
v___y_3547_ = v___x_3670_;
v___y_3548_ = v_a_3625_;
v___y_3549_ = v_a_3675_;
v___y_3550_ = v___x_3651_;
v___y_3551_ = v_val_3838_;
v___y_3552_ = v___x_3672_;
v___y_3553_ = v_type_3810_;
goto v___jp_3539_;
}
}
else
{
uint8_t v___x_3840_; 
lean_dec(v_a_3813_);
lean_dec_ref(v_type_3810_);
lean_dec_ref(v_type_3808_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3840_ = 0;
v___y_3448_ = v___x_3767_;
v___y_3449_ = v_a_3625_;
v___y_3450_ = v_a_3675_;
v_a_3451_ = v___x_3840_;
goto v___jp_3447_;
}
}
else
{
lean_object* v_a_3841_; 
lean_dec_ref(v_type_3810_);
lean_dec_ref(v_type_3808_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3841_ = lean_ctor_get(v___x_3812_, 0);
lean_inc(v_a_3841_);
lean_dec_ref_known(v___x_3812_, 1);
v___y_3442_ = v___x_3767_;
v___y_3443_ = v_a_3625_;
v___y_3444_ = v_a_3675_;
v_a_3445_ = v_a_3841_;
goto v___jp_3441_;
}
}
else
{
lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v___x_3844_; lean_object* v___x_3845_; lean_object* v___x_258746__overap_3846_; lean_object* v___x_3847_; 
lean_dec(v___x_3805_);
lean_dec(v_val_3803_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3842_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3843_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3843_, 0, v___x_3842_);
lean_ctor_set(v___x_3843_, 1, v___x_3657_);
v___x_3844_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3845_, 0, v___x_3843_);
lean_ctor_set(v___x_3845_, 1, v___x_3844_);
lean_inc_ref(v___x_2695_);
v___x_258746__overap_3846_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3845_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3847_ = lean_apply_7(v___x_258746__overap_3846_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3467_ = v___x_3767_;
v___y_3468_ = v_a_3625_;
v___y_3469_ = v_a_3675_;
v___y_3470_ = v___x_3847_;
goto v___jp_3466_;
}
}
else
{
lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_258753__overap_3852_; lean_object* v___x_3853_; 
lean_dec(v___x_3802_);
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3848_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3849_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3849_, 0, v___x_3848_);
lean_ctor_set(v___x_3849_, 1, v___x_3653_);
v___x_3850_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3851_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3851_, 0, v___x_3849_);
lean_ctor_set(v___x_3851_, 1, v___x_3850_);
lean_inc_ref(v___x_2695_);
v___x_258753__overap_3852_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3851_);
lean_inc(v_a_2652_);
lean_inc_ref(v_a_2651_);
lean_inc(v_a_2650_);
lean_inc_ref(v_a_2649_);
lean_inc(v_a_2648_);
lean_inc_ref(v_a_2647_);
v___x_3853_ = lean_apply_7(v___x_258753__overap_3852_, v_a_2647_, v_a_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, lean_box(0));
v___y_3467_ = v___x_3767_;
v___y_3468_ = v_a_3625_;
v___y_3469_ = v_a_3675_;
v___y_3470_ = v___x_3853_;
goto v___jp_3466_;
}
}
}
}
else
{
lean_object* v_a_3854_; 
lean_dec(v___x_3678_);
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref(v___x_3653_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3854_ = lean_ctor_get(v___x_3768_, 0);
lean_inc(v_a_3854_);
lean_dec_ref_known(v___x_3768_, 1);
v___y_3442_ = v___x_3767_;
v___y_3443_ = v_a_3625_;
v___y_3444_ = v_a_3675_;
v_a_3445_ = v_a_3854_;
goto v___jp_3441_;
}
}
}
else
{
lean_object* v_a_3855_; lean_object* v___x_3857_; uint8_t v_isShared_3858_; uint8_t v_isSharedCheck_3862_; 
lean_dec_ref(v___x_3668_);
lean_dec_ref(v___x_3657_);
lean_dec_ref(v___x_3653_);
lean_dec_ref_known(v___x_3651_, 14);
lean_dec(v_a_3625_);
lean_dec_ref_known(v___x_2703_, 3);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3855_ = lean_ctor_get(v___x_3674_, 0);
v_isSharedCheck_3862_ = !lean_is_exclusive(v___x_3674_);
if (v_isSharedCheck_3862_ == 0)
{
v___x_3857_ = v___x_3674_;
v_isShared_3858_ = v_isSharedCheck_3862_;
goto v_resetjp_3856_;
}
else
{
lean_inc(v_a_3855_);
lean_dec(v___x_3674_);
v___x_3857_ = lean_box(0);
v_isShared_3858_ = v_isSharedCheck_3862_;
goto v_resetjp_3856_;
}
v_resetjp_3856_:
{
lean_object* v___x_3860_; 
if (v_isShared_3858_ == 0)
{
v___x_3860_ = v___x_3857_;
goto v_reusejp_3859_;
}
else
{
lean_object* v_reuseFailAlloc_3861_; 
v_reuseFailAlloc_3861_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3861_, 0, v_a_3855_);
v___x_3860_ = v_reuseFailAlloc_3861_;
goto v_reusejp_3859_;
}
v_reusejp_3859_:
{
return v___x_3860_;
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
lean_object* v_a_3869_; lean_object* v___x_3871_; uint8_t v_isShared_3872_; uint8_t v_isSharedCheck_3876_; 
lean_dec_ref_known(v___x_2703_, 3);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3869_ = lean_ctor_get(v___x_3623_, 0);
v_isSharedCheck_3876_ = !lean_is_exclusive(v___x_3623_);
if (v_isSharedCheck_3876_ == 0)
{
v___x_3871_ = v___x_3623_;
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
else
{
lean_inc(v_a_3869_);
lean_dec(v___x_3623_);
v___x_3871_ = lean_box(0);
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
v_resetjp_3870_:
{
lean_object* v___x_3874_; 
if (v_isShared_3872_ == 0)
{
v___x_3874_ = v___x_3871_;
goto v_reusejp_3873_;
}
else
{
lean_object* v_reuseFailAlloc_3875_; 
v_reuseFailAlloc_3875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3875_, 0, v_a_3869_);
v___x_3874_ = v_reuseFailAlloc_3875_;
goto v_reusejp_3873_;
}
v_reusejp_3873_:
{
return v___x_3874_;
}
}
}
}
}
v___jp_2724_:
{
if (v_____do__lift_2725_ == 0)
{
lean_object* v___x_2727_; lean_object* v___x_2728_; 
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_2727_ = lean_box(v_____do__lift_2725_);
v___x_2728_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2728_, 0, v___x_2727_);
return v___x_2728_;
}
else
{
lean_object* v___x_2729_; lean_object* v_equalMVarIds_2730_; lean_object* v_equalLMVarIds_2731_; lean_object* v_leftUnassignedMVarValues_2732_; lean_object* v_rightUnassignedMVarValues_2733_; lean_object* v___x_2735_; uint8_t v_isShared_2736_; uint8_t v_isSharedCheck_2744_; 
v___x_2729_ = lean_st_ref_take(v___y_2726_);
v_equalMVarIds_2730_ = lean_ctor_get(v___x_2729_, 0);
v_equalLMVarIds_2731_ = lean_ctor_get(v___x_2729_, 1);
v_leftUnassignedMVarValues_2732_ = lean_ctor_get(v___x_2729_, 2);
v_rightUnassignedMVarValues_2733_ = lean_ctor_get(v___x_2729_, 3);
v_isSharedCheck_2744_ = !lean_is_exclusive(v___x_2729_);
if (v_isSharedCheck_2744_ == 0)
{
v___x_2735_ = v___x_2729_;
v_isShared_2736_ = v_isSharedCheck_2744_;
goto v_resetjp_2734_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_2733_);
lean_inc(v_leftUnassignedMVarValues_2732_);
lean_inc(v_equalLMVarIds_2731_);
lean_inc(v_equalMVarIds_2730_);
lean_dec(v___x_2729_);
v___x_2735_ = lean_box(0);
v_isShared_2736_ = v_isSharedCheck_2744_;
goto v_resetjp_2734_;
}
v_resetjp_2734_:
{
lean_object* v___x_2737_; lean_object* v___x_2739_; 
v___x_2737_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_2730_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_2736_ == 0)
{
lean_ctor_set(v___x_2735_, 0, v___x_2737_);
v___x_2739_ = v___x_2735_;
goto v_reusejp_2738_;
}
else
{
lean_object* v_reuseFailAlloc_2743_; 
v_reuseFailAlloc_2743_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2743_, 0, v___x_2737_);
lean_ctor_set(v_reuseFailAlloc_2743_, 1, v_equalLMVarIds_2731_);
lean_ctor_set(v_reuseFailAlloc_2743_, 2, v_leftUnassignedMVarValues_2732_);
lean_ctor_set(v_reuseFailAlloc_2743_, 3, v_rightUnassignedMVarValues_2733_);
v___x_2739_ = v_reuseFailAlloc_2743_;
goto v_reusejp_2738_;
}
v_reusejp_2738_:
{
lean_object* v___x_2740_; lean_object* v___x_2741_; lean_object* v___x_2742_; 
v___x_2740_ = lean_st_ref_set(v___y_2726_, v___x_2739_);
v___x_2741_ = lean_box(v_____do__lift_2725_);
v___x_2742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2742_, 0, v___x_2741_);
return v___x_2742_;
}
}
}
}
v___jp_2746_:
{
lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_245730__overap_2758_; lean_object* v___x_2759_; 
lean_inc_ref(v___y_2755_);
v___x_2756_ = l_Lean_stringToMessageData(v___y_2755_);
lean_inc_ref(v___y_2749_);
v___x_2757_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2757_, 0, v___y_2749_);
lean_ctor_set(v___x_2757_, 1, v___x_2756_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_245730__overap_2758_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_2757_);
lean_inc(v___y_2752_);
lean_inc_ref(v___y_2754_);
lean_inc(v___y_2750_);
lean_inc_ref(v___y_2753_);
lean_inc(v___y_2747_);
lean_inc_ref(v___y_2751_);
v___x_2759_ = lean_apply_7(v___x_245730__overap_2758_, v___y_2751_, v___y_2747_, v___y_2753_, v___y_2750_, v___y_2754_, v___y_2752_, lean_box(0));
if (lean_obj_tag(v___x_2759_) == 0)
{
lean_object* v___x_2761_; uint8_t v_isShared_2762_; uint8_t v_isSharedCheck_2767_; 
v_isSharedCheck_2767_ = !lean_is_exclusive(v___x_2759_);
if (v_isSharedCheck_2767_ == 0)
{
lean_object* v_unused_2768_; 
v_unused_2768_ = lean_ctor_get(v___x_2759_, 0);
lean_dec(v_unused_2768_);
v___x_2761_ = v___x_2759_;
v_isShared_2762_ = v_isSharedCheck_2767_;
goto v_resetjp_2760_;
}
else
{
lean_dec(v___x_2759_);
v___x_2761_ = lean_box(0);
v_isShared_2762_ = v_isSharedCheck_2767_;
goto v_resetjp_2760_;
}
v_resetjp_2760_:
{
lean_object* v___x_2763_; lean_object* v___x_2765_; 
v___x_2763_ = lean_box(v___y_2748_);
if (v_isShared_2762_ == 0)
{
lean_ctor_set(v___x_2761_, 0, v___x_2763_);
v___x_2765_ = v___x_2761_;
goto v_reusejp_2764_;
}
else
{
lean_object* v_reuseFailAlloc_2766_; 
v_reuseFailAlloc_2766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2766_, 0, v___x_2763_);
v___x_2765_ = v_reuseFailAlloc_2766_;
goto v_reusejp_2764_;
}
v_reusejp_2764_:
{
return v___x_2765_;
}
}
}
else
{
lean_object* v_a_2769_; lean_object* v___x_2771_; uint8_t v_isShared_2772_; uint8_t v_isSharedCheck_2776_; 
v_a_2769_ = lean_ctor_get(v___x_2759_, 0);
v_isSharedCheck_2776_ = !lean_is_exclusive(v___x_2759_);
if (v_isSharedCheck_2776_ == 0)
{
v___x_2771_ = v___x_2759_;
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
else
{
lean_inc(v_a_2769_);
lean_dec(v___x_2759_);
v___x_2771_ = lean_box(0);
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
v_resetjp_2770_:
{
lean_object* v___x_2774_; 
if (v_isShared_2772_ == 0)
{
v___x_2774_ = v___x_2771_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2775_; 
v_reuseFailAlloc_2775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2775_, 0, v_a_2769_);
v___x_2774_ = v_reuseFailAlloc_2775_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
return v___x_2774_;
}
}
}
}
v___jp_2777_:
{
lean_object* v___x_2793_; double v___x_2794_; double v___x_2795_; double v___x_2796_; double v___x_2797_; double v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_258098__overap_2803_; lean_object* v___x_2804_; 
v___x_2793_ = lean_io_mono_nanos_now();
v___x_2794_ = lean_float_of_nat(v___y_2778_);
v___x_2795_ = lean_float_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__1);
v___x_2796_ = lean_float_div(v___x_2794_, v___x_2795_);
v___x_2797_ = lean_float_of_nat(v___x_2793_);
v___x_2798_ = lean_float_div(v___x_2797_, v___x_2795_);
v___x_2799_ = lean_box_float(v___x_2796_);
v___x_2800_ = lean_box_float(v___x_2798_);
v___x_2801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2801_, 0, v___x_2799_);
lean_ctor_set(v___x_2801_, 1, v___x_2800_);
v___x_2802_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2802_, 0, v_a_2792_);
lean_ctor_set(v___x_2802_, 1, v___x_2801_);
lean_inc(v___y_2781_);
lean_inc_ref(v___y_2790_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_258098__overap_2803_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v___y_2786_, v___y_2790_, v___y_2788_, v___y_2791_, v___y_2783_, v___y_2781_, v___y_2784_, v___x_2802_);
lean_inc(v___y_2780_);
lean_inc_ref(v___y_2789_);
lean_inc(v___y_2785_);
lean_inc_ref(v___y_2787_);
lean_inc(v___y_2782_);
lean_inc_ref(v___y_2779_);
v___x_2804_ = lean_apply_7(v___x_258098__overap_2803_, v___y_2779_, v___y_2782_, v___y_2787_, v___y_2785_, v___y_2789_, v___y_2780_, lean_box(0));
return v___x_2804_;
}
v___jp_2805_:
{
lean_object* v___x_2821_; lean_object* v___x_2822_; 
v___x_2821_ = lean_box(v_a_2820_);
v___x_2822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2822_, 0, v___x_2821_);
v___y_2778_ = v___y_2806_;
v___y_2779_ = v___y_2807_;
v___y_2780_ = v___y_2808_;
v___y_2781_ = v___y_2809_;
v___y_2782_ = v___y_2810_;
v___y_2783_ = v___y_2811_;
v___y_2784_ = v___y_2812_;
v___y_2785_ = v___y_2813_;
v___y_2786_ = v___y_2814_;
v___y_2787_ = v___y_2815_;
v___y_2788_ = v___y_2816_;
v___y_2789_ = v___y_2817_;
v___y_2790_ = v___y_2819_;
v___y_2791_ = v___y_2818_;
v_a_2792_ = v___x_2822_;
goto v___jp_2777_;
}
v___jp_2823_:
{
lean_object* v___x_2839_; double v___x_2840_; double v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_258138__overap_2846_; lean_object* v___x_2847_; 
v___x_2839_ = lean_io_get_num_heartbeats();
v___x_2840_ = lean_float_of_nat(v___y_2827_);
v___x_2841_ = lean_float_of_nat(v___x_2839_);
v___x_2842_ = lean_box_float(v___x_2840_);
v___x_2843_ = lean_box_float(v___x_2841_);
v___x_2844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2844_, 0, v___x_2842_);
lean_ctor_set(v___x_2844_, 1, v___x_2843_);
v___x_2845_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2845_, 0, v_a_2838_);
lean_ctor_set(v___x_2845_, 1, v___x_2844_);
lean_inc(v___y_2826_);
lean_inc_ref(v___y_2836_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_258138__overap_2846_ = l___private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback(lean_box(0), lean_box(0), v___x_2695_, v___x_2696_, lean_box(0), v_toMonadRef_2698_, v___f_2699_, v___x_2700_, v___f_2721_, v_cls_2745_, v___y_2832_, v___y_2836_, v___y_2834_, v___y_2837_, v___y_2829_, v___y_2826_, v___y_2830_, v___x_2845_);
lean_inc(v___y_2825_);
lean_inc_ref(v___y_2835_);
lean_inc(v___y_2831_);
lean_inc_ref(v___y_2833_);
lean_inc(v___y_2828_);
lean_inc_ref(v___y_2824_);
v___x_2847_ = lean_apply_7(v___x_258138__overap_2846_, v___y_2824_, v___y_2828_, v___y_2833_, v___y_2831_, v___y_2835_, v___y_2825_, lean_box(0));
return v___x_2847_;
}
v___jp_2848_:
{
lean_object* v___x_2864_; lean_object* v___x_2865_; 
v___x_2864_ = lean_box(v_a_2863_);
v___x_2865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2865_, 0, v___x_2864_);
v___y_2824_ = v___y_2849_;
v___y_2825_ = v___y_2850_;
v___y_2826_ = v___y_2851_;
v___y_2827_ = v___y_2852_;
v___y_2828_ = v___y_2853_;
v___y_2829_ = v___y_2854_;
v___y_2830_ = v___y_2855_;
v___y_2831_ = v___y_2856_;
v___y_2832_ = v___y_2857_;
v___y_2833_ = v___y_2858_;
v___y_2834_ = v___y_2859_;
v___y_2835_ = v___y_2860_;
v___y_2836_ = v___y_2862_;
v___y_2837_ = v___y_2861_;
v_a_2838_ = v___x_2865_;
goto v___jp_2823_;
}
v___jp_2866_:
{
lean_object* v___x_258025__overap_2895_; lean_object* v___x_2896_; 
lean_inc_ref(v___x_2695_);
v___x_258025__overap_2895_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2695_, v___x_2696_);
lean_inc(v___y_2880_);
lean_inc_ref(v___y_2890_);
lean_inc(v___y_2887_);
lean_inc_ref(v___y_2888_);
lean_inc(v___y_2883_);
lean_inc_ref(v___y_2870_);
v___x_2896_ = lean_apply_7(v___x_258025__overap_2895_, v___y_2870_, v___y_2883_, v___y_2888_, v___y_2887_, v___y_2890_, v___y_2880_, lean_box(0));
if (lean_obj_tag(v___x_2896_) == 0)
{
lean_object* v_toApplicative_2897_; lean_object* v_a_2898_; lean_object* v_toFunctor_2899_; lean_object* v_toSeq_2900_; lean_object* v_toSeqLeft_2901_; lean_object* v_toSeqRight_2902_; lean_object* v___f_2903_; lean_object* v___f_2904_; lean_object* v___x_2905_; lean_object* v___f_2906_; lean_object* v___f_2907_; lean_object* v___f_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; lean_object* v_toApplicative_2912_; lean_object* v___x_2914_; uint8_t v_isShared_2915_; uint8_t v_isSharedCheck_3018_; 
v_toApplicative_2897_ = lean_ctor_get(v___x_2654_, 0);
v_a_2898_ = lean_ctor_get(v___x_2896_, 0);
lean_inc(v_a_2898_);
lean_dec_ref_known(v___x_2896_, 1);
v_toFunctor_2899_ = lean_ctor_get(v_toApplicative_2897_, 0);
v_toSeq_2900_ = lean_ctor_get(v_toApplicative_2897_, 2);
v_toSeqLeft_2901_ = lean_ctor_get(v_toApplicative_2897_, 3);
v_toSeqRight_2902_ = lean_ctor_get(v_toApplicative_2897_, 4);
lean_inc_ref_n(v_toFunctor_2899_, 2);
v___f_2903_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2903_, 0, v_toFunctor_2899_);
v___f_2904_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2904_, 0, v_toFunctor_2899_);
v___x_2905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2905_, 0, v___f_2903_);
lean_ctor_set(v___x_2905_, 1, v___f_2904_);
lean_inc(v_toSeqRight_2902_);
v___f_2906_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2906_, 0, v_toSeqRight_2902_);
lean_inc(v_toSeqLeft_2901_);
v___f_2907_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2907_, 0, v_toSeqLeft_2901_);
lean_inc(v_toSeq_2900_);
v___f_2908_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2908_, 0, v_toSeq_2900_);
v___x_2909_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2909_, 0, v___x_2905_);
lean_ctor_set(v___x_2909_, 1, v___f_2660_);
lean_ctor_set(v___x_2909_, 2, v___f_2908_);
lean_ctor_set(v___x_2909_, 3, v___f_2907_);
lean_ctor_set(v___x_2909_, 4, v___f_2906_);
v___x_2910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2910_, 0, v___x_2909_);
lean_ctor_set(v___x_2910_, 1, v___f_2661_);
v___x_2911_ = l_StateRefT_x27_instMonad___redArg(v___x_2910_);
v_toApplicative_2912_ = lean_ctor_get(v___x_2911_, 0);
v_isSharedCheck_3018_ = !lean_is_exclusive(v___x_2911_);
if (v_isSharedCheck_3018_ == 0)
{
lean_object* v_unused_3019_; 
v_unused_3019_ = lean_ctor_get(v___x_2911_, 1);
lean_dec(v_unused_3019_);
v___x_2914_ = v___x_2911_;
v_isShared_2915_ = v_isSharedCheck_3018_;
goto v_resetjp_2913_;
}
else
{
lean_inc(v_toApplicative_2912_);
lean_dec(v___x_2911_);
v___x_2914_ = lean_box(0);
v_isShared_2915_ = v_isSharedCheck_3018_;
goto v_resetjp_2913_;
}
v_resetjp_2913_:
{
lean_object* v_toFunctor_2916_; lean_object* v_toSeq_2917_; lean_object* v_toSeqLeft_2918_; lean_object* v_toSeqRight_2919_; lean_object* v___x_2921_; uint8_t v_isShared_2922_; uint8_t v_isSharedCheck_3016_; 
v_toFunctor_2916_ = lean_ctor_get(v_toApplicative_2912_, 0);
v_toSeq_2917_ = lean_ctor_get(v_toApplicative_2912_, 2);
v_toSeqLeft_2918_ = lean_ctor_get(v_toApplicative_2912_, 3);
v_toSeqRight_2919_ = lean_ctor_get(v_toApplicative_2912_, 4);
v_isSharedCheck_3016_ = !lean_is_exclusive(v_toApplicative_2912_);
if (v_isSharedCheck_3016_ == 0)
{
lean_object* v_unused_3017_; 
v_unused_3017_ = lean_ctor_get(v_toApplicative_2912_, 1);
lean_dec(v_unused_3017_);
v___x_2921_ = v_toApplicative_2912_;
v_isShared_2922_ = v_isSharedCheck_3016_;
goto v_resetjp_2920_;
}
else
{
lean_inc(v_toSeqRight_2919_);
lean_inc(v_toSeqLeft_2918_);
lean_inc(v_toSeq_2917_);
lean_inc(v_toFunctor_2916_);
lean_dec(v_toApplicative_2912_);
v___x_2921_ = lean_box(0);
v_isShared_2922_ = v_isSharedCheck_3016_;
goto v_resetjp_2920_;
}
v_resetjp_2920_:
{
lean_object* v_ref_2923_; lean_object* v___x_2924_; lean_object* v___x_2925_; lean_object* v___f_2926_; lean_object* v___f_2927_; lean_object* v___x_2928_; lean_object* v___f_2929_; lean_object* v___f_2930_; lean_object* v___f_2931_; lean_object* v___x_2933_; 
v_ref_2923_ = l_Lean_replaceRef(v___y_2871_, v___y_2871_);
lean_inc_ref(v___y_2873_);
lean_inc(v___y_2872_);
lean_inc(v___y_2881_);
lean_inc(v___y_2891_);
lean_inc(v___y_2885_);
lean_inc(v___y_2894_);
lean_inc(v___y_2884_);
lean_inc(v___y_2879_);
lean_inc(v___y_2875_);
lean_inc(v___y_2886_);
lean_inc_ref(v___y_2889_);
lean_inc_ref(v___y_2868_);
lean_inc_ref(v___y_2876_);
v___x_2924_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2924_, 0, v___y_2876_);
lean_ctor_set(v___x_2924_, 1, v___y_2868_);
lean_ctor_set(v___x_2924_, 2, v___y_2889_);
lean_ctor_set(v___x_2924_, 3, v___y_2886_);
lean_ctor_set(v___x_2924_, 4, v___y_2875_);
lean_ctor_set(v___x_2924_, 5, v_ref_2923_);
lean_ctor_set(v___x_2924_, 6, v___y_2879_);
lean_ctor_set(v___x_2924_, 7, v___y_2884_);
lean_ctor_set(v___x_2924_, 8, v___y_2894_);
lean_ctor_set(v___x_2924_, 9, v___y_2885_);
lean_ctor_set(v___x_2924_, 10, v___y_2891_);
lean_ctor_set(v___x_2924_, 11, v___y_2881_);
lean_ctor_set(v___x_2924_, 12, v___y_2872_);
lean_ctor_set(v___x_2924_, 13, v___y_2873_);
lean_ctor_set_uint8(v___x_2924_, sizeof(void*)*14, v___y_2869_);
lean_ctor_set_uint8(v___x_2924_, sizeof(void*)*14 + 1, v___y_2867_);
v___x_2925_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__3);
lean_inc_ref(v_toFunctor_2916_);
v___f_2926_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2926_, 0, v_toFunctor_2916_);
v___f_2927_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2927_, 0, v_toFunctor_2916_);
v___x_2928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2928_, 0, v___f_2926_);
lean_ctor_set(v___x_2928_, 1, v___f_2927_);
v___f_2929_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2929_, 0, v_toSeqRight_2919_);
v___f_2930_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2930_, 0, v_toSeqLeft_2918_);
v___f_2931_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2931_, 0, v_toSeq_2917_);
if (v_isShared_2922_ == 0)
{
lean_ctor_set(v___x_2921_, 4, v___f_2929_);
lean_ctor_set(v___x_2921_, 3, v___f_2930_);
lean_ctor_set(v___x_2921_, 2, v___f_2931_);
lean_ctor_set(v___x_2921_, 1, v___f_2682_);
lean_ctor_set(v___x_2921_, 0, v___x_2928_);
v___x_2933_ = v___x_2921_;
goto v_reusejp_2932_;
}
else
{
lean_object* v_reuseFailAlloc_3015_; 
v_reuseFailAlloc_3015_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3015_, 0, v___x_2928_);
lean_ctor_set(v_reuseFailAlloc_3015_, 1, v___f_2682_);
lean_ctor_set(v_reuseFailAlloc_3015_, 2, v___f_2931_);
lean_ctor_set(v_reuseFailAlloc_3015_, 3, v___f_2930_);
lean_ctor_set(v_reuseFailAlloc_3015_, 4, v___f_2929_);
v___x_2933_ = v_reuseFailAlloc_3015_;
goto v_reusejp_2932_;
}
v_reusejp_2932_:
{
lean_object* v___x_2935_; 
if (v_isShared_2915_ == 0)
{
lean_ctor_set(v___x_2914_, 1, v___f_2683_);
lean_ctor_set(v___x_2914_, 0, v___x_2933_);
v___x_2935_ = v___x_2914_;
goto v_reusejp_2934_;
}
else
{
lean_object* v_reuseFailAlloc_3014_; 
v_reuseFailAlloc_3014_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3014_, 0, v___x_2933_);
lean_ctor_set(v_reuseFailAlloc_3014_, 1, v___f_2683_);
v___x_2935_ = v_reuseFailAlloc_3014_;
goto v_reusejp_2934_;
}
v_reusejp_2934_:
{
lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___f_2938_; lean_object* v___x_2939_; lean_object* v___x_258075__overap_2940_; lean_object* v___x_2941_; 
v___x_2936_ = l_Lean_Meta_instMonadEnvMetaM;
v___x_2937_ = l_Lean_Meta_instMonadMCtxMetaM;
v___f_2938_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__27));
v___x_2939_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___closed__30));
v___x_258075__overap_2940_ = l_Lean_addMessageContextFull___redArg(v___x_2935_, v___x_2936_, v___x_2937_, v___f_2938_, v___x_2939_, v___x_2925_);
lean_inc(v___y_2880_);
lean_inc(v___y_2887_);
lean_inc_ref(v___y_2888_);
v___x_2941_ = lean_apply_5(v___x_258075__overap_2940_, v___y_2888_, v___y_2887_, v___x_2924_, v___y_2880_, lean_box(0));
if (lean_obj_tag(v___x_2941_) == 0)
{
lean_object* v_a_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; uint8_t v___x_2946_; 
v_a_2942_ = lean_ctor_get(v___x_2941_, 0);
lean_inc(v_a_2942_);
lean_dec_ref_known(v___x_2941_, 1);
v___x_2943_ = l_Lean_KVMap_instValueBool;
v___x_2944_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2945_ = l_Lean_Option_get___redArg(v___x_2943_, v___y_2889_, v___x_2944_);
v___x_2946_ = lean_unbox(v___x_2945_);
if (v___x_2946_ == 0)
{
lean_object* v___x_2947_; lean_object* v___x_2948_; 
v___x_2947_ = lean_io_mono_nanos_now();
v___x_2948_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_2877_, v___y_2882_, v___y_2878_, v___y_2870_, v___y_2883_, v___y_2888_, v___y_2887_, v___y_2890_, v___y_2880_);
lean_dec_ref(v___y_2878_);
if (lean_obj_tag(v___x_2948_) == 0)
{
lean_object* v_a_2949_; uint8_t v___x_2950_; 
v_a_2949_ = lean_ctor_get(v___x_2948_, 0);
lean_inc(v_a_2949_);
lean_dec_ref_known(v___x_2948_, 1);
v___x_2950_ = lean_unbox(v_a_2949_);
lean_dec(v_a_2949_);
if (v___x_2950_ == 0)
{
uint8_t v___x_2951_; 
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_2951_ = lean_unbox(v___x_2945_);
lean_dec(v___x_2945_);
v___y_2806_ = v___x_2947_;
v___y_2807_ = v___y_2870_;
v___y_2808_ = v___y_2880_;
v___y_2809_ = v___y_2871_;
v___y_2810_ = v___y_2883_;
v___y_2811_ = v_a_2898_;
v___y_2812_ = v_a_2942_;
v___y_2813_ = v___y_2887_;
v___y_2814_ = v___y_2874_;
v___y_2815_ = v___y_2888_;
v___y_2816_ = v___y_2889_;
v___y_2817_ = v___y_2890_;
v___y_2818_ = v___y_2892_;
v___y_2819_ = v___y_2893_;
v_a_2820_ = v___x_2951_;
goto v___jp_2805_;
}
else
{
lean_object* v___x_2952_; lean_object* v_equalMVarIds_2953_; lean_object* v_equalLMVarIds_2954_; lean_object* v_leftUnassignedMVarValues_2955_; lean_object* v_rightUnassignedMVarValues_2956_; lean_object* v___x_2958_; uint8_t v_isShared_2959_; uint8_t v_isSharedCheck_2965_; 
lean_dec(v___x_2945_);
v___x_2952_ = lean_st_ref_take(v___y_2883_);
v_equalMVarIds_2953_ = lean_ctor_get(v___x_2952_, 0);
v_equalLMVarIds_2954_ = lean_ctor_get(v___x_2952_, 1);
v_leftUnassignedMVarValues_2955_ = lean_ctor_get(v___x_2952_, 2);
v_rightUnassignedMVarValues_2956_ = lean_ctor_get(v___x_2952_, 3);
v_isSharedCheck_2965_ = !lean_is_exclusive(v___x_2952_);
if (v_isSharedCheck_2965_ == 0)
{
v___x_2958_ = v___x_2952_;
v_isShared_2959_ = v_isSharedCheck_2965_;
goto v_resetjp_2957_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_2956_);
lean_inc(v_leftUnassignedMVarValues_2955_);
lean_inc(v_equalLMVarIds_2954_);
lean_inc(v_equalMVarIds_2953_);
lean_dec(v___x_2952_);
v___x_2958_ = lean_box(0);
v_isShared_2959_ = v_isSharedCheck_2965_;
goto v_resetjp_2957_;
}
v_resetjp_2957_:
{
lean_object* v___x_2960_; lean_object* v___x_2962_; 
v___x_2960_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_2953_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_2959_ == 0)
{
lean_ctor_set(v___x_2958_, 0, v___x_2960_);
v___x_2962_ = v___x_2958_;
goto v_reusejp_2961_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v___x_2960_);
lean_ctor_set(v_reuseFailAlloc_2964_, 1, v_equalLMVarIds_2954_);
lean_ctor_set(v_reuseFailAlloc_2964_, 2, v_leftUnassignedMVarValues_2955_);
lean_ctor_set(v_reuseFailAlloc_2964_, 3, v_rightUnassignedMVarValues_2956_);
v___x_2962_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2961_;
}
v_reusejp_2961_:
{
lean_object* v___x_2963_; 
v___x_2963_ = lean_st_ref_set(v___y_2883_, v___x_2962_);
v___y_2806_ = v___x_2947_;
v___y_2807_ = v___y_2870_;
v___y_2808_ = v___y_2880_;
v___y_2809_ = v___y_2871_;
v___y_2810_ = v___y_2883_;
v___y_2811_ = v_a_2898_;
v___y_2812_ = v_a_2942_;
v___y_2813_ = v___y_2887_;
v___y_2814_ = v___y_2874_;
v___y_2815_ = v___y_2888_;
v___y_2816_ = v___y_2889_;
v___y_2817_ = v___y_2890_;
v___y_2818_ = v___y_2892_;
v___y_2819_ = v___y_2893_;
v_a_2820_ = v___y_2874_;
goto v___jp_2805_;
}
}
}
}
else
{
lean_dec(v___x_2945_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_2948_) == 0)
{
lean_object* v_a_2966_; uint8_t v___x_2967_; 
v_a_2966_ = lean_ctor_get(v___x_2948_, 0);
lean_inc(v_a_2966_);
lean_dec_ref_known(v___x_2948_, 1);
v___x_2967_ = lean_unbox(v_a_2966_);
lean_dec(v_a_2966_);
v___y_2806_ = v___x_2947_;
v___y_2807_ = v___y_2870_;
v___y_2808_ = v___y_2880_;
v___y_2809_ = v___y_2871_;
v___y_2810_ = v___y_2883_;
v___y_2811_ = v_a_2898_;
v___y_2812_ = v_a_2942_;
v___y_2813_ = v___y_2887_;
v___y_2814_ = v___y_2874_;
v___y_2815_ = v___y_2888_;
v___y_2816_ = v___y_2889_;
v___y_2817_ = v___y_2890_;
v___y_2818_ = v___y_2892_;
v___y_2819_ = v___y_2893_;
v_a_2820_ = v___x_2967_;
goto v___jp_2805_;
}
else
{
lean_object* v_a_2968_; lean_object* v___x_2970_; uint8_t v_isShared_2971_; uint8_t v_isSharedCheck_2975_; 
v_a_2968_ = lean_ctor_get(v___x_2948_, 0);
v_isSharedCheck_2975_ = !lean_is_exclusive(v___x_2948_);
if (v_isSharedCheck_2975_ == 0)
{
v___x_2970_ = v___x_2948_;
v_isShared_2971_ = v_isSharedCheck_2975_;
goto v_resetjp_2969_;
}
else
{
lean_inc(v_a_2968_);
lean_dec(v___x_2948_);
v___x_2970_ = lean_box(0);
v_isShared_2971_ = v_isSharedCheck_2975_;
goto v_resetjp_2969_;
}
v_resetjp_2969_:
{
lean_object* v___x_2973_; 
if (v_isShared_2971_ == 0)
{
lean_ctor_set_tag(v___x_2970_, 0);
v___x_2973_ = v___x_2970_;
goto v_reusejp_2972_;
}
else
{
lean_object* v_reuseFailAlloc_2974_; 
v_reuseFailAlloc_2974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2974_, 0, v_a_2968_);
v___x_2973_ = v_reuseFailAlloc_2974_;
goto v_reusejp_2972_;
}
v_reusejp_2972_:
{
v___y_2778_ = v___x_2947_;
v___y_2779_ = v___y_2870_;
v___y_2780_ = v___y_2880_;
v___y_2781_ = v___y_2871_;
v___y_2782_ = v___y_2883_;
v___y_2783_ = v_a_2898_;
v___y_2784_ = v_a_2942_;
v___y_2785_ = v___y_2887_;
v___y_2786_ = v___y_2874_;
v___y_2787_ = v___y_2888_;
v___y_2788_ = v___y_2889_;
v___y_2789_ = v___y_2890_;
v___y_2790_ = v___y_2893_;
v___y_2791_ = v___y_2892_;
v_a_2792_ = v___x_2973_;
goto v___jp_2777_;
}
}
}
}
}
else
{
lean_object* v___x_2976_; lean_object* v___x_2977_; 
v___x_2976_ = lean_io_get_num_heartbeats();
v___x_2977_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v___y_2877_, v___y_2882_, v___y_2878_, v___y_2870_, v___y_2883_, v___y_2888_, v___y_2887_, v___y_2890_, v___y_2880_);
lean_dec_ref(v___y_2878_);
if (lean_obj_tag(v___x_2977_) == 0)
{
lean_object* v_a_2978_; uint8_t v___x_2979_; 
v_a_2978_ = lean_ctor_get(v___x_2977_, 0);
lean_inc(v_a_2978_);
lean_dec_ref_known(v___x_2977_, 1);
v___x_2979_ = lean_unbox(v_a_2978_);
if (v___x_2979_ == 0)
{
uint8_t v___x_2980_; 
lean_dec(v___x_2945_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_2980_ = lean_unbox(v_a_2978_);
lean_dec(v_a_2978_);
v___y_2849_ = v___y_2870_;
v___y_2850_ = v___y_2880_;
v___y_2851_ = v___y_2871_;
v___y_2852_ = v___x_2976_;
v___y_2853_ = v___y_2883_;
v___y_2854_ = v_a_2898_;
v___y_2855_ = v_a_2942_;
v___y_2856_ = v___y_2887_;
v___y_2857_ = v___y_2874_;
v___y_2858_ = v___y_2888_;
v___y_2859_ = v___y_2889_;
v___y_2860_ = v___y_2890_;
v___y_2861_ = v___y_2892_;
v___y_2862_ = v___y_2893_;
v_a_2863_ = v___x_2980_;
goto v___jp_2848_;
}
else
{
lean_object* v___x_2981_; lean_object* v_equalMVarIds_2982_; lean_object* v_equalLMVarIds_2983_; lean_object* v_leftUnassignedMVarValues_2984_; lean_object* v_rightUnassignedMVarValues_2985_; lean_object* v___x_2987_; uint8_t v_isShared_2988_; uint8_t v_isSharedCheck_2995_; 
lean_dec(v_a_2978_);
v___x_2981_ = lean_st_ref_take(v___y_2883_);
v_equalMVarIds_2982_ = lean_ctor_get(v___x_2981_, 0);
v_equalLMVarIds_2983_ = lean_ctor_get(v___x_2981_, 1);
v_leftUnassignedMVarValues_2984_ = lean_ctor_get(v___x_2981_, 2);
v_rightUnassignedMVarValues_2985_ = lean_ctor_get(v___x_2981_, 3);
v_isSharedCheck_2995_ = !lean_is_exclusive(v___x_2981_);
if (v_isSharedCheck_2995_ == 0)
{
v___x_2987_ = v___x_2981_;
v_isShared_2988_ = v_isSharedCheck_2995_;
goto v_resetjp_2986_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_2985_);
lean_inc(v_leftUnassignedMVarValues_2984_);
lean_inc(v_equalLMVarIds_2983_);
lean_inc(v_equalMVarIds_2982_);
lean_dec(v___x_2981_);
v___x_2987_ = lean_box(0);
v_isShared_2988_ = v_isSharedCheck_2995_;
goto v_resetjp_2986_;
}
v_resetjp_2986_:
{
lean_object* v___x_2989_; lean_object* v___x_2991_; 
v___x_2989_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_2982_, v_mvarId_u2081_2645_, v_mvarId_u2082_2646_);
if (v_isShared_2988_ == 0)
{
lean_ctor_set(v___x_2987_, 0, v___x_2989_);
v___x_2991_ = v___x_2987_;
goto v_reusejp_2990_;
}
else
{
lean_object* v_reuseFailAlloc_2994_; 
v_reuseFailAlloc_2994_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2994_, 0, v___x_2989_);
lean_ctor_set(v_reuseFailAlloc_2994_, 1, v_equalLMVarIds_2983_);
lean_ctor_set(v_reuseFailAlloc_2994_, 2, v_leftUnassignedMVarValues_2984_);
lean_ctor_set(v_reuseFailAlloc_2994_, 3, v_rightUnassignedMVarValues_2985_);
v___x_2991_ = v_reuseFailAlloc_2994_;
goto v_reusejp_2990_;
}
v_reusejp_2990_:
{
lean_object* v___x_2992_; uint8_t v___x_2993_; 
v___x_2992_ = lean_st_ref_set(v___y_2883_, v___x_2991_);
v___x_2993_ = lean_unbox(v___x_2945_);
lean_dec(v___x_2945_);
v___y_2849_ = v___y_2870_;
v___y_2850_ = v___y_2880_;
v___y_2851_ = v___y_2871_;
v___y_2852_ = v___x_2976_;
v___y_2853_ = v___y_2883_;
v___y_2854_ = v_a_2898_;
v___y_2855_ = v_a_2942_;
v___y_2856_ = v___y_2887_;
v___y_2857_ = v___y_2874_;
v___y_2858_ = v___y_2888_;
v___y_2859_ = v___y_2889_;
v___y_2860_ = v___y_2890_;
v___y_2861_ = v___y_2892_;
v___y_2862_ = v___y_2893_;
v_a_2863_ = v___x_2993_;
goto v___jp_2848_;
}
}
}
}
else
{
lean_dec(v___x_2945_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_2977_) == 0)
{
lean_object* v_a_2996_; uint8_t v___x_2997_; 
v_a_2996_ = lean_ctor_get(v___x_2977_, 0);
lean_inc(v_a_2996_);
lean_dec_ref_known(v___x_2977_, 1);
v___x_2997_ = lean_unbox(v_a_2996_);
lean_dec(v_a_2996_);
v___y_2849_ = v___y_2870_;
v___y_2850_ = v___y_2880_;
v___y_2851_ = v___y_2871_;
v___y_2852_ = v___x_2976_;
v___y_2853_ = v___y_2883_;
v___y_2854_ = v_a_2898_;
v___y_2855_ = v_a_2942_;
v___y_2856_ = v___y_2887_;
v___y_2857_ = v___y_2874_;
v___y_2858_ = v___y_2888_;
v___y_2859_ = v___y_2889_;
v___y_2860_ = v___y_2890_;
v___y_2861_ = v___y_2892_;
v___y_2862_ = v___y_2893_;
v_a_2863_ = v___x_2997_;
goto v___jp_2848_;
}
else
{
lean_object* v_a_2998_; lean_object* v___x_3000_; uint8_t v_isShared_3001_; uint8_t v_isSharedCheck_3005_; 
v_a_2998_ = lean_ctor_get(v___x_2977_, 0);
v_isSharedCheck_3005_ = !lean_is_exclusive(v___x_2977_);
if (v_isSharedCheck_3005_ == 0)
{
v___x_3000_ = v___x_2977_;
v_isShared_3001_ = v_isSharedCheck_3005_;
goto v_resetjp_2999_;
}
else
{
lean_inc(v_a_2998_);
lean_dec(v___x_2977_);
v___x_3000_ = lean_box(0);
v_isShared_3001_ = v_isSharedCheck_3005_;
goto v_resetjp_2999_;
}
v_resetjp_2999_:
{
lean_object* v___x_3003_; 
if (v_isShared_3001_ == 0)
{
lean_ctor_set_tag(v___x_3000_, 0);
v___x_3003_ = v___x_3000_;
goto v_reusejp_3002_;
}
else
{
lean_object* v_reuseFailAlloc_3004_; 
v_reuseFailAlloc_3004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3004_, 0, v_a_2998_);
v___x_3003_ = v_reuseFailAlloc_3004_;
goto v_reusejp_3002_;
}
v_reusejp_3002_:
{
v___y_2824_ = v___y_2870_;
v___y_2825_ = v___y_2880_;
v___y_2826_ = v___y_2871_;
v___y_2827_ = v___x_2976_;
v___y_2828_ = v___y_2883_;
v___y_2829_ = v_a_2898_;
v___y_2830_ = v_a_2942_;
v___y_2831_ = v___y_2887_;
v___y_2832_ = v___y_2874_;
v___y_2833_ = v___y_2888_;
v___y_2834_ = v___y_2889_;
v___y_2835_ = v___y_2890_;
v___y_2836_ = v___y_2893_;
v___y_2837_ = v___y_2892_;
v_a_2838_ = v___x_3003_;
goto v___jp_2823_;
}
}
}
}
}
}
else
{
lean_object* v_a_3006_; lean_object* v___x_3008_; uint8_t v_isShared_3009_; uint8_t v_isSharedCheck_3013_; 
lean_dec(v_a_2898_);
lean_dec_ref(v___y_2882_);
lean_dec_ref(v___y_2878_);
lean_dec_ref(v___y_2877_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3006_ = lean_ctor_get(v___x_2941_, 0);
v_isSharedCheck_3013_ = !lean_is_exclusive(v___x_2941_);
if (v_isSharedCheck_3013_ == 0)
{
v___x_3008_ = v___x_2941_;
v_isShared_3009_ = v_isSharedCheck_3013_;
goto v_resetjp_3007_;
}
else
{
lean_inc(v_a_3006_);
lean_dec(v___x_2941_);
v___x_3008_ = lean_box(0);
v_isShared_3009_ = v_isSharedCheck_3013_;
goto v_resetjp_3007_;
}
v_resetjp_3007_:
{
lean_object* v___x_3011_; 
if (v_isShared_3009_ == 0)
{
v___x_3011_ = v___x_3008_;
goto v_reusejp_3010_;
}
else
{
lean_object* v_reuseFailAlloc_3012_; 
v_reuseFailAlloc_3012_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3012_, 0, v_a_3006_);
v___x_3011_ = v_reuseFailAlloc_3012_;
goto v_reusejp_3010_;
}
v_reusejp_3010_:
{
return v___x_3011_;
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
lean_object* v_a_3020_; lean_object* v___x_3022_; uint8_t v_isShared_3023_; uint8_t v_isSharedCheck_3027_; 
lean_dec_ref(v___y_2882_);
lean_dec_ref(v___y_2878_);
lean_dec_ref(v___y_2877_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3020_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_3027_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_3027_ == 0)
{
v___x_3022_ = v___x_2896_;
v_isShared_3023_ = v_isSharedCheck_3027_;
goto v_resetjp_3021_;
}
else
{
lean_inc(v_a_3020_);
lean_dec(v___x_2896_);
v___x_3022_ = lean_box(0);
v_isShared_3023_ = v_isSharedCheck_3027_;
goto v_resetjp_3021_;
}
v_resetjp_3021_:
{
lean_object* v___x_3025_; 
if (v_isShared_3023_ == 0)
{
v___x_3025_ = v___x_3022_;
goto v_reusejp_3024_;
}
else
{
lean_object* v_reuseFailAlloc_3026_; 
v_reuseFailAlloc_3026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3026_, 0, v_a_3020_);
v___x_3025_ = v_reuseFailAlloc_3026_;
goto v_reusejp_3024_;
}
v_reusejp_3024_:
{
return v___x_3025_;
}
}
}
}
v___jp_3028_:
{
if (lean_obj_tag(v_____do__lift_3029_) == 1)
{
lean_object* v_options_3036_; uint8_t v_hasTrace_3037_; 
lean_dec_ref_known(v___x_2703_, 3);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_options_3036_ = lean_ctor_get(v___y_3034_, 2);
v_hasTrace_3037_ = lean_ctor_get_uint8(v_options_3036_, sizeof(void*)*1);
if (v_hasTrace_3037_ == 0)
{
lean_object* v_val_3038_; lean_object* v___x_3040_; uint8_t v_isShared_3041_; uint8_t v_isSharedCheck_3045_; 
lean_dec_ref(v___x_2695_);
v_val_3038_ = lean_ctor_get(v_____do__lift_3029_, 0);
v_isSharedCheck_3045_ = !lean_is_exclusive(v_____do__lift_3029_);
if (v_isSharedCheck_3045_ == 0)
{
v___x_3040_ = v_____do__lift_3029_;
v_isShared_3041_ = v_isSharedCheck_3045_;
goto v_resetjp_3039_;
}
else
{
lean_inc(v_val_3038_);
lean_dec(v_____do__lift_3029_);
v___x_3040_ = lean_box(0);
v_isShared_3041_ = v_isSharedCheck_3045_;
goto v_resetjp_3039_;
}
v_resetjp_3039_:
{
lean_object* v___x_3043_; 
if (v_isShared_3041_ == 0)
{
lean_ctor_set_tag(v___x_3040_, 0);
v___x_3043_ = v___x_3040_;
goto v_reusejp_3042_;
}
else
{
lean_object* v_reuseFailAlloc_3044_; 
v_reuseFailAlloc_3044_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3044_, 0, v_val_3038_);
v___x_3043_ = v_reuseFailAlloc_3044_;
goto v_reusejp_3042_;
}
v_reusejp_3042_:
{
return v___x_3043_;
}
}
}
else
{
lean_object* v_val_3046_; lean_object* v___x_3048_; uint8_t v_isShared_3049_; uint8_t v_isSharedCheck_3062_; 
v_val_3046_ = lean_ctor_get(v_____do__lift_3029_, 0);
v_isSharedCheck_3062_ = !lean_is_exclusive(v_____do__lift_3029_);
if (v_isSharedCheck_3062_ == 0)
{
v___x_3048_ = v_____do__lift_3029_;
v_isShared_3049_ = v_isSharedCheck_3062_;
goto v_resetjp_3047_;
}
else
{
lean_inc(v_val_3046_);
lean_dec(v_____do__lift_3029_);
v___x_3048_ = lean_box(0);
v_isShared_3049_ = v_isSharedCheck_3062_;
goto v_resetjp_3047_;
}
v_resetjp_3047_:
{
lean_object* v_inheritedTraceOptions_3050_; lean_object* v___x_3051_; uint8_t v___x_3052_; 
v_inheritedTraceOptions_3050_ = lean_ctor_get(v___y_3034_, 13);
v___x_3051_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_3052_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3050_, v_options_3036_, v___x_3051_);
if (v___x_3052_ == 0)
{
lean_object* v___x_3054_; 
lean_dec_ref(v___x_2695_);
if (v_isShared_3049_ == 0)
{
lean_ctor_set_tag(v___x_3048_, 0);
v___x_3054_ = v___x_3048_;
goto v_reusejp_3053_;
}
else
{
lean_object* v_reuseFailAlloc_3055_; 
v_reuseFailAlloc_3055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3055_, 0, v_val_3046_);
v___x_3054_ = v_reuseFailAlloc_3055_;
goto v_reusejp_3053_;
}
v_reusejp_3053_:
{
return v___x_3054_;
}
}
else
{
lean_object* v___x_3056_; uint8_t v___x_3057_; 
lean_del_object(v___x_3048_);
v___x_3056_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__5);
v___x_3057_ = lean_unbox(v_val_3046_);
if (v___x_3057_ == 0)
{
lean_object* v___x_3058_; uint8_t v___x_3059_; 
v___x_3058_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__6));
v___x_3059_ = lean_unbox(v_val_3046_);
lean_dec(v_val_3046_);
v___y_2747_ = v___y_3031_;
v___y_2748_ = v___x_3059_;
v___y_2749_ = v___x_3056_;
v___y_2750_ = v___y_3033_;
v___y_2751_ = v___y_3030_;
v___y_2752_ = v___y_3035_;
v___y_2753_ = v___y_3032_;
v___y_2754_ = v___y_3034_;
v___y_2755_ = v___x_3058_;
goto v___jp_2746_;
}
else
{
lean_object* v___x_3060_; uint8_t v___x_3061_; 
v___x_3060_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__7));
v___x_3061_ = lean_unbox(v_val_3046_);
lean_dec(v_val_3046_);
v___y_2747_ = v___y_3031_;
v___y_2748_ = v___x_3061_;
v___y_2749_ = v___x_3056_;
v___y_2750_ = v___y_3033_;
v___y_2751_ = v___y_3030_;
v___y_2752_ = v___y_3035_;
v___y_2753_ = v___y_3032_;
v___y_2754_ = v___y_3034_;
v___y_2755_ = v___x_3060_;
goto v___jp_2746_;
}
}
}
}
}
else
{
lean_object* v___x_3063_; lean_object* v_equalMVarIds_3064_; lean_object* v___x_3065_; 
lean_dec(v_____do__lift_3029_);
v___x_3063_ = lean_st_ref_get(v___y_3031_);
v_equalMVarIds_3064_ = lean_ctor_get(v___x_3063_, 0);
lean_inc_ref(v_equalMVarIds_3064_);
lean_dec(v___x_3063_);
lean_inc(v_mvarId_u2081_2645_);
v___x_3065_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_2722_, v___x_2723_, v_equalMVarIds_3064_, v_mvarId_u2081_2645_);
lean_dec_ref(v_equalMVarIds_3064_);
if (lean_obj_tag(v___x_3065_) == 1)
{
lean_object* v_val_3066_; lean_object* v___x_3068_; uint8_t v_isShared_3069_; uint8_t v_isSharedCheck_3143_; 
lean_dec_ref_known(v___x_2703_, 3);
v_val_3066_ = lean_ctor_get(v___x_3065_, 0);
v_isSharedCheck_3143_ = !lean_is_exclusive(v___x_3065_);
if (v_isSharedCheck_3143_ == 0)
{
v___x_3068_ = v___x_3065_;
v_isShared_3069_ = v_isSharedCheck_3143_;
goto v_resetjp_3067_;
}
else
{
lean_inc(v_val_3066_);
lean_dec(v___x_3065_);
v___x_3068_ = lean_box(0);
v_isShared_3069_ = v_isSharedCheck_3143_;
goto v_resetjp_3067_;
}
v_resetjp_3067_:
{
uint8_t v___x_3070_; 
v___x_3070_ = l_Lean_instBEqMVarId_beq(v_mvarId_u2082_2646_, v_val_3066_);
lean_dec(v_mvarId_u2082_2646_);
if (v___x_3070_ == 0)
{
lean_object* v_options_3071_; uint8_t v_hasTrace_3072_; 
v_options_3071_ = lean_ctor_get(v___y_3034_, 2);
v_hasTrace_3072_ = lean_ctor_get_uint8(v_options_3071_, sizeof(void*)*1);
if (v_hasTrace_3072_ == 0)
{
lean_object* v___x_3073_; lean_object* v___x_3075_; 
lean_dec(v_val_3066_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3073_ = lean_box(v___x_3070_);
if (v_isShared_3069_ == 0)
{
lean_ctor_set_tag(v___x_3068_, 0);
lean_ctor_set(v___x_3068_, 0, v___x_3073_);
v___x_3075_ = v___x_3068_;
goto v_reusejp_3074_;
}
else
{
lean_object* v_reuseFailAlloc_3076_; 
v_reuseFailAlloc_3076_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3076_, 0, v___x_3073_);
v___x_3075_ = v_reuseFailAlloc_3076_;
goto v_reusejp_3074_;
}
v_reusejp_3074_:
{
return v___x_3075_;
}
}
else
{
lean_object* v_inheritedTraceOptions_3077_; lean_object* v___x_3078_; uint8_t v___x_3079_; 
v_inheritedTraceOptions_3077_ = lean_ctor_get(v___y_3034_, 13);
v___x_3078_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_3079_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3077_, v_options_3071_, v___x_3078_);
if (v___x_3079_ == 0)
{
lean_object* v___x_3080_; lean_object* v___x_3082_; 
lean_dec(v_val_3066_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3080_ = lean_box(v___x_3070_);
if (v_isShared_3069_ == 0)
{
lean_ctor_set_tag(v___x_3068_, 0);
lean_ctor_set(v___x_3068_, 0, v___x_3080_);
v___x_3082_ = v___x_3068_;
goto v_reusejp_3081_;
}
else
{
lean_object* v_reuseFailAlloc_3083_; 
v_reuseFailAlloc_3083_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3083_, 0, v___x_3080_);
v___x_3082_ = v_reuseFailAlloc_3083_;
goto v_reusejp_3081_;
}
v_reusejp_3081_:
{
return v___x_3082_;
}
}
else
{
lean_object* v___x_3084_; lean_object* v___x_3085_; lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_245761__overap_3091_; lean_object* v___x_3092_; 
lean_del_object(v___x_3068_);
v___x_3084_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__9);
v___x_3085_ = l_Lean_MessageData_ofName(v_mvarId_u2081_2645_);
v___x_3086_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3086_, 0, v___x_3084_);
lean_ctor_set(v___x_3086_, 1, v___x_3085_);
v___x_3087_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__11);
v___x_3088_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3088_, 0, v___x_3086_);
lean_ctor_set(v___x_3088_, 1, v___x_3087_);
v___x_3089_ = l_Lean_MessageData_ofName(v_val_3066_);
v___x_3090_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3090_, 0, v___x_3088_);
lean_ctor_set(v___x_3090_, 1, v___x_3089_);
lean_inc_ref(v_toMonadRef_2698_);
v___x_245761__overap_3091_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3090_);
lean_inc(v___y_3035_);
lean_inc_ref(v___y_3034_);
lean_inc(v___y_3033_);
lean_inc_ref(v___y_3032_);
lean_inc(v___y_3031_);
lean_inc_ref(v___y_3030_);
v___x_3092_ = lean_apply_7(v___x_245761__overap_3091_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_, lean_box(0));
if (lean_obj_tag(v___x_3092_) == 0)
{
lean_object* v___x_3094_; uint8_t v_isShared_3095_; uint8_t v_isSharedCheck_3100_; 
v_isSharedCheck_3100_ = !lean_is_exclusive(v___x_3092_);
if (v_isSharedCheck_3100_ == 0)
{
lean_object* v_unused_3101_; 
v_unused_3101_ = lean_ctor_get(v___x_3092_, 0);
lean_dec(v_unused_3101_);
v___x_3094_ = v___x_3092_;
v_isShared_3095_ = v_isSharedCheck_3100_;
goto v_resetjp_3093_;
}
else
{
lean_dec(v___x_3092_);
v___x_3094_ = lean_box(0);
v_isShared_3095_ = v_isSharedCheck_3100_;
goto v_resetjp_3093_;
}
v_resetjp_3093_:
{
lean_object* v___x_3096_; lean_object* v___x_3098_; 
v___x_3096_ = lean_box(v___x_3070_);
if (v_isShared_3095_ == 0)
{
lean_ctor_set(v___x_3094_, 0, v___x_3096_);
v___x_3098_ = v___x_3094_;
goto v_reusejp_3097_;
}
else
{
lean_object* v_reuseFailAlloc_3099_; 
v_reuseFailAlloc_3099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3099_, 0, v___x_3096_);
v___x_3098_ = v_reuseFailAlloc_3099_;
goto v_reusejp_3097_;
}
v_reusejp_3097_:
{
return v___x_3098_;
}
}
}
else
{
lean_object* v_a_3102_; lean_object* v___x_3104_; uint8_t v_isShared_3105_; uint8_t v_isSharedCheck_3109_; 
v_a_3102_ = lean_ctor_get(v___x_3092_, 0);
v_isSharedCheck_3109_ = !lean_is_exclusive(v___x_3092_);
if (v_isSharedCheck_3109_ == 0)
{
v___x_3104_ = v___x_3092_;
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
else
{
lean_inc(v_a_3102_);
lean_dec(v___x_3092_);
v___x_3104_ = lean_box(0);
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
v_resetjp_3103_:
{
lean_object* v___x_3107_; 
if (v_isShared_3105_ == 0)
{
v___x_3107_ = v___x_3104_;
goto v_reusejp_3106_;
}
else
{
lean_object* v_reuseFailAlloc_3108_; 
v_reuseFailAlloc_3108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3108_, 0, v_a_3102_);
v___x_3107_ = v_reuseFailAlloc_3108_;
goto v_reusejp_3106_;
}
v_reusejp_3106_:
{
return v___x_3107_;
}
}
}
}
}
}
else
{
lean_object* v_options_3110_; uint8_t v_hasTrace_3111_; 
lean_dec(v_val_3066_);
lean_dec(v_mvarId_u2081_2645_);
v_options_3110_ = lean_ctor_get(v___y_3034_, 2);
v_hasTrace_3111_ = lean_ctor_get_uint8(v_options_3110_, sizeof(void*)*1);
if (v_hasTrace_3111_ == 0)
{
lean_object* v___x_3112_; lean_object* v___x_3114_; 
lean_dec_ref(v___x_2695_);
v___x_3112_ = lean_box(v___x_3070_);
if (v_isShared_3069_ == 0)
{
lean_ctor_set_tag(v___x_3068_, 0);
lean_ctor_set(v___x_3068_, 0, v___x_3112_);
v___x_3114_ = v___x_3068_;
goto v_reusejp_3113_;
}
else
{
lean_object* v_reuseFailAlloc_3115_; 
v_reuseFailAlloc_3115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3115_, 0, v___x_3112_);
v___x_3114_ = v_reuseFailAlloc_3115_;
goto v_reusejp_3113_;
}
v_reusejp_3113_:
{
return v___x_3114_;
}
}
else
{
lean_object* v_inheritedTraceOptions_3116_; lean_object* v___x_3117_; uint8_t v___x_3118_; 
v_inheritedTraceOptions_3116_ = lean_ctor_get(v___y_3034_, 13);
v___x_3117_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_3118_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3116_, v_options_3110_, v___x_3117_);
if (v___x_3118_ == 0)
{
lean_object* v___x_3119_; lean_object* v___x_3121_; 
lean_dec_ref(v___x_2695_);
v___x_3119_ = lean_box(v___x_3070_);
if (v_isShared_3069_ == 0)
{
lean_ctor_set_tag(v___x_3068_, 0);
lean_ctor_set(v___x_3068_, 0, v___x_3119_);
v___x_3121_ = v___x_3068_;
goto v_reusejp_3120_;
}
else
{
lean_object* v_reuseFailAlloc_3122_; 
v_reuseFailAlloc_3122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3122_, 0, v___x_3119_);
v___x_3121_ = v_reuseFailAlloc_3122_;
goto v_reusejp_3120_;
}
v_reusejp_3120_:
{
return v___x_3121_;
}
}
else
{
lean_object* v___x_3123_; lean_object* v___x_245772__overap_3124_; lean_object* v___x_3125_; 
lean_del_object(v___x_3068_);
v___x_3123_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__13);
lean_inc_ref(v_toMonadRef_2698_);
v___x_245772__overap_3124_ = l_Lean_addTrace___redArg(v___x_2695_, v___x_2696_, v_toMonadRef_2698_, v___f_2699_, v_cls_2745_, v___x_3123_);
lean_inc(v___y_3035_);
lean_inc_ref(v___y_3034_);
lean_inc(v___y_3033_);
lean_inc_ref(v___y_3032_);
lean_inc(v___y_3031_);
lean_inc_ref(v___y_3030_);
v___x_3125_ = lean_apply_7(v___x_245772__overap_3124_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_, lean_box(0));
if (lean_obj_tag(v___x_3125_) == 0)
{
lean_object* v___x_3127_; uint8_t v_isShared_3128_; uint8_t v_isSharedCheck_3133_; 
v_isSharedCheck_3133_ = !lean_is_exclusive(v___x_3125_);
if (v_isSharedCheck_3133_ == 0)
{
lean_object* v_unused_3134_; 
v_unused_3134_ = lean_ctor_get(v___x_3125_, 0);
lean_dec(v_unused_3134_);
v___x_3127_ = v___x_3125_;
v_isShared_3128_ = v_isSharedCheck_3133_;
goto v_resetjp_3126_;
}
else
{
lean_dec(v___x_3125_);
v___x_3127_ = lean_box(0);
v_isShared_3128_ = v_isSharedCheck_3133_;
goto v_resetjp_3126_;
}
v_resetjp_3126_:
{
lean_object* v___x_3129_; lean_object* v___x_3131_; 
v___x_3129_ = lean_box(v___x_3070_);
if (v_isShared_3128_ == 0)
{
lean_ctor_set(v___x_3127_, 0, v___x_3129_);
v___x_3131_ = v___x_3127_;
goto v_reusejp_3130_;
}
else
{
lean_object* v_reuseFailAlloc_3132_; 
v_reuseFailAlloc_3132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3132_, 0, v___x_3129_);
v___x_3131_ = v_reuseFailAlloc_3132_;
goto v_reusejp_3130_;
}
v_reusejp_3130_:
{
return v___x_3131_;
}
}
}
else
{
lean_object* v_a_3135_; lean_object* v___x_3137_; uint8_t v_isShared_3138_; uint8_t v_isSharedCheck_3142_; 
v_a_3135_ = lean_ctor_get(v___x_3125_, 0);
v_isSharedCheck_3142_ = !lean_is_exclusive(v___x_3125_);
if (v_isSharedCheck_3142_ == 0)
{
v___x_3137_ = v___x_3125_;
v_isShared_3138_ = v_isSharedCheck_3142_;
goto v_resetjp_3136_;
}
else
{
lean_inc(v_a_3135_);
lean_dec(v___x_3125_);
v___x_3137_ = lean_box(0);
v_isShared_3138_ = v_isSharedCheck_3142_;
goto v_resetjp_3136_;
}
v_resetjp_3136_:
{
lean_object* v___x_3140_; 
if (v_isShared_3138_ == 0)
{
v___x_3140_ = v___x_3137_;
goto v_reusejp_3139_;
}
else
{
lean_object* v_reuseFailAlloc_3141_; 
v_reuseFailAlloc_3141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3141_, 0, v_a_3135_);
v___x_3140_ = v_reuseFailAlloc_3141_;
goto v_reusejp_3139_;
}
v_reusejp_3139_:
{
return v___x_3140_;
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
lean_object* v_mctx_u2081_3144_; lean_object* v_mctx_u2082_3145_; lean_object* v_decls_3146_; lean_object* v___x_3147_; 
lean_dec(v___x_3065_);
v_mctx_u2081_3144_ = lean_ctor_get(v___y_3030_, 1);
v_mctx_u2082_3145_ = lean_ctor_get(v___y_3030_, 2);
v_decls_3146_ = lean_ctor_get(v_mctx_u2081_3144_, 5);
lean_inc(v_mvarId_u2081_2645_);
v___x_3147_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3146_, v_mvarId_u2081_2645_);
if (lean_obj_tag(v___x_3147_) == 1)
{
lean_object* v_val_3148_; lean_object* v_decls_3149_; lean_object* v___x_3150_; 
v_val_3148_ = lean_ctor_get(v___x_3147_, 0);
lean_inc(v_val_3148_);
lean_dec_ref_known(v___x_3147_, 1);
v_decls_3149_ = lean_ctor_get(v_mctx_u2082_3145_, 5);
lean_inc(v_mvarId_u2082_2646_);
v___x_3150_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_2722_, v___x_2723_, v_decls_3149_, v_mvarId_u2082_2646_);
if (lean_obj_tag(v___x_3150_) == 1)
{
lean_object* v_val_3151_; lean_object* v_lctx_3152_; lean_object* v_type_3153_; lean_object* v_lctx_3154_; lean_object* v_type_3155_; lean_object* v_localInstances_3156_; lean_object* v___x_3157_; 
lean_dec_ref_known(v___x_2703_, 3);
v_val_3151_ = lean_ctor_get(v___x_3150_, 0);
lean_inc(v_val_3151_);
lean_dec_ref_known(v___x_3150_, 1);
v_lctx_3152_ = lean_ctor_get(v_val_3148_, 1);
lean_inc_ref(v_lctx_3152_);
v_type_3153_ = lean_ctor_get(v_val_3148_, 2);
lean_inc_ref(v_type_3153_);
lean_dec(v_val_3148_);
v_lctx_3154_ = lean_ctor_get(v_val_3151_, 1);
lean_inc_ref(v_lctx_3154_);
v_type_3155_ = lean_ctor_get(v_val_3151_, 2);
lean_inc_ref(v_type_3155_);
v_localInstances_3156_ = lean_ctor_get(v_val_3151_, 4);
lean_inc_ref_n(v_localInstances_3156_, 2);
lean_dec(v_val_3151_);
v___x_3157_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(v_lctx_3152_, v_lctx_3154_, v_localInstances_3156_, v_localInstances_3156_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
if (lean_obj_tag(v___x_3157_) == 0)
{
lean_object* v_a_3158_; lean_object* v___x_3160_; uint8_t v_isShared_3161_; uint8_t v_isSharedCheck_3199_; 
v_a_3158_ = lean_ctor_get(v___x_3157_, 0);
v_isSharedCheck_3199_ = !lean_is_exclusive(v___x_3157_);
if (v_isSharedCheck_3199_ == 0)
{
v___x_3160_ = v___x_3157_;
v_isShared_3161_ = v_isSharedCheck_3199_;
goto v_resetjp_3159_;
}
else
{
lean_inc(v_a_3158_);
lean_dec(v___x_3157_);
v___x_3160_ = lean_box(0);
v_isShared_3161_ = v_isSharedCheck_3199_;
goto v_resetjp_3159_;
}
v_resetjp_3159_:
{
if (lean_obj_tag(v_a_3158_) == 1)
{
lean_object* v_options_3162_; uint8_t v_hasTrace_3163_; 
lean_del_object(v___x_3160_);
v_options_3162_ = lean_ctor_get(v___y_3034_, 2);
v_hasTrace_3163_ = lean_ctor_get_uint8(v_options_3162_, sizeof(void*)*1);
if (v_hasTrace_3163_ == 0)
{
lean_object* v_val_3164_; lean_object* v___x_3165_; 
lean_dec_ref(v___x_2695_);
v_val_3164_ = lean_ctor_get(v_a_3158_, 0);
lean_inc(v_val_3164_);
lean_dec_ref_known(v_a_3158_, 1);
v___x_3165_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v_type_3153_, v_type_3155_, v_val_3164_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
lean_dec(v_val_3164_);
if (lean_obj_tag(v___x_3165_) == 0)
{
lean_object* v_a_3166_; uint8_t v___x_3167_; 
v_a_3166_ = lean_ctor_get(v___x_3165_, 0);
lean_inc(v_a_3166_);
lean_dec_ref_known(v___x_3165_, 1);
v___x_3167_ = lean_unbox(v_a_3166_);
lean_dec(v_a_3166_);
v_____do__lift_2725_ = v___x_3167_;
v___y_2726_ = v___y_3031_;
goto v___jp_2724_;
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
return v___x_3165_;
}
}
else
{
lean_object* v_val_3168_; lean_object* v_fileName_3169_; lean_object* v_fileMap_3170_; lean_object* v_currRecDepth_3171_; lean_object* v_maxRecDepth_3172_; lean_object* v_ref_3173_; lean_object* v_currNamespace_3174_; lean_object* v_openDecls_3175_; lean_object* v_initHeartbeats_3176_; lean_object* v_maxHeartbeats_3177_; lean_object* v_quotContext_3178_; lean_object* v_currMacroScope_3179_; uint8_t v_diag_3180_; lean_object* v_cancelTk_x3f_3181_; uint8_t v_suppressElabErrors_3182_; lean_object* v_inheritedTraceOptions_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; uint8_t v___x_3186_; 
v_val_3168_ = lean_ctor_get(v_a_3158_, 0);
lean_inc(v_val_3168_);
lean_dec_ref_known(v_a_3158_, 1);
v_fileName_3169_ = lean_ctor_get(v___y_3034_, 0);
v_fileMap_3170_ = lean_ctor_get(v___y_3034_, 1);
v_currRecDepth_3171_ = lean_ctor_get(v___y_3034_, 3);
v_maxRecDepth_3172_ = lean_ctor_get(v___y_3034_, 4);
v_ref_3173_ = lean_ctor_get(v___y_3034_, 5);
v_currNamespace_3174_ = lean_ctor_get(v___y_3034_, 6);
v_openDecls_3175_ = lean_ctor_get(v___y_3034_, 7);
v_initHeartbeats_3176_ = lean_ctor_get(v___y_3034_, 8);
v_maxHeartbeats_3177_ = lean_ctor_get(v___y_3034_, 9);
v_quotContext_3178_ = lean_ctor_get(v___y_3034_, 10);
v_currMacroScope_3179_ = lean_ctor_get(v___y_3034_, 11);
v_diag_3180_ = lean_ctor_get_uint8(v___y_3034_, sizeof(void*)*14);
v_cancelTk_x3f_3181_ = lean_ctor_get(v___y_3034_, 12);
v_suppressElabErrors_3182_ = lean_ctor_get_uint8(v___y_3034_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3183_ = lean_ctor_get(v___y_3034_, 13);
v___x_3184_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___closed__0));
v___x_3185_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___closed__5);
v___x_3186_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3183_, v_options_3162_, v___x_3185_);
if (v___x_3186_ == 0)
{
lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; uint8_t v___x_3190_; 
v___x_3187_ = l_Lean_KVMap_instValueBool;
v___x_3188_ = l_Lean_trace_profiler;
v___x_3189_ = l_Lean_Option_get___redArg(v___x_3187_, v_options_3162_, v___x_3188_);
v___x_3190_ = lean_unbox(v___x_3189_);
lean_dec(v___x_3189_);
if (v___x_3190_ == 0)
{
lean_object* v___x_3191_; 
lean_dec_ref(v___x_2695_);
v___x_3191_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v_type_3153_, v_type_3155_, v_val_3168_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
lean_dec(v_val_3168_);
if (lean_obj_tag(v___x_3191_) == 0)
{
lean_object* v_a_3192_; uint8_t v___x_3193_; 
v_a_3192_ = lean_ctor_get(v___x_3191_, 0);
lean_inc(v_a_3192_);
lean_dec_ref_known(v___x_3191_, 1);
v___x_3193_ = lean_unbox(v_a_3192_);
lean_dec(v_a_3192_);
v_____do__lift_2725_ = v___x_3193_;
v___y_2726_ = v___y_3031_;
goto v___jp_2724_;
}
else
{
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
return v___x_3191_;
}
}
else
{
v___y_2867_ = v_suppressElabErrors_3182_;
v___y_2868_ = v_fileMap_3170_;
v___y_2869_ = v_diag_3180_;
v___y_2870_ = v___y_3030_;
v___y_2871_ = v_ref_3173_;
v___y_2872_ = v_cancelTk_x3f_3181_;
v___y_2873_ = v_inheritedTraceOptions_3183_;
v___y_2874_ = v_hasTrace_3163_;
v___y_2875_ = v_maxRecDepth_3172_;
v___y_2876_ = v_fileName_3169_;
v___y_2877_ = v_type_3153_;
v___y_2878_ = v_val_3168_;
v___y_2879_ = v_currNamespace_3174_;
v___y_2880_ = v___y_3035_;
v___y_2881_ = v_currMacroScope_3179_;
v___y_2882_ = v_type_3155_;
v___y_2883_ = v___y_3031_;
v___y_2884_ = v_openDecls_3175_;
v___y_2885_ = v_maxHeartbeats_3177_;
v___y_2886_ = v_currRecDepth_3171_;
v___y_2887_ = v___y_3033_;
v___y_2888_ = v___y_3032_;
v___y_2889_ = v_options_3162_;
v___y_2890_ = v___y_3034_;
v___y_2891_ = v_quotContext_3178_;
v___y_2892_ = v___x_3186_;
v___y_2893_ = v___x_3184_;
v___y_2894_ = v_initHeartbeats_3176_;
goto v___jp_2866_;
}
}
else
{
v___y_2867_ = v_suppressElabErrors_3182_;
v___y_2868_ = v_fileMap_3170_;
v___y_2869_ = v_diag_3180_;
v___y_2870_ = v___y_3030_;
v___y_2871_ = v_ref_3173_;
v___y_2872_ = v_cancelTk_x3f_3181_;
v___y_2873_ = v_inheritedTraceOptions_3183_;
v___y_2874_ = v_hasTrace_3163_;
v___y_2875_ = v_maxRecDepth_3172_;
v___y_2876_ = v_fileName_3169_;
v___y_2877_ = v_type_3153_;
v___y_2878_ = v_val_3168_;
v___y_2879_ = v_currNamespace_3174_;
v___y_2880_ = v___y_3035_;
v___y_2881_ = v_currMacroScope_3179_;
v___y_2882_ = v_type_3155_;
v___y_2883_ = v___y_3031_;
v___y_2884_ = v_openDecls_3175_;
v___y_2885_ = v_maxHeartbeats_3177_;
v___y_2886_ = v_currRecDepth_3171_;
v___y_2887_ = v___y_3033_;
v___y_2888_ = v___y_3032_;
v___y_2889_ = v_options_3162_;
v___y_2890_ = v___y_3034_;
v___y_2891_ = v_quotContext_3178_;
v___y_2892_ = v___x_3186_;
v___y_2893_ = v___x_3184_;
v___y_2894_ = v_initHeartbeats_3176_;
goto v___jp_2866_;
}
}
}
else
{
uint8_t v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3197_; 
lean_dec(v_a_3158_);
lean_dec_ref(v_type_3155_);
lean_dec_ref(v_type_3153_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3194_ = 0;
v___x_3195_ = lean_box(v___x_3194_);
if (v_isShared_3161_ == 0)
{
lean_ctor_set(v___x_3160_, 0, v___x_3195_);
v___x_3197_ = v___x_3160_;
goto v_reusejp_3196_;
}
else
{
lean_object* v_reuseFailAlloc_3198_; 
v_reuseFailAlloc_3198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3198_, 0, v___x_3195_);
v___x_3197_ = v_reuseFailAlloc_3198_;
goto v_reusejp_3196_;
}
v_reusejp_3196_:
{
return v___x_3197_;
}
}
}
}
else
{
lean_object* v_a_3200_; lean_object* v___x_3202_; uint8_t v_isShared_3203_; uint8_t v_isSharedCheck_3207_; 
lean_dec_ref(v_type_3155_);
lean_dec_ref(v_type_3153_);
lean_dec_ref(v___x_2695_);
lean_dec(v_mvarId_u2082_2646_);
lean_dec(v_mvarId_u2081_2645_);
v_a_3200_ = lean_ctor_get(v___x_3157_, 0);
v_isSharedCheck_3207_ = !lean_is_exclusive(v___x_3157_);
if (v_isSharedCheck_3207_ == 0)
{
v___x_3202_ = v___x_3157_;
v_isShared_3203_ = v_isSharedCheck_3207_;
goto v_resetjp_3201_;
}
else
{
lean_inc(v_a_3200_);
lean_dec(v___x_3157_);
v___x_3202_ = lean_box(0);
v_isShared_3203_ = v_isSharedCheck_3207_;
goto v_resetjp_3201_;
}
v_resetjp_3201_:
{
lean_object* v___x_3205_; 
if (v_isShared_3203_ == 0)
{
v___x_3205_ = v___x_3202_;
goto v_reusejp_3204_;
}
else
{
lean_object* v_reuseFailAlloc_3206_; 
v_reuseFailAlloc_3206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3206_, 0, v_a_3200_);
v___x_3205_ = v_reuseFailAlloc_3206_;
goto v_reusejp_3204_;
}
v_reusejp_3204_:
{
return v___x_3205_;
}
}
}
}
else
{
lean_object* v___x_3208_; lean_object* v___x_3209_; lean_object* v___x_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; lean_object* v___x_245025__overap_3213_; lean_object* v___x_3214_; 
lean_dec(v___x_3150_);
lean_dec(v_val_3148_);
lean_dec(v_mvarId_u2081_2645_);
v___x_3208_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3209_ = l_Lean_MessageData_ofName(v_mvarId_u2082_2646_);
v___x_3210_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3210_, 0, v___x_3208_);
lean_ctor_set(v___x_3210_, 1, v___x_3209_);
v___x_3211_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3212_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3212_, 0, v___x_3210_);
lean_ctor_set(v___x_3212_, 1, v___x_3211_);
v___x_245025__overap_3213_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3212_);
lean_inc(v___y_3035_);
lean_inc_ref(v___y_3034_);
lean_inc(v___y_3033_);
lean_inc_ref(v___y_3032_);
lean_inc(v___y_3031_);
lean_inc_ref(v___y_3030_);
v___x_3214_ = lean_apply_7(v___x_245025__overap_3213_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_, lean_box(0));
return v___x_3214_;
}
}
else
{
lean_object* v___x_3215_; lean_object* v___x_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_245034__overap_3220_; lean_object* v___x_3221_; 
lean_dec(v___x_3147_);
lean_dec(v_mvarId_u2082_2646_);
v___x_3215_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__15);
v___x_3216_ = l_Lean_MessageData_ofName(v_mvarId_u2081_2645_);
v___x_3217_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3217_, 0, v___x_3215_);
lean_ctor_set(v___x_3217_, 1, v___x_3216_);
v___x_3218_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17, &lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___closed__17);
v___x_3219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3219_, 0, v___x_3217_);
lean_ctor_set(v___x_3219_, 1, v___x_3218_);
v___x_245034__overap_3220_ = l_Lean_throwError___redArg(v___x_2695_, v___x_2703_, v___x_3219_);
lean_inc(v___y_3035_);
lean_inc_ref(v___y_3034_);
lean_inc(v___y_3033_);
lean_inc_ref(v___y_3032_);
lean_inc(v___y_3031_);
lean_inc_ref(v___y_3030_);
v___x_3221_ = lean_apply_7(v___x_245034__overap_3220_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_, lean_box(0));
return v___x_3221_;
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
static lean_object* _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3(void){
_start:
{
lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; 
v___x_3900_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__2));
v___x_3901_ = lean_unsigned_to_nat(28u);
v___x_3902_ = lean_unsigned_to_nat(255u);
v___x_3903_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__1));
v___x_3904_ = ((lean_object*)(lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__0));
v___x_3905_ = l_mkPanicMessageWithDecl(v___x_3904_, v___x_3903_, v___x_3902_, v___x_3901_, v___x_3900_);
return v___x_3905_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(lean_object* v_a_3906_, lean_object* v_a_3907_, lean_object* v_a_3908_, lean_object* v_a_3909_, lean_object* v_a_3910_, lean_object* v_a_3911_, lean_object* v_a_3912_, lean_object* v_a_3913_, lean_object* v_a_3914_, lean_object* v_a_3915_){
_start:
{
switch(lean_obj_tag(v_a_3906_))
{
case 0:
{
switch(lean_obj_tag(v_a_3907_))
{
case 0:
{
lean_object* v_mvarId_3921_; lean_object* v_mvarId_3922_; lean_object* v___x_3923_; 
v_mvarId_3921_ = lean_ctor_get(v_a_3906_, 0);
lean_inc(v_mvarId_3921_);
lean_dec_ref_known(v_a_3906_, 1);
v_mvarId_3922_ = lean_ctor_get(v_a_3907_, 0);
lean_inc(v_mvarId_3922_);
lean_dec_ref_known(v_a_3907_, 1);
v___x_3923_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(v_mvarId_3921_, v_mvarId_3922_, v_a_3910_, v_a_3911_, v_a_3912_, v_a_3913_, v_a_3914_, v_a_3915_);
return v___x_3923_;
}
case 1:
{
uint8_t v_allowAssignmentDiff_3924_; 
v_allowAssignmentDiff_3924_ = lean_ctor_get_uint8(v_a_3910_, sizeof(void*)*3);
if (v_allowAssignmentDiff_3924_ == 0)
{
lean_object* v___x_3926_; uint8_t v_isShared_3927_; uint8_t v_isSharedCheck_3932_; 
lean_dec_ref_known(v_a_3906_, 1);
v_isSharedCheck_3932_ = !lean_is_exclusive(v_a_3907_);
if (v_isSharedCheck_3932_ == 0)
{
lean_object* v_unused_3933_; 
v_unused_3933_ = lean_ctor_get(v_a_3907_, 0);
lean_dec(v_unused_3933_);
v___x_3926_ = v_a_3907_;
v_isShared_3927_ = v_isSharedCheck_3932_;
goto v_resetjp_3925_;
}
else
{
lean_dec(v_a_3907_);
v___x_3926_ = lean_box(0);
v_isShared_3927_ = v_isSharedCheck_3932_;
goto v_resetjp_3925_;
}
v_resetjp_3925_:
{
lean_object* v___x_3928_; lean_object* v___x_3930_; 
v___x_3928_ = lean_box(v_allowAssignmentDiff_3924_);
if (v_isShared_3927_ == 0)
{
lean_ctor_set_tag(v___x_3926_, 0);
lean_ctor_set(v___x_3926_, 0, v___x_3928_);
v___x_3930_ = v___x_3926_;
goto v_reusejp_3929_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v___x_3928_);
v___x_3930_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3929_;
}
v_reusejp_3929_:
{
return v___x_3930_;
}
}
}
else
{
lean_object* v_mvarId_3934_; lean_object* v_e_3935_; lean_object* v___x_3937_; uint8_t v_isShared_3938_; uint8_t v_isSharedCheck_3962_; 
v_mvarId_3934_ = lean_ctor_get(v_a_3906_, 0);
lean_inc(v_mvarId_3934_);
lean_dec_ref_known(v_a_3906_, 1);
v_e_3935_ = lean_ctor_get(v_a_3907_, 0);
v_isSharedCheck_3962_ = !lean_is_exclusive(v_a_3907_);
if (v_isSharedCheck_3962_ == 0)
{
v___x_3937_ = v_a_3907_;
v_isShared_3938_ = v_isSharedCheck_3962_;
goto v_resetjp_3936_;
}
else
{
lean_inc(v_e_3935_);
lean_dec(v_a_3907_);
v___x_3937_ = lean_box(0);
v_isShared_3938_ = v_isSharedCheck_3962_;
goto v_resetjp_3936_;
}
v_resetjp_3936_:
{
lean_object* v___x_3939_; lean_object* v_leftUnassignedMVarValues_3940_; lean_object* v___x_3941_; 
v___x_3939_ = lean_st_ref_get(v_a_3911_);
v_leftUnassignedMVarValues_3940_ = lean_ctor_get(v___x_3939_, 2);
lean_inc_ref(v_leftUnassignedMVarValues_3940_);
lean_dec(v___x_3939_);
v___x_3941_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(v_leftUnassignedMVarValues_3940_, v_mvarId_3934_);
lean_dec_ref(v_leftUnassignedMVarValues_3940_);
if (lean_obj_tag(v___x_3941_) == 1)
{
lean_object* v_val_3942_; lean_object* v___x_3943_; 
lean_del_object(v___x_3937_);
lean_dec(v_mvarId_3934_);
v_val_3942_ = lean_ctor_get(v___x_3941_, 0);
lean_inc(v_val_3942_);
lean_dec_ref_known(v___x_3941_, 1);
v___x_3943_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_val_3942_, v_e_3935_, v_a_3908_, v_a_3909_, v_a_3910_, v_a_3911_, v_a_3912_, v_a_3913_, v_a_3914_, v_a_3915_);
return v___x_3943_;
}
else
{
lean_object* v___x_3944_; lean_object* v_equalMVarIds_3945_; lean_object* v_equalLMVarIds_3946_; lean_object* v_leftUnassignedMVarValues_3947_; lean_object* v_rightUnassignedMVarValues_3948_; lean_object* v___x_3950_; uint8_t v_isShared_3951_; uint8_t v_isSharedCheck_3961_; 
lean_dec(v___x_3941_);
v___x_3944_ = lean_st_ref_take(v_a_3911_);
v_equalMVarIds_3945_ = lean_ctor_get(v___x_3944_, 0);
v_equalLMVarIds_3946_ = lean_ctor_get(v___x_3944_, 1);
v_leftUnassignedMVarValues_3947_ = lean_ctor_get(v___x_3944_, 2);
v_rightUnassignedMVarValues_3948_ = lean_ctor_get(v___x_3944_, 3);
v_isSharedCheck_3961_ = !lean_is_exclusive(v___x_3944_);
if (v_isSharedCheck_3961_ == 0)
{
v___x_3950_ = v___x_3944_;
v_isShared_3951_ = v_isSharedCheck_3961_;
goto v_resetjp_3949_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3948_);
lean_inc(v_leftUnassignedMVarValues_3947_);
lean_inc(v_equalLMVarIds_3946_);
lean_inc(v_equalMVarIds_3945_);
lean_dec(v___x_3944_);
v___x_3950_ = lean_box(0);
v_isShared_3951_ = v_isSharedCheck_3961_;
goto v_resetjp_3949_;
}
v_resetjp_3949_:
{
lean_object* v___x_3952_; lean_object* v___x_3954_; 
v___x_3952_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10___redArg(v_leftUnassignedMVarValues_3947_, v_mvarId_3934_, v_e_3935_);
if (v_isShared_3951_ == 0)
{
lean_ctor_set(v___x_3950_, 2, v___x_3952_);
v___x_3954_ = v___x_3950_;
goto v_reusejp_3953_;
}
else
{
lean_object* v_reuseFailAlloc_3960_; 
v_reuseFailAlloc_3960_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3960_, 0, v_equalMVarIds_3945_);
lean_ctor_set(v_reuseFailAlloc_3960_, 1, v_equalLMVarIds_3946_);
lean_ctor_set(v_reuseFailAlloc_3960_, 2, v___x_3952_);
lean_ctor_set(v_reuseFailAlloc_3960_, 3, v_rightUnassignedMVarValues_3948_);
v___x_3954_ = v_reuseFailAlloc_3960_;
goto v_reusejp_3953_;
}
v_reusejp_3953_:
{
lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v___x_3958_; 
v___x_3955_ = lean_st_ref_set(v_a_3911_, v___x_3954_);
v___x_3956_ = lean_box(v_allowAssignmentDiff_3924_);
if (v_isShared_3938_ == 0)
{
lean_ctor_set_tag(v___x_3937_, 0);
lean_ctor_set(v___x_3937_, 0, v___x_3956_);
v___x_3958_ = v___x_3937_;
goto v_reusejp_3957_;
}
else
{
lean_object* v_reuseFailAlloc_3959_; 
v_reuseFailAlloc_3959_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3959_, 0, v___x_3956_);
v___x_3958_ = v_reuseFailAlloc_3959_;
goto v_reusejp_3957_;
}
v_reusejp_3957_:
{
return v___x_3958_;
}
}
}
}
}
}
}
default: 
{
lean_dec_ref_known(v_a_3906_, 1);
lean_dec_ref(v_a_3907_);
goto v___jp_3917_;
}
}
}
case 1:
{
switch(lean_obj_tag(v_a_3907_))
{
case 1:
{
lean_object* v___x_3963_; lean_object* v___x_3964_; 
lean_dec_ref_known(v_a_3907_, 1);
lean_dec_ref_known(v_a_3906_, 1);
v___x_3963_ = lean_obj_once(&lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3, &lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3_once, _init_lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___closed__3);
v___x_3964_ = lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11(v___x_3963_, v_a_3908_, v_a_3909_, v_a_3910_, v_a_3911_, v_a_3912_, v_a_3913_, v_a_3914_, v_a_3915_);
return v___x_3964_;
}
case 0:
{
uint8_t v_allowAssignmentDiff_3965_; 
v_allowAssignmentDiff_3965_ = lean_ctor_get_uint8(v_a_3910_, sizeof(void*)*3);
if (v_allowAssignmentDiff_3965_ == 0)
{
lean_object* v___x_3967_; uint8_t v_isShared_3968_; uint8_t v_isSharedCheck_3973_; 
lean_dec_ref_known(v_a_3906_, 1);
v_isSharedCheck_3973_ = !lean_is_exclusive(v_a_3907_);
if (v_isSharedCheck_3973_ == 0)
{
lean_object* v_unused_3974_; 
v_unused_3974_ = lean_ctor_get(v_a_3907_, 0);
lean_dec(v_unused_3974_);
v___x_3967_ = v_a_3907_;
v_isShared_3968_ = v_isSharedCheck_3973_;
goto v_resetjp_3966_;
}
else
{
lean_dec(v_a_3907_);
v___x_3967_ = lean_box(0);
v_isShared_3968_ = v_isSharedCheck_3973_;
goto v_resetjp_3966_;
}
v_resetjp_3966_:
{
lean_object* v___x_3969_; lean_object* v___x_3971_; 
v___x_3969_ = lean_box(v_allowAssignmentDiff_3965_);
if (v_isShared_3968_ == 0)
{
lean_ctor_set(v___x_3967_, 0, v___x_3969_);
v___x_3971_ = v___x_3967_;
goto v_reusejp_3970_;
}
else
{
lean_object* v_reuseFailAlloc_3972_; 
v_reuseFailAlloc_3972_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3972_, 0, v___x_3969_);
v___x_3971_ = v_reuseFailAlloc_3972_;
goto v_reusejp_3970_;
}
v_reusejp_3970_:
{
return v___x_3971_;
}
}
}
else
{
lean_object* v_e_3975_; lean_object* v_mvarId_3976_; lean_object* v___x_3978_; uint8_t v_isShared_3979_; uint8_t v_isSharedCheck_4003_; 
v_e_3975_ = lean_ctor_get(v_a_3906_, 0);
lean_inc_ref(v_e_3975_);
lean_dec_ref_known(v_a_3906_, 1);
v_mvarId_3976_ = lean_ctor_get(v_a_3907_, 0);
v_isSharedCheck_4003_ = !lean_is_exclusive(v_a_3907_);
if (v_isSharedCheck_4003_ == 0)
{
v___x_3978_ = v_a_3907_;
v_isShared_3979_ = v_isSharedCheck_4003_;
goto v_resetjp_3977_;
}
else
{
lean_inc(v_mvarId_3976_);
lean_dec(v_a_3907_);
v___x_3978_ = lean_box(0);
v_isShared_3979_ = v_isSharedCheck_4003_;
goto v_resetjp_3977_;
}
v_resetjp_3977_:
{
lean_object* v___x_3980_; lean_object* v_rightUnassignedMVarValues_3981_; lean_object* v___x_3982_; 
v___x_3980_ = lean_st_ref_get(v_a_3911_);
v_rightUnassignedMVarValues_3981_ = lean_ctor_get(v___x_3980_, 3);
lean_inc_ref(v_rightUnassignedMVarValues_3981_);
lean_dec(v___x_3980_);
v___x_3982_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(v_rightUnassignedMVarValues_3981_, v_mvarId_3976_);
lean_dec_ref(v_rightUnassignedMVarValues_3981_);
if (lean_obj_tag(v___x_3982_) == 1)
{
lean_object* v_val_3983_; lean_object* v___x_3984_; 
lean_del_object(v___x_3978_);
lean_dec(v_mvarId_3976_);
v_val_3983_ = lean_ctor_get(v___x_3982_, 0);
lean_inc(v_val_3983_);
lean_dec_ref_known(v___x_3982_, 1);
v___x_3984_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_e_3975_, v_val_3983_, v_a_3908_, v_a_3909_, v_a_3910_, v_a_3911_, v_a_3912_, v_a_3913_, v_a_3914_, v_a_3915_);
return v___x_3984_;
}
else
{
lean_object* v___x_3985_; lean_object* v_equalMVarIds_3986_; lean_object* v_equalLMVarIds_3987_; lean_object* v_leftUnassignedMVarValues_3988_; lean_object* v_rightUnassignedMVarValues_3989_; lean_object* v___x_3991_; uint8_t v_isShared_3992_; uint8_t v_isSharedCheck_4002_; 
lean_dec(v___x_3982_);
v___x_3985_ = lean_st_ref_take(v_a_3911_);
v_equalMVarIds_3986_ = lean_ctor_get(v___x_3985_, 0);
v_equalLMVarIds_3987_ = lean_ctor_get(v___x_3985_, 1);
v_leftUnassignedMVarValues_3988_ = lean_ctor_get(v___x_3985_, 2);
v_rightUnassignedMVarValues_3989_ = lean_ctor_get(v___x_3985_, 3);
v_isSharedCheck_4002_ = !lean_is_exclusive(v___x_3985_);
if (v_isSharedCheck_4002_ == 0)
{
v___x_3991_ = v___x_3985_;
v_isShared_3992_ = v_isSharedCheck_4002_;
goto v_resetjp_3990_;
}
else
{
lean_inc(v_rightUnassignedMVarValues_3989_);
lean_inc(v_leftUnassignedMVarValues_3988_);
lean_inc(v_equalLMVarIds_3987_);
lean_inc(v_equalMVarIds_3986_);
lean_dec(v___x_3985_);
v___x_3991_ = lean_box(0);
v_isShared_3992_ = v_isSharedCheck_4002_;
goto v_resetjp_3990_;
}
v_resetjp_3990_:
{
lean_object* v___x_3993_; lean_object* v___x_3995_; 
v___x_3993_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10___redArg(v_rightUnassignedMVarValues_3989_, v_mvarId_3976_, v_e_3975_);
if (v_isShared_3992_ == 0)
{
lean_ctor_set(v___x_3991_, 3, v___x_3993_);
v___x_3995_ = v___x_3991_;
goto v_reusejp_3994_;
}
else
{
lean_object* v_reuseFailAlloc_4001_; 
v_reuseFailAlloc_4001_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4001_, 0, v_equalMVarIds_3986_);
lean_ctor_set(v_reuseFailAlloc_4001_, 1, v_equalLMVarIds_3987_);
lean_ctor_set(v_reuseFailAlloc_4001_, 2, v_leftUnassignedMVarValues_3988_);
lean_ctor_set(v_reuseFailAlloc_4001_, 3, v___x_3993_);
v___x_3995_ = v_reuseFailAlloc_4001_;
goto v_reusejp_3994_;
}
v_reusejp_3994_:
{
lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3999_; 
v___x_3996_ = lean_st_ref_set(v_a_3911_, v___x_3995_);
v___x_3997_ = lean_box(v_allowAssignmentDiff_3965_);
if (v_isShared_3979_ == 0)
{
lean_ctor_set(v___x_3978_, 0, v___x_3997_);
v___x_3999_ = v___x_3978_;
goto v_reusejp_3998_;
}
else
{
lean_object* v_reuseFailAlloc_4000_; 
v_reuseFailAlloc_4000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4000_, 0, v___x_3997_);
v___x_3999_ = v_reuseFailAlloc_4000_;
goto v_reusejp_3998_;
}
v_reusejp_3998_:
{
return v___x_3999_;
}
}
}
}
}
}
}
default: 
{
lean_dec_ref_known(v_a_3906_, 1);
lean_dec_ref(v_a_3907_);
goto v___jp_3917_;
}
}
}
default: 
{
if (lean_obj_tag(v_a_3907_) == 2)
{
lean_object* v_da_4004_; lean_object* v_da_4005_; lean_object* v_mvarIdPending_4006_; lean_object* v_mvarIdPending_4007_; lean_object* v___x_4008_; 
v_da_4004_ = lean_ctor_get(v_a_3906_, 0);
lean_inc_ref(v_da_4004_);
lean_dec_ref_known(v_a_3906_, 1);
v_da_4005_ = lean_ctor_get(v_a_3907_, 0);
lean_inc_ref(v_da_4005_);
lean_dec_ref_known(v_a_3907_, 1);
v_mvarIdPending_4006_ = lean_ctor_get(v_da_4004_, 1);
lean_inc(v_mvarIdPending_4006_);
lean_dec_ref(v_da_4004_);
v_mvarIdPending_4007_ = lean_ctor_get(v_da_4005_, 1);
lean_inc(v_mvarIdPending_4007_);
lean_dec_ref(v_da_4005_);
v___x_4008_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(v_mvarIdPending_4006_, v_mvarIdPending_4007_, v_a_3910_, v_a_3911_, v_a_3912_, v_a_3913_, v_a_3914_, v_a_3915_);
return v___x_4008_;
}
else
{
lean_dec_ref_known(v_a_3906_, 1);
lean_dec_ref(v_a_3907_);
goto v___jp_3917_;
}
}
}
v___jp_3917_:
{
uint8_t v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; 
v___x_3918_ = 0;
v___x_3919_ = lean_box(v___x_3918_);
v___x_3920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3920_, 0, v___x_3919_);
return v___x_3920_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083(lean_object* v_x_4013_, lean_object* v_x_4014_, lean_object* v_a_4015_, lean_object* v_a_4016_, lean_object* v_a_4017_, lean_object* v_a_4018_, lean_object* v_a_4019_, lean_object* v_a_4020_, lean_object* v_a_4021_, lean_object* v_a_4022_){
_start:
{
lean_object* v___y_4025_; lean_object* v___y_4026_; lean_object* v___y_4027_; lean_object* v___y_4028_; lean_object* v___y_4029_; lean_object* v___y_4030_; lean_object* v___y_4031_; lean_object* v___y_4032_; lean_object* v___y_4033_; lean_object* v___y_4034_; uint8_t v___y_4035_; lean_object* v___y_4036_; uint8_t v___y_4037_; lean_object* v___y_4038_; lean_object* v___y_4039_; lean_object* v___y_4040_; lean_object* v___y_4054_; lean_object* v___y_4055_; lean_object* v___y_4056_; lean_object* v___y_4057_; lean_object* v___y_4058_; lean_object* v___y_4059_; lean_object* v___y_4060_; lean_object* v___y_4061_; lean_object* v___y_4062_; lean_object* v___y_4063_; uint8_t v___y_4064_; lean_object* v___y_4065_; uint8_t v___y_4066_; lean_object* v___y_4067_; lean_object* v___y_4068_; lean_object* v___y_4069_; uint8_t v___y_4070_; lean_object* v_n_u2081_4074_; lean_object* v_t_u2081_4075_; lean_object* v_e_u2081_4076_; uint8_t v_bi_u2081_4077_; lean_object* v_n_u2082_4078_; lean_object* v_t_u2082_4079_; lean_object* v_e_u2082_4080_; uint8_t v_bi_u2082_4081_; lean_object* v___y_4082_; lean_object* v___y_4083_; lean_object* v___y_4084_; lean_object* v___y_4085_; lean_object* v___y_4086_; lean_object* v___y_4087_; lean_object* v___y_4088_; lean_object* v___y_4089_; lean_object* v_e_u2081_4093_; lean_object* v_m_u2082_4094_; lean_object* v___y_4095_; lean_object* v___y_4096_; lean_object* v___y_4097_; lean_object* v___y_4098_; lean_object* v___y_4099_; lean_object* v___y_4100_; lean_object* v___y_4101_; lean_object* v___y_4102_; lean_object* v___x_4120_; lean_object* v___x_4121_; lean_object* v___f_4122_; lean_object* v___x_4123_; lean_object* v___f_4124_; lean_object* v___x_4125_; lean_object* v_toApplicative_4126_; lean_object* v_toFunctor_4127_; lean_object* v_toSeq_4128_; lean_object* v_toSeqLeft_4129_; lean_object* v_toSeqRight_4130_; lean_object* v___f_4131_; lean_object* v___f_4132_; lean_object* v___f_4133_; lean_object* v___f_4134_; lean_object* v___x_4135_; lean_object* v___f_4136_; lean_object* v___f_4137_; lean_object* v___f_4138_; lean_object* v___x_4139_; lean_object* v___x_4140_; lean_object* v___x_4141_; lean_object* v_toApplicative_4142_; lean_object* v___x_4144_; uint8_t v_isShared_4145_; uint8_t v_isSharedCheck_4375_; 
v___x_4120_ = lean_box(0);
v___x_4121_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__0));
v___f_4122_ = lean_alloc_closure((void*)(l_instBEqProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_4122_, 0, v___x_4121_);
lean_closure_set(v___f_4122_, 1, v___x_4121_);
v___x_4123_ = ((lean_object*)(lp_aesop_panic___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__11___closed__1));
v___f_4124_ = lean_alloc_closure((void*)(l_instHashableProd___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_4124_, 0, v___x_4123_);
lean_closure_set(v___f_4124_, 1, v___x_4123_);
v___x_4125_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_4126_ = lean_ctor_get(v___x_4125_, 0);
v_toFunctor_4127_ = lean_ctor_get(v_toApplicative_4126_, 0);
v_toSeq_4128_ = lean_ctor_get(v_toApplicative_4126_, 2);
v_toSeqLeft_4129_ = lean_ctor_get(v_toApplicative_4126_, 3);
v_toSeqRight_4130_ = lean_ctor_get(v_toApplicative_4126_, 4);
v___f_4131_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_4132_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_4127_, 2);
v___f_4133_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4133_, 0, v_toFunctor_4127_);
v___f_4134_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4134_, 0, v_toFunctor_4127_);
v___x_4135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4135_, 0, v___f_4133_);
lean_ctor_set(v___x_4135_, 1, v___f_4134_);
lean_inc(v_toSeqRight_4130_);
v___f_4136_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4136_, 0, v_toSeqRight_4130_);
lean_inc(v_toSeqLeft_4129_);
v___f_4137_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4137_, 0, v_toSeqLeft_4129_);
lean_inc(v_toSeq_4128_);
v___f_4138_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4138_, 0, v_toSeq_4128_);
v___x_4139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4139_, 0, v___x_4135_);
lean_ctor_set(v___x_4139_, 1, v___f_4131_);
lean_ctor_set(v___x_4139_, 2, v___f_4138_);
lean_ctor_set(v___x_4139_, 3, v___f_4137_);
lean_ctor_set(v___x_4139_, 4, v___f_4136_);
v___x_4140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4140_, 0, v___x_4139_);
lean_ctor_set(v___x_4140_, 1, v___f_4132_);
v___x_4141_ = l_StateRefT_x27_instMonad___redArg(v___x_4140_);
v_toApplicative_4142_ = lean_ctor_get(v___x_4141_, 0);
v_isSharedCheck_4375_ = !lean_is_exclusive(v___x_4141_);
if (v_isSharedCheck_4375_ == 0)
{
lean_object* v_unused_4376_; 
v_unused_4376_ = lean_ctor_get(v___x_4141_, 1);
lean_dec(v_unused_4376_);
v___x_4144_ = v___x_4141_;
v_isShared_4145_ = v_isSharedCheck_4375_;
goto v_resetjp_4143_;
}
else
{
lean_inc(v_toApplicative_4142_);
lean_dec(v___x_4141_);
v___x_4144_ = lean_box(0);
v_isShared_4145_ = v_isSharedCheck_4375_;
goto v_resetjp_4143_;
}
v___jp_4024_:
{
lean_object* v___x_4041_; lean_object* v___x_4042_; uint8_t v___x_4043_; 
v___x_4041_ = l_Lean_Name_eraseMacroScopes(v___y_4040_);
lean_dec(v___y_4040_);
v___x_4042_ = l_Lean_Name_eraseMacroScopes(v___y_4033_);
lean_dec(v___y_4033_);
v___x_4043_ = lean_name_eq(v___x_4041_, v___x_4042_);
lean_dec(v___x_4042_);
lean_dec(v___x_4041_);
if (v___x_4043_ == 0)
{
lean_object* v___x_4044_; lean_object* v___x_4045_; 
lean_dec_ref(v___y_4032_);
lean_dec_ref(v___y_4031_);
lean_dec_ref(v___y_4028_);
lean_dec_ref(v___y_4027_);
v___x_4044_ = lean_box(v___x_4043_);
v___x_4045_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4045_, 0, v___x_4044_);
return v___x_4045_;
}
else
{
uint8_t v___x_4046_; 
v___x_4046_ = l_Lean_instBEqBinderInfo_beq(v___y_4035_, v___y_4037_);
if (v___x_4046_ == 0)
{
lean_object* v___x_4047_; lean_object* v___x_4048_; 
lean_dec_ref(v___y_4032_);
lean_dec_ref(v___y_4031_);
lean_dec_ref(v___y_4028_);
lean_dec_ref(v___y_4027_);
v___x_4047_ = lean_box(v___x_4046_);
v___x_4048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4048_, 0, v___x_4047_);
return v___x_4048_;
}
else
{
lean_object* v___x_4049_; 
v___x_4049_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v___y_4031_, v___y_4032_, v___y_4039_, v___y_4038_, v___y_4025_, v___y_4029_, v___y_4036_, v___y_4034_, v___y_4030_, v___y_4026_);
if (lean_obj_tag(v___x_4049_) == 0)
{
lean_object* v_a_4050_; uint8_t v___x_4051_; 
v_a_4050_ = lean_ctor_get(v___x_4049_, 0);
lean_inc(v_a_4050_);
v___x_4051_ = lean_unbox(v_a_4050_);
lean_dec(v_a_4050_);
if (v___x_4051_ == 0)
{
lean_dec_ref(v___y_4028_);
lean_dec_ref(v___y_4027_);
return v___x_4049_;
}
else
{
lean_object* v___x_4052_; 
lean_dec_ref_known(v___x_4049_, 1);
v___x_4052_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v___y_4027_, v___y_4028_, v___y_4039_, v___y_4038_, v___y_4025_, v___y_4029_, v___y_4036_, v___y_4034_, v___y_4030_, v___y_4026_);
return v___x_4052_;
}
}
else
{
lean_dec_ref(v___y_4028_);
lean_dec_ref(v___y_4027_);
return v___x_4049_;
}
}
}
}
v___jp_4053_:
{
if (v___y_4070_ == 0)
{
lean_object* v___x_4071_; lean_object* v___x_4072_; 
lean_dec(v___y_4069_);
lean_dec(v___y_4062_);
lean_dec_ref(v___y_4061_);
lean_dec_ref(v___y_4060_);
lean_dec_ref(v___y_4057_);
lean_dec_ref(v___y_4056_);
v___x_4071_ = lean_box(v___y_4070_);
v___x_4072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4072_, 0, v___x_4071_);
return v___x_4072_;
}
else
{
v___y_4025_ = v___y_4054_;
v___y_4026_ = v___y_4055_;
v___y_4027_ = v___y_4056_;
v___y_4028_ = v___y_4057_;
v___y_4029_ = v___y_4058_;
v___y_4030_ = v___y_4059_;
v___y_4031_ = v___y_4060_;
v___y_4032_ = v___y_4061_;
v___y_4033_ = v___y_4062_;
v___y_4034_ = v___y_4063_;
v___y_4035_ = v___y_4064_;
v___y_4036_ = v___y_4065_;
v___y_4037_ = v___y_4066_;
v___y_4038_ = v___y_4067_;
v___y_4039_ = v___y_4068_;
v___y_4040_ = v___y_4069_;
goto v___jp_4024_;
}
}
v___jp_4073_:
{
uint8_t v___x_4090_; uint8_t v___x_4091_; 
v___x_4090_ = l_Lean_Name_hasMacroScopes(v_n_u2081_4074_);
v___x_4091_ = l_Lean_Name_hasMacroScopes(v_n_u2082_4078_);
if (v___x_4090_ == 0)
{
if (v___x_4091_ == 0)
{
v___y_4025_ = v___y_4084_;
v___y_4026_ = v___y_4089_;
v___y_4027_ = v_e_u2081_4076_;
v___y_4028_ = v_e_u2082_4080_;
v___y_4029_ = v___y_4085_;
v___y_4030_ = v___y_4088_;
v___y_4031_ = v_t_u2081_4075_;
v___y_4032_ = v_t_u2082_4079_;
v___y_4033_ = v_n_u2082_4078_;
v___y_4034_ = v___y_4087_;
v___y_4035_ = v_bi_u2081_4077_;
v___y_4036_ = v___y_4086_;
v___y_4037_ = v_bi_u2082_4081_;
v___y_4038_ = v___y_4083_;
v___y_4039_ = v___y_4082_;
v___y_4040_ = v_n_u2081_4074_;
goto v___jp_4024_;
}
else
{
v___y_4054_ = v___y_4084_;
v___y_4055_ = v___y_4089_;
v___y_4056_ = v_e_u2081_4076_;
v___y_4057_ = v_e_u2082_4080_;
v___y_4058_ = v___y_4085_;
v___y_4059_ = v___y_4088_;
v___y_4060_ = v_t_u2081_4075_;
v___y_4061_ = v_t_u2082_4079_;
v___y_4062_ = v_n_u2082_4078_;
v___y_4063_ = v___y_4087_;
v___y_4064_ = v_bi_u2081_4077_;
v___y_4065_ = v___y_4086_;
v___y_4066_ = v_bi_u2082_4081_;
v___y_4067_ = v___y_4083_;
v___y_4068_ = v___y_4082_;
v___y_4069_ = v_n_u2081_4074_;
v___y_4070_ = v___x_4090_;
goto v___jp_4053_;
}
}
else
{
v___y_4054_ = v___y_4084_;
v___y_4055_ = v___y_4089_;
v___y_4056_ = v_e_u2081_4076_;
v___y_4057_ = v_e_u2082_4080_;
v___y_4058_ = v___y_4085_;
v___y_4059_ = v___y_4088_;
v___y_4060_ = v_t_u2081_4075_;
v___y_4061_ = v_t_u2082_4079_;
v___y_4062_ = v_n_u2082_4078_;
v___y_4063_ = v___y_4087_;
v___y_4064_ = v_bi_u2081_4077_;
v___y_4065_ = v___y_4086_;
v___y_4066_ = v_bi_u2082_4081_;
v___y_4067_ = v___y_4083_;
v___y_4068_ = v___y_4082_;
v___y_4069_ = v_n_u2081_4074_;
v___y_4070_ = v___x_4091_;
goto v___jp_4053_;
}
}
v___jp_4092_:
{
lean_object* v_mctx_u2082_4103_; lean_object* v___x_4104_; 
v_mctx_u2082_4103_ = lean_ctor_get(v___y_4097_, 2);
lean_inc_ref(v_mctx_u2082_4103_);
v___x_4104_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(v_mctx_u2082_4103_, v_m_u2082_4094_, v___y_4099_, v___y_4100_, v___y_4101_, v___y_4102_);
if (lean_obj_tag(v___x_4104_) == 0)
{
lean_object* v_a_4105_; lean_object* v___x_4106_; lean_object* v___x_4107_; 
v_a_4105_ = lean_ctor_get(v___x_4104_, 0);
lean_inc(v_a_4105_);
lean_dec_ref_known(v___x_4104_, 1);
v___x_4106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4106_, 0, v_e_u2081_4093_);
v___x_4107_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(v___x_4106_, v_a_4105_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_, v___y_4099_, v___y_4100_, v___y_4101_, v___y_4102_);
return v___x_4107_;
}
else
{
lean_object* v_a_4108_; lean_object* v___x_4110_; uint8_t v_isShared_4111_; uint8_t v_isSharedCheck_4115_; 
lean_dec_ref(v_e_u2081_4093_);
v_a_4108_ = lean_ctor_get(v___x_4104_, 0);
v_isSharedCheck_4115_ = !lean_is_exclusive(v___x_4104_);
if (v_isSharedCheck_4115_ == 0)
{
v___x_4110_ = v___x_4104_;
v_isShared_4111_ = v_isSharedCheck_4115_;
goto v_resetjp_4109_;
}
else
{
lean_inc(v_a_4108_);
lean_dec(v___x_4104_);
v___x_4110_ = lean_box(0);
v_isShared_4111_ = v_isSharedCheck_4115_;
goto v_resetjp_4109_;
}
v_resetjp_4109_:
{
lean_object* v___x_4113_; 
if (v_isShared_4111_ == 0)
{
v___x_4113_ = v___x_4110_;
goto v_reusejp_4112_;
}
else
{
lean_object* v_reuseFailAlloc_4114_; 
v_reuseFailAlloc_4114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4114_, 0, v_a_4108_);
v___x_4113_ = v_reuseFailAlloc_4114_;
goto v_reusejp_4112_;
}
v_reusejp_4112_:
{
return v___x_4113_;
}
}
}
}
v___jp_4116_:
{
uint8_t v___x_4117_; lean_object* v___x_4118_; lean_object* v___x_4119_; 
v___x_4117_ = 0;
v___x_4118_ = lean_box(v___x_4117_);
v___x_4119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4119_, 0, v___x_4118_);
return v___x_4119_;
}
v_resetjp_4143_:
{
lean_object* v_toFunctor_4146_; lean_object* v_toSeq_4147_; lean_object* v_toSeqLeft_4148_; lean_object* v_toSeqRight_4149_; lean_object* v___x_4151_; uint8_t v_isShared_4152_; uint8_t v_isSharedCheck_4373_; 
v_toFunctor_4146_ = lean_ctor_get(v_toApplicative_4142_, 0);
v_toSeq_4147_ = lean_ctor_get(v_toApplicative_4142_, 2);
v_toSeqLeft_4148_ = lean_ctor_get(v_toApplicative_4142_, 3);
v_toSeqRight_4149_ = lean_ctor_get(v_toApplicative_4142_, 4);
v_isSharedCheck_4373_ = !lean_is_exclusive(v_toApplicative_4142_);
if (v_isSharedCheck_4373_ == 0)
{
lean_object* v_unused_4374_; 
v_unused_4374_ = lean_ctor_get(v_toApplicative_4142_, 1);
lean_dec(v_unused_4374_);
v___x_4151_ = v_toApplicative_4142_;
v_isShared_4152_ = v_isSharedCheck_4373_;
goto v_resetjp_4150_;
}
else
{
lean_inc(v_toSeqRight_4149_);
lean_inc(v_toSeqLeft_4148_);
lean_inc(v_toSeq_4147_);
lean_inc(v_toFunctor_4146_);
lean_dec(v_toApplicative_4142_);
v___x_4151_ = lean_box(0);
v_isShared_4152_ = v_isSharedCheck_4373_;
goto v_resetjp_4150_;
}
v_resetjp_4150_:
{
lean_object* v___f_4153_; lean_object* v___f_4154_; lean_object* v___f_4155_; lean_object* v___f_4156_; lean_object* v___x_4157_; lean_object* v___f_4158_; lean_object* v___f_4159_; lean_object* v___f_4160_; lean_object* v___x_4162_; 
v___f_4153_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_4154_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_4146_);
v___f_4155_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4155_, 0, v_toFunctor_4146_);
v___f_4156_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4156_, 0, v_toFunctor_4146_);
v___x_4157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4157_, 0, v___f_4155_);
lean_ctor_set(v___x_4157_, 1, v___f_4156_);
v___f_4158_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4158_, 0, v_toSeqRight_4149_);
v___f_4159_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4159_, 0, v_toSeqLeft_4148_);
v___f_4160_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4160_, 0, v_toSeq_4147_);
if (v_isShared_4152_ == 0)
{
lean_ctor_set(v___x_4151_, 4, v___f_4158_);
lean_ctor_set(v___x_4151_, 3, v___f_4159_);
lean_ctor_set(v___x_4151_, 2, v___f_4160_);
lean_ctor_set(v___x_4151_, 1, v___f_4153_);
lean_ctor_set(v___x_4151_, 0, v___x_4157_);
v___x_4162_ = v___x_4151_;
goto v_reusejp_4161_;
}
else
{
lean_object* v_reuseFailAlloc_4372_; 
v_reuseFailAlloc_4372_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4372_, 0, v___x_4157_);
lean_ctor_set(v_reuseFailAlloc_4372_, 1, v___f_4153_);
lean_ctor_set(v_reuseFailAlloc_4372_, 2, v___f_4160_);
lean_ctor_set(v_reuseFailAlloc_4372_, 3, v___f_4159_);
lean_ctor_set(v_reuseFailAlloc_4372_, 4, v___f_4158_);
v___x_4162_ = v_reuseFailAlloc_4372_;
goto v_reusejp_4161_;
}
v_reusejp_4161_:
{
lean_object* v___x_4164_; 
if (v_isShared_4145_ == 0)
{
lean_ctor_set(v___x_4144_, 1, v___f_4154_);
lean_ctor_set(v___x_4144_, 0, v___x_4162_);
v___x_4164_ = v___x_4144_;
goto v_reusejp_4163_;
}
else
{
lean_object* v_reuseFailAlloc_4371_; 
v_reuseFailAlloc_4371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4371_, 0, v___x_4162_);
lean_ctor_set(v_reuseFailAlloc_4371_, 1, v___f_4154_);
v___x_4164_ = v_reuseFailAlloc_4371_;
goto v_reusejp_4163_;
}
v_reusejp_4163_:
{
lean_object* v___x_4165_; lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_4168_; 
v___x_4165_ = l_StateRefT_x27_instMonad___redArg(v___x_4164_);
v___x_4166_ = l_ReaderT_instMonad___redArg(v___x_4165_);
v___x_4167_ = l_ReaderT_instMonad___redArg(v___x_4166_);
v___x_4168_ = l_Lean_MonadCacheT_instMonad___redArg(v___x_4120_, v___f_4122_, v___f_4124_, v___x_4167_);
switch(lean_obj_tag(v_x_4013_))
{
case 0:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 0:
{
lean_object* v_deBruijnIndex_4169_; lean_object* v_deBruijnIndex_4170_; uint8_t v___x_4171_; lean_object* v___x_4172_; lean_object* v___x_4173_; 
v_deBruijnIndex_4169_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_deBruijnIndex_4169_);
lean_dec_ref_known(v_x_4013_, 1);
v_deBruijnIndex_4170_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_deBruijnIndex_4170_);
lean_dec_ref_known(v_x_4014_, 1);
v___x_4171_ = lean_nat_dec_eq(v_deBruijnIndex_4169_, v_deBruijnIndex_4170_);
lean_dec(v_deBruijnIndex_4170_);
lean_dec(v_deBruijnIndex_4169_);
v___x_4172_ = lean_box(v___x_4171_);
v___x_4173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4173_, 0, v___x_4172_);
return v___x_4173_;
}
case 10:
{
lean_object* v_expr_4174_; lean_object* v___x_4175_; 
v_expr_4174_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4174_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4175_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4174_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4175_;
}
case 2:
{
lean_object* v_mvarId_4176_; 
v_mvarId_4176_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4176_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4176_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 1);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 1:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 1:
{
lean_object* v_fvarId_4177_; lean_object* v_fvarId_4178_; uint8_t v___x_4179_; 
v_fvarId_4177_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_fvarId_4177_);
lean_dec_ref_known(v_x_4013_, 1);
v_fvarId_4178_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_fvarId_4178_);
lean_dec_ref_known(v_x_4014_, 1);
v___x_4179_ = l_Lean_instBEqFVarId_beq(v_fvarId_4177_, v_fvarId_4178_);
if (v___x_4179_ == 0)
{
lean_object* v_equalFVarIds_4180_; lean_object* v___x_4181_; lean_object* v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; uint8_t v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; 
v_equalFVarIds_4180_ = lean_ctor_get(v_a_4016_, 4);
v___x_4181_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__0));
v___x_4182_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__1));
v___x_4183_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_4181_, v___x_4182_, v_equalFVarIds_4180_, v_fvarId_4177_);
v___x_4184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4184_, 0, v_fvarId_4178_);
v___x_4185_ = l_Option_instBEq_beq___redArg(v___x_4181_, v___x_4183_, v___x_4184_);
v___x_4186_ = lean_box(v___x_4185_);
v___x_4187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4187_, 0, v___x_4186_);
return v___x_4187_;
}
else
{
lean_object* v___x_4188_; lean_object* v___x_4189_; 
lean_dec(v_fvarId_4178_);
lean_dec(v_fvarId_4177_);
v___x_4188_ = lean_box(v___x_4179_);
v___x_4189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4189_, 0, v___x_4188_);
return v___x_4189_;
}
}
case 10:
{
lean_object* v_expr_4190_; lean_object* v___x_4191_; 
v_expr_4190_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4190_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4191_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4190_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4191_;
}
case 2:
{
lean_object* v_mvarId_4192_; 
v_mvarId_4192_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4192_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4192_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 1);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 2:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 10:
{
lean_object* v_expr_4193_; lean_object* v___x_4194_; 
v_expr_4193_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4193_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4194_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4193_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4194_;
}
case 2:
{
lean_object* v_mvarId_4195_; lean_object* v_mvarId_4196_; lean_object* v_mctx_u2081_4197_; lean_object* v_mctx_u2082_4198_; lean_object* v___x_4199_; 
v_mvarId_4195_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_mvarId_4195_);
lean_dec_ref_known(v_x_4013_, 1);
v_mvarId_4196_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4196_);
lean_dec_ref_known(v_x_4014_, 1);
v_mctx_u2081_4197_ = lean_ctor_get(v_a_4017_, 1);
v_mctx_u2082_4198_ = lean_ctor_get(v_a_4017_, 2);
lean_inc_ref(v_mctx_u2081_4197_);
v___x_4199_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(v_mctx_u2081_4197_, v_mvarId_4195_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4199_) == 0)
{
lean_object* v_a_4200_; lean_object* v___x_4201_; 
v_a_4200_ = lean_ctor_get(v___x_4199_, 0);
lean_inc(v_a_4200_);
lean_dec_ref_known(v___x_4199_, 1);
lean_inc_ref(v_mctx_u2082_4198_);
v___x_4201_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(v_mctx_u2082_4198_, v_mvarId_4196_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4201_) == 0)
{
lean_object* v_a_4202_; lean_object* v___x_4203_; 
v_a_4202_ = lean_ctor_get(v___x_4201_, 0);
lean_inc(v_a_4202_);
lean_dec_ref_known(v___x_4201_, 1);
v___x_4203_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(v_a_4200_, v_a_4202_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4203_;
}
else
{
lean_object* v_a_4204_; lean_object* v___x_4206_; uint8_t v_isShared_4207_; uint8_t v_isSharedCheck_4211_; 
lean_dec(v_a_4200_);
v_a_4204_ = lean_ctor_get(v___x_4201_, 0);
v_isSharedCheck_4211_ = !lean_is_exclusive(v___x_4201_);
if (v_isSharedCheck_4211_ == 0)
{
v___x_4206_ = v___x_4201_;
v_isShared_4207_ = v_isSharedCheck_4211_;
goto v_resetjp_4205_;
}
else
{
lean_inc(v_a_4204_);
lean_dec(v___x_4201_);
v___x_4206_ = lean_box(0);
v_isShared_4207_ = v_isSharedCheck_4211_;
goto v_resetjp_4205_;
}
v_resetjp_4205_:
{
lean_object* v___x_4209_; 
if (v_isShared_4207_ == 0)
{
v___x_4209_ = v___x_4206_;
goto v_reusejp_4208_;
}
else
{
lean_object* v_reuseFailAlloc_4210_; 
v_reuseFailAlloc_4210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4210_, 0, v_a_4204_);
v___x_4209_ = v_reuseFailAlloc_4210_;
goto v_reusejp_4208_;
}
v_reusejp_4208_:
{
return v___x_4209_;
}
}
}
}
else
{
lean_object* v_a_4212_; lean_object* v___x_4214_; uint8_t v_isShared_4215_; uint8_t v_isSharedCheck_4219_; 
lean_dec(v_mvarId_4196_);
v_a_4212_ = lean_ctor_get(v___x_4199_, 0);
v_isSharedCheck_4219_ = !lean_is_exclusive(v___x_4199_);
if (v_isSharedCheck_4219_ == 0)
{
v___x_4214_ = v___x_4199_;
v_isShared_4215_ = v_isSharedCheck_4219_;
goto v_resetjp_4213_;
}
else
{
lean_inc(v_a_4212_);
lean_dec(v___x_4199_);
v___x_4214_ = lean_box(0);
v_isShared_4215_ = v_isSharedCheck_4219_;
goto v_resetjp_4213_;
}
v_resetjp_4213_:
{
lean_object* v___x_4217_; 
if (v_isShared_4215_ == 0)
{
v___x_4217_ = v___x_4214_;
goto v_reusejp_4216_;
}
else
{
lean_object* v_reuseFailAlloc_4218_; 
v_reuseFailAlloc_4218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4218_, 0, v_a_4212_);
v___x_4217_ = v_reuseFailAlloc_4218_;
goto v_reusejp_4216_;
}
v_reusejp_4216_:
{
return v___x_4217_;
}
}
}
}
default: 
{
lean_object* v_mvarId_4220_; lean_object* v_mctx_u2081_4221_; lean_object* v___x_4222_; 
v_mvarId_4220_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_mvarId_4220_);
lean_dec_ref_known(v_x_4013_, 1);
v_mctx_u2081_4221_ = lean_ctor_get(v_a_4017_, 1);
lean_inc_ref(v_mctx_u2081_4221_);
v___x_4222_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_normalizeMVar(v_mctx_u2081_4221_, v_mvarId_4220_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4222_) == 0)
{
lean_object* v_a_4223_; lean_object* v___x_4224_; lean_object* v___x_4225_; 
v_a_4223_ = lean_ctor_get(v___x_4222_, 0);
lean_inc(v_a_4223_);
lean_dec_ref_known(v___x_4222_, 1);
v___x_4224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4224_, 0, v_x_4014_);
v___x_4225_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(v_a_4223_, v___x_4224_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4225_;
}
else
{
lean_object* v_a_4226_; lean_object* v___x_4228_; uint8_t v_isShared_4229_; uint8_t v_isSharedCheck_4233_; 
lean_dec_ref(v_x_4014_);
v_a_4226_ = lean_ctor_get(v___x_4222_, 0);
v_isSharedCheck_4233_ = !lean_is_exclusive(v___x_4222_);
if (v_isSharedCheck_4233_ == 0)
{
v___x_4228_ = v___x_4222_;
v_isShared_4229_ = v_isSharedCheck_4233_;
goto v_resetjp_4227_;
}
else
{
lean_inc(v_a_4226_);
lean_dec(v___x_4222_);
v___x_4228_ = lean_box(0);
v_isShared_4229_ = v_isSharedCheck_4233_;
goto v_resetjp_4227_;
}
v_resetjp_4227_:
{
lean_object* v___x_4231_; 
if (v_isShared_4229_ == 0)
{
v___x_4231_ = v___x_4228_;
goto v_reusejp_4230_;
}
else
{
lean_object* v_reuseFailAlloc_4232_; 
v_reuseFailAlloc_4232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4232_, 0, v_a_4226_);
v___x_4231_ = v_reuseFailAlloc_4232_;
goto v_reusejp_4230_;
}
v_reusejp_4230_:
{
return v___x_4231_;
}
}
}
}
}
}
case 3:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 3:
{
lean_object* v_u_4234_; lean_object* v_u_4235_; lean_object* v___x_4236_; 
v_u_4234_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_u_4234_);
lean_dec_ref_known(v_x_4013_, 1);
v_u_4235_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_u_4235_);
lean_dec_ref_known(v_x_4014_, 1);
v___x_4236_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_levelsEqualUpToIdsCore(v_u_4234_, v_u_4235_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4236_;
}
case 10:
{
lean_object* v_expr_4237_; lean_object* v___x_4238_; 
v_expr_4237_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4237_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4238_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4237_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4238_;
}
case 2:
{
lean_object* v_mvarId_4239_; 
v_mvarId_4239_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4239_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4239_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 1);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 4:
{
switch(lean_obj_tag(v_x_4014_))
{
case 4:
{
lean_object* v_declName_4240_; lean_object* v_us_4241_; lean_object* v_declName_4242_; lean_object* v_us_4243_; uint8_t v___y_4245_; uint8_t v___x_4275_; 
v_declName_4240_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_declName_4240_);
v_us_4241_ = lean_ctor_get(v_x_4013_, 1);
lean_inc(v_us_4241_);
lean_dec_ref_known(v_x_4013_, 2);
v_declName_4242_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_declName_4242_);
v_us_4243_ = lean_ctor_get(v_x_4014_, 1);
lean_inc(v_us_4243_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4275_ = lean_name_eq(v_declName_4240_, v_declName_4242_);
lean_dec(v_declName_4242_);
lean_dec(v_declName_4240_);
if (v___x_4275_ == 0)
{
v___y_4245_ = v___x_4275_;
goto v___jp_4244_;
}
else
{
lean_object* v___x_4276_; lean_object* v___x_4277_; uint8_t v___x_4278_; 
v___x_4276_ = l_List_lengthTR___redArg(v_us_4241_);
v___x_4277_ = l_List_lengthTR___redArg(v_us_4243_);
v___x_4278_ = lean_nat_dec_eq(v___x_4276_, v___x_4277_);
lean_dec(v___x_4277_);
lean_dec(v___x_4276_);
v___y_4245_ = v___x_4278_;
goto v___jp_4244_;
}
v___jp_4244_:
{
if (v___y_4245_ == 0)
{
lean_object* v___x_4246_; lean_object* v___x_4247_; 
lean_dec(v_us_4243_);
lean_dec(v_us_4241_);
lean_dec_ref(v___x_4168_);
v___x_4246_ = lean_box(v___y_4245_);
v___x_4247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4247_, 0, v___x_4246_);
return v___x_4247_;
}
else
{
lean_object* v___x_4248_; lean_object* v___f_4249_; lean_object* v___x_4250_; lean_object* v___x_100649__overap_4251_; lean_object* v___x_4252_; 
v___x_4248_ = lean_box(0);
v___f_4249_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___closed__2));
v___x_4250_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4250_, 0, v___x_4248_);
lean_ctor_set(v___x_4250_, 1, v_us_4243_);
v___x_100649__overap_4251_ = l_List_forIn_x27_loop___redArg(v___x_4168_, v___f_4249_, v_us_4241_, v___x_4250_);
lean_dec(v_us_4241_);
lean_inc(v_a_4022_);
lean_inc_ref(v_a_4021_);
lean_inc(v_a_4020_);
lean_inc_ref(v_a_4019_);
lean_inc(v_a_4018_);
lean_inc_ref(v_a_4017_);
lean_inc_ref(v_a_4016_);
lean_inc(v_a_4015_);
v___x_4252_ = lean_apply_9(v___x_100649__overap_4251_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_, lean_box(0));
if (lean_obj_tag(v___x_4252_) == 0)
{
lean_object* v_a_4253_; lean_object* v___x_4255_; uint8_t v_isShared_4256_; uint8_t v_isSharedCheck_4266_; 
v_a_4253_ = lean_ctor_get(v___x_4252_, 0);
v_isSharedCheck_4266_ = !lean_is_exclusive(v___x_4252_);
if (v_isSharedCheck_4266_ == 0)
{
v___x_4255_ = v___x_4252_;
v_isShared_4256_ = v_isSharedCheck_4266_;
goto v_resetjp_4254_;
}
else
{
lean_inc(v_a_4253_);
lean_dec(v___x_4252_);
v___x_4255_ = lean_box(0);
v_isShared_4256_ = v_isSharedCheck_4266_;
goto v_resetjp_4254_;
}
v_resetjp_4254_:
{
lean_object* v_fst_4257_; 
v_fst_4257_ = lean_ctor_get(v_a_4253_, 0);
lean_inc(v_fst_4257_);
lean_dec(v_a_4253_);
if (lean_obj_tag(v_fst_4257_) == 0)
{
lean_object* v___x_4258_; lean_object* v___x_4260_; 
v___x_4258_ = lean_box(v___y_4245_);
if (v_isShared_4256_ == 0)
{
lean_ctor_set(v___x_4255_, 0, v___x_4258_);
v___x_4260_ = v___x_4255_;
goto v_reusejp_4259_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v___x_4258_);
v___x_4260_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4259_;
}
v_reusejp_4259_:
{
return v___x_4260_;
}
}
else
{
lean_object* v_val_4262_; lean_object* v___x_4264_; 
v_val_4262_ = lean_ctor_get(v_fst_4257_, 0);
lean_inc(v_val_4262_);
lean_dec_ref_known(v_fst_4257_, 1);
if (v_isShared_4256_ == 0)
{
lean_ctor_set(v___x_4255_, 0, v_val_4262_);
v___x_4264_ = v___x_4255_;
goto v_reusejp_4263_;
}
else
{
lean_object* v_reuseFailAlloc_4265_; 
v_reuseFailAlloc_4265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4265_, 0, v_val_4262_);
v___x_4264_ = v_reuseFailAlloc_4265_;
goto v_reusejp_4263_;
}
v_reusejp_4263_:
{
return v___x_4264_;
}
}
}
}
else
{
lean_object* v_a_4267_; lean_object* v___x_4269_; uint8_t v_isShared_4270_; uint8_t v_isSharedCheck_4274_; 
v_a_4267_ = lean_ctor_get(v___x_4252_, 0);
v_isSharedCheck_4274_ = !lean_is_exclusive(v___x_4252_);
if (v_isSharedCheck_4274_ == 0)
{
v___x_4269_ = v___x_4252_;
v_isShared_4270_ = v_isSharedCheck_4274_;
goto v_resetjp_4268_;
}
else
{
lean_inc(v_a_4267_);
lean_dec(v___x_4252_);
v___x_4269_ = lean_box(0);
v_isShared_4270_ = v_isSharedCheck_4274_;
goto v_resetjp_4268_;
}
v_resetjp_4268_:
{
lean_object* v___x_4272_; 
if (v_isShared_4270_ == 0)
{
v___x_4272_ = v___x_4269_;
goto v_reusejp_4271_;
}
else
{
lean_object* v_reuseFailAlloc_4273_; 
v_reuseFailAlloc_4273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4273_, 0, v_a_4267_);
v___x_4272_ = v_reuseFailAlloc_4273_;
goto v_reusejp_4271_;
}
v_reusejp_4271_:
{
return v___x_4272_;
}
}
}
}
}
}
case 10:
{
lean_object* v_expr_4279_; lean_object* v___x_4280_; 
lean_dec_ref(v___x_4168_);
v_expr_4279_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4279_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4280_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4279_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4280_;
}
case 2:
{
lean_object* v_mvarId_4281_; 
lean_dec_ref(v___x_4168_);
v_mvarId_4281_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4281_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4281_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 2);
lean_dec_ref(v___x_4168_);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 5:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 5:
{
lean_object* v_fn_4282_; lean_object* v_arg_4283_; lean_object* v_fn_4284_; lean_object* v_arg_4285_; lean_object* v___x_4286_; 
v_fn_4282_ = lean_ctor_get(v_x_4013_, 0);
lean_inc_ref(v_fn_4282_);
v_arg_4283_ = lean_ctor_get(v_x_4013_, 1);
lean_inc_ref(v_arg_4283_);
lean_dec_ref_known(v_x_4013_, 2);
v_fn_4284_ = lean_ctor_get(v_x_4014_, 0);
lean_inc_ref(v_fn_4284_);
v_arg_4285_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_arg_4285_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4286_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_fn_4282_, v_fn_4284_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4286_) == 0)
{
lean_object* v_a_4287_; uint8_t v___x_4288_; 
v_a_4287_ = lean_ctor_get(v___x_4286_, 0);
lean_inc(v_a_4287_);
v___x_4288_ = lean_unbox(v_a_4287_);
lean_dec(v_a_4287_);
if (v___x_4288_ == 0)
{
lean_dec_ref(v_arg_4285_);
lean_dec_ref(v_arg_4283_);
return v___x_4286_;
}
else
{
lean_object* v___x_4289_; 
lean_dec_ref_known(v___x_4286_, 1);
v___x_4289_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_arg_4283_, v_arg_4285_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4289_;
}
}
else
{
lean_dec_ref(v_arg_4285_);
lean_dec_ref(v_arg_4283_);
return v___x_4286_;
}
}
case 10:
{
lean_object* v_expr_4290_; lean_object* v___x_4291_; 
v_expr_4290_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4290_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4291_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4290_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4291_;
}
case 2:
{
lean_object* v_mvarId_4292_; 
v_mvarId_4292_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4292_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4292_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 2);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 6:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 6:
{
lean_object* v_binderName_4293_; lean_object* v_binderType_4294_; lean_object* v_body_4295_; uint8_t v_binderInfo_4296_; lean_object* v_binderName_4297_; lean_object* v_binderType_4298_; lean_object* v_body_4299_; uint8_t v_binderInfo_4300_; 
v_binderName_4293_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_binderName_4293_);
v_binderType_4294_ = lean_ctor_get(v_x_4013_, 1);
lean_inc_ref(v_binderType_4294_);
v_body_4295_ = lean_ctor_get(v_x_4013_, 2);
lean_inc_ref(v_body_4295_);
v_binderInfo_4296_ = lean_ctor_get_uint8(v_x_4013_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_4013_, 3);
v_binderName_4297_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_binderName_4297_);
v_binderType_4298_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_binderType_4298_);
v_body_4299_ = lean_ctor_get(v_x_4014_, 2);
lean_inc_ref(v_body_4299_);
v_binderInfo_4300_ = lean_ctor_get_uint8(v_x_4014_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_4014_, 3);
v_n_u2081_4074_ = v_binderName_4293_;
v_t_u2081_4075_ = v_binderType_4294_;
v_e_u2081_4076_ = v_body_4295_;
v_bi_u2081_4077_ = v_binderInfo_4296_;
v_n_u2082_4078_ = v_binderName_4297_;
v_t_u2082_4079_ = v_binderType_4298_;
v_e_u2082_4080_ = v_body_4299_;
v_bi_u2082_4081_ = v_binderInfo_4300_;
v___y_4082_ = v_a_4015_;
v___y_4083_ = v_a_4016_;
v___y_4084_ = v_a_4017_;
v___y_4085_ = v_a_4018_;
v___y_4086_ = v_a_4019_;
v___y_4087_ = v_a_4020_;
v___y_4088_ = v_a_4021_;
v___y_4089_ = v_a_4022_;
goto v___jp_4073_;
}
case 10:
{
lean_object* v_expr_4301_; lean_object* v___x_4302_; 
v_expr_4301_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4301_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4302_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4301_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4302_;
}
case 2:
{
lean_object* v_mvarId_4303_; 
v_mvarId_4303_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4303_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4303_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 3);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 7:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 7:
{
lean_object* v_binderName_4304_; lean_object* v_binderType_4305_; lean_object* v_body_4306_; uint8_t v_binderInfo_4307_; lean_object* v_binderName_4308_; lean_object* v_binderType_4309_; lean_object* v_body_4310_; uint8_t v_binderInfo_4311_; 
v_binderName_4304_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_binderName_4304_);
v_binderType_4305_ = lean_ctor_get(v_x_4013_, 1);
lean_inc_ref(v_binderType_4305_);
v_body_4306_ = lean_ctor_get(v_x_4013_, 2);
lean_inc_ref(v_body_4306_);
v_binderInfo_4307_ = lean_ctor_get_uint8(v_x_4013_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_4013_, 3);
v_binderName_4308_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_binderName_4308_);
v_binderType_4309_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_binderType_4309_);
v_body_4310_ = lean_ctor_get(v_x_4014_, 2);
lean_inc_ref(v_body_4310_);
v_binderInfo_4311_ = lean_ctor_get_uint8(v_x_4014_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_4014_, 3);
v_n_u2081_4074_ = v_binderName_4304_;
v_t_u2081_4075_ = v_binderType_4305_;
v_e_u2081_4076_ = v_body_4306_;
v_bi_u2081_4077_ = v_binderInfo_4307_;
v_n_u2082_4078_ = v_binderName_4308_;
v_t_u2082_4079_ = v_binderType_4309_;
v_e_u2082_4080_ = v_body_4310_;
v_bi_u2082_4081_ = v_binderInfo_4311_;
v___y_4082_ = v_a_4015_;
v___y_4083_ = v_a_4016_;
v___y_4084_ = v_a_4017_;
v___y_4085_ = v_a_4018_;
v___y_4086_ = v_a_4019_;
v___y_4087_ = v_a_4020_;
v___y_4088_ = v_a_4021_;
v___y_4089_ = v_a_4022_;
goto v___jp_4073_;
}
case 10:
{
lean_object* v_expr_4312_; lean_object* v___x_4313_; 
v_expr_4312_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4312_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4313_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4312_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4313_;
}
case 2:
{
lean_object* v_mvarId_4314_; 
v_mvarId_4314_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4314_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4314_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 3);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 8:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 8:
{
lean_object* v_declName_4315_; lean_object* v_type_4316_; lean_object* v_value_4317_; lean_object* v_body_4318_; lean_object* v_declName_4319_; lean_object* v_type_4320_; lean_object* v_value_4321_; lean_object* v_body_4322_; uint8_t v___y_4337_; uint8_t v___x_4340_; uint8_t v___x_4341_; 
v_declName_4315_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_declName_4315_);
v_type_4316_ = lean_ctor_get(v_x_4013_, 1);
lean_inc_ref(v_type_4316_);
v_value_4317_ = lean_ctor_get(v_x_4013_, 2);
lean_inc_ref(v_value_4317_);
v_body_4318_ = lean_ctor_get(v_x_4013_, 3);
lean_inc_ref(v_body_4318_);
lean_dec_ref_known(v_x_4013_, 4);
v_declName_4319_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_declName_4319_);
v_type_4320_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_type_4320_);
v_value_4321_ = lean_ctor_get(v_x_4014_, 2);
lean_inc_ref(v_value_4321_);
v_body_4322_ = lean_ctor_get(v_x_4014_, 3);
lean_inc_ref(v_body_4322_);
lean_dec_ref_known(v_x_4014_, 4);
v___x_4340_ = l_Lean_Name_hasMacroScopes(v_declName_4315_);
v___x_4341_ = l_Lean_Name_hasMacroScopes(v_declName_4319_);
if (v___x_4340_ == 0)
{
if (v___x_4341_ == 0)
{
goto v___jp_4323_;
}
else
{
v___y_4337_ = v___x_4340_;
goto v___jp_4336_;
}
}
else
{
v___y_4337_ = v___x_4341_;
goto v___jp_4336_;
}
v___jp_4323_:
{
lean_object* v___x_4324_; lean_object* v___x_4325_; uint8_t v___x_4326_; 
v___x_4324_ = l_Lean_Name_eraseMacroScopes(v_declName_4315_);
lean_dec(v_declName_4315_);
v___x_4325_ = l_Lean_Name_eraseMacroScopes(v_declName_4319_);
lean_dec(v_declName_4319_);
v___x_4326_ = lean_name_eq(v___x_4324_, v___x_4325_);
lean_dec(v___x_4325_);
lean_dec(v___x_4324_);
if (v___x_4326_ == 0)
{
lean_object* v___x_4327_; lean_object* v___x_4328_; 
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_value_4321_);
lean_dec_ref(v_type_4320_);
lean_dec_ref(v_body_4318_);
lean_dec_ref(v_value_4317_);
lean_dec_ref(v_type_4316_);
v___x_4327_ = lean_box(v___x_4326_);
v___x_4328_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4328_, 0, v___x_4327_);
return v___x_4328_;
}
else
{
lean_object* v___x_4329_; 
v___x_4329_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_type_4316_, v_type_4320_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4329_) == 0)
{
lean_object* v_a_4330_; uint8_t v___x_4331_; 
v_a_4330_ = lean_ctor_get(v___x_4329_, 0);
lean_inc(v_a_4330_);
v___x_4331_ = lean_unbox(v_a_4330_);
lean_dec(v_a_4330_);
if (v___x_4331_ == 0)
{
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_value_4321_);
lean_dec_ref(v_body_4318_);
lean_dec_ref(v_value_4317_);
return v___x_4329_;
}
else
{
lean_object* v___x_4332_; 
lean_dec_ref_known(v___x_4329_, 1);
v___x_4332_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_value_4317_, v_value_4321_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
if (lean_obj_tag(v___x_4332_) == 0)
{
lean_object* v_a_4333_; uint8_t v___x_4334_; 
v_a_4333_ = lean_ctor_get(v___x_4332_, 0);
lean_inc(v_a_4333_);
v___x_4334_ = lean_unbox(v_a_4333_);
lean_dec(v_a_4333_);
if (v___x_4334_ == 0)
{
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_body_4318_);
return v___x_4332_;
}
else
{
lean_object* v___x_4335_; 
lean_dec_ref_known(v___x_4332_, 1);
v___x_4335_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_body_4318_, v_body_4322_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4335_;
}
}
else
{
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_body_4318_);
return v___x_4332_;
}
}
}
else
{
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_value_4321_);
lean_dec_ref(v_body_4318_);
lean_dec_ref(v_value_4317_);
return v___x_4329_;
}
}
}
v___jp_4336_:
{
if (v___y_4337_ == 0)
{
lean_object* v___x_4338_; lean_object* v___x_4339_; 
lean_dec_ref(v_body_4322_);
lean_dec_ref(v_value_4321_);
lean_dec_ref(v_type_4320_);
lean_dec(v_declName_4319_);
lean_dec_ref(v_body_4318_);
lean_dec_ref(v_value_4317_);
lean_dec_ref(v_type_4316_);
lean_dec(v_declName_4315_);
v___x_4338_ = lean_box(v___y_4337_);
v___x_4339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4339_, 0, v___x_4338_);
return v___x_4339_;
}
else
{
goto v___jp_4323_;
}
}
}
case 10:
{
lean_object* v_expr_4342_; lean_object* v___x_4343_; 
v_expr_4342_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4342_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4343_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4342_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4343_;
}
case 2:
{
lean_object* v_mvarId_4344_; 
v_mvarId_4344_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4344_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4344_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 4);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 9:
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 9:
{
lean_object* v_a_4345_; lean_object* v_a_4346_; uint8_t v___x_4347_; lean_object* v___x_4348_; lean_object* v___x_4349_; 
v_a_4345_ = lean_ctor_get(v_x_4013_, 0);
lean_inc_ref(v_a_4345_);
lean_dec_ref_known(v_x_4013_, 1);
v_a_4346_ = lean_ctor_get(v_x_4014_, 0);
lean_inc_ref(v_a_4346_);
lean_dec_ref_known(v_x_4014_, 1);
v___x_4347_ = l_Lean_instBEqLiteral_beq(v_a_4345_, v_a_4346_);
lean_dec_ref(v_a_4346_);
lean_dec_ref(v_a_4345_);
v___x_4348_ = lean_box(v___x_4347_);
v___x_4349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4349_, 0, v___x_4348_);
return v___x_4349_;
}
case 10:
{
lean_object* v_expr_4350_; lean_object* v___x_4351_; 
v_expr_4350_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4350_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4351_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4350_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4351_;
}
case 2:
{
lean_object* v_mvarId_4352_; 
v_mvarId_4352_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4352_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4352_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 1);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
}
}
}
case 10:
{
lean_object* v_expr_4353_; lean_object* v___x_4354_; 
lean_dec_ref(v___x_4168_);
v_expr_4353_ = lean_ctor_get(v_x_4013_, 1);
lean_inc_ref(v_expr_4353_);
lean_dec_ref_known(v_x_4013_, 2);
v___x_4354_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_expr_4353_, v_x_4014_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4354_;
}
default: 
{
lean_dec_ref(v___x_4168_);
switch(lean_obj_tag(v_x_4014_))
{
case 10:
{
lean_object* v_expr_4355_; lean_object* v___x_4356_; 
v_expr_4355_ = lean_ctor_get(v_x_4014_, 1);
lean_inc_ref(v_expr_4355_);
lean_dec_ref_known(v_x_4014_, 2);
v___x_4356_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_x_4013_, v_expr_4355_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4356_;
}
case 11:
{
lean_object* v_typeName_4357_; lean_object* v_idx_4358_; lean_object* v_struct_4359_; lean_object* v_typeName_4360_; lean_object* v_idx_4361_; lean_object* v_struct_4362_; uint8_t v___y_4364_; uint8_t v___x_4368_; 
v_typeName_4357_ = lean_ctor_get(v_x_4013_, 0);
lean_inc(v_typeName_4357_);
v_idx_4358_ = lean_ctor_get(v_x_4013_, 1);
lean_inc(v_idx_4358_);
v_struct_4359_ = lean_ctor_get(v_x_4013_, 2);
lean_inc_ref(v_struct_4359_);
lean_dec_ref_known(v_x_4013_, 3);
v_typeName_4360_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_typeName_4360_);
v_idx_4361_ = lean_ctor_get(v_x_4014_, 1);
lean_inc(v_idx_4361_);
v_struct_4362_ = lean_ctor_get(v_x_4014_, 2);
lean_inc_ref(v_struct_4362_);
lean_dec_ref_known(v_x_4014_, 3);
v___x_4368_ = lean_name_eq(v_typeName_4357_, v_typeName_4360_);
lean_dec(v_typeName_4360_);
lean_dec(v_typeName_4357_);
if (v___x_4368_ == 0)
{
lean_dec(v_idx_4361_);
lean_dec(v_idx_4358_);
v___y_4364_ = v___x_4368_;
goto v___jp_4363_;
}
else
{
uint8_t v___x_4369_; 
v___x_4369_ = lean_nat_dec_eq(v_idx_4358_, v_idx_4361_);
lean_dec(v_idx_4361_);
lean_dec(v_idx_4358_);
v___y_4364_ = v___x_4369_;
goto v___jp_4363_;
}
v___jp_4363_:
{
if (v___y_4364_ == 0)
{
lean_object* v___x_4365_; lean_object* v___x_4366_; 
lean_dec_ref(v_struct_4362_);
lean_dec_ref(v_struct_4359_);
v___x_4365_ = lean_box(v___y_4364_);
v___x_4366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4366_, 0, v___x_4365_);
return v___x_4366_;
}
else
{
lean_object* v___x_4367_; 
v___x_4367_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_struct_4359_, v_struct_4362_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
return v___x_4367_;
}
}
}
case 2:
{
lean_object* v_mvarId_4370_; 
v_mvarId_4370_ = lean_ctor_get(v_x_4014_, 0);
lean_inc(v_mvarId_4370_);
lean_dec_ref_known(v_x_4014_, 1);
v_e_u2081_4093_ = v_x_4013_;
v_m_u2082_4094_ = v_mvarId_4370_;
v___y_4095_ = v_a_4015_;
v___y_4096_ = v_a_4016_;
v___y_4097_ = v_a_4017_;
v___y_4098_ = v_a_4018_;
v___y_4099_ = v_a_4019_;
v___y_4100_ = v_a_4020_;
v___y_4101_ = v_a_4021_;
v___y_4102_ = v_a_4022_;
goto v___jp_4092_;
}
default: 
{
lean_dec_ref_known(v_x_4013_, 3);
lean_dec_ref(v_x_4014_);
goto v___jp_4116_;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(uint8_t v___x_4377_, lean_object* v_e_u2081_4378_, lean_object* v_e_u2082_4379_, lean_object* v___f_4380_, lean_object* v___f_4381_, uint8_t v___x_4382_, lean_object* v_cls_4383_, lean_object* v___x_4384_, lean_object* v___x_4385_, lean_object* v___x_4386_, lean_object* v___f_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_, lean_object* v___y_4392_, lean_object* v___y_4393_, lean_object* v___y_4394_, lean_object* v___y_4395_){
_start:
{
if (v___x_4377_ == 0)
{
uint8_t v___x_4421_; 
v___x_4421_ = l_Lean_Expr_hasMVar(v_e_u2081_4378_);
if (v___x_4421_ == 0)
{
uint8_t v___x_4422_; 
v___x_4422_ = l_Lean_Expr_hasMVar(v_e_u2082_4379_);
if (v___x_4422_ == 0)
{
uint8_t v___x_4423_; 
v___x_4423_ = lean_expr_equal(v_e_u2081_4378_, v_e_u2082_4379_);
if (v___x_4423_ == 0)
{
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
goto v___jp_4397_;
}
else
{
lean_object* v_options_4424_; uint8_t v_hasTrace_4425_; 
lean_dec_ref(v___f_4381_);
lean_dec_ref(v___f_4380_);
lean_dec_ref(v_e_u2082_4379_);
lean_dec_ref(v_e_u2081_4378_);
v_options_4424_ = lean_ctor_get(v___y_4394_, 2);
v_hasTrace_4425_ = lean_ctor_get_uint8(v_options_4424_, sizeof(void*)*1);
if (v_hasTrace_4425_ == 0)
{
lean_object* v___x_4426_; lean_object* v___x_4427_; 
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
v___x_4426_ = lean_box(v___x_4382_);
v___x_4427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4427_, 0, v___x_4426_);
return v___x_4427_;
}
else
{
lean_object* v_inheritedTraceOptions_4428_; lean_object* v___x_4429_; lean_object* v___x_4430_; uint8_t v___x_4431_; 
v_inheritedTraceOptions_4428_ = lean_ctor_get(v___y_4394_, 13);
v___x_4429_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1));
lean_inc(v_cls_4383_);
v___x_4430_ = l_Lean_Name_append(v___x_4429_, v_cls_4383_);
v___x_4431_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4428_, v_options_4424_, v___x_4430_);
lean_dec(v___x_4430_);
if (v___x_4431_ == 0)
{
lean_object* v___x_4432_; lean_object* v___x_4433_; 
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
v___x_4432_ = lean_box(v___x_4382_);
v___x_4433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4433_, 0, v___x_4432_);
return v___x_4433_;
}
else
{
lean_object* v___x_4434_; lean_object* v___x_259616__overap_4435_; lean_object* v___x_4436_; 
v___x_4434_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__3);
v___x_259616__overap_4435_ = l_Lean_addTrace___redArg(v___x_4384_, v___x_4385_, v___x_4386_, v___f_4387_, v_cls_4383_, v___x_4434_);
lean_inc(v___y_4395_);
lean_inc_ref(v___y_4394_);
lean_inc(v___y_4393_);
lean_inc_ref(v___y_4392_);
lean_inc(v___y_4391_);
lean_inc_ref(v___y_4390_);
lean_inc_ref(v___y_4389_);
lean_inc(v___y_4388_);
v___x_4436_ = lean_apply_9(v___x_259616__overap_4435_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, v___y_4392_, v___y_4393_, v___y_4394_, v___y_4395_, lean_box(0));
if (lean_obj_tag(v___x_4436_) == 0)
{
lean_object* v___x_4438_; uint8_t v_isShared_4439_; uint8_t v_isSharedCheck_4444_; 
v_isSharedCheck_4444_ = !lean_is_exclusive(v___x_4436_);
if (v_isSharedCheck_4444_ == 0)
{
lean_object* v_unused_4445_; 
v_unused_4445_ = lean_ctor_get(v___x_4436_, 0);
lean_dec(v_unused_4445_);
v___x_4438_ = v___x_4436_;
v_isShared_4439_ = v_isSharedCheck_4444_;
goto v_resetjp_4437_;
}
else
{
lean_dec(v___x_4436_);
v___x_4438_ = lean_box(0);
v_isShared_4439_ = v_isSharedCheck_4444_;
goto v_resetjp_4437_;
}
v_resetjp_4437_:
{
lean_object* v___x_4440_; lean_object* v___x_4442_; 
v___x_4440_ = lean_box(v___x_4382_);
if (v_isShared_4439_ == 0)
{
lean_ctor_set(v___x_4438_, 0, v___x_4440_);
v___x_4442_ = v___x_4438_;
goto v_reusejp_4441_;
}
else
{
lean_object* v_reuseFailAlloc_4443_; 
v_reuseFailAlloc_4443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4443_, 0, v___x_4440_);
v___x_4442_ = v_reuseFailAlloc_4443_;
goto v_reusejp_4441_;
}
v_reusejp_4441_:
{
return v___x_4442_;
}
}
}
else
{
lean_object* v_a_4446_; lean_object* v___x_4448_; uint8_t v_isShared_4449_; uint8_t v_isSharedCheck_4453_; 
v_a_4446_ = lean_ctor_get(v___x_4436_, 0);
v_isSharedCheck_4453_ = !lean_is_exclusive(v___x_4436_);
if (v_isSharedCheck_4453_ == 0)
{
v___x_4448_ = v___x_4436_;
v_isShared_4449_ = v_isSharedCheck_4453_;
goto v_resetjp_4447_;
}
else
{
lean_inc(v_a_4446_);
lean_dec(v___x_4436_);
v___x_4448_ = lean_box(0);
v_isShared_4449_ = v_isSharedCheck_4453_;
goto v_resetjp_4447_;
}
v_resetjp_4447_:
{
lean_object* v___x_4451_; 
if (v_isShared_4449_ == 0)
{
v___x_4451_ = v___x_4448_;
goto v_reusejp_4450_;
}
else
{
lean_object* v_reuseFailAlloc_4452_; 
v_reuseFailAlloc_4452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4452_, 0, v_a_4446_);
v___x_4451_ = v_reuseFailAlloc_4452_;
goto v_reusejp_4450_;
}
v_reusejp_4450_:
{
return v___x_4451_;
}
}
}
}
}
}
}
else
{
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
goto v___jp_4397_;
}
}
else
{
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
goto v___jp_4397_;
}
}
else
{
lean_object* v_options_4454_; uint8_t v_hasTrace_4455_; 
lean_dec_ref(v___f_4381_);
lean_dec_ref(v___f_4380_);
lean_dec_ref(v_e_u2082_4379_);
lean_dec_ref(v_e_u2081_4378_);
v_options_4454_ = lean_ctor_get(v___y_4394_, 2);
v_hasTrace_4455_ = lean_ctor_get_uint8(v_options_4454_, sizeof(void*)*1);
if (v_hasTrace_4455_ == 0)
{
lean_object* v___x_4456_; lean_object* v___x_4457_; 
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
v___x_4456_ = lean_box(v___x_4382_);
v___x_4457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4457_, 0, v___x_4456_);
return v___x_4457_;
}
else
{
lean_object* v_inheritedTraceOptions_4458_; lean_object* v___x_4459_; lean_object* v___x_4460_; uint8_t v___x_4461_; 
v_inheritedTraceOptions_4458_ = lean_ctor_get(v___y_4394_, 13);
v___x_4459_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__1));
lean_inc(v_cls_4383_);
v___x_4460_ = l_Lean_Name_append(v___x_4459_, v_cls_4383_);
v___x_4461_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4458_, v_options_4454_, v___x_4460_);
lean_dec(v___x_4460_);
if (v___x_4461_ == 0)
{
lean_object* v___x_4462_; lean_object* v___x_4463_; 
lean_dec(v___f_4387_);
lean_dec_ref(v___x_4386_);
lean_dec_ref(v___x_4385_);
lean_dec_ref(v___x_4384_);
lean_dec(v_cls_4383_);
v___x_4462_ = lean_box(v___x_4382_);
v___x_4463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4463_, 0, v___x_4462_);
return v___x_4463_;
}
else
{
lean_object* v___x_4464_; lean_object* v___x_259631__overap_4465_; lean_object* v___x_4466_; 
v___x_4464_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5, &lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5_once, _init_lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___closed__5);
v___x_259631__overap_4465_ = l_Lean_addTrace___redArg(v___x_4384_, v___x_4385_, v___x_4386_, v___f_4387_, v_cls_4383_, v___x_4464_);
lean_inc(v___y_4395_);
lean_inc_ref(v___y_4394_);
lean_inc(v___y_4393_);
lean_inc_ref(v___y_4392_);
lean_inc(v___y_4391_);
lean_inc_ref(v___y_4390_);
lean_inc_ref(v___y_4389_);
lean_inc(v___y_4388_);
v___x_4466_ = lean_apply_9(v___x_259631__overap_4465_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, v___y_4392_, v___y_4393_, v___y_4394_, v___y_4395_, lean_box(0));
if (lean_obj_tag(v___x_4466_) == 0)
{
lean_object* v___x_4468_; uint8_t v_isShared_4469_; uint8_t v_isSharedCheck_4474_; 
v_isSharedCheck_4474_ = !lean_is_exclusive(v___x_4466_);
if (v_isSharedCheck_4474_ == 0)
{
lean_object* v_unused_4475_; 
v_unused_4475_ = lean_ctor_get(v___x_4466_, 0);
lean_dec(v_unused_4475_);
v___x_4468_ = v___x_4466_;
v_isShared_4469_ = v_isSharedCheck_4474_;
goto v_resetjp_4467_;
}
else
{
lean_dec(v___x_4466_);
v___x_4468_ = lean_box(0);
v_isShared_4469_ = v_isSharedCheck_4474_;
goto v_resetjp_4467_;
}
v_resetjp_4467_:
{
lean_object* v___x_4470_; lean_object* v___x_4472_; 
v___x_4470_ = lean_box(v___x_4382_);
if (v_isShared_4469_ == 0)
{
lean_ctor_set(v___x_4468_, 0, v___x_4470_);
v___x_4472_ = v___x_4468_;
goto v_reusejp_4471_;
}
else
{
lean_object* v_reuseFailAlloc_4473_; 
v_reuseFailAlloc_4473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4473_, 0, v___x_4470_);
v___x_4472_ = v_reuseFailAlloc_4473_;
goto v_reusejp_4471_;
}
v_reusejp_4471_:
{
return v___x_4472_;
}
}
}
else
{
lean_object* v_a_4476_; lean_object* v___x_4478_; uint8_t v_isShared_4479_; uint8_t v_isSharedCheck_4483_; 
v_a_4476_ = lean_ctor_get(v___x_4466_, 0);
v_isSharedCheck_4483_ = !lean_is_exclusive(v___x_4466_);
if (v_isSharedCheck_4483_ == 0)
{
v___x_4478_ = v___x_4466_;
v_isShared_4479_ = v_isSharedCheck_4483_;
goto v_resetjp_4477_;
}
else
{
lean_inc(v_a_4476_);
lean_dec(v___x_4466_);
v___x_4478_ = lean_box(0);
v_isShared_4479_ = v_isSharedCheck_4483_;
goto v_resetjp_4477_;
}
v_resetjp_4477_:
{
lean_object* v___x_4481_; 
if (v_isShared_4479_ == 0)
{
v___x_4481_ = v___x_4478_;
goto v_reusejp_4480_;
}
else
{
lean_object* v_reuseFailAlloc_4482_; 
v_reuseFailAlloc_4482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4482_, 0, v_a_4476_);
v___x_4481_ = v_reuseFailAlloc_4482_;
goto v_reusejp_4480_;
}
v_reusejp_4480_:
{
return v___x_4481_;
}
}
}
}
}
}
v___jp_4397_:
{
lean_object* v___x_4398_; lean_object* v___x_4399_; lean_object* v___x_4400_; 
v___x_4398_ = lean_st_ref_get(v___y_4388_);
lean_inc_ref(v_e_u2082_4379_);
lean_inc_ref(v_e_u2081_4378_);
v___x_4399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4399_, 0, v_e_u2081_4378_);
lean_ctor_set(v___x_4399_, 1, v_e_u2082_4379_);
lean_inc_ref(v___x_4399_);
lean_inc_ref(v___f_4381_);
lean_inc_ref(v___f_4380_);
v___x_4400_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___f_4380_, v___f_4381_, v___x_4398_, v___x_4399_);
lean_dec(v___x_4398_);
if (lean_obj_tag(v___x_4400_) == 0)
{
lean_object* v___x_4401_; 
v___x_4401_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083(v_e_u2081_4378_, v_e_u2082_4379_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, v___y_4392_, v___y_4393_, v___y_4394_, v___y_4395_);
if (lean_obj_tag(v___x_4401_) == 0)
{
lean_object* v_a_4402_; lean_object* v___x_4404_; uint8_t v_isShared_4405_; uint8_t v_isSharedCheck_4412_; 
v_a_4402_ = lean_ctor_get(v___x_4401_, 0);
v_isSharedCheck_4412_ = !lean_is_exclusive(v___x_4401_);
if (v_isSharedCheck_4412_ == 0)
{
v___x_4404_ = v___x_4401_;
v_isShared_4405_ = v_isSharedCheck_4412_;
goto v_resetjp_4403_;
}
else
{
lean_inc(v_a_4402_);
lean_dec(v___x_4401_);
v___x_4404_ = lean_box(0);
v_isShared_4405_ = v_isSharedCheck_4412_;
goto v_resetjp_4403_;
}
v_resetjp_4403_:
{
lean_object* v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4410_; 
v___x_4406_ = lean_st_ref_take(v___y_4388_);
lean_inc(v_a_4402_);
v___x_4407_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___f_4380_, v___f_4381_, v___x_4406_, v___x_4399_, v_a_4402_);
v___x_4408_ = lean_st_ref_set(v___y_4388_, v___x_4407_);
if (v_isShared_4405_ == 0)
{
v___x_4410_ = v___x_4404_;
goto v_reusejp_4409_;
}
else
{
lean_object* v_reuseFailAlloc_4411_; 
v_reuseFailAlloc_4411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4411_, 0, v_a_4402_);
v___x_4410_ = v_reuseFailAlloc_4411_;
goto v_reusejp_4409_;
}
v_reusejp_4409_:
{
return v___x_4410_;
}
}
}
else
{
lean_dec_ref_known(v___x_4399_, 2);
lean_dec_ref(v___f_4381_);
lean_dec_ref(v___f_4380_);
return v___x_4401_;
}
}
else
{
lean_object* v_val_4413_; lean_object* v___x_4415_; uint8_t v_isShared_4416_; uint8_t v_isSharedCheck_4420_; 
lean_dec_ref_known(v___x_4399_, 2);
lean_dec_ref(v___f_4381_);
lean_dec_ref(v___f_4380_);
lean_dec_ref(v_e_u2082_4379_);
lean_dec_ref(v_e_u2081_4378_);
v_val_4413_ = lean_ctor_get(v___x_4400_, 0);
v_isSharedCheck_4420_ = !lean_is_exclusive(v___x_4400_);
if (v_isSharedCheck_4420_ == 0)
{
v___x_4415_ = v___x_4400_;
v_isShared_4416_ = v_isSharedCheck_4420_;
goto v_resetjp_4414_;
}
else
{
lean_inc(v_val_4413_);
lean_dec(v___x_4400_);
v___x_4415_ = lean_box(0);
v_isShared_4416_ = v_isSharedCheck_4420_;
goto v_resetjp_4414_;
}
v_resetjp_4414_:
{
lean_object* v___x_4418_; 
if (v_isShared_4416_ == 0)
{
lean_ctor_set_tag(v___x_4415_, 0);
v___x_4418_ = v___x_4415_;
goto v_reusejp_4417_;
}
else
{
lean_object* v_reuseFailAlloc_4419_; 
v_reuseFailAlloc_4419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4419_, 0, v_val_4413_);
v___x_4418_ = v_reuseFailAlloc_4419_;
goto v_reusejp_4417_;
}
v_reusejp_4417_:
{
return v___x_4418_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0___boxed(lean_object** _args){
lean_object* v___x_4484_ = _args[0];
lean_object* v_e_u2081_4485_ = _args[1];
lean_object* v_e_u2082_4486_ = _args[2];
lean_object* v___f_4487_ = _args[3];
lean_object* v___f_4488_ = _args[4];
lean_object* v___x_4489_ = _args[5];
lean_object* v_cls_4490_ = _args[6];
lean_object* v___x_4491_ = _args[7];
lean_object* v___x_4492_ = _args[8];
lean_object* v___x_4493_ = _args[9];
lean_object* v___f_4494_ = _args[10];
lean_object* v___y_4495_ = _args[11];
lean_object* v___y_4496_ = _args[12];
lean_object* v___y_4497_ = _args[13];
lean_object* v___y_4498_ = _args[14];
lean_object* v___y_4499_ = _args[15];
lean_object* v___y_4500_ = _args[16];
lean_object* v___y_4501_ = _args[17];
lean_object* v___y_4502_ = _args[18];
lean_object* v___y_4503_ = _args[19];
_start:
{
uint8_t v___x_261329__boxed_4504_; uint8_t v___x_261332__boxed_4505_; lean_object* v_res_4506_; 
v___x_261329__boxed_4504_ = lean_unbox(v___x_4484_);
v___x_261332__boxed_4505_ = lean_unbox(v___x_4489_);
v_res_4506_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___lam__0(v___x_261329__boxed_4504_, v_e_u2081_4485_, v_e_u2082_4486_, v___f_4487_, v___f_4488_, v___x_261332__boxed_4505_, v_cls_4490_, v___x_4491_, v___x_4492_, v___x_4493_, v___f_4494_, v___y_4495_, v___y_4496_, v___y_4497_, v___y_4498_, v___y_4499_, v___y_4500_, v___y_4501_, v___y_4502_);
lean_dec(v___y_4502_);
lean_dec_ref(v___y_4501_);
lean_dec(v___y_4500_);
lean_dec_ref(v___y_4499_);
lean_dec(v___y_4498_);
lean_dec_ref(v___y_4497_);
lean_dec_ref(v___y_4496_);
lean_dec(v___y_4495_);
return v_res_4506_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues___boxed(lean_object* v_a_4507_, lean_object* v_a_4508_, lean_object* v_a_4509_, lean_object* v_a_4510_, lean_object* v_a_4511_, lean_object* v_a_4512_, lean_object* v_a_4513_, lean_object* v_a_4514_, lean_object* v_a_4515_, lean_object* v_a_4516_, lean_object* v_a_4517_){
_start:
{
lean_object* v_res_4518_; 
v_res_4518_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues(v_a_4507_, v_a_4508_, v_a_4509_, v_a_4510_, v_a_4511_, v_a_4512_, v_a_4513_, v_a_4514_, v_a_4515_, v_a_4516_);
lean_dec(v_a_4516_);
lean_dec_ref(v_a_4515_);
lean_dec(v_a_4514_);
lean_dec_ref(v_a_4513_);
lean_dec(v_a_4512_);
lean_dec_ref(v_a_4511_);
lean_dec_ref(v_a_4510_);
lean_dec(v_a_4509_);
return v_res_4518_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore___boxed(lean_object* v_ldecl_u2081_4519_, lean_object* v_ldecl_u2082_4520_, lean_object* v_a_4521_, lean_object* v_a_4522_, lean_object* v_a_4523_, lean_object* v_a_4524_, lean_object* v_a_4525_, lean_object* v_a_4526_, lean_object* v_a_4527_, lean_object* v_a_4528_){
_start:
{
lean_object* v_res_4529_; 
v_res_4529_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localDeclsEqualUpToIdsCore(v_ldecl_u2081_4519_, v_ldecl_u2082_4520_, v_a_4521_, v_a_4522_, v_a_4523_, v_a_4524_, v_a_4525_, v_a_4526_, v_a_4527_);
lean_dec(v_a_4527_);
lean_dec_ref(v_a_4526_);
lean_dec(v_a_4525_);
lean_dec_ref(v_a_4524_);
lean_dec(v_a_4523_);
lean_dec_ref(v_a_4522_);
lean_dec_ref(v_a_4521_);
lean_dec_ref(v_ldecl_u2082_4520_);
lean_dec_ref(v_ldecl_u2081_4519_);
return v_res_4529_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___boxed(lean_object* v_e_u2081_4530_, lean_object* v_e_u2082_4531_, lean_object* v_a_4532_, lean_object* v_a_4533_, lean_object* v_a_4534_, lean_object* v_a_4535_, lean_object* v_a_4536_, lean_object* v_a_4537_, lean_object* v_a_4538_, lean_object* v_a_4539_){
_start:
{
lean_object* v_res_4540_; 
v_res_4540_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081(v_e_u2081_4530_, v_e_u2082_4531_, v_a_4532_, v_a_4533_, v_a_4534_, v_a_4535_, v_a_4536_, v_a_4537_, v_a_4538_);
lean_dec(v_a_4538_);
lean_dec_ref(v_a_4537_);
lean_dec(v_a_4536_);
lean_dec_ref(v_a_4535_);
lean_dec(v_a_4534_);
lean_dec_ref(v_a_4533_);
lean_dec_ref(v_a_4532_);
return v_res_4540_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore___boxed(lean_object* v_lctx_u2081_4541_, lean_object* v_lctx_u2082_4542_, lean_object* v_localInstances_u2081_4543_, lean_object* v_localInstances_u2082_4544_, lean_object* v_a_4545_, lean_object* v_a_4546_, lean_object* v_a_4547_, lean_object* v_a_4548_, lean_object* v_a_4549_, lean_object* v_a_4550_, lean_object* v_a_4551_){
_start:
{
lean_object* v_res_4552_; 
v_res_4552_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore(v_lctx_u2081_4541_, v_lctx_u2082_4542_, v_localInstances_u2081_4543_, v_localInstances_u2082_4544_, v_a_4545_, v_a_4546_, v_a_4547_, v_a_4548_, v_a_4549_, v_a_4550_);
lean_dec(v_a_4550_);
lean_dec_ref(v_a_4549_);
lean_dec(v_a_4548_);
lean_dec_ref(v_a_4547_);
lean_dec(v_a_4546_);
lean_dec_ref(v_a_4545_);
return v_res_4552_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg___boxed(lean_object* v_decls_u2081_4553_, lean_object* v_decls_u2082_4554_, lean_object* v_i_4555_, lean_object* v_gctx_4556_, lean_object* v_a_4557_, lean_object* v_a_4558_, lean_object* v_a_4559_, lean_object* v_a_4560_, lean_object* v_a_4561_, lean_object* v_a_4562_, lean_object* v_a_4563_){
_start:
{
lean_object* v_res_4564_; 
v_res_4564_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg(v_decls_u2081_4553_, v_decls_u2082_4554_, v_i_4555_, v_gctx_4556_, v_a_4557_, v_a_4558_, v_a_4559_, v_a_4560_, v_a_4561_, v_a_4562_);
lean_dec(v_a_4562_);
lean_dec_ref(v_a_4561_);
lean_dec(v_a_4560_);
lean_dec_ref(v_a_4559_);
lean_dec(v_a_4558_);
lean_dec_ref(v_a_4557_);
lean_dec_ref(v_decls_u2082_4554_);
lean_dec_ref(v_decls_u2081_4553_);
return v_res_4564_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082___boxed(lean_object* v_e_u2081_4565_, lean_object* v_e_u2082_4566_, lean_object* v_a_4567_, lean_object* v_a_4568_, lean_object* v_a_4569_, lean_object* v_a_4570_, lean_object* v_a_4571_, lean_object* v_a_4572_, lean_object* v_a_4573_, lean_object* v_a_4574_, lean_object* v_a_4575_){
_start:
{
lean_object* v_res_4576_; 
v_res_4576_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2082(v_e_u2081_4565_, v_e_u2082_4566_, v_a_4567_, v_a_4568_, v_a_4569_, v_a_4570_, v_a_4571_, v_a_4572_, v_a_4573_, v_a_4574_);
lean_dec(v_a_4574_);
lean_dec_ref(v_a_4573_);
lean_dec(v_a_4572_);
lean_dec_ref(v_a_4571_);
lean_dec(v_a_4570_);
lean_dec_ref(v_a_4569_);
lean_dec_ref(v_a_4568_);
lean_dec(v_a_4567_);
return v_res_4576_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083___boxed(lean_object* v_x_4577_, lean_object* v_x_4578_, lean_object* v_a_4579_, lean_object* v_a_4580_, lean_object* v_a_4581_, lean_object* v_a_4582_, lean_object* v_a_4583_, lean_object* v_a_4584_, lean_object* v_a_4585_, lean_object* v_a_4586_, lean_object* v_a_4587_){
_start:
{
lean_object* v_res_4588_; 
v_res_4588_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083(v_x_4577_, v_x_4578_, v_a_4579_, v_a_4580_, v_a_4581_, v_a_4582_, v_a_4583_, v_a_4584_, v_a_4585_, v_a_4586_);
lean_dec(v_a_4586_);
lean_dec_ref(v_a_4585_);
lean_dec(v_a_4584_);
lean_dec_ref(v_a_4583_);
lean_dec(v_a_4582_);
lean_dec_ref(v_a_4581_);
lean_dec_ref(v_a_4580_);
lean_dec(v_a_4579_);
return v_res_4588_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___boxed(lean_object* v_mvarId_u2081_4589_, lean_object* v_mvarId_u2082_4590_, lean_object* v_a_4591_, lean_object* v_a_4592_, lean_object* v_a_4593_, lean_object* v_a_4594_, lean_object* v_a_4595_, lean_object* v_a_4596_, lean_object* v_a_4597_){
_start:
{
lean_object* v_res_4598_; 
v_res_4598_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(v_mvarId_u2081_4589_, v_mvarId_u2082_4590_, v_a_4591_, v_a_4592_, v_a_4593_, v_a_4594_, v_a_4595_, v_a_4596_);
lean_dec(v_a_4596_);
lean_dec_ref(v_a_4595_);
lean_dec(v_a_4594_);
lean_dec_ref(v_a_4593_);
lean_dec(v_a_4592_);
lean_dec_ref(v_a_4591_);
return v_res_4598_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1(lean_object* v___y_4599_, lean_object* v___y_4600_, lean_object* v___y_4601_, lean_object* v___y_4602_, lean_object* v___y_4603_, lean_object* v___y_4604_){
_start:
{
lean_object* v___x_4606_; 
v___x_4606_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___redArg(v___y_4604_);
return v___x_4606_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1___boxed(lean_object* v___y_4607_, lean_object* v___y_4608_, lean_object* v___y_4609_, lean_object* v___y_4610_, lean_object* v___y_4611_, lean_object* v___y_4612_, lean_object* v___y_4613_){
_start:
{
lean_object* v_res_4614_; 
v_res_4614_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__1(v___y_4607_, v___y_4608_, v___y_4609_, v___y_4610_, v___y_4611_, v___y_4612_);
lean_dec(v___y_4612_);
lean_dec_ref(v___y_4611_);
lean_dec(v___y_4610_);
lean_dec_ref(v___y_4609_);
lean_dec(v___y_4608_);
lean_dec_ref(v___y_4607_);
return v_res_4614_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go(lean_object* v_decls_u2081_4615_, lean_object* v_decls_u2082_4616_, lean_object* v_h_4617_, lean_object* v_i_4618_, lean_object* v_gctx_4619_, lean_object* v_a_4620_, lean_object* v_a_4621_, lean_object* v_a_4622_, lean_object* v_a_4623_, lean_object* v_a_4624_, lean_object* v_a_4625_){
_start:
{
lean_object* v___x_4627_; 
v___x_4627_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___redArg(v_decls_u2081_4615_, v_decls_u2082_4616_, v_i_4618_, v_gctx_4619_, v_a_4620_, v_a_4621_, v_a_4622_, v_a_4623_, v_a_4624_, v_a_4625_);
return v___x_4627_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go___boxed(lean_object* v_decls_u2081_4628_, lean_object* v_decls_u2082_4629_, lean_object* v_h_4630_, lean_object* v_i_4631_, lean_object* v_gctx_4632_, lean_object* v_a_4633_, lean_object* v_a_4634_, lean_object* v_a_4635_, lean_object* v_a_4636_, lean_object* v_a_4637_, lean_object* v_a_4638_, lean_object* v_a_4639_){
_start:
{
lean_object* v_res_4640_; 
v_res_4640_ = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go(v_decls_u2081_4628_, v_decls_u2082_4629_, v_h_4630_, v_i_4631_, v_gctx_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v_a_4638_);
lean_dec(v_a_4638_);
lean_dec_ref(v_a_4637_);
lean_dec(v_a_4636_);
lean_dec_ref(v_a_4635_);
lean_dec(v_a_4634_);
lean_dec_ref(v_a_4633_);
lean_dec_ref(v_decls_u2082_4629_);
lean_dec_ref(v_decls_u2081_4628_);
return v_res_4640_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0(lean_object* v_00_u03b2_4641_, lean_object* v_m_4642_, lean_object* v_a_4643_, lean_object* v_b_4644_){
_start:
{
lean_object* v___x_4645_; 
v___x_4645_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0___redArg(v_m_4642_, v_a_4643_, v_b_4644_);
return v___x_4645_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13(lean_object* v_00_u03b1_4646_, lean_object* v_x_4647_, lean_object* v___y_4648_, lean_object* v___y_4649_, lean_object* v___y_4650_, lean_object* v___y_4651_, lean_object* v___y_4652_, lean_object* v___y_4653_){
_start:
{
lean_object* v___x_4655_; 
v___x_4655_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___redArg(v_x_4647_);
return v___x_4655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13___boxed(lean_object* v_00_u03b1_4656_, lean_object* v_x_4657_, lean_object* v___y_4658_, lean_object* v___y_4659_, lean_object* v___y_4660_, lean_object* v___y_4661_, lean_object* v___y_4662_, lean_object* v___y_4663_, lean_object* v___y_4664_){
_start:
{
lean_object* v_res_4665_; 
v_res_4665_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__13(v_00_u03b1_4656_, v_x_4657_, v___y_4658_, v___y_4659_, v___y_4660_, v___y_4661_, v___y_4662_, v___y_4663_);
lean_dec(v___y_4663_);
lean_dec_ref(v___y_4662_);
lean_dec(v___y_4661_);
lean_dec_ref(v___y_4660_);
lean_dec(v___y_4659_);
lean_dec_ref(v___y_4658_);
return v_res_4665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9(lean_object* v_00_u03b2_4666_, lean_object* v_m_4667_, lean_object* v_a_4668_){
_start:
{
lean_object* v___x_4669_; 
v___x_4669_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___redArg(v_m_4667_, v_a_4668_);
return v___x_4669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9___boxed(lean_object* v_00_u03b2_4670_, lean_object* v_m_4671_, lean_object* v_a_4672_){
_start:
{
lean_object* v_res_4673_; 
v_res_4673_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9(v_00_u03b2_4670_, v_m_4671_, v_a_4672_);
lean_dec(v_a_4672_);
lean_dec_ref(v_m_4671_);
return v_res_4673_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10(lean_object* v_00_u03b2_4674_, lean_object* v_m_4675_, lean_object* v_a_4676_, lean_object* v_b_4677_){
_start:
{
lean_object* v___x_4678_; 
v___x_4678_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10___redArg(v_m_4675_, v_a_4676_, v_b_4677_);
return v___x_4678_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6(lean_object* v_00_u03b2_4679_, lean_object* v_a_4680_, lean_object* v_x_4681_){
_start:
{
uint8_t v___x_4682_; 
v___x_4682_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___redArg(v_a_4680_, v_x_4681_);
return v___x_4682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6___boxed(lean_object* v_00_u03b2_4683_, lean_object* v_a_4684_, lean_object* v_x_4685_){
_start:
{
uint8_t v_res_4686_; lean_object* v_r_4687_; 
v_res_4686_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__6(v_00_u03b2_4683_, v_a_4684_, v_x_4685_);
lean_dec(v_x_4685_);
lean_dec(v_a_4684_);
v_r_4687_ = lean_box(v_res_4686_);
return v_r_4687_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7(lean_object* v_00_u03b2_4688_, lean_object* v_data_4689_){
_start:
{
lean_object* v___x_4690_; 
v___x_4690_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7___redArg(v_data_4689_);
return v___x_4690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8(lean_object* v_00_u03b2_4691_, lean_object* v_a_4692_, lean_object* v_b_4693_, lean_object* v_x_4694_){
_start:
{
lean_object* v___x_4695_; 
v___x_4695_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__8___redArg(v_a_4692_, v_b_4693_, v_x_4694_);
return v___x_4695_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12(lean_object* v_oldTraces_4696_, lean_object* v_data_4697_, lean_object* v_ref_4698_, lean_object* v_msg_4699_, lean_object* v___y_4700_, lean_object* v___y_4701_, lean_object* v___y_4702_, lean_object* v___y_4703_, lean_object* v___y_4704_, lean_object* v___y_4705_){
_start:
{
lean_object* v___x_4707_; 
v___x_4707_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___redArg(v_oldTraces_4696_, v_data_4697_, v_ref_4698_, v_msg_4699_, v___y_4702_, v___y_4703_, v___y_4704_, v___y_4705_);
return v___x_4707_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12___boxed(lean_object* v_oldTraces_4708_, lean_object* v_data_4709_, lean_object* v_ref_4710_, lean_object* v_msg_4711_, lean_object* v___y_4712_, lean_object* v___y_4713_, lean_object* v___y_4714_, lean_object* v___y_4715_, lean_object* v___y_4716_, lean_object* v___y_4717_, lean_object* v___y_4718_){
_start:
{
lean_object* v_res_4719_; 
v_res_4719_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNodeBefore_postCallback___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__3_spec__12(v_oldTraces_4708_, v_data_4709_, v_ref_4710_, v_msg_4711_, v___y_4712_, v___y_4713_, v___y_4714_, v___y_4715_, v___y_4716_, v___y_4717_);
lean_dec(v___y_4717_);
lean_dec_ref(v___y_4716_);
lean_dec(v___y_4715_);
lean_dec_ref(v___y_4714_);
lean_dec(v___y_4713_);
lean_dec_ref(v___y_4712_);
return v_res_4719_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18(lean_object* v_00_u03b2_4720_, lean_object* v_a_4721_, lean_object* v_x_4722_){
_start:
{
lean_object* v___x_4723_; 
v___x_4723_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___redArg(v_a_4721_, v_x_4722_);
return v___x_4723_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18___boxed(lean_object* v_00_u03b2_4724_, lean_object* v_a_4725_, lean_object* v_x_4726_){
_start:
{
lean_object* v_res_4727_; 
v_res_4727_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__9_spec__18(v_00_u03b2_4724_, v_a_4725_, v_x_4726_);
lean_dec(v_x_4726_);
lean_dec(v_a_4725_);
return v_res_4727_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20(lean_object* v_00_u03b2_4728_, lean_object* v_a_4729_, lean_object* v_x_4730_){
_start:
{
uint8_t v___x_4731_; 
v___x_4731_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___redArg(v_a_4729_, v_x_4730_);
return v___x_4731_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20___boxed(lean_object* v_00_u03b2_4732_, lean_object* v_a_4733_, lean_object* v_x_4734_){
_start:
{
uint8_t v_res_4735_; lean_object* v_r_4736_; 
v_res_4735_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__20(v_00_u03b2_4732_, v_a_4733_, v_x_4734_);
lean_dec(v_x_4734_);
lean_dec(v_a_4733_);
v_r_4736_ = lean_box(v_res_4735_);
return v_r_4736_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21(lean_object* v_00_u03b2_4737_, lean_object* v_data_4738_){
_start:
{
lean_object* v___x_4739_; 
v___x_4739_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21___redArg(v_data_4738_);
return v___x_4739_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22(lean_object* v_00_u03b2_4740_, lean_object* v_a_4741_, lean_object* v_b_4742_, lean_object* v_x_4743_){
_start:
{
lean_object* v___x_4744_; 
v___x_4744_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__22___redArg(v_a_4741_, v_b_4742_, v_x_4743_);
return v___x_4744_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12(lean_object* v_00_u03b2_4745_, lean_object* v_i_4746_, lean_object* v_source_4747_, lean_object* v_target_4748_){
_start:
{
lean_object* v___x_4749_; 
v___x_4749_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12___redArg(v_i_4746_, v_source_4747_, v_target_4748_);
return v___x_4749_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25(lean_object* v_00_u03b2_4750_, lean_object* v_i_4751_, lean_object* v_source_4752_, lean_object* v_target_4753_){
_start:
{
lean_object* v___x_4754_; 
v___x_4754_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25___redArg(v_i_4751_, v_source_4752_, v_target_4753_);
return v___x_4754_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19(lean_object* v_00_u03b2_4755_, lean_object* v_x_4756_, lean_object* v_x_4757_){
_start:
{
lean_object* v___x_4758_; 
v___x_4758_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_localContextsEqualUpToIdsCore_go_spec__0_spec__7_spec__12_spec__19___redArg(v_x_4756_, v_x_4757_);
return v___x_4758_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27(lean_object* v_00_u03b2_4759_, lean_object* v_x_4760_, lean_object* v_x_4761_){
_start:
{
lean_object* v___x_4762_; 
v___x_4762_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Util_EqualUpToIds_0__Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2083_compareMVarValues_spec__10_spec__21_spec__25_spec__27___redArg(v_x_4760_, v_x_4761_);
return v___x_4762_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0(lean_object* v___x_4763_, uint8_t v___x_4764_, lean_object* v_a_4765_, lean_object* v_x_4766_, lean_object* v___y_4767_, lean_object* v___y_4768_, lean_object* v___y_4769_, lean_object* v___y_4770_, lean_object* v___y_4771_, lean_object* v___y_4772_, lean_object* v___y_4773_){
_start:
{
lean_object* v_snd_4775_; lean_object* v___x_4777_; uint8_t v_isShared_4778_; uint8_t v_isSharedCheck_4832_; 
v_snd_4775_ = lean_ctor_get(v___y_4767_, 1);
v_isSharedCheck_4832_ = !lean_is_exclusive(v___y_4767_);
if (v_isSharedCheck_4832_ == 0)
{
lean_object* v_unused_4833_; 
v_unused_4833_ = lean_ctor_get(v___y_4767_, 0);
lean_dec(v_unused_4833_);
v___x_4777_ = v___y_4767_;
v_isShared_4778_ = v_isSharedCheck_4832_;
goto v_resetjp_4776_;
}
else
{
lean_inc(v_snd_4775_);
lean_dec(v___y_4767_);
v___x_4777_ = lean_box(0);
v_isShared_4778_ = v_isSharedCheck_4832_;
goto v_resetjp_4776_;
}
v_resetjp_4776_:
{
lean_object* v_array_4779_; lean_object* v_start_4780_; lean_object* v_stop_4781_; uint8_t v___x_4782_; 
v_array_4779_ = lean_ctor_get(v_snd_4775_, 0);
v_start_4780_ = lean_ctor_get(v_snd_4775_, 1);
v_stop_4781_ = lean_ctor_get(v_snd_4775_, 2);
v___x_4782_ = lean_nat_dec_lt(v_start_4780_, v_stop_4781_);
if (v___x_4782_ == 0)
{
lean_object* v___x_4784_; 
lean_dec(v_a_4765_);
if (v_isShared_4778_ == 0)
{
lean_ctor_set(v___x_4777_, 0, v___x_4763_);
v___x_4784_ = v___x_4777_;
goto v_reusejp_4783_;
}
else
{
lean_object* v_reuseFailAlloc_4787_; 
v_reuseFailAlloc_4787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4787_, 0, v___x_4763_);
lean_ctor_set(v_reuseFailAlloc_4787_, 1, v_snd_4775_);
v___x_4784_ = v_reuseFailAlloc_4787_;
goto v_reusejp_4783_;
}
v_reusejp_4783_:
{
lean_object* v___x_4785_; lean_object* v___x_4786_; 
v___x_4785_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4785_, 0, v___x_4784_);
v___x_4786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4786_, 0, v___x_4785_);
return v___x_4786_;
}
}
else
{
lean_object* v___x_4789_; uint8_t v_isShared_4790_; uint8_t v_isSharedCheck_4828_; 
lean_inc(v_stop_4781_);
lean_inc(v_start_4780_);
lean_inc_ref(v_array_4779_);
v_isSharedCheck_4828_ = !lean_is_exclusive(v_snd_4775_);
if (v_isSharedCheck_4828_ == 0)
{
lean_object* v_unused_4829_; lean_object* v_unused_4830_; lean_object* v_unused_4831_; 
v_unused_4829_ = lean_ctor_get(v_snd_4775_, 2);
lean_dec(v_unused_4829_);
v_unused_4830_ = lean_ctor_get(v_snd_4775_, 1);
lean_dec(v_unused_4830_);
v_unused_4831_ = lean_ctor_get(v_snd_4775_, 0);
lean_dec(v_unused_4831_);
v___x_4789_ = v_snd_4775_;
v_isShared_4790_ = v_isSharedCheck_4828_;
goto v_resetjp_4788_;
}
else
{
lean_dec(v_snd_4775_);
v___x_4789_ = lean_box(0);
v_isShared_4790_ = v_isSharedCheck_4828_;
goto v_resetjp_4788_;
}
v_resetjp_4788_:
{
lean_object* v___x_4791_; lean_object* v___x_4792_; 
v___x_4791_ = lean_array_fget_borrowed(v_array_4779_, v_start_4780_);
lean_inc(v___x_4791_);
v___x_4792_ = lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore(v_a_4765_, v___x_4791_, v___y_4768_, v___y_4769_, v___y_4770_, v___y_4771_, v___y_4772_, v___y_4773_);
if (lean_obj_tag(v___x_4792_) == 0)
{
lean_object* v_a_4793_; lean_object* v___x_4795_; uint8_t v_isShared_4796_; uint8_t v_isSharedCheck_4819_; 
v_a_4793_ = lean_ctor_get(v___x_4792_, 0);
v_isSharedCheck_4819_ = !lean_is_exclusive(v___x_4792_);
if (v_isSharedCheck_4819_ == 0)
{
v___x_4795_ = v___x_4792_;
v_isShared_4796_ = v_isSharedCheck_4819_;
goto v_resetjp_4794_;
}
else
{
lean_inc(v_a_4793_);
lean_dec(v___x_4792_);
v___x_4795_ = lean_box(0);
v_isShared_4796_ = v_isSharedCheck_4819_;
goto v_resetjp_4794_;
}
v_resetjp_4794_:
{
lean_object* v___x_4797_; lean_object* v___x_4798_; lean_object* v___x_4800_; 
v___x_4797_ = lean_unsigned_to_nat(1u);
v___x_4798_ = lean_nat_add(v_start_4780_, v___x_4797_);
lean_dec(v_start_4780_);
if (v_isShared_4790_ == 0)
{
lean_ctor_set(v___x_4789_, 1, v___x_4798_);
v___x_4800_ = v___x_4789_;
goto v_reusejp_4799_;
}
else
{
lean_object* v_reuseFailAlloc_4818_; 
v_reuseFailAlloc_4818_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4818_, 0, v_array_4779_);
lean_ctor_set(v_reuseFailAlloc_4818_, 1, v___x_4798_);
lean_ctor_set(v_reuseFailAlloc_4818_, 2, v_stop_4781_);
v___x_4800_ = v_reuseFailAlloc_4818_;
goto v_reusejp_4799_;
}
v_reusejp_4799_:
{
uint8_t v___x_4801_; 
v___x_4801_ = lean_unbox(v_a_4793_);
lean_dec(v_a_4793_);
if (v___x_4801_ == 0)
{
lean_object* v___x_4802_; lean_object* v___x_4803_; lean_object* v___x_4805_; 
lean_dec(v___x_4763_);
v___x_4802_ = lean_box(v___x_4764_);
v___x_4803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4803_, 0, v___x_4802_);
if (v_isShared_4778_ == 0)
{
lean_ctor_set(v___x_4777_, 1, v___x_4800_);
lean_ctor_set(v___x_4777_, 0, v___x_4803_);
v___x_4805_ = v___x_4777_;
goto v_reusejp_4804_;
}
else
{
lean_object* v_reuseFailAlloc_4810_; 
v_reuseFailAlloc_4810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4810_, 0, v___x_4803_);
lean_ctor_set(v_reuseFailAlloc_4810_, 1, v___x_4800_);
v___x_4805_ = v_reuseFailAlloc_4810_;
goto v_reusejp_4804_;
}
v_reusejp_4804_:
{
lean_object* v___x_4806_; lean_object* v___x_4808_; 
v___x_4806_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4806_, 0, v___x_4805_);
if (v_isShared_4796_ == 0)
{
lean_ctor_set(v___x_4795_, 0, v___x_4806_);
v___x_4808_ = v___x_4795_;
goto v_reusejp_4807_;
}
else
{
lean_object* v_reuseFailAlloc_4809_; 
v_reuseFailAlloc_4809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4809_, 0, v___x_4806_);
v___x_4808_ = v_reuseFailAlloc_4809_;
goto v_reusejp_4807_;
}
v_reusejp_4807_:
{
return v___x_4808_;
}
}
}
else
{
lean_object* v___x_4812_; 
if (v_isShared_4778_ == 0)
{
lean_ctor_set(v___x_4777_, 1, v___x_4800_);
lean_ctor_set(v___x_4777_, 0, v___x_4763_);
v___x_4812_ = v___x_4777_;
goto v_reusejp_4811_;
}
else
{
lean_object* v_reuseFailAlloc_4817_; 
v_reuseFailAlloc_4817_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4817_, 0, v___x_4763_);
lean_ctor_set(v_reuseFailAlloc_4817_, 1, v___x_4800_);
v___x_4812_ = v_reuseFailAlloc_4817_;
goto v_reusejp_4811_;
}
v_reusejp_4811_:
{
lean_object* v___x_4813_; lean_object* v___x_4815_; 
v___x_4813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4813_, 0, v___x_4812_);
if (v_isShared_4796_ == 0)
{
lean_ctor_set(v___x_4795_, 0, v___x_4813_);
v___x_4815_ = v___x_4795_;
goto v_reusejp_4814_;
}
else
{
lean_object* v_reuseFailAlloc_4816_; 
v_reuseFailAlloc_4816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4816_, 0, v___x_4813_);
v___x_4815_ = v_reuseFailAlloc_4816_;
goto v_reusejp_4814_;
}
v_reusejp_4814_:
{
return v___x_4815_;
}
}
}
}
}
}
else
{
lean_object* v_a_4820_; lean_object* v___x_4822_; uint8_t v_isShared_4823_; uint8_t v_isSharedCheck_4827_; 
lean_del_object(v___x_4789_);
lean_dec(v_stop_4781_);
lean_dec(v_start_4780_);
lean_dec_ref(v_array_4779_);
lean_del_object(v___x_4777_);
lean_dec(v___x_4763_);
v_a_4820_ = lean_ctor_get(v___x_4792_, 0);
v_isSharedCheck_4827_ = !lean_is_exclusive(v___x_4792_);
if (v_isSharedCheck_4827_ == 0)
{
v___x_4822_ = v___x_4792_;
v_isShared_4823_ = v_isSharedCheck_4827_;
goto v_resetjp_4821_;
}
else
{
lean_inc(v_a_4820_);
lean_dec(v___x_4792_);
v___x_4822_ = lean_box(0);
v_isShared_4823_ = v_isSharedCheck_4827_;
goto v_resetjp_4821_;
}
v_resetjp_4821_:
{
lean_object* v___x_4825_; 
if (v_isShared_4823_ == 0)
{
v___x_4825_ = v___x_4822_;
goto v_reusejp_4824_;
}
else
{
lean_object* v_reuseFailAlloc_4826_; 
v_reuseFailAlloc_4826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4826_, 0, v_a_4820_);
v___x_4825_ = v_reuseFailAlloc_4826_;
goto v_reusejp_4824_;
}
v_reusejp_4824_:
{
return v___x_4825_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0___boxed(lean_object* v___x_4834_, lean_object* v___x_4835_, lean_object* v_a_4836_, lean_object* v_x_4837_, lean_object* v___y_4838_, lean_object* v___y_4839_, lean_object* v___y_4840_, lean_object* v___y_4841_, lean_object* v___y_4842_, lean_object* v___y_4843_, lean_object* v___y_4844_, lean_object* v___y_4845_){
_start:
{
uint8_t v___x_2313__boxed_4846_; lean_object* v_res_4847_; 
v___x_2313__boxed_4846_ = lean_unbox(v___x_4835_);
v_res_4847_ = lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___lam__0(v___x_4834_, v___x_2313__boxed_4846_, v_a_4836_, v_x_4837_, v___y_4838_, v___y_4839_, v___y_4840_, v___y_4841_, v___y_4842_, v___y_4843_, v___y_4844_);
lean_dec(v___y_4844_);
lean_dec_ref(v___y_4843_);
lean_dec(v___y_4842_);
lean_dec_ref(v___y_4841_);
lean_dec(v___y_4840_);
lean_dec_ref(v___y_4839_);
return v_res_4847_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore(lean_object* v_goals_u2081_4852_, lean_object* v_goals_u2082_4853_, lean_object* v_a_4854_, lean_object* v_a_4855_, lean_object* v_a_4856_, lean_object* v_a_4857_, lean_object* v_a_4858_, lean_object* v_a_4859_){
_start:
{
lean_object* v___x_4861_; lean_object* v___x_4862_; uint8_t v___x_4863_; 
v___x_4861_ = lean_array_get_size(v_goals_u2081_4852_);
v___x_4862_ = lean_array_get_size(v_goals_u2082_4853_);
v___x_4863_ = lean_nat_dec_eq(v___x_4861_, v___x_4862_);
if (v___x_4863_ == 0)
{
lean_object* v___x_4864_; lean_object* v___x_4865_; 
lean_dec_ref(v_goals_u2082_4853_);
lean_dec_ref(v_goals_u2081_4852_);
v___x_4864_ = lean_box(v___x_4863_);
v___x_4865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4865_, 0, v___x_4864_);
return v___x_4865_;
}
else
{
lean_object* v___x_4866_; lean_object* v_toApplicative_4867_; lean_object* v_toFunctor_4868_; lean_object* v_toSeq_4869_; lean_object* v_toSeqLeft_4870_; lean_object* v_toSeqRight_4871_; lean_object* v___f_4872_; lean_object* v___f_4873_; lean_object* v___f_4874_; lean_object* v___f_4875_; lean_object* v___x_4876_; lean_object* v___f_4877_; lean_object* v___f_4878_; lean_object* v___f_4879_; lean_object* v___x_4880_; lean_object* v___x_4881_; lean_object* v___x_4882_; lean_object* v_toApplicative_4883_; lean_object* v___x_4885_; uint8_t v_isShared_4886_; uint8_t v_isSharedCheck_4943_; 
v___x_4866_ = lean_obj_once(&lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1, &lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1_once, _init_lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__1);
v_toApplicative_4867_ = lean_ctor_get(v___x_4866_, 0);
v_toFunctor_4868_ = lean_ctor_get(v_toApplicative_4867_, 0);
v_toSeq_4869_ = lean_ctor_get(v_toApplicative_4867_, 2);
v_toSeqLeft_4870_ = lean_ctor_get(v_toApplicative_4867_, 3);
v_toSeqRight_4871_ = lean_ctor_get(v_toApplicative_4867_, 4);
v___f_4872_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__2));
v___f_4873_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__3));
lean_inc_ref_n(v_toFunctor_4868_, 2);
v___f_4874_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4874_, 0, v_toFunctor_4868_);
v___f_4875_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4875_, 0, v_toFunctor_4868_);
v___x_4876_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4876_, 0, v___f_4874_);
lean_ctor_set(v___x_4876_, 1, v___f_4875_);
lean_inc(v_toSeqRight_4871_);
v___f_4877_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4877_, 0, v_toSeqRight_4871_);
lean_inc(v_toSeqLeft_4870_);
v___f_4878_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4878_, 0, v_toSeqLeft_4870_);
lean_inc(v_toSeq_4869_);
v___f_4879_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4879_, 0, v_toSeq_4869_);
v___x_4880_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4880_, 0, v___x_4876_);
lean_ctor_set(v___x_4880_, 1, v___f_4872_);
lean_ctor_set(v___x_4880_, 2, v___f_4879_);
lean_ctor_set(v___x_4880_, 3, v___f_4878_);
lean_ctor_set(v___x_4880_, 4, v___f_4877_);
v___x_4881_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4881_, 0, v___x_4880_);
lean_ctor_set(v___x_4881_, 1, v___f_4873_);
v___x_4882_ = l_StateRefT_x27_instMonad___redArg(v___x_4881_);
v_toApplicative_4883_ = lean_ctor_get(v___x_4882_, 0);
v_isSharedCheck_4943_ = !lean_is_exclusive(v___x_4882_);
if (v_isSharedCheck_4943_ == 0)
{
lean_object* v_unused_4944_; 
v_unused_4944_ = lean_ctor_get(v___x_4882_, 1);
lean_dec(v_unused_4944_);
v___x_4885_ = v___x_4882_;
v_isShared_4886_ = v_isSharedCheck_4943_;
goto v_resetjp_4884_;
}
else
{
lean_inc(v_toApplicative_4883_);
lean_dec(v___x_4882_);
v___x_4885_ = lean_box(0);
v_isShared_4886_ = v_isSharedCheck_4943_;
goto v_resetjp_4884_;
}
v_resetjp_4884_:
{
lean_object* v_toFunctor_4887_; lean_object* v_toSeq_4888_; lean_object* v_toSeqLeft_4889_; lean_object* v_toSeqRight_4890_; lean_object* v___x_4892_; uint8_t v_isShared_4893_; uint8_t v_isSharedCheck_4941_; 
v_toFunctor_4887_ = lean_ctor_get(v_toApplicative_4883_, 0);
v_toSeq_4888_ = lean_ctor_get(v_toApplicative_4883_, 2);
v_toSeqLeft_4889_ = lean_ctor_get(v_toApplicative_4883_, 3);
v_toSeqRight_4890_ = lean_ctor_get(v_toApplicative_4883_, 4);
v_isSharedCheck_4941_ = !lean_is_exclusive(v_toApplicative_4883_);
if (v_isSharedCheck_4941_ == 0)
{
lean_object* v_unused_4942_; 
v_unused_4942_ = lean_ctor_get(v_toApplicative_4883_, 1);
lean_dec(v_unused_4942_);
v___x_4892_ = v_toApplicative_4883_;
v_isShared_4893_ = v_isSharedCheck_4941_;
goto v_resetjp_4891_;
}
else
{
lean_inc(v_toSeqRight_4890_);
lean_inc(v_toSeqLeft_4889_);
lean_inc(v_toSeq_4888_);
lean_inc(v_toFunctor_4887_);
lean_dec(v_toApplicative_4883_);
v___x_4892_ = lean_box(0);
v_isShared_4893_ = v_isSharedCheck_4941_;
goto v_resetjp_4891_;
}
v_resetjp_4891_:
{
lean_object* v___x_4894_; lean_object* v___x_4895_; lean_object* v___f_4896_; lean_object* v___f_4897_; lean_object* v___f_4898_; lean_object* v___f_4899_; lean_object* v___x_4900_; lean_object* v___f_4901_; lean_object* v___f_4902_; lean_object* v___f_4903_; lean_object* v___x_4905_; 
v___x_4894_ = lean_unsigned_to_nat(0u);
v___x_4895_ = l_Array_toSubarray___redArg(v_goals_u2082_4853_, v___x_4894_, v___x_4862_);
v___f_4896_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__4));
v___f_4897_ = ((lean_object*)(lp_aesop_Aesop_instMonadEqualUpToIdsM___closed__5));
lean_inc_ref(v_toFunctor_4887_);
v___f_4898_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4898_, 0, v_toFunctor_4887_);
v___f_4899_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4899_, 0, v_toFunctor_4887_);
v___x_4900_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4900_, 0, v___f_4898_);
lean_ctor_set(v___x_4900_, 1, v___f_4899_);
v___f_4901_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4901_, 0, v_toSeqRight_4890_);
v___f_4902_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4902_, 0, v_toSeqLeft_4889_);
v___f_4903_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4903_, 0, v_toSeq_4888_);
if (v_isShared_4893_ == 0)
{
lean_ctor_set(v___x_4892_, 4, v___f_4901_);
lean_ctor_set(v___x_4892_, 3, v___f_4902_);
lean_ctor_set(v___x_4892_, 2, v___f_4903_);
lean_ctor_set(v___x_4892_, 1, v___f_4896_);
lean_ctor_set(v___x_4892_, 0, v___x_4900_);
v___x_4905_ = v___x_4892_;
goto v_reusejp_4904_;
}
else
{
lean_object* v_reuseFailAlloc_4940_; 
v_reuseFailAlloc_4940_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4940_, 0, v___x_4900_);
lean_ctor_set(v_reuseFailAlloc_4940_, 1, v___f_4896_);
lean_ctor_set(v_reuseFailAlloc_4940_, 2, v___f_4903_);
lean_ctor_set(v_reuseFailAlloc_4940_, 3, v___f_4902_);
lean_ctor_set(v_reuseFailAlloc_4940_, 4, v___f_4901_);
v___x_4905_ = v_reuseFailAlloc_4940_;
goto v_reusejp_4904_;
}
v_reusejp_4904_:
{
lean_object* v___x_4907_; 
if (v_isShared_4886_ == 0)
{
lean_ctor_set(v___x_4885_, 1, v___f_4897_);
lean_ctor_set(v___x_4885_, 0, v___x_4905_);
v___x_4907_ = v___x_4885_;
goto v_reusejp_4906_;
}
else
{
lean_object* v_reuseFailAlloc_4939_; 
v_reuseFailAlloc_4939_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4939_, 0, v___x_4905_);
lean_ctor_set(v_reuseFailAlloc_4939_, 1, v___f_4897_);
v___x_4907_ = v_reuseFailAlloc_4939_;
goto v_reusejp_4906_;
}
v_reusejp_4906_:
{
lean_object* v___x_4908_; lean_object* v___x_4909_; lean_object* v___x_4910_; lean_object* v___f_4911_; lean_object* v___x_4912_; size_t v_sz_4913_; size_t v___x_4914_; lean_object* v___x_2237__overap_4915_; lean_object* v___x_4916_; 
v___x_4908_ = l_StateRefT_x27_instMonad___redArg(v___x_4907_);
v___x_4909_ = l_ReaderT_instMonad___redArg(v___x_4908_);
v___x_4910_ = lean_box(0);
v___f_4911_ = ((lean_object*)(lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___closed__0));
v___x_4912_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4912_, 0, v___x_4910_);
lean_ctor_set(v___x_4912_, 1, v___x_4895_);
v_sz_4913_ = lean_array_size(v_goals_u2081_4852_);
v___x_4914_ = ((size_t)0ULL);
v___x_2237__overap_4915_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_4909_, v_goals_u2081_4852_, v___f_4911_, v_sz_4913_, v___x_4914_, v___x_4912_);
lean_inc(v_a_4859_);
lean_inc_ref(v_a_4858_);
lean_inc(v_a_4857_);
lean_inc_ref(v_a_4856_);
lean_inc(v_a_4855_);
lean_inc_ref(v_a_4854_);
v___x_4916_ = lean_apply_7(v___x_2237__overap_4915_, v_a_4854_, v_a_4855_, v_a_4856_, v_a_4857_, v_a_4858_, v_a_4859_, lean_box(0));
if (lean_obj_tag(v___x_4916_) == 0)
{
lean_object* v_a_4917_; lean_object* v___x_4919_; uint8_t v_isShared_4920_; uint8_t v_isSharedCheck_4930_; 
v_a_4917_ = lean_ctor_get(v___x_4916_, 0);
v_isSharedCheck_4930_ = !lean_is_exclusive(v___x_4916_);
if (v_isSharedCheck_4930_ == 0)
{
v___x_4919_ = v___x_4916_;
v_isShared_4920_ = v_isSharedCheck_4930_;
goto v_resetjp_4918_;
}
else
{
lean_inc(v_a_4917_);
lean_dec(v___x_4916_);
v___x_4919_ = lean_box(0);
v_isShared_4920_ = v_isSharedCheck_4930_;
goto v_resetjp_4918_;
}
v_resetjp_4918_:
{
lean_object* v_fst_4921_; 
v_fst_4921_ = lean_ctor_get(v_a_4917_, 0);
lean_inc(v_fst_4921_);
lean_dec(v_a_4917_);
if (lean_obj_tag(v_fst_4921_) == 0)
{
lean_object* v___x_4922_; lean_object* v___x_4924_; 
v___x_4922_ = lean_box(v___x_4863_);
if (v_isShared_4920_ == 0)
{
lean_ctor_set(v___x_4919_, 0, v___x_4922_);
v___x_4924_ = v___x_4919_;
goto v_reusejp_4923_;
}
else
{
lean_object* v_reuseFailAlloc_4925_; 
v_reuseFailAlloc_4925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4925_, 0, v___x_4922_);
v___x_4924_ = v_reuseFailAlloc_4925_;
goto v_reusejp_4923_;
}
v_reusejp_4923_:
{
return v___x_4924_;
}
}
else
{
lean_object* v_val_4926_; lean_object* v___x_4928_; 
v_val_4926_ = lean_ctor_get(v_fst_4921_, 0);
lean_inc(v_val_4926_);
lean_dec_ref_known(v_fst_4921_, 1);
if (v_isShared_4920_ == 0)
{
lean_ctor_set(v___x_4919_, 0, v_val_4926_);
v___x_4928_ = v___x_4919_;
goto v_reusejp_4927_;
}
else
{
lean_object* v_reuseFailAlloc_4929_; 
v_reuseFailAlloc_4929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4929_, 0, v_val_4926_);
v___x_4928_ = v_reuseFailAlloc_4929_;
goto v_reusejp_4927_;
}
v_reusejp_4927_:
{
return v___x_4928_;
}
}
}
}
else
{
lean_object* v_a_4931_; lean_object* v___x_4933_; uint8_t v_isShared_4934_; uint8_t v_isSharedCheck_4938_; 
v_a_4931_ = lean_ctor_get(v___x_4916_, 0);
v_isSharedCheck_4938_ = !lean_is_exclusive(v___x_4916_);
if (v_isSharedCheck_4938_ == 0)
{
v___x_4933_ = v___x_4916_;
v_isShared_4934_ = v_isSharedCheck_4938_;
goto v_resetjp_4932_;
}
else
{
lean_inc(v_a_4931_);
lean_dec(v___x_4916_);
v___x_4933_ = lean_box(0);
v_isShared_4934_ = v_isSharedCheck_4938_;
goto v_resetjp_4932_;
}
v_resetjp_4932_:
{
lean_object* v___x_4936_; 
if (v_isShared_4934_ == 0)
{
v___x_4936_ = v___x_4933_;
goto v_reusejp_4935_;
}
else
{
lean_object* v_reuseFailAlloc_4937_; 
v_reuseFailAlloc_4937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4937_, 0, v_a_4931_);
v___x_4936_ = v_reuseFailAlloc_4937_;
goto v_reusejp_4935_;
}
v_reusejp_4935_:
{
return v___x_4936_;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___boxed(lean_object* v_goals_u2081_4945_, lean_object* v_goals_u2082_4946_, lean_object* v_a_4947_, lean_object* v_a_4948_, lean_object* v_a_4949_, lean_object* v_a_4950_, lean_object* v_a_4951_, lean_object* v_a_4952_, lean_object* v_a_4953_){
_start:
{
lean_object* v_res_4954_; 
v_res_4954_ = lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore(v_goals_u2081_4945_, v_goals_u2082_4946_, v_a_4947_, v_a_4948_, v_a_4949_, v_a_4950_, v_a_4951_, v_a_4952_);
lean_dec(v_a_4952_);
lean_dec_ref(v_a_4951_);
lean_dec(v_a_4950_);
lean_dec_ref(v_a_4949_);
lean_dec(v_a_4948_);
lean_dec_ref(v_a_4947_);
return v_res_4954_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds(lean_object* v_mctx_u2081_4955_, lean_object* v_mctx_u2082_4956_, lean_object* v_lctx_u2081_4957_, lean_object* v_lctx_u2082_4958_, lean_object* v_localInstances_u2081_4959_, lean_object* v_localInstances_u2082_4960_, lean_object* v_e_u2081_4961_, lean_object* v_e_u2082_4962_, uint8_t v_allowAssignmentDiff_4963_, lean_object* v_a_4964_, lean_object* v_a_4965_, lean_object* v_a_4966_, lean_object* v_a_4967_){
_start:
{
lean_object* v___x_4969_; lean_object* v___x_4970_; lean_object* v___x_4971_; lean_object* v___x_4972_; lean_object* v___x_4973_; 
v___x_4969_ = lean_obj_once(&lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1, &lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1_once, _init_lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg___closed__1);
v___x_4970_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4970_, 0, v_lctx_u2081_4957_);
lean_ctor_set(v___x_4970_, 1, v_localInstances_u2081_4959_);
lean_ctor_set(v___x_4970_, 2, v_lctx_u2082_4958_);
lean_ctor_set(v___x_4970_, 3, v_localInstances_u2082_4960_);
lean_ctor_set(v___x_4970_, 4, v___x_4969_);
v___x_4971_ = lean_alloc_closure((void*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_exprsEqualUpToIdsCore_u2081___boxed), 10, 3);
lean_closure_set(v___x_4971_, 0, v_e_u2081_4961_);
lean_closure_set(v___x_4971_, 1, v_e_u2082_4962_);
lean_closure_set(v___x_4971_, 2, v___x_4970_);
v___x_4972_ = lean_box(0);
v___x_4973_ = lp_aesop_Aesop_EqualUpToIdsM_run___redArg(v___x_4971_, v___x_4972_, v_mctx_u2081_4955_, v_mctx_u2082_4956_, v_allowAssignmentDiff_4963_, v_a_4964_, v_a_4965_, v_a_4966_, v_a_4967_);
return v___x_4973_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds___boxed(lean_object* v_mctx_u2081_4974_, lean_object* v_mctx_u2082_4975_, lean_object* v_lctx_u2081_4976_, lean_object* v_lctx_u2082_4977_, lean_object* v_localInstances_u2081_4978_, lean_object* v_localInstances_u2082_4979_, lean_object* v_e_u2081_4980_, lean_object* v_e_u2082_4981_, lean_object* v_allowAssignmentDiff_4982_, lean_object* v_a_4983_, lean_object* v_a_4984_, lean_object* v_a_4985_, lean_object* v_a_4986_, lean_object* v_a_4987_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_4988_; lean_object* v_res_4989_; 
v_allowAssignmentDiff_boxed_4988_ = lean_unbox(v_allowAssignmentDiff_4982_);
v_res_4989_ = lp_aesop_Aesop_exprsEqualUpToIds(v_mctx_u2081_4974_, v_mctx_u2082_4975_, v_lctx_u2081_4976_, v_lctx_u2082_4977_, v_localInstances_u2081_4978_, v_localInstances_u2082_4979_, v_e_u2081_4980_, v_e_u2082_4981_, v_allowAssignmentDiff_boxed_4988_, v_a_4983_, v_a_4984_, v_a_4985_, v_a_4986_);
lean_dec(v_a_4986_);
lean_dec_ref(v_a_4985_);
lean_dec(v_a_4984_);
lean_dec_ref(v_a_4983_);
return v_res_4989_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds_x27(lean_object* v_e_u2081_4990_, lean_object* v_e_u2082_4991_, uint8_t v_allowAssignmentDiff_4992_, lean_object* v_a_4993_, lean_object* v_a_4994_, lean_object* v_a_4995_, lean_object* v_a_4996_){
_start:
{
lean_object* v___x_4998_; lean_object* v_mctx_4999_; lean_object* v_lctx_5000_; lean_object* v_localInstances_5001_; lean_object* v___x_5002_; 
v___x_4998_ = lean_st_ref_get(v_a_4994_);
v_mctx_4999_ = lean_ctor_get(v___x_4998_, 0);
lean_inc_ref_n(v_mctx_4999_, 2);
lean_dec(v___x_4998_);
v_lctx_5000_ = lean_ctor_get(v_a_4993_, 2);
v_localInstances_5001_ = lean_ctor_get(v_a_4993_, 3);
lean_inc_ref_n(v_localInstances_5001_, 2);
lean_inc_ref_n(v_lctx_5000_, 2);
v___x_5002_ = lp_aesop_Aesop_exprsEqualUpToIds(v_mctx_4999_, v_mctx_4999_, v_lctx_5000_, v_lctx_5000_, v_localInstances_5001_, v_localInstances_5001_, v_e_u2081_4990_, v_e_u2082_4991_, v_allowAssignmentDiff_4992_, v_a_4993_, v_a_4994_, v_a_4995_, v_a_4996_);
return v___x_5002_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_exprsEqualUpToIds_x27___boxed(lean_object* v_e_u2081_5003_, lean_object* v_e_u2082_5004_, lean_object* v_allowAssignmentDiff_5005_, lean_object* v_a_5006_, lean_object* v_a_5007_, lean_object* v_a_5008_, lean_object* v_a_5009_, lean_object* v_a_5010_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_5011_; lean_object* v_res_5012_; 
v_allowAssignmentDiff_boxed_5011_ = lean_unbox(v_allowAssignmentDiff_5005_);
v_res_5012_ = lp_aesop_Aesop_exprsEqualUpToIds_x27(v_e_u2081_5003_, v_e_u2082_5004_, v_allowAssignmentDiff_boxed_5011_, v_a_5006_, v_a_5007_, v_a_5008_, v_a_5009_);
lean_dec(v_a_5009_);
lean_dec_ref(v_a_5008_);
lean_dec(v_a_5007_);
lean_dec_ref(v_a_5006_);
return v_res_5012_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds(lean_object* v_commonMCtx_x3f_5013_, lean_object* v_mctx_u2081_5014_, lean_object* v_mctx_u2082_5015_, lean_object* v_mvarId_u2081_5016_, lean_object* v_mvarId_u2082_5017_, uint8_t v_allowAssignmentDiff_5018_, lean_object* v_a_5019_, lean_object* v_a_5020_, lean_object* v_a_5021_, lean_object* v_a_5022_){
_start:
{
lean_object* v___x_5024_; lean_object* v___x_5025_; 
v___x_5024_ = lean_alloc_closure((void*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___boxed), 9, 2);
lean_closure_set(v___x_5024_, 0, v_mvarId_u2081_5016_);
lean_closure_set(v___x_5024_, 1, v_mvarId_u2082_5017_);
v___x_5025_ = lp_aesop_Aesop_EqualUpToIdsM_run___redArg(v___x_5024_, v_commonMCtx_x3f_5013_, v_mctx_u2081_5014_, v_mctx_u2082_5015_, v_allowAssignmentDiff_5018_, v_a_5019_, v_a_5020_, v_a_5021_, v_a_5022_);
return v___x_5025_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds___boxed(lean_object* v_commonMCtx_x3f_5026_, lean_object* v_mctx_u2081_5027_, lean_object* v_mctx_u2082_5028_, lean_object* v_mvarId_u2081_5029_, lean_object* v_mvarId_u2082_5030_, lean_object* v_allowAssignmentDiff_5031_, lean_object* v_a_5032_, lean_object* v_a_5033_, lean_object* v_a_5034_, lean_object* v_a_5035_, lean_object* v_a_5036_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_5037_; lean_object* v_res_5038_; 
v_allowAssignmentDiff_boxed_5037_ = lean_unbox(v_allowAssignmentDiff_5031_);
v_res_5038_ = lp_aesop_Aesop_unassignedMVarsEqualUptoIds(v_commonMCtx_x3f_5026_, v_mctx_u2081_5027_, v_mctx_u2082_5028_, v_mvarId_u2081_5029_, v_mvarId_u2082_5030_, v_allowAssignmentDiff_boxed_5037_, v_a_5032_, v_a_5033_, v_a_5034_, v_a_5035_);
lean_dec(v_a_5035_);
lean_dec_ref(v_a_5034_);
lean_dec(v_a_5033_);
lean_dec_ref(v_a_5032_);
return v_res_5038_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds_x27(lean_object* v_commonMCtx_x3f_5039_, lean_object* v_mctx_u2081_5040_, lean_object* v_mctx_u2082_5041_, lean_object* v_mvarId_u2081_5042_, lean_object* v_mvarId_u2082_5043_, uint8_t v_allowAssignmentDiff_5044_, lean_object* v_a_5045_, lean_object* v_a_5046_, lean_object* v_a_5047_, lean_object* v_a_5048_){
_start:
{
lean_object* v___x_5050_; lean_object* v___x_5051_; 
v___x_5050_ = lean_alloc_closure((void*)(lp_aesop_Aesop_EqualUpToIds_Unsafe_unassignedMVarsEqualUpToIdsCore___boxed), 9, 2);
lean_closure_set(v___x_5050_, 0, v_mvarId_u2081_5042_);
lean_closure_set(v___x_5050_, 1, v_mvarId_u2082_5043_);
v___x_5051_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(v___x_5050_, v_commonMCtx_x3f_5039_, v_mctx_u2081_5040_, v_mctx_u2082_5041_, v_allowAssignmentDiff_5044_, v_a_5045_, v_a_5046_, v_a_5047_, v_a_5048_);
return v___x_5051_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unassignedMVarsEqualUptoIds_x27___boxed(lean_object* v_commonMCtx_x3f_5052_, lean_object* v_mctx_u2081_5053_, lean_object* v_mctx_u2082_5054_, lean_object* v_mvarId_u2081_5055_, lean_object* v_mvarId_u2082_5056_, lean_object* v_allowAssignmentDiff_5057_, lean_object* v_a_5058_, lean_object* v_a_5059_, lean_object* v_a_5060_, lean_object* v_a_5061_, lean_object* v_a_5062_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_5063_; lean_object* v_res_5064_; 
v_allowAssignmentDiff_boxed_5063_ = lean_unbox(v_allowAssignmentDiff_5057_);
v_res_5064_ = lp_aesop_Aesop_unassignedMVarsEqualUptoIds_x27(v_commonMCtx_x3f_5052_, v_mctx_u2081_5053_, v_mctx_u2082_5054_, v_mvarId_u2081_5055_, v_mvarId_u2082_5056_, v_allowAssignmentDiff_boxed_5063_, v_a_5058_, v_a_5059_, v_a_5060_, v_a_5061_);
lean_dec(v_a_5061_);
lean_dec_ref(v_a_5060_);
lean_dec(v_a_5059_);
lean_dec_ref(v_a_5058_);
return v_res_5064_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds(lean_object* v_commonMCtx_x3f_5065_, lean_object* v_mctx_u2081_5066_, lean_object* v_mctx_u2082_5067_, lean_object* v_goals_u2081_5068_, lean_object* v_goals_u2082_5069_, uint8_t v_allowAssignmentDiff_5070_, lean_object* v_a_5071_, lean_object* v_a_5072_, lean_object* v_a_5073_, lean_object* v_a_5074_){
_start:
{
lean_object* v___x_5076_; lean_object* v___x_5077_; 
v___x_5076_ = lean_alloc_closure((void*)(lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___boxed), 9, 2);
lean_closure_set(v___x_5076_, 0, v_goals_u2081_5068_);
lean_closure_set(v___x_5076_, 1, v_goals_u2082_5069_);
v___x_5077_ = lp_aesop_Aesop_EqualUpToIdsM_run___redArg(v___x_5076_, v_commonMCtx_x3f_5065_, v_mctx_u2081_5066_, v_mctx_u2082_5067_, v_allowAssignmentDiff_5070_, v_a_5071_, v_a_5072_, v_a_5073_, v_a_5074_);
return v___x_5077_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds___boxed(lean_object* v_commonMCtx_x3f_5078_, lean_object* v_mctx_u2081_5079_, lean_object* v_mctx_u2082_5080_, lean_object* v_goals_u2081_5081_, lean_object* v_goals_u2082_5082_, lean_object* v_allowAssignmentDiff_5083_, lean_object* v_a_5084_, lean_object* v_a_5085_, lean_object* v_a_5086_, lean_object* v_a_5087_, lean_object* v_a_5088_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_5089_; lean_object* v_res_5090_; 
v_allowAssignmentDiff_boxed_5089_ = lean_unbox(v_allowAssignmentDiff_5083_);
v_res_5090_ = lp_aesop_Aesop_tacticStatesEqualUpToIds(v_commonMCtx_x3f_5078_, v_mctx_u2081_5079_, v_mctx_u2082_5080_, v_goals_u2081_5081_, v_goals_u2082_5082_, v_allowAssignmentDiff_boxed_5089_, v_a_5084_, v_a_5085_, v_a_5086_, v_a_5087_);
lean_dec(v_a_5087_);
lean_dec_ref(v_a_5086_);
lean_dec(v_a_5085_);
lean_dec_ref(v_a_5084_);
return v_res_5090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds_x27(lean_object* v_commonMCtx_x3f_5091_, lean_object* v_mctx_u2081_5092_, lean_object* v_mctx_u2082_5093_, lean_object* v_goals_u2081_5094_, lean_object* v_goals_u2082_5095_, uint8_t v_allowAssignmentDiff_5096_, lean_object* v_a_5097_, lean_object* v_a_5098_, lean_object* v_a_5099_, lean_object* v_a_5100_){
_start:
{
lean_object* v___x_5102_; lean_object* v___x_5103_; 
v___x_5102_ = lean_alloc_closure((void*)(lp_aesop_Aesop_EqualUpToIds_tacticStatesEqualUpToIdsCore___boxed), 9, 2);
lean_closure_set(v___x_5102_, 0, v_goals_u2081_5094_);
lean_closure_set(v___x_5102_, 1, v_goals_u2082_5095_);
v___x_5103_ = lp_aesop_Aesop_EqualUpToIdsM_run_x27___redArg(v___x_5102_, v_commonMCtx_x3f_5091_, v_mctx_u2081_5092_, v_mctx_u2082_5093_, v_allowAssignmentDiff_5096_, v_a_5097_, v_a_5098_, v_a_5099_, v_a_5100_);
return v___x_5103_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds_x27___boxed(lean_object* v_commonMCtx_x3f_5104_, lean_object* v_mctx_u2081_5105_, lean_object* v_mctx_u2082_5106_, lean_object* v_goals_u2081_5107_, lean_object* v_goals_u2082_5108_, lean_object* v_allowAssignmentDiff_5109_, lean_object* v_a_5110_, lean_object* v_a_5111_, lean_object* v_a_5112_, lean_object* v_a_5113_, lean_object* v_a_5114_){
_start:
{
uint8_t v_allowAssignmentDiff_boxed_5115_; lean_object* v_res_5116_; 
v_allowAssignmentDiff_boxed_5115_ = lean_unbox(v_allowAssignmentDiff_5109_);
v_res_5116_ = lp_aesop_Aesop_tacticStatesEqualUpToIds_x27(v_commonMCtx_x3f_5104_, v_mctx_u2081_5105_, v_mctx_u2082_5106_, v_goals_u2081_5107_, v_goals_u2082_5108_, v_allowAssignmentDiff_boxed_5115_, v_a_5110_, v_a_5111_, v_a_5112_, v_a_5113_);
lean_dec(v_a_5113_);
lean_dec_ref(v_a_5112_);
lean_dec(v_a_5111_);
lean_dec_ref(v_a_5110_);
return v_res_5116_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Util_EqualUpToIds_0__Aesop_initFn_00___x40_Aesop_Util_EqualUpToIds_1192428161____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instMonadEqualUpToIdsM = _init_lp_aesop_Aesop_instMonadEqualUpToIdsM();
lean_mark_persistent(lp_aesop_Aesop_instMonadEqualUpToIdsM);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
}
#ifdef __cplusplus
}
#endif
