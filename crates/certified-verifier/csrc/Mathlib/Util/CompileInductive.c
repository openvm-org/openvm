// Lean compiler output
// Module: Mathlib.Util.CompileInductive
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Command public meta import Lean.Compiler.CSimpAttr public meta import Lean.Util.FoldConsts public meta import Lean.Data.AssocList
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
lean_object* lean_uint32_to_nat(uint32_t);
uint64_t lean_float_to_bits(double);
extern lean_object* l_Lean_instInhabitedExpr;
extern lean_object* l_Lean_instInhabitedRecursorRule_default;
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_RecursorVal_getFirstMinorIdx(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_RecursorVal_getMajorIdx(lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addAndCompile(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instInhabitedCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Level_param___override(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addDecl(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Compiler_CSimp_add(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_mkRecOnName(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantInfo(lean_object*);
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
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_compileDecl(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqRefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_isInductiveCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Compiler_CSimp_ext;
extern lean_object* l_Lean_Compiler_CSimp_instInhabitedState_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_usize_to_nat(size_t);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* lean_uint8_to_nat(uint8_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_byte_array_data(lean_object*);
lean_object* lean_string_to_utf8(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkRecName(lean_object*);
lean_object* l_Lean_RecursorVal_getFirstIndexIdx(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Lean_mkCasesOnName(lean_object*);
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_replace_expr(lean_object*, lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_InductiveVal_numCtors(lean_object*);
lean_object* l_Lean_mkBRecOnName(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_uint16_to_nat(uint16_t);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_mkPtrSet___redArg(lean_object*);
lean_object* l_Lean_Expr_FoldConstsImpl_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint32_t lean_float32_to_bits(float);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "rec_"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_mkRecNames(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_mkRecNames___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instInhabitedCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__1(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "while compiling "};
static const lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Mathlib.Util.CompileInductive"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "_private.Mathlib.Util.CompileInductive.0.Mathlib.Util.addAndCompile'"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileDefn_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_compileDefn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "eq"};
static const lean_object* lp_mathlib_Mathlib_Util_compileDefn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_compileDefn___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileDefn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileDefn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Util_hasCSimpLemma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_hasCSimpLemma___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "commandCompile_def%_"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 78, 33, 145, 2, 247, 91, 5)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "compile_def% "};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_commandCompile__def_x25__ = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__5 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__5_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__6 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "` is not a definition"};
static const lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3;
static const lean_string_object lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.MonadEnv"};
static const lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__4_value;
static const lean_string_object lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Lean.isDefn\?"};
static const lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__5 = (const lean_object*)&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "already compiled "};
static const lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "` is not a recursor"};
static const lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1;
static const lean_string_object lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Lean.isRec\?"};
static const lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "go"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Util_compileInductiveOnly_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is not an inductive type"};
static const lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileInductiveOnly_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not an inductive"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "not compiling "};
static const lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "_sizeOf_inst"};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_sizeOf_"};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductive(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileSizeOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileSizeOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "commandCompile_inductive%_"};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 238, 230, 231, 134, 228, 173, 255)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "compile_inductive% "};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_commandCompile__inductive_x25__ = (const lean_object*)&lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_rec_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_5_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_recOn_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_rec_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_rec_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_recOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_7_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9_(lean_object*);
static const lean_closure_object lp_mathlib_PUnit___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9_ = (const lean_object*)&lp_mathlib_PUnit___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9__value;
LEAN_EXPORT const lean_object* lp_mathlib_PUnit___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9_ = (const lean_object*)&lp_mathlib_PUnit___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9__value;
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_rec_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_3_(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_rec_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_3____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_recOn_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_5_(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_recOn_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_5____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_rec_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_recOn_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum_rec_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum_recOn_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_And_rec___redArg_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_3_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_And_rec_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_And_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_5_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_And_recOn_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_7_(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_7____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9_(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Bool___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9_ = (const lean_object*)&lp_mathlib_Bool___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9__value;
LEAN_EXPORT const lean_object* lp_mathlib_Bool___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9_ = (const lean_object*)&lp_mathlib_Bool___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9__value;
LEAN_EXPORT lean_object* lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_rec_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_5_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_recOn_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_rec_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_rec_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_False_recOn_00___x40_Mathlib_Util_CompileInductive_2706509475____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Empty_recOn_00___x40_Mathlib_Util_CompileInductive_2909641248____hygCtx___hyg_3_(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Empty_recOn_00___x40_Mathlib_Util_CompileInductive_2909641248____hygCtx___hyg_3____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fin___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233_ = (const lean_object*)&lp_mathlib_Fin___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233__value;
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_BitVec___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_BitVec___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237_ = (const lean_object*)&lp_mathlib_BitVec___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237__value;
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed(lean_object*);
static const lean_closure_object lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_ = (const lean_object*)&lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239__value;
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed(lean_object*);
static const lean_closure_object lp_mathlib_UInt8___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_241__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UInt8___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_241_ = (const lean_object*)&lp_mathlib_UInt8___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_241__value;
LEAN_EXPORT const lean_object* lp_mathlib_UInt8___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_241_ = (const lean_object*)&lp_mathlib_UInt8___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_241__value;
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(lean_object*, uint16_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(lean_object*, lean_object*, uint16_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(uint16_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(lean_object*, uint16_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254_(uint16_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254____boxed(lean_object*);
static const lean_closure_object lp_mathlib_UInt16___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_256__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UInt16___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_256_ = (const lean_object*)&lp_mathlib_UInt16___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_256__value;
LEAN_EXPORT const lean_object* lp_mathlib_UInt16___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_256_ = (const lean_object*)&lp_mathlib_UInt16___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_256__value;
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(lean_object*, lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(lean_object*, uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269_(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269____boxed(lean_object*);
static const lean_closure_object lp_mathlib_UInt32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_271__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UInt32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_271_ = (const lean_object*)&lp_mathlib_UInt32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_271__value;
LEAN_EXPORT const lean_object* lp_mathlib_UInt32___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_271_ = (const lean_object*)&lp_mathlib_UInt32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_271__value;
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284_(uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284____boxed(lean_object*);
static const lean_closure_object lp_mathlib_UInt64___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_286__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UInt64___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_286_ = (const lean_object*)&lp_mathlib_UInt64___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_286__value;
LEAN_EXPORT const lean_object* lp_mathlib_UInt64___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_286_ = (const lean_object*)&lp_mathlib_UInt64___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_286__value;
LEAN_EXPORT lean_object* lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(lean_object*, size_t);
LEAN_EXPORT lean_object* lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(lean_object*, lean_object*, size_t);
LEAN_EXPORT lean_object* lp_mathlib_USize_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299_(size_t);
LEAN_EXPORT lean_object* lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299____boxed(lean_object*);
static const lean_closure_object lp_mathlib_USize___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_301__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_USize___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_301_ = (const lean_object*)&lp_mathlib_USize___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_301__value;
LEAN_EXPORT const lean_object* lp_mathlib_USize___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_301_ = (const lean_object*)&lp_mathlib_USize___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_301__value;
LEAN_EXPORT lean_object* lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(lean_object*, double);
LEAN_EXPORT lean_object* lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(lean_object*, lean_object*, double);
LEAN_EXPORT lean_object* lp_mathlib_Float_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(double, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(lean_object*, double, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318_(uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_320__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_320_ = (const lean_object*)&lp_mathlib_Float_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_320__value;
LEAN_EXPORT const lean_object* lp_mathlib_Float_Model___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_320_ = (const lean_object*)&lp_mathlib_Float_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_320__value;
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_ = (const lean_object*)&lp_mathlib_Float___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322__value;
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(double);
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_324__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_324_ = (const lean_object*)&lp_mathlib_Float___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_324__value;
LEAN_EXPORT const lean_object* lp_mathlib_Float___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_324_ = (const lean_object*)&lp_mathlib_Float___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_324__value;
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(lean_object*, float);
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(lean_object*, lean_object*, float);
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(float, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(lean_object*, float, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(lean_object*, lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(lean_object*, uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341_(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float32_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_343__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float32_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_343_ = (const lean_object*)&lp_mathlib_Float32_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_343__value;
LEAN_EXPORT const lean_object* lp_mathlib_Float32_Model___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_343_ = (const lean_object*)&lp_mathlib_Float32_Model___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_343__value;
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float32___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float32___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_ = (const lean_object*)&lp_mathlib_Float32___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345__value;
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(float);
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed(lean_object*);
static const lean_closure_object lp_mathlib_Float32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_347__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Float32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_347_ = (const lean_object*)&lp_mathlib_Float32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_347__value;
LEAN_EXPORT const lean_object* lp_mathlib_Float32___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_347_ = (const lean_object*)&lp_mathlib_Float32___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_347__value;
LEAN_EXPORT lean_object* lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_5_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_13_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_13_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___closed__0 = (const lean_object*)&lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ByteArray___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_(lean_object*);
static const lean_closure_object lp_mathlib_ByteArray___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ByteArray___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ByteArray___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_ = (const lean_object*)&lp_mathlib_ByteArray___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__value;
LEAN_EXPORT lean_object* lp_mathlib_ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_(lean_object*);
static const lean_closure_object lp_mathlib_ByteArray___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_21__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ByteArray___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_21_ = (const lean_object*)&lp_mathlib_ByteArray___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_21__value;
LEAN_EXPORT const lean_object* lp_mathlib_ByteArray___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_21_ = (const lean_object*)&lp_mathlib_ByteArray___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_21__value;
LEAN_EXPORT lean_object* lp_mathlib_String___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_String___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_String___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_String___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_ = (const lean_object*)&lp_mathlib_String___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23__value;
LEAN_EXPORT lean_object* lp_mathlib_String___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_(lean_object*);
static const lean_closure_object lp_mathlib_String___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_25__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_String___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_String___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_25_ = (const lean_object*)&lp_mathlib_String___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_25__value;
LEAN_EXPORT const lean_object* lp_mathlib_String___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_25_ = (const lean_object*)&lp_mathlib_String___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_25__value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg___lam__1_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Name_sizeOf___closed__0_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Name_sizeOf___closed__0_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib_Lean_Name_sizeOf___closed__0_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3__value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3_(lean_object*);
static const lean_closure_object lp_mathlib_Lean_instSizeOfName___closed__0_00___x40_Mathlib_Util_CompileInductive_1320556760____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Name_sizeOf_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_instSizeOfName___closed__0_00___x40_Mathlib_Util_CompileInductive_1320556760____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib_Lean_instSizeOfName___closed__0_00___x40_Mathlib_Util_CompileInductive_1320556760____hygCtx___hyg_3__value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_instSizeOfName_00___x40_Mathlib_Util_CompileInductive_1320556760____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib_Lean_instSizeOfName___closed__0_00___x40_Mathlib_Util_CompileInductive_1320556760____hygCtx___hyg_3__value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg(lean_object* v_a_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
else
{
lean_object* v_key_4_; lean_object* v_value_5_; lean_object* v_tail_6_; uint8_t v___x_7_; 
v_key_4_ = lean_ctor_get(v_x_2_, 0);
v_value_5_ = lean_ctor_get(v_x_2_, 1);
v_tail_6_ = lean_ctor_get(v_x_2_, 2);
v___x_7_ = lean_name_eq(v_key_4_, v_a_1_);
if (v___x_7_ == 0)
{
v_x_2_ = v_tail_6_;
goto _start;
}
else
{
lean_object* v___x_9_; 
lean_inc(v_value_5_);
v___x_9_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_9_, 0, v_value_5_);
return v___x_9_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg___boxed(lean_object* v_a_10_, lean_object* v_x_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg(v_a_10_, v_x_11_);
lean_dec(v_x_11_);
lean_dec(v_a_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0(lean_object* v_repl_13_, lean_object* v_x_14_){
_start:
{
if (lean_obj_tag(v_x_14_) == 4)
{
lean_object* v_declName_15_; lean_object* v_us_16_; lean_object* v___x_17_; 
v_declName_15_ = lean_ctor_get(v_x_14_, 0);
lean_inc(v_declName_15_);
v_us_16_ = lean_ctor_get(v_x_14_, 1);
lean_inc(v_us_16_);
lean_dec_ref_known(v_x_14_, 2);
v___x_17_ = lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg(v_declName_15_, v_repl_13_);
lean_dec(v_declName_15_);
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v___x_18_; 
lean_dec(v_us_16_);
v___x_18_ = lean_box(0);
return v___x_18_;
}
else
{
lean_object* v_val_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_27_; 
v_val_19_ = lean_ctor_get(v___x_17_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_27_ == 0)
{
v___x_21_ = v___x_17_;
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_val_19_);
lean_dec(v___x_17_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_27_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
lean_object* v___x_23_; lean_object* v___x_25_; 
v___x_23_ = l_Lean_Expr_const___override(v_val_19_, v_us_16_);
if (v_isShared_22_ == 0)
{
lean_ctor_set(v___x_21_, 0, v___x_23_);
v___x_25_ = v___x_21_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_23_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
else
{
lean_object* v___x_28_; 
lean_dec_ref(v_x_14_);
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0___boxed(lean_object* v_repl_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0(v_repl_29_, v_x_30_);
lean_dec(v_repl_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(lean_object* v_repl_32_, lean_object* v_e_33_){
_start:
{
lean_object* v___f_34_; lean_object* v___x_35_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___lam__0___boxed), 2, 1);
lean_closure_set(v___f_34_, 0, v_repl_32_);
v___x_35_ = lean_replace_expr(v___f_34_, v_e_33_);
lean_dec_ref(v___f_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst___boxed(lean_object* v_repl_36_, lean_object* v_e_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(v_repl_36_, v_e_37_);
lean_dec_ref(v_e_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0(lean_object* v_00_u03b2_39_, lean_object* v_a_40_, lean_object* v_x_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___redArg(v_a_40_, v_x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0___boxed(lean_object* v_00_u03b2_43_, lean_object* v_a_44_, lean_object* v_x_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_AssocList_find_x3f___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst_spec__0(v_00_u03b2_43_, v_a_44_, v_x_45_);
lean_dec(v_x_45_);
lean_dec(v_a_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1(lean_object* v_main_48_, lean_object* v_a_49_, lean_object* v_a_50_){
_start:
{
if (lean_obj_tag(v_a_49_) == 0)
{
lean_object* v___x_51_; 
lean_dec(v_main_48_);
v___x_51_ = l_List_reverse___redArg(v_a_50_);
return v___x_51_;
}
else
{
lean_object* v_head_52_; lean_object* v_tail_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_67_; 
v_head_52_ = lean_ctor_get(v_a_49_, 0);
v_tail_53_ = lean_ctor_get(v_a_49_, 1);
v_isSharedCheck_67_ = !lean_is_exclusive(v_a_49_);
if (v_isSharedCheck_67_ == 0)
{
v___x_55_ = v_a_49_;
v_isShared_56_ = v_isSharedCheck_67_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_tail_53_);
lean_inc(v_head_52_);
lean_dec(v_a_49_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_67_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_64_; 
v___x_57_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1___closed__0));
v___x_58_ = lean_unsigned_to_nat(1u);
v___x_59_ = lean_nat_add(v_head_52_, v___x_58_);
lean_dec(v_head_52_);
v___x_60_ = l_Nat_reprFast(v___x_59_);
v___x_61_ = lean_string_append(v___x_57_, v___x_60_);
lean_dec_ref(v___x_60_);
lean_inc(v_main_48_);
v___x_62_ = l_Lean_Name_str___override(v_main_48_, v___x_61_);
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 1, v_a_50_);
lean_ctor_set(v___x_55_, 0, v___x_62_);
v___x_64_ = v___x_55_;
goto v_reusejp_63_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v___x_62_);
lean_ctor_set(v_reuseFailAlloc_66_, 1, v_a_50_);
v___x_64_ = v_reuseFailAlloc_66_;
goto v_reusejp_63_;
}
v_reusejp_63_:
{
v_a_49_ = v_tail_53_;
v_a_50_ = v___x_64_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__0(lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
if (lean_obj_tag(v_a_68_) == 0)
{
lean_object* v___x_70_; 
v___x_70_ = l_List_reverse___redArg(v_a_69_);
return v___x_70_;
}
else
{
lean_object* v_head_71_; lean_object* v_tail_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_81_; 
v_head_71_ = lean_ctor_get(v_a_68_, 0);
v_tail_72_ = lean_ctor_get(v_a_68_, 1);
v_isSharedCheck_81_ = !lean_is_exclusive(v_a_68_);
if (v_isSharedCheck_81_ == 0)
{
v___x_74_ = v_a_68_;
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_tail_72_);
lean_inc(v_head_71_);
lean_dec(v_a_68_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___x_76_; lean_object* v___x_78_; 
v___x_76_ = l_Lean_mkRecName(v_head_71_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 1, v_a_69_);
lean_ctor_set(v___x_74_, 0, v___x_76_);
v___x_78_ = v___x_74_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v___x_76_);
lean_ctor_set(v_reuseFailAlloc_80_, 1, v_a_69_);
v___x_78_ = v_reuseFailAlloc_80_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
v_a_68_ = v_tail_72_;
v_a_69_ = v___x_78_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_mkRecNames(lean_object* v_all_82_, lean_object* v_numMotives_83_){
_start:
{
lean_object* v___x_84_; uint8_t v___x_85_; 
v___x_84_ = l_List_lengthTR___redArg(v_all_82_);
v___x_85_ = lean_nat_dec_le(v_numMotives_83_, v___x_84_);
if (v___x_85_ == 0)
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v_main_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_86_ = lean_box(0);
v___x_87_ = lean_unsigned_to_nat(0u);
v_main_88_ = l_List_get_x21Internal___redArg(v___x_86_, v_all_82_, v___x_87_);
v___x_89_ = lean_box(0);
v___x_90_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__0(v_all_82_, v___x_89_);
v___x_91_ = lean_nat_sub(v_numMotives_83_, v___x_84_);
lean_dec(v___x_84_);
v___x_92_ = l_List_range(v___x_91_);
v___x_93_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__1(v_main_88_, v___x_92_, v___x_89_);
v___x_94_ = l_List_appendTR___redArg(v___x_90_, v___x_93_);
return v___x_94_;
}
else
{
lean_object* v___x_95_; lean_object* v___x_96_; 
lean_dec(v___x_84_);
v___x_95_ = lean_box(0);
v___x_96_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_mkRecNames_spec__0(v_all_82_, v___x_95_);
return v___x_96_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_mkRecNames___boxed(lean_object* v_all_97_, lean_object* v_numMotives_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Mathlib_Util_mkRecNames(v_all_97_, v_numMotives_98_);
lean_dec(v_numMotives_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3(lean_object* v_msg_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v___f_105_; lean_object* v___x_792__overap_106_; lean_object* v___x_107_; 
v___f_105_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___closed__0));
v___x_792__overap_106_ = lean_panic_fn_borrowed(v___f_105_, v_msg_101_);
lean_inc(v___y_103_);
lean_inc_ref(v___y_102_);
v___x_107_ = lean_apply_3(v___x_792__overap_106_, v___y_102_, v___y_103_, lean_box(0));
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3___boxed(lean_object* v_msg_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3(v_msg_108_, v___y_109_, v___y_110_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__1(lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
if (lean_obj_tag(v_a_113_) == 0)
{
lean_object* v___x_115_; 
v___x_115_ = l_List_reverse___redArg(v_a_114_);
return v___x_115_;
}
else
{
lean_object* v_head_116_; lean_object* v_toConstantVal_117_; lean_object* v_tail_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_127_; 
v_head_116_ = lean_ctor_get(v_a_113_, 0);
v_toConstantVal_117_ = lean_ctor_get(v_head_116_, 0);
lean_inc_ref(v_toConstantVal_117_);
v_tail_118_ = lean_ctor_get(v_a_113_, 1);
v_isSharedCheck_127_ = !lean_is_exclusive(v_a_113_);
if (v_isSharedCheck_127_ == 0)
{
lean_object* v_unused_128_; 
v_unused_128_ = lean_ctor_get(v_a_113_, 0);
lean_dec(v_unused_128_);
v___x_120_ = v_a_113_;
v_isShared_121_ = v_isSharedCheck_127_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_tail_118_);
lean_dec(v_a_113_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_127_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v_name_122_; lean_object* v___x_124_; 
v_name_122_ = lean_ctor_get(v_toConstantVal_117_, 0);
lean_inc(v_name_122_);
lean_dec_ref(v_toConstantVal_117_);
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 1, v_a_114_);
lean_ctor_set(v___x_120_, 0, v_name_122_);
v___x_124_ = v___x_120_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_name_122_);
lean_ctor_set(v_reuseFailAlloc_126_, 1, v_a_114_);
v___x_124_ = v_reuseFailAlloc_126_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
v_a_113_ = v_tail_118_;
v_a_114_ = v___x_124_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__0);
v___x_131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
return v___x_131_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_132_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1);
v___x_133_ = lean_unsigned_to_nat(0u);
v___x_134_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
lean_ctor_set(v___x_134_, 2, v___x_133_);
lean_ctor_set(v___x_134_, 3, v___x_133_);
lean_ctor_set(v___x_134_, 4, v___x_132_);
lean_ctor_set(v___x_134_, 5, v___x_132_);
lean_ctor_set(v___x_134_, 6, v___x_132_);
lean_ctor_set(v___x_134_, 7, v___x_132_);
lean_ctor_set(v___x_134_, 8, v___x_132_);
lean_ctor_set(v___x_134_, 9, v___x_132_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_135_ = lean_unsigned_to_nat(32u);
v___x_136_ = lean_mk_empty_array_with_capacity(v___x_135_);
v___x_137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4(void){
_start:
{
size_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_138_ = ((size_t)5ULL);
v___x_139_ = lean_unsigned_to_nat(0u);
v___x_140_ = lean_unsigned_to_nat(32u);
v___x_141_ = lean_mk_empty_array_with_capacity(v___x_140_);
v___x_142_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__3);
v___x_143_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v___x_141_);
lean_ctor_set(v___x_143_, 2, v___x_139_);
lean_ctor_set(v___x_143_, 3, v___x_139_);
lean_ctor_set_usize(v___x_143_, 4, v___x_138_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_144_ = lean_box(1);
v___x_145_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__4);
v___x_146_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__1);
v___x_147_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___x_145_);
lean_ctor_set(v___x_147_, 2, v___x_144_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0(lean_object* v_msgData_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v___x_152_; lean_object* v_env_153_; lean_object* v_options_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_152_ = lean_st_ref_get(v___y_150_);
v_env_153_ = lean_ctor_get(v___x_152_, 0);
lean_inc_ref(v_env_153_);
lean_dec(v___x_152_);
v_options_154_ = lean_ctor_get(v___y_149_, 2);
v___x_155_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__2);
v___x_156_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___closed__5);
lean_inc_ref(v_options_154_);
v___x_157_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_157_, 0, v_env_153_);
lean_ctor_set(v___x_157_, 1, v___x_155_);
lean_ctor_set(v___x_157_, 2, v___x_156_);
lean_ctor_set(v___x_157_, 3, v_options_154_);
v___x_158_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v_msgData_148_);
v___x_159_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0___boxed(lean_object* v_msgData_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0(v_msgData_160_, v___y_161_, v___y_162_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(lean_object* v_msg_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_ref_169_; lean_object* v___x_170_; lean_object* v_a_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_179_; 
v_ref_169_ = lean_ctor_get(v___y_166_, 5);
v___x_170_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0_spec__0(v_msg_165_, v___y_166_, v___y_167_);
v_a_171_ = lean_ctor_get(v___x_170_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_179_ == 0)
{
v___x_173_ = v___x_170_;
v_isShared_174_ = v_isSharedCheck_179_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_a_171_);
lean_dec(v___x_170_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_179_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_175_; lean_object* v___x_177_; 
lean_inc(v_ref_169_);
v___x_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_175_, 0, v_ref_169_);
lean_ctor_set(v___x_175_, 1, v_a_171_);
if (v_isShared_174_ == 0)
{
lean_ctor_set_tag(v___x_173_, 1);
lean_ctor_set(v___x_173_, 0, v___x_175_);
v___x_177_ = v___x_173_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v___x_175_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg___boxed(lean_object* v_msg_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(v_msg_180_, v___y_181_, v___y_182_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__2(lean_object* v_a_185_, lean_object* v_a_186_){
_start:
{
if (lean_obj_tag(v_a_185_) == 0)
{
lean_object* v___x_187_; 
v___x_187_ = l_List_reverse___redArg(v_a_186_);
return v___x_187_;
}
else
{
lean_object* v_head_188_; lean_object* v_tail_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_198_; 
v_head_188_ = lean_ctor_get(v_a_185_, 0);
v_tail_189_ = lean_ctor_get(v_a_185_, 1);
v_isSharedCheck_198_ = !lean_is_exclusive(v_a_185_);
if (v_isSharedCheck_198_ == 0)
{
v___x_191_ = v_a_185_;
v_isShared_192_ = v_isSharedCheck_198_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_tail_189_);
lean_inc(v_head_188_);
lean_dec(v_a_185_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_198_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_193_; lean_object* v___x_195_; 
v___x_193_ = l_Lean_MessageData_ofName(v_head_188_);
if (v_isShared_192_ == 0)
{
lean_ctor_set(v___x_191_, 1, v_a_186_);
lean_ctor_set(v___x_191_, 0, v___x_193_);
v___x_195_ = v___x_191_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_193_);
lean_ctor_set(v_reuseFailAlloc_197_, 1, v_a_186_);
v___x_195_ = v_reuseFailAlloc_197_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
v_a_185_ = v_tail_189_;
v_a_186_ = v___x_195_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__0));
v___x_201_ = l_Lean_stringToMessageData(v___x_200_);
return v___x_201_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3(void){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__2));
v___x_204_ = l_Lean_stringToMessageData(v___x_203_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_208_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6));
v___x_209_ = lean_unsigned_to_nat(11u);
v___x_210_ = lean_unsigned_to_nat(55u);
v___x_211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__5));
v___x_212_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__4));
v___x_213_ = l_mkPanicMessageWithDecl(v___x_212_, v___x_211_, v___x_210_, v___x_209_, v___x_208_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(lean_object* v_decl_214_, lean_object* v_a_215_, lean_object* v_a_216_){
_start:
{
uint8_t v___x_218_; uint8_t v___x_219_; lean_object* v___x_220_; 
v___x_218_ = 1;
v___x_219_ = 0;
lean_inc(v_decl_214_);
v___x_220_ = l_Lean_addAndCompile(v_decl_214_, v___x_218_, v___x_219_, v_a_215_, v_a_216_);
if (lean_obj_tag(v___x_220_) == 0)
{
lean_dec(v_decl_214_);
return v___x_220_;
}
else
{
lean_object* v_a_221_; uint8_t v___y_223_; uint8_t v___x_249_; 
v_a_221_ = lean_ctor_get(v___x_220_, 0);
lean_inc(v_a_221_);
v___x_249_ = l_Lean_Exception_isInterrupt(v_a_221_);
if (v___x_249_ == 0)
{
uint8_t v___x_250_; 
lean_inc(v_a_221_);
v___x_250_ = l_Lean_Exception_isRuntime(v_a_221_);
v___y_223_ = v___x_250_;
goto v___jp_222_;
}
else
{
v___y_223_ = v___x_249_;
goto v___jp_222_;
}
v___jp_222_:
{
if (v___y_223_ == 0)
{
lean_dec_ref_known(v___x_220_, 1);
switch(lean_obj_tag(v_decl_214_))
{
case 1:
{
lean_object* v_val_224_; lean_object* v_toConstantVal_225_; lean_object* v_name_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v_val_224_ = lean_ctor_get(v_decl_214_, 0);
lean_inc_ref(v_val_224_);
lean_dec_ref_known(v_decl_214_, 1);
v_toConstantVal_225_ = lean_ctor_get(v_val_224_, 0);
lean_inc_ref(v_toConstantVal_225_);
lean_dec_ref(v_val_224_);
v_name_226_ = lean_ctor_get(v_toConstantVal_225_, 0);
lean_inc(v_name_226_);
lean_dec_ref(v_toConstantVal_225_);
v___x_227_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1, &lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1_once, _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1);
v___x_228_ = l_Lean_MessageData_ofName(v_name_226_);
v___x_229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_227_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3, &lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3_once, _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3);
v___x_231_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_229_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
v___x_232_ = l_Lean_Exception_toMessageData(v_a_221_);
v___x_233_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_231_);
lean_ctor_set(v___x_233_, 1, v___x_232_);
v___x_234_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(v___x_233_, v_a_215_, v_a_216_);
return v___x_234_;
}
case 5:
{
lean_object* v_defns_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v_defns_235_ = lean_ctor_get(v_decl_214_, 0);
lean_inc(v_defns_235_);
lean_dec_ref_known(v_decl_214_, 1);
v___x_236_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1, &lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1_once, _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__1);
v___x_237_ = lean_box(0);
v___x_238_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__1(v_defns_235_, v___x_237_);
v___x_239_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__2(v___x_238_, v___x_237_);
v___x_240_ = l_Lean_MessageData_ofList(v___x_239_);
v___x_241_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_236_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3, &lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3_once, _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__3);
v___x_243_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
v___x_244_ = l_Lean_Exception_toMessageData(v_a_221_);
v___x_245_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_243_);
lean_ctor_set(v___x_245_, 1, v___x_244_);
v___x_246_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(v___x_245_, v_a_215_, v_a_216_);
return v___x_246_;
}
default: 
{
lean_object* v___x_247_; lean_object* v___x_248_; 
lean_dec(v_a_221_);
lean_dec(v_decl_214_);
v___x_247_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7, &lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7_once, _init_lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__7);
v___x_248_ = lp_mathlib_panic___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__3(v___x_247_, v_a_215_, v_a_216_);
return v___x_248_;
}
}
}
else
{
lean_dec(v_a_221_);
lean_dec(v_decl_214_);
return v___x_220_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___boxed(lean_object* v_decl_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(v_decl_251_, v_a_252_, v_a_253_);
lean_dec(v_a_253_);
lean_dec_ref(v_a_252_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0(lean_object* v_00_u03b1_256_, lean_object* v_msg_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___redArg(v_msg_257_, v___y_258_, v___y_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0___boxed(lean_object* v_00_u03b1_262_, lean_object* v_msg_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27_spec__0(v_00_u03b1_262_, v_msg_263_, v___y_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileDefn_spec__0(lean_object* v_a_268_, lean_object* v_a_269_){
_start:
{
if (lean_obj_tag(v_a_268_) == 0)
{
lean_object* v___x_270_; 
v___x_270_ = l_List_reverse___redArg(v_a_269_);
return v___x_270_;
}
else
{
lean_object* v_head_271_; lean_object* v_tail_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_281_; 
v_head_271_ = lean_ctor_get(v_a_268_, 0);
v_tail_272_ = lean_ctor_get(v_a_268_, 1);
v_isSharedCheck_281_ = !lean_is_exclusive(v_a_268_);
if (v_isSharedCheck_281_ == 0)
{
v___x_274_ = v_a_268_;
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_tail_272_);
lean_inc(v_head_271_);
lean_dec(v_a_268_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_276_; lean_object* v___x_278_; 
v___x_276_ = l_Lean_Level_param___override(v_head_271_);
if (v_isShared_275_ == 0)
{
lean_ctor_set(v___x_274_, 1, v_a_269_);
lean_ctor_set(v___x_274_, 0, v___x_276_);
v___x_278_ = v___x_274_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_280_, 1, v_a_269_);
v___x_278_ = v_reuseFailAlloc_280_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
v_a_268_ = v_tail_272_;
v_a_269_ = v___x_278_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileDefn(lean_object* v_dv_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v___x_289_; lean_object* v_toConstantVal_290_; lean_object* v_env_291_; lean_object* v_value_292_; lean_object* v_hints_293_; uint8_t v_safety_294_; lean_object* v_all_295_; lean_object* v_name_296_; lean_object* v_levelParams_297_; lean_object* v_type_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_390_; 
v___x_289_ = lean_st_ref_get(v_a_287_);
v_toConstantVal_290_ = lean_ctor_get(v_dv_283_, 0);
lean_inc_ref(v_toConstantVal_290_);
v_env_291_ = lean_ctor_get(v___x_289_, 0);
lean_inc_ref(v_env_291_);
lean_dec(v___x_289_);
v_value_292_ = lean_ctor_get(v_dv_283_, 1);
v_hints_293_ = lean_ctor_get(v_dv_283_, 2);
v_safety_294_ = lean_ctor_get_uint8(v_dv_283_, sizeof(void*)*4);
v_all_295_ = lean_ctor_get(v_dv_283_, 3);
v_name_296_ = lean_ctor_get(v_toConstantVal_290_, 0);
v_levelParams_297_ = lean_ctor_get(v_toConstantVal_290_, 1);
v_type_298_ = lean_ctor_get(v_toConstantVal_290_, 2);
v_isSharedCheck_390_ = !lean_is_exclusive(v_toConstantVal_290_);
if (v_isSharedCheck_390_ == 0)
{
v___x_300_ = v_toConstantVal_290_;
v_isShared_301_ = v_isSharedCheck_390_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_type_298_);
lean_inc(v_levelParams_297_);
lean_inc(v_name_296_);
lean_dec(v_toConstantVal_290_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_390_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_302_; 
v___x_302_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_291_, v_name_296_);
lean_dec_ref(v_env_291_);
if (lean_obj_tag(v___x_302_) == 0)
{
uint8_t v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
lean_del_object(v___x_300_);
lean_dec_ref(v_type_298_);
lean_dec(v_levelParams_297_);
lean_dec(v_name_296_);
v___x_303_ = 1;
v___x_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_304_, 0, v_dv_283_);
v___x_305_ = l_Lean_compileDecl(v___x_304_, v___x_303_, v_a_286_, v_a_287_);
return v___x_305_;
}
else
{
lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_385_; 
lean_inc(v_all_295_);
lean_inc(v_hints_293_);
lean_inc_ref(v_value_292_);
v_isSharedCheck_385_ = !lean_is_exclusive(v_dv_283_);
if (v_isSharedCheck_385_ == 0)
{
lean_object* v_unused_386_; lean_object* v_unused_387_; lean_object* v_unused_388_; lean_object* v_unused_389_; 
v_unused_386_ = lean_ctor_get(v_dv_283_, 3);
lean_dec(v_unused_386_);
v_unused_387_ = lean_ctor_get(v_dv_283_, 2);
lean_dec(v_unused_387_);
v_unused_388_ = lean_ctor_get(v_dv_283_, 1);
lean_dec(v_unused_388_);
v_unused_389_ = lean_ctor_get(v_dv_283_, 0);
lean_dec(v_unused_389_);
v___x_307_ = v_dv_283_;
v_isShared_308_ = v_isSharedCheck_385_;
goto v_resetjp_306_;
}
else
{
lean_dec(v_dv_283_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_385_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_383_; 
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_383_ == 0)
{
lean_object* v_unused_384_; 
v_unused_384_ = lean_ctor_get(v___x_302_, 0);
lean_dec(v_unused_384_);
v___x_310_ = v___x_302_;
v_isShared_311_ = v_isSharedCheck_383_;
goto v_resetjp_309_;
}
else
{
lean_dec(v___x_302_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_383_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; 
lean_inc(v_name_296_);
v___x_312_ = l_Lean_Core_mkFreshUserName(v_name_296_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_312_) == 0)
{
lean_object* v_a_313_; lean_object* v___x_315_; 
v_a_313_ = lean_ctor_get(v___x_312_, 0);
lean_inc_n(v_a_313_, 2);
lean_dec_ref_known(v___x_312_, 1);
lean_inc(v_levelParams_297_);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v_a_313_);
v___x_315_ = v___x_300_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_a_313_);
lean_ctor_set(v_reuseFailAlloc_374_, 1, v_levelParams_297_);
lean_ctor_set(v_reuseFailAlloc_374_, 2, v_type_298_);
v___x_315_ = v_reuseFailAlloc_374_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
lean_object* v___x_317_; 
if (v_isShared_308_ == 0)
{
lean_ctor_set(v___x_307_, 0, v___x_315_);
v___x_317_ = v___x_307_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v___x_315_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v_value_292_);
lean_ctor_set(v_reuseFailAlloc_373_, 2, v_hints_293_);
lean_ctor_set(v_reuseFailAlloc_373_, 3, v_all_295_);
lean_ctor_set_uint8(v_reuseFailAlloc_373_, sizeof(void*)*4, v_safety_294_);
v___x_317_ = v_reuseFailAlloc_373_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
lean_object* v___x_319_; 
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v___x_317_);
v___x_319_ = v___x_310_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_317_);
v___x_319_ = v_reuseFailAlloc_372_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(v___x_319_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_320_) == 0)
{
lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_370_; 
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_320_);
if (v_isSharedCheck_370_ == 0)
{
lean_object* v_unused_371_; 
v_unused_371_ = lean_ctor_get(v___x_320_, 0);
lean_dec(v_unused_371_);
v___x_322_ = v___x_320_;
v_isShared_323_ = v_isSharedCheck_370_;
goto v_resetjp_321_;
}
else
{
lean_dec(v___x_320_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_370_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_324_ = lean_box(0);
lean_inc(v_levelParams_297_);
v___x_325_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileDefn_spec__0(v_levelParams_297_, v___x_324_);
lean_inc(v___x_325_);
lean_inc(v_name_296_);
v___x_326_ = l_Lean_Expr_const___override(v_name_296_, v___x_325_);
v___x_327_ = ((lean_object*)(lp_mathlib_Mathlib_Util_compileDefn___closed__0));
v___x_328_ = l_Lean_Name_str___override(v_name_296_, v___x_327_);
v___x_329_ = l_Lean_Core_mkFreshUserName(v___x_328_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_329_) == 0)
{
lean_object* v_a_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v_a_330_ = lean_ctor_get(v___x_329_, 0);
lean_inc(v_a_330_);
lean_dec_ref_known(v___x_329_, 1);
v___x_331_ = l_Lean_Expr_const___override(v_a_313_, v___x_325_);
lean_inc_ref(v___x_326_);
v___x_332_ = l_Lean_Meta_mkEq(v___x_326_, v___x_331_, v_a_284_, v_a_285_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v_a_333_; lean_object* v___x_334_; 
v_a_333_ = lean_ctor_get(v___x_332_, 0);
lean_inc(v_a_333_);
lean_dec_ref_known(v___x_332_, 1);
v___x_334_ = l_Lean_Meta_mkEqRefl(v___x_326_, v_a_284_, v_a_285_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_334_) == 0)
{
lean_object* v_a_335_; uint8_t v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_341_; 
v_a_335_ = lean_ctor_get(v___x_334_, 0);
lean_inc(v_a_335_);
lean_dec_ref_known(v___x_334_, 1);
v___x_336_ = 0;
lean_inc_n(v_a_330_, 2);
v___x_337_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_337_, 0, v_a_330_);
lean_ctor_set(v___x_337_, 1, v_levelParams_297_);
lean_ctor_set(v___x_337_, 2, v_a_333_);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v_a_330_);
lean_ctor_set(v___x_338_, 1, v___x_324_);
v___x_339_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_339_, 0, v___x_337_);
lean_ctor_set(v___x_339_, 1, v_a_335_);
lean_ctor_set(v___x_339_, 2, v___x_338_);
if (v_isShared_323_ == 0)
{
lean_ctor_set_tag(v___x_322_, 2);
lean_ctor_set(v___x_322_, 0, v___x_339_);
v___x_341_ = v___x_322_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_345_; 
v_reuseFailAlloc_345_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_345_, 0, v___x_339_);
v___x_341_ = v_reuseFailAlloc_345_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
lean_object* v___x_342_; 
v___x_342_ = l_Lean_addDecl(v___x_341_, v___x_336_, v_a_286_, v_a_287_);
if (lean_obj_tag(v___x_342_) == 0)
{
uint8_t v___x_343_; lean_object* v___x_344_; 
lean_dec_ref_known(v___x_342_, 1);
v___x_343_ = 0;
v___x_344_ = l_Lean_Compiler_CSimp_add(v_a_330_, v___x_343_, v_a_286_, v_a_287_);
return v___x_344_;
}
else
{
lean_dec(v_a_330_);
return v___x_342_;
}
}
}
else
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_353_; 
lean_dec(v_a_333_);
lean_dec(v_a_330_);
lean_del_object(v___x_322_);
lean_dec(v_levelParams_297_);
v_a_346_ = lean_ctor_get(v___x_334_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v___x_334_);
if (v_isSharedCheck_353_ == 0)
{
v___x_348_ = v___x_334_;
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_334_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_351_; 
if (v_isShared_349_ == 0)
{
v___x_351_ = v___x_348_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_a_346_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
lean_dec(v_a_330_);
lean_dec_ref(v___x_326_);
lean_del_object(v___x_322_);
lean_dec(v_levelParams_297_);
v_a_354_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_332_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_332_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v_a_354_);
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
else
{
lean_object* v_a_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_369_; 
lean_dec_ref(v___x_326_);
lean_dec(v___x_325_);
lean_del_object(v___x_322_);
lean_dec(v_a_313_);
lean_dec(v_levelParams_297_);
v_a_362_ = lean_ctor_get(v___x_329_, 0);
v_isSharedCheck_369_ = !lean_is_exclusive(v___x_329_);
if (v_isSharedCheck_369_ == 0)
{
v___x_364_ = v___x_329_;
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_a_362_);
lean_dec(v___x_329_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_367_; 
if (v_isShared_365_ == 0)
{
v___x_367_ = v___x_364_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_368_; 
v_reuseFailAlloc_368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_368_, 0, v_a_362_);
v___x_367_ = v_reuseFailAlloc_368_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
return v___x_367_;
}
}
}
}
}
else
{
lean_dec(v_a_313_);
lean_dec(v_levelParams_297_);
lean_dec(v_name_296_);
return v___x_320_;
}
}
}
}
}
else
{
lean_object* v_a_375_; lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_382_; 
lean_del_object(v___x_310_);
lean_del_object(v___x_307_);
lean_del_object(v___x_300_);
lean_dec_ref(v_type_298_);
lean_dec(v_levelParams_297_);
lean_dec(v_name_296_);
lean_dec(v_all_295_);
lean_dec(v_hints_293_);
lean_dec_ref(v_value_292_);
v_a_375_ = lean_ctor_get(v___x_312_, 0);
v_isSharedCheck_382_ = !lean_is_exclusive(v___x_312_);
if (v_isSharedCheck_382_ == 0)
{
v___x_377_ = v___x_312_;
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
else
{
lean_inc(v_a_375_);
lean_dec(v___x_312_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
lean_object* v___x_380_; 
if (v_isShared_378_ == 0)
{
v___x_380_ = v___x_377_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_a_375_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileDefn___boxed(lean_object* v_dv_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_Mathlib_Util_compileDefn(v_dv_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
lean_dec(v_a_395_);
lean_dec_ref(v_a_394_);
lean_dec(v_a_393_);
lean_dec_ref(v_a_392_);
return v_res_397_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg(lean_object* v_a_398_, lean_object* v_x_399_){
_start:
{
if (lean_obj_tag(v_x_399_) == 0)
{
uint8_t v___x_400_; 
v___x_400_ = 0;
return v___x_400_;
}
else
{
lean_object* v_key_401_; lean_object* v_tail_402_; uint8_t v___x_403_; 
v_key_401_ = lean_ctor_get(v_x_399_, 0);
v_tail_402_ = lean_ctor_get(v_x_399_, 2);
v___x_403_ = lean_name_eq(v_key_401_, v_a_398_);
if (v___x_403_ == 0)
{
v_x_399_ = v_tail_402_;
goto _start;
}
else
{
return v___x_403_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_405_, lean_object* v_x_406_){
_start:
{
uint8_t v_res_407_; lean_object* v_r_408_; 
v_res_407_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg(v_a_405_, v_x_406_);
lean_dec(v_x_406_);
lean_dec(v_a_405_);
v_r_408_ = lean_box(v_res_407_);
return v_r_408_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(lean_object* v_m_409_, lean_object* v_a_410_){
_start:
{
lean_object* v_buckets_411_; lean_object* v___x_412_; uint64_t v___y_414_; 
v_buckets_411_ = lean_ctor_get(v_m_409_, 1);
v___x_412_ = lean_array_get_size(v_buckets_411_);
if (lean_obj_tag(v_a_410_) == 0)
{
uint64_t v___x_428_; 
v___x_428_ = 1723ULL;
v___y_414_ = v___x_428_;
goto v___jp_413_;
}
else
{
uint64_t v_hash_429_; 
v_hash_429_ = lean_ctor_get_uint64(v_a_410_, sizeof(void*)*2);
v___y_414_ = v_hash_429_;
goto v___jp_413_;
}
v___jp_413_:
{
uint64_t v___x_415_; uint64_t v___x_416_; uint64_t v_fold_417_; uint64_t v___x_418_; uint64_t v___x_419_; uint64_t v___x_420_; size_t v___x_421_; size_t v___x_422_; size_t v___x_423_; size_t v___x_424_; size_t v___x_425_; lean_object* v___x_426_; uint8_t v___x_427_; 
v___x_415_ = 32ULL;
v___x_416_ = lean_uint64_shift_right(v___y_414_, v___x_415_);
v_fold_417_ = lean_uint64_xor(v___y_414_, v___x_416_);
v___x_418_ = 16ULL;
v___x_419_ = lean_uint64_shift_right(v_fold_417_, v___x_418_);
v___x_420_ = lean_uint64_xor(v_fold_417_, v___x_419_);
v___x_421_ = lean_uint64_to_usize(v___x_420_);
v___x_422_ = lean_usize_of_nat(v___x_412_);
v___x_423_ = ((size_t)1ULL);
v___x_424_ = lean_usize_sub(v___x_422_, v___x_423_);
v___x_425_ = lean_usize_land(v___x_421_, v___x_424_);
v___x_426_ = lean_array_uget_borrowed(v_buckets_411_, v___x_425_);
v___x_427_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg(v_a_410_, v___x_426_);
return v___x_427_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg___boxed(lean_object* v_m_430_, lean_object* v_a_431_){
_start:
{
uint8_t v_res_432_; lean_object* v_r_433_; 
v_res_432_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(v_m_430_, v_a_431_);
lean_dec(v_a_431_);
lean_dec_ref(v_m_430_);
v_r_433_ = lean_box(v_res_432_);
return v_r_433_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg(lean_object* v_keys_434_, lean_object* v_i_435_, lean_object* v_k_436_){
_start:
{
lean_object* v___x_437_; uint8_t v___x_438_; 
v___x_437_ = lean_array_get_size(v_keys_434_);
v___x_438_ = lean_nat_dec_lt(v_i_435_, v___x_437_);
if (v___x_438_ == 0)
{
lean_dec(v_i_435_);
return v___x_438_;
}
else
{
lean_object* v_k_x27_439_; uint8_t v___x_440_; 
v_k_x27_439_ = lean_array_fget_borrowed(v_keys_434_, v_i_435_);
v___x_440_ = lean_name_eq(v_k_436_, v_k_x27_439_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_441_ = lean_unsigned_to_nat(1u);
v___x_442_ = lean_nat_add(v_i_435_, v___x_441_);
lean_dec(v_i_435_);
v_i_435_ = v___x_442_;
goto _start;
}
else
{
lean_dec(v_i_435_);
return v___x_440_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg___boxed(lean_object* v_keys_444_, lean_object* v_i_445_, lean_object* v_k_446_){
_start:
{
uint8_t v_res_447_; lean_object* v_r_448_; 
v_res_447_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg(v_keys_444_, v_i_445_, v_k_446_);
lean_dec(v_k_446_);
lean_dec_ref(v_keys_444_);
v_r_448_ = lean_box(v_res_447_);
return v_r_448_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg(lean_object* v_x_449_, size_t v_x_450_, lean_object* v_x_451_){
_start:
{
if (lean_obj_tag(v_x_449_) == 0)
{
lean_object* v_es_452_; lean_object* v___x_453_; size_t v___x_454_; size_t v___x_455_; lean_object* v_j_456_; lean_object* v___x_457_; 
v_es_452_ = lean_ctor_get(v_x_449_, 0);
v___x_453_ = lean_box(2);
v___x_454_ = ((size_t)31ULL);
v___x_455_ = lean_usize_land(v_x_450_, v___x_454_);
v_j_456_ = lean_usize_to_nat(v___x_455_);
v___x_457_ = lean_array_get_borrowed(v___x_453_, v_es_452_, v_j_456_);
lean_dec(v_j_456_);
switch(lean_obj_tag(v___x_457_))
{
case 0:
{
lean_object* v_key_458_; uint8_t v___x_459_; 
v_key_458_ = lean_ctor_get(v___x_457_, 0);
v___x_459_ = lean_name_eq(v_x_451_, v_key_458_);
return v___x_459_;
}
case 1:
{
lean_object* v_node_460_; size_t v___x_461_; size_t v___x_462_; 
v_node_460_ = lean_ctor_get(v___x_457_, 0);
v___x_461_ = ((size_t)5ULL);
v___x_462_ = lean_usize_shift_right(v_x_450_, v___x_461_);
v_x_449_ = v_node_460_;
v_x_450_ = v___x_462_;
goto _start;
}
default: 
{
uint8_t v___x_464_; 
v___x_464_ = 0;
return v___x_464_;
}
}
}
else
{
lean_object* v_ks_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v_ks_465_ = lean_ctor_get(v_x_449_, 0);
v___x_466_ = lean_unsigned_to_nat(0u);
v___x_467_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg(v_ks_465_, v___x_466_, v_x_451_);
return v___x_467_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_x_468_, lean_object* v_x_469_, lean_object* v_x_470_){
_start:
{
size_t v_x_287__boxed_471_; uint8_t v_res_472_; lean_object* v_r_473_; 
v_x_287__boxed_471_ = lean_unbox_usize(v_x_469_);
lean_dec(v_x_469_);
v_res_472_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg(v_x_468_, v_x_287__boxed_471_, v_x_470_);
lean_dec(v_x_470_);
lean_dec_ref(v_x_468_);
v_r_473_ = lean_box(v_res_472_);
return v_r_473_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg(lean_object* v_x_474_, lean_object* v_x_475_){
_start:
{
uint64_t v___y_477_; 
if (lean_obj_tag(v_x_475_) == 0)
{
uint64_t v___x_480_; 
v___x_480_ = 1723ULL;
v___y_477_ = v___x_480_;
goto v___jp_476_;
}
else
{
uint64_t v_hash_481_; 
v_hash_481_ = lean_ctor_get_uint64(v_x_475_, sizeof(void*)*2);
v___y_477_ = v_hash_481_;
goto v___jp_476_;
}
v___jp_476_:
{
size_t v___x_478_; uint8_t v___x_479_; 
v___x_478_ = lean_uint64_to_usize(v___y_477_);
v___x_479_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg(v_x_474_, v___x_478_, v_x_475_);
return v___x_479_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg___boxed(lean_object* v_x_482_, lean_object* v_x_483_){
_start:
{
uint8_t v_res_484_; lean_object* v_r_485_; 
v_res_484_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg(v_x_482_, v_x_483_);
lean_dec(v_x_483_);
lean_dec_ref(v_x_482_);
v_r_485_ = lean_box(v_res_484_);
return v_r_485_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg(lean_object* v_x_486_, lean_object* v_x_487_){
_start:
{
uint8_t v_stage_u2081_488_; 
v_stage_u2081_488_ = lean_ctor_get_uint8(v_x_486_, sizeof(void*)*2);
if (v_stage_u2081_488_ == 0)
{
lean_object* v_map_u2081_489_; lean_object* v_map_u2082_490_; uint8_t v___x_491_; 
v_map_u2081_489_ = lean_ctor_get(v_x_486_, 0);
v_map_u2082_490_ = lean_ctor_get(v_x_486_, 1);
v___x_491_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(v_map_u2081_489_, v_x_487_);
if (v___x_491_ == 0)
{
uint8_t v___x_492_; 
v___x_492_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg(v_map_u2082_490_, v_x_487_);
return v___x_492_;
}
else
{
return v___x_491_;
}
}
else
{
lean_object* v_map_u2081_493_; uint8_t v___x_494_; 
v_map_u2081_493_ = lean_ctor_get(v_x_486_, 0);
v___x_494_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(v_map_u2081_493_, v_x_487_);
return v___x_494_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg___boxed(lean_object* v_x_495_, lean_object* v_x_496_){
_start:
{
uint8_t v_res_497_; lean_object* v_r_498_; 
v_res_497_ = lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg(v_x_495_, v_x_496_);
lean_dec(v_x_496_);
lean_dec_ref(v_x_495_);
v_r_498_ = lean_box(v_res_497_);
return v_r_498_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Util_hasCSimpLemma(lean_object* v_env_499_, lean_object* v_n_500_){
_start:
{
lean_object* v___x_501_; lean_object* v_ext_502_; lean_object* v_toEnvExtension_503_; lean_object* v_asyncMode_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v_map_507_; uint8_t v___x_508_; 
v___x_501_ = l_Lean_Compiler_CSimp_ext;
v_ext_502_ = lean_ctor_get(v___x_501_, 1);
v_toEnvExtension_503_ = lean_ctor_get(v_ext_502_, 0);
v_asyncMode_504_ = lean_ctor_get(v_toEnvExtension_503_, 2);
v___x_505_ = l_Lean_Compiler_CSimp_instInhabitedState_default;
v___x_506_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_505_, v___x_501_, v_env_499_, v_asyncMode_504_);
v_map_507_ = lean_ctor_get(v___x_506_, 0);
lean_inc_ref(v_map_507_);
lean_dec(v___x_506_);
v___x_508_ = lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg(v_map_507_, v_n_500_);
lean_dec_ref(v_map_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_hasCSimpLemma___boxed(lean_object* v_env_509_, lean_object* v_n_510_){
_start:
{
uint8_t v_res_511_; lean_object* v_r_512_; 
v_res_511_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_509_, v_n_510_);
lean_dec(v_n_510_);
v_r_512_ = lean_box(v_res_511_);
return v_r_512_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0(lean_object* v_00_u03b2_513_, lean_object* v_x_514_, lean_object* v_x_515_){
_start:
{
uint8_t v___x_516_; 
v___x_516_ = lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___redArg(v_x_514_, v_x_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0___boxed(lean_object* v_00_u03b2_517_, lean_object* v_x_518_, lean_object* v_x_519_){
_start:
{
uint8_t v_res_520_; lean_object* v_r_521_; 
v_res_520_ = lp_mathlib_Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0(v_00_u03b2_517_, v_x_518_, v_x_519_);
lean_dec(v_x_519_);
lean_dec_ref(v_x_518_);
v_r_521_ = lean_box(v_res_520_);
return v_r_521_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0(lean_object* v_00_u03b2_522_, lean_object* v_m_523_, lean_object* v_a_524_){
_start:
{
uint8_t v___x_525_; 
v___x_525_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___redArg(v_m_523_, v_a_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0___boxed(lean_object* v_00_u03b2_526_, lean_object* v_m_527_, lean_object* v_a_528_){
_start:
{
uint8_t v_res_529_; lean_object* v_r_530_; 
v_res_529_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0(v_00_u03b2_526_, v_m_527_, v_a_528_);
lean_dec(v_a_528_);
lean_dec_ref(v_m_527_);
v_r_530_ = lean_box(v_res_529_);
return v_r_530_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1(lean_object* v_00_u03b2_531_, lean_object* v_x_532_, lean_object* v_x_533_){
_start:
{
uint8_t v___x_534_; 
v___x_534_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___redArg(v_x_532_, v_x_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1___boxed(lean_object* v_00_u03b2_535_, lean_object* v_x_536_, lean_object* v_x_537_){
_start:
{
uint8_t v_res_538_; lean_object* v_r_539_; 
v_res_538_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1(v_00_u03b2_535_, v_x_536_, v_x_537_);
lean_dec(v_x_537_);
lean_dec_ref(v_x_536_);
v_r_539_ = lean_box(v_res_538_);
return v_r_539_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_540_, lean_object* v_a_541_, lean_object* v_x_542_){
_start:
{
uint8_t v___x_543_; 
v___x_543_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___redArg(v_a_541_, v_x_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_544_, lean_object* v_a_545_, lean_object* v_x_546_){
_start:
{
uint8_t v_res_547_; lean_object* v_r_548_; 
v_res_547_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__0_spec__1(v_00_u03b2_544_, v_a_545_, v_x_546_);
lean_dec(v_x_546_);
lean_dec(v_a_545_);
v_r_548_ = lean_box(v_res_547_);
return v_r_548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_549_, lean_object* v_x_550_, size_t v_x_551_, lean_object* v_x_552_){
_start:
{
uint8_t v___x_553_; 
v___x_553_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___redArg(v_x_550_, v_x_551_, v_x_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b2_554_, lean_object* v_x_555_, lean_object* v_x_556_, lean_object* v_x_557_){
_start:
{
size_t v_x_395__boxed_558_; uint8_t v_res_559_; lean_object* v_r_560_; 
v_x_395__boxed_558_ = lean_unbox_usize(v_x_556_);
lean_dec(v_x_556_);
v_res_559_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3(v_00_u03b2_554_, v_x_555_, v_x_395__boxed_558_, v_x_557_);
lean_dec(v_x_557_);
lean_dec_ref(v_x_555_);
v_r_560_ = lean_box(v_res_559_);
return v_r_560_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4(lean_object* v_00_u03b2_561_, lean_object* v_keys_562_, lean_object* v_vals_563_, lean_object* v_heq_564_, lean_object* v_i_565_, lean_object* v_k_566_){
_start:
{
uint8_t v___x_567_; 
v___x_567_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___redArg(v_keys_562_, v_i_565_, v_k_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4___boxed(lean_object* v_00_u03b2_568_, lean_object* v_keys_569_, lean_object* v_vals_570_, lean_object* v_heq_571_, lean_object* v_i_572_, lean_object* v_k_573_){
_start:
{
uint8_t v_res_574_; lean_object* v_r_575_; 
v_res_574_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_SMap_contains___at___00Mathlib_Util_hasCSimpLemma_spec__0_spec__1_spec__3_spec__4(v_00_u03b2_568_, v_keys_569_, v_vals_570_, v_heq_571_, v_i_572_, v_k_573_);
lean_dec(v_k_573_);
lean_dec_ref(v_vals_570_);
lean_dec_ref(v_keys_569_);
v_r_575_ = lean_box(v_res_574_);
return v_r_575_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_603_ = lean_box(0);
v___x_604_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_605_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_604_);
lean_ctor_set(v___x_605_, 1, v___x_603_);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg(){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_607_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___closed__0);
v___x_608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_608_, 0, v___x_607_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg___boxed(lean_object* v___y_609_){
_start:
{
lean_object* v_res_610_; 
v_res_610_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg();
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0(lean_object* v_00_u03b1_611_, lean_object* v___y_612_, lean_object* v___y_613_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg();
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___boxed(lean_object* v_00_u03b1_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0(v_00_u03b1_616_, v___y_617_, v___y_618_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
return v_res_620_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(lean_object* v_opts_621_, lean_object* v_opt_622_){
_start:
{
lean_object* v_name_623_; lean_object* v_defValue_624_; lean_object* v_map_625_; lean_object* v___x_626_; 
v_name_623_ = lean_ctor_get(v_opt_622_, 0);
v_defValue_624_ = lean_ctor_get(v_opt_622_, 1);
v_map_625_ = lean_ctor_get(v_opts_621_, 0);
v___x_626_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_625_, v_name_623_);
if (lean_obj_tag(v___x_626_) == 0)
{
uint8_t v___x_627_; 
v___x_627_ = lean_unbox(v_defValue_624_);
return v___x_627_;
}
else
{
lean_object* v_val_628_; 
v_val_628_ = lean_ctor_get(v___x_626_, 0);
lean_inc(v_val_628_);
lean_dec_ref_known(v___x_626_, 1);
if (lean_obj_tag(v_val_628_) == 1)
{
uint8_t v_v_629_; 
v_v_629_ = lean_ctor_get_uint8(v_val_628_, 0);
lean_dec_ref_known(v_val_628_, 0);
return v_v_629_;
}
else
{
uint8_t v___x_630_; 
lean_dec(v_val_628_);
v___x_630_ = lean_unbox(v_defValue_624_);
return v___x_630_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7___boxed(lean_object* v_opts_631_, lean_object* v_opt_632_){
_start:
{
uint8_t v_res_633_; lean_object* v_r_634_; 
v_res_633_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(v_opts_631_, v_opt_632_);
lean_dec_ref(v_opt_632_);
lean_dec_ref(v_opts_631_);
v_r_634_ = lean_box(v_res_633_);
return v_r_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(lean_object* v_msgData_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
lean_object* v___x_641_; lean_object* v_env_642_; lean_object* v___x_643_; lean_object* v_mctx_644_; lean_object* v_lctx_645_; lean_object* v_options_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v___x_641_ = lean_st_ref_get(v___y_639_);
v_env_642_ = lean_ctor_get(v___x_641_, 0);
lean_inc_ref(v_env_642_);
lean_dec(v___x_641_);
v___x_643_ = lean_st_ref_get(v___y_637_);
v_mctx_644_ = lean_ctor_get(v___x_643_, 0);
lean_inc_ref(v_mctx_644_);
lean_dec(v___x_643_);
v_lctx_645_ = lean_ctor_get(v___y_636_, 2);
v_options_646_ = lean_ctor_get(v___y_638_, 2);
lean_inc_ref(v_options_646_);
lean_inc_ref(v_lctx_645_);
v___x_647_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_647_, 0, v_env_642_);
lean_ctor_set(v___x_647_, 1, v_mctx_644_);
lean_ctor_set(v___x_647_, 2, v_lctx_645_);
lean_ctor_set(v___x_647_, 3, v_options_646_);
v___x_648_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_648_, 0, v___x_647_);
lean_ctor_set(v___x_648_, 1, v_msgData_635_);
v___x_649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_649_, 0, v___x_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(v_msgData_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_653_);
lean_dec(v___y_652_);
lean_dec_ref(v___y_651_);
return v_res_656_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0(uint8_t v___y_665_, uint8_t v_suppressElabErrors_666_, lean_object* v_x_667_){
_start:
{
if (lean_obj_tag(v_x_667_) == 1)
{
lean_object* v_pre_668_; 
v_pre_668_ = lean_ctor_get(v_x_667_, 0);
switch(lean_obj_tag(v_pre_668_))
{
case 1:
{
lean_object* v_pre_669_; 
v_pre_669_ = lean_ctor_get(v_pre_668_, 0);
switch(lean_obj_tag(v_pre_669_))
{
case 0:
{
lean_object* v_str_670_; lean_object* v_str_671_; lean_object* v___x_672_; uint8_t v___x_673_; 
v_str_670_ = lean_ctor_get(v_x_667_, 1);
v_str_671_ = lean_ctor_get(v_pre_668_, 1);
v___x_672_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__0));
v___x_673_ = lean_string_dec_eq(v_str_671_, v___x_672_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; uint8_t v___x_675_; 
v___x_674_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__1));
v___x_675_ = lean_string_dec_eq(v_str_671_, v___x_674_);
if (v___x_675_ == 0)
{
return v___y_665_;
}
else
{
lean_object* v___x_676_; uint8_t v___x_677_; 
v___x_676_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__2));
v___x_677_ = lean_string_dec_eq(v_str_670_, v___x_676_);
if (v___x_677_ == 0)
{
return v___y_665_;
}
else
{
return v_suppressElabErrors_666_;
}
}
}
else
{
lean_object* v___x_678_; uint8_t v___x_679_; 
v___x_678_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__3));
v___x_679_ = lean_string_dec_eq(v_str_670_, v___x_678_);
if (v___x_679_ == 0)
{
return v___y_665_;
}
else
{
return v_suppressElabErrors_666_;
}
}
}
case 1:
{
lean_object* v_pre_680_; 
v_pre_680_ = lean_ctor_get(v_pre_669_, 0);
if (lean_obj_tag(v_pre_680_) == 0)
{
lean_object* v_str_681_; lean_object* v_str_682_; lean_object* v_str_683_; lean_object* v___x_684_; uint8_t v___x_685_; 
v_str_681_ = lean_ctor_get(v_x_667_, 1);
v_str_682_ = lean_ctor_get(v_pre_668_, 1);
v_str_683_ = lean_ctor_get(v_pre_669_, 1);
v___x_684_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__4));
v___x_685_ = lean_string_dec_eq(v_str_683_, v___x_684_);
if (v___x_685_ == 0)
{
return v___y_665_;
}
else
{
lean_object* v___x_686_; uint8_t v___x_687_; 
v___x_686_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__5));
v___x_687_ = lean_string_dec_eq(v_str_682_, v___x_686_);
if (v___x_687_ == 0)
{
return v___y_665_;
}
else
{
lean_object* v___x_688_; uint8_t v___x_689_; 
v___x_688_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__6));
v___x_689_ = lean_string_dec_eq(v_str_681_, v___x_688_);
if (v___x_689_ == 0)
{
return v___y_665_;
}
else
{
return v_suppressElabErrors_666_;
}
}
}
}
else
{
return v___y_665_;
}
}
default: 
{
return v___y_665_;
}
}
}
case 0:
{
lean_object* v_str_690_; lean_object* v___x_691_; uint8_t v___x_692_; 
v_str_690_ = lean_ctor_get(v_x_667_, 1);
v___x_691_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___closed__7));
v___x_692_ = lean_string_dec_eq(v_str_690_, v___x_691_);
if (v___x_692_ == 0)
{
return v___y_665_;
}
else
{
return v_suppressElabErrors_666_;
}
}
default: 
{
return v___y_665_;
}
}
}
else
{
return v___y_665_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___boxed(lean_object* v___y_693_, lean_object* v_suppressElabErrors_694_, lean_object* v_x_695_){
_start:
{
uint8_t v___y_9373__boxed_696_; uint8_t v_suppressElabErrors_boxed_697_; uint8_t v_res_698_; lean_object* v_r_699_; 
v___y_9373__boxed_696_ = lean_unbox(v___y_693_);
v_suppressElabErrors_boxed_697_ = lean_unbox(v_suppressElabErrors_694_);
v_res_698_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0(v___y_9373__boxed_696_, v_suppressElabErrors_boxed_697_, v_x_695_);
lean_dec(v_x_695_);
v_r_699_ = lean_box(v_res_698_);
return v_r_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg(lean_object* v_ref_701_, lean_object* v_msgData_702_, uint8_t v_severity_703_, uint8_t v_isSilent_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_){
_start:
{
lean_object* v___y_711_; uint8_t v___y_712_; lean_object* v___y_713_; uint8_t v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_717_; lean_object* v___y_718_; lean_object* v___y_719_; lean_object* v___y_747_; lean_object* v___y_748_; uint8_t v___y_749_; uint8_t v___y_750_; lean_object* v___y_751_; uint8_t v___y_752_; lean_object* v___y_753_; lean_object* v___y_754_; lean_object* v___y_772_; lean_object* v___y_773_; uint8_t v___y_774_; uint8_t v___y_775_; uint8_t v___y_776_; lean_object* v___y_777_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_783_; lean_object* v___y_784_; uint8_t v___y_785_; uint8_t v___y_786_; lean_object* v___y_787_; lean_object* v___y_788_; uint8_t v___y_789_; uint8_t v___x_794_; lean_object* v___y_796_; uint8_t v___y_797_; lean_object* v___y_798_; lean_object* v___y_799_; lean_object* v___y_800_; uint8_t v___y_801_; uint8_t v___y_802_; uint8_t v___y_804_; uint8_t v___x_819_; 
v___x_794_ = 2;
v___x_819_ = l_Lean_instBEqMessageSeverity_beq(v_severity_703_, v___x_794_);
if (v___x_819_ == 0)
{
v___y_804_ = v___x_819_;
goto v___jp_803_;
}
else
{
uint8_t v___x_820_; 
lean_inc_ref(v_msgData_702_);
v___x_820_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_702_);
v___y_804_ = v___x_820_;
goto v___jp_803_;
}
v___jp_710_:
{
lean_object* v___x_720_; lean_object* v_currNamespace_721_; lean_object* v_openDecls_722_; lean_object* v_env_723_; lean_object* v_nextMacroScope_724_; lean_object* v_ngen_725_; lean_object* v_auxDeclNGen_726_; lean_object* v_traceState_727_; lean_object* v_cache_728_; lean_object* v_messages_729_; lean_object* v_infoState_730_; lean_object* v_snapshotTasks_731_; lean_object* v___x_733_; uint8_t v_isShared_734_; uint8_t v_isSharedCheck_745_; 
v___x_720_ = lean_st_ref_take(v___y_719_);
v_currNamespace_721_ = lean_ctor_get(v___y_718_, 6);
v_openDecls_722_ = lean_ctor_get(v___y_718_, 7);
v_env_723_ = lean_ctor_get(v___x_720_, 0);
v_nextMacroScope_724_ = lean_ctor_get(v___x_720_, 1);
v_ngen_725_ = lean_ctor_get(v___x_720_, 2);
v_auxDeclNGen_726_ = lean_ctor_get(v___x_720_, 3);
v_traceState_727_ = lean_ctor_get(v___x_720_, 4);
v_cache_728_ = lean_ctor_get(v___x_720_, 5);
v_messages_729_ = lean_ctor_get(v___x_720_, 6);
v_infoState_730_ = lean_ctor_get(v___x_720_, 7);
v_snapshotTasks_731_ = lean_ctor_get(v___x_720_, 8);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_720_);
if (v_isSharedCheck_745_ == 0)
{
v___x_733_ = v___x_720_;
v_isShared_734_ = v_isSharedCheck_745_;
goto v_resetjp_732_;
}
else
{
lean_inc(v_snapshotTasks_731_);
lean_inc(v_infoState_730_);
lean_inc(v_messages_729_);
lean_inc(v_cache_728_);
lean_inc(v_traceState_727_);
lean_inc(v_auxDeclNGen_726_);
lean_inc(v_ngen_725_);
lean_inc(v_nextMacroScope_724_);
lean_inc(v_env_723_);
lean_dec(v___x_720_);
v___x_733_ = lean_box(0);
v_isShared_734_ = v_isSharedCheck_745_;
goto v_resetjp_732_;
}
v_resetjp_732_:
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_740_; 
lean_inc(v_openDecls_722_);
lean_inc(v_currNamespace_721_);
v___x_735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_735_, 0, v_currNamespace_721_);
lean_ctor_set(v___x_735_, 1, v_openDecls_722_);
v___x_736_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_735_);
lean_ctor_set(v___x_736_, 1, v___y_717_);
lean_inc_ref(v___y_713_);
lean_inc_ref(v___y_715_);
v___x_737_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_737_, 0, v___y_715_);
lean_ctor_set(v___x_737_, 1, v___y_711_);
lean_ctor_set(v___x_737_, 2, v___y_716_);
lean_ctor_set(v___x_737_, 3, v___y_713_);
lean_ctor_set(v___x_737_, 4, v___x_736_);
lean_ctor_set_uint8(v___x_737_, sizeof(void*)*5, v___y_712_);
lean_ctor_set_uint8(v___x_737_, sizeof(void*)*5 + 1, v___y_714_);
lean_ctor_set_uint8(v___x_737_, sizeof(void*)*5 + 2, v_isSilent_704_);
v___x_738_ = l_Lean_MessageLog_add(v___x_737_, v_messages_729_);
if (v_isShared_734_ == 0)
{
lean_ctor_set(v___x_733_, 6, v___x_738_);
v___x_740_ = v___x_733_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_env_723_);
lean_ctor_set(v_reuseFailAlloc_744_, 1, v_nextMacroScope_724_);
lean_ctor_set(v_reuseFailAlloc_744_, 2, v_ngen_725_);
lean_ctor_set(v_reuseFailAlloc_744_, 3, v_auxDeclNGen_726_);
lean_ctor_set(v_reuseFailAlloc_744_, 4, v_traceState_727_);
lean_ctor_set(v_reuseFailAlloc_744_, 5, v_cache_728_);
lean_ctor_set(v_reuseFailAlloc_744_, 6, v___x_738_);
lean_ctor_set(v_reuseFailAlloc_744_, 7, v_infoState_730_);
lean_ctor_set(v_reuseFailAlloc_744_, 8, v_snapshotTasks_731_);
v___x_740_ = v_reuseFailAlloc_744_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; 
v___x_741_ = lean_st_ref_set(v___y_719_, v___x_740_);
v___x_742_ = lean_box(0);
v___x_743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_743_, 0, v___x_742_);
return v___x_743_;
}
}
}
v___jp_746_:
{
lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v_a_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_770_; 
v___x_755_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_702_);
v___x_756_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(v___x_755_, v___y_705_, v___y_706_, v___y_707_, v___y_708_);
v_a_757_ = lean_ctor_get(v___x_756_, 0);
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_756_);
if (v_isSharedCheck_770_ == 0)
{
v___x_759_ = v___x_756_;
v_isShared_760_ = v_isSharedCheck_770_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_a_757_);
lean_dec(v___x_756_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_770_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
lean_inc_ref_n(v___y_748_, 2);
v___x_761_ = l_Lean_FileMap_toPosition(v___y_748_, v___y_751_);
lean_dec(v___y_751_);
v___x_762_ = l_Lean_FileMap_toPosition(v___y_748_, v___y_754_);
lean_dec(v___y_754_);
v___x_763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_763_, 0, v___x_762_);
v___x_764_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___closed__0));
if (v___y_749_ == 0)
{
lean_del_object(v___x_759_);
lean_dec_ref(v___y_747_);
v___y_711_ = v___x_761_;
v___y_712_ = v___y_750_;
v___y_713_ = v___x_764_;
v___y_714_ = v___y_752_;
v___y_715_ = v___y_753_;
v___y_716_ = v___x_763_;
v___y_717_ = v_a_757_;
v___y_718_ = v___y_707_;
v___y_719_ = v___y_708_;
goto v___jp_710_;
}
else
{
uint8_t v___x_765_; 
lean_inc(v_a_757_);
v___x_765_ = l_Lean_MessageData_hasTag(v___y_747_, v_a_757_);
if (v___x_765_ == 0)
{
lean_object* v___x_766_; lean_object* v___x_768_; 
lean_dec_ref_known(v___x_763_, 1);
lean_dec_ref(v___x_761_);
lean_dec(v_a_757_);
v___x_766_ = lean_box(0);
if (v_isShared_760_ == 0)
{
lean_ctor_set(v___x_759_, 0, v___x_766_);
v___x_768_ = v___x_759_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v___x_766_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
else
{
lean_del_object(v___x_759_);
v___y_711_ = v___x_761_;
v___y_712_ = v___y_750_;
v___y_713_ = v___x_764_;
v___y_714_ = v___y_752_;
v___y_715_ = v___y_753_;
v___y_716_ = v___x_763_;
v___y_717_ = v_a_757_;
v___y_718_ = v___y_707_;
v___y_719_ = v___y_708_;
goto v___jp_710_;
}
}
}
}
v___jp_771_:
{
lean_object* v___x_780_; 
v___x_780_ = l_Lean_Syntax_getTailPos_x3f(v___y_777_, v___y_775_);
lean_dec(v___y_777_);
if (lean_obj_tag(v___x_780_) == 0)
{
lean_inc(v___y_779_);
v___y_747_ = v___y_772_;
v___y_748_ = v___y_773_;
v___y_749_ = v___y_774_;
v___y_750_ = v___y_775_;
v___y_751_ = v___y_779_;
v___y_752_ = v___y_776_;
v___y_753_ = v___y_778_;
v___y_754_ = v___y_779_;
goto v___jp_746_;
}
else
{
lean_object* v_val_781_; 
v_val_781_ = lean_ctor_get(v___x_780_, 0);
lean_inc(v_val_781_);
lean_dec_ref_known(v___x_780_, 1);
v___y_747_ = v___y_772_;
v___y_748_ = v___y_773_;
v___y_749_ = v___y_774_;
v___y_750_ = v___y_775_;
v___y_751_ = v___y_779_;
v___y_752_ = v___y_776_;
v___y_753_ = v___y_778_;
v___y_754_ = v_val_781_;
goto v___jp_746_;
}
}
v___jp_782_:
{
lean_object* v_ref_790_; lean_object* v___x_791_; 
v_ref_790_ = l_Lean_replaceRef(v_ref_701_, v___y_787_);
v___x_791_ = l_Lean_Syntax_getPos_x3f(v_ref_790_, v___y_786_);
if (lean_obj_tag(v___x_791_) == 0)
{
lean_object* v___x_792_; 
v___x_792_ = lean_unsigned_to_nat(0u);
v___y_772_ = v___y_783_;
v___y_773_ = v___y_784_;
v___y_774_ = v___y_785_;
v___y_775_ = v___y_786_;
v___y_776_ = v___y_789_;
v___y_777_ = v_ref_790_;
v___y_778_ = v___y_788_;
v___y_779_ = v___x_792_;
goto v___jp_771_;
}
else
{
lean_object* v_val_793_; 
v_val_793_ = lean_ctor_get(v___x_791_, 0);
lean_inc(v_val_793_);
lean_dec_ref_known(v___x_791_, 1);
v___y_772_ = v___y_783_;
v___y_773_ = v___y_784_;
v___y_774_ = v___y_785_;
v___y_775_ = v___y_786_;
v___y_776_ = v___y_789_;
v___y_777_ = v_ref_790_;
v___y_778_ = v___y_788_;
v___y_779_ = v_val_793_;
goto v___jp_771_;
}
}
v___jp_795_:
{
if (v___y_802_ == 0)
{
v___y_783_ = v___y_798_;
v___y_784_ = v___y_796_;
v___y_785_ = v___y_797_;
v___y_786_ = v___y_801_;
v___y_787_ = v___y_799_;
v___y_788_ = v___y_800_;
v___y_789_ = v_severity_703_;
goto v___jp_782_;
}
else
{
v___y_783_ = v___y_798_;
v___y_784_ = v___y_796_;
v___y_785_ = v___y_797_;
v___y_786_ = v___y_801_;
v___y_787_ = v___y_799_;
v___y_788_ = v___y_800_;
v___y_789_ = v___x_794_;
goto v___jp_782_;
}
}
v___jp_803_:
{
if (v___y_804_ == 0)
{
lean_object* v_fileName_805_; lean_object* v_fileMap_806_; lean_object* v_options_807_; lean_object* v_ref_808_; uint8_t v_suppressElabErrors_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___f_812_; uint8_t v___x_813_; uint8_t v___x_814_; 
v_fileName_805_ = lean_ctor_get(v___y_707_, 0);
v_fileMap_806_ = lean_ctor_get(v___y_707_, 1);
v_options_807_ = lean_ctor_get(v___y_707_, 2);
v_ref_808_ = lean_ctor_get(v___y_707_, 5);
v_suppressElabErrors_809_ = lean_ctor_get_uint8(v___y_707_, sizeof(void*)*14 + 1);
v___x_810_ = lean_box(v___y_804_);
v___x_811_ = lean_box(v_suppressElabErrors_809_);
v___f_812_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_812_, 0, v___x_810_);
lean_closure_set(v___f_812_, 1, v___x_811_);
v___x_813_ = 1;
v___x_814_ = l_Lean_instBEqMessageSeverity_beq(v_severity_703_, v___x_813_);
if (v___x_814_ == 0)
{
v___y_796_ = v_fileMap_806_;
v___y_797_ = v_suppressElabErrors_809_;
v___y_798_ = v___f_812_;
v___y_799_ = v_ref_808_;
v___y_800_ = v_fileName_805_;
v___y_801_ = v___y_804_;
v___y_802_ = v___x_814_;
goto v___jp_795_;
}
else
{
lean_object* v___x_815_; uint8_t v___x_816_; 
v___x_815_ = l_Lean_warningAsError;
v___x_816_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(v_options_807_, v___x_815_);
v___y_796_ = v_fileMap_806_;
v___y_797_ = v_suppressElabErrors_809_;
v___y_798_ = v___f_812_;
v___y_799_ = v_ref_808_;
v___y_800_ = v_fileName_805_;
v___y_801_ = v___y_804_;
v___y_802_ = v___x_816_;
goto v___jp_795_;
}
}
else
{
lean_object* v___x_817_; lean_object* v___x_818_; 
lean_dec_ref(v_msgData_702_);
v___x_817_ = lean_box(0);
v___x_818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
return v___x_818_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___boxed(lean_object* v_ref_821_, lean_object* v_msgData_822_, lean_object* v_severity_823_, lean_object* v_isSilent_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_){
_start:
{
uint8_t v_severity_boxed_830_; uint8_t v_isSilent_boxed_831_; lean_object* v_res_832_; 
v_severity_boxed_830_ = lean_unbox(v_severity_823_);
v_isSilent_boxed_831_ = lean_unbox(v_isSilent_824_);
v_res_832_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg(v_ref_821_, v_msgData_822_, v_severity_boxed_830_, v_isSilent_boxed_831_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
lean_dec(v___y_828_);
lean_dec_ref(v___y_827_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
lean_dec(v_ref_821_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2(lean_object* v_ref_833_, lean_object* v_msgData_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_){
_start:
{
uint8_t v___x_842_; uint8_t v___x_843_; lean_object* v___x_844_; 
v___x_842_ = 1;
v___x_843_ = 0;
v___x_844_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg(v_ref_833_, v_msgData_834_, v___x_842_, v___x_843_, v___y_837_, v___y_838_, v___y_839_, v___y_840_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2___boxed(lean_object* v_ref_845_, lean_object* v_msgData_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2(v_ref_845_, v_msgData_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v_ref_845_);
return v_res_854_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_855_; 
v___x_855_ = l_instMonadEIO(lean_box(0));
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2(lean_object* v_msg_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_){
_start:
{
lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v_toApplicative_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_963_; 
v___x_870_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0, &lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0);
v___x_871_ = l_StateRefT_x27_instMonad___redArg(v___x_870_);
v_toApplicative_872_ = lean_ctor_get(v___x_871_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_871_);
if (v_isSharedCheck_963_ == 0)
{
lean_object* v_unused_964_; 
v_unused_964_ = lean_ctor_get(v___x_871_, 1);
lean_dec(v_unused_964_);
v___x_874_ = v___x_871_;
v_isShared_875_ = v_isSharedCheck_963_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_toApplicative_872_);
lean_dec(v___x_871_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_963_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v_toFunctor_876_; lean_object* v_toSeq_877_; lean_object* v_toSeqLeft_878_; lean_object* v_toSeqRight_879_; lean_object* v___x_881_; uint8_t v_isShared_882_; uint8_t v_isSharedCheck_961_; 
v_toFunctor_876_ = lean_ctor_get(v_toApplicative_872_, 0);
v_toSeq_877_ = lean_ctor_get(v_toApplicative_872_, 2);
v_toSeqLeft_878_ = lean_ctor_get(v_toApplicative_872_, 3);
v_toSeqRight_879_ = lean_ctor_get(v_toApplicative_872_, 4);
v_isSharedCheck_961_ = !lean_is_exclusive(v_toApplicative_872_);
if (v_isSharedCheck_961_ == 0)
{
lean_object* v_unused_962_; 
v_unused_962_ = lean_ctor_get(v_toApplicative_872_, 1);
lean_dec(v_unused_962_);
v___x_881_ = v_toApplicative_872_;
v_isShared_882_ = v_isSharedCheck_961_;
goto v_resetjp_880_;
}
else
{
lean_inc(v_toSeqRight_879_);
lean_inc(v_toSeqLeft_878_);
lean_inc(v_toSeq_877_);
lean_inc(v_toFunctor_876_);
lean_dec(v_toApplicative_872_);
v___x_881_ = lean_box(0);
v_isShared_882_ = v_isSharedCheck_961_;
goto v_resetjp_880_;
}
v_resetjp_880_:
{
lean_object* v___f_883_; lean_object* v___f_884_; lean_object* v___f_885_; lean_object* v___f_886_; lean_object* v___x_887_; lean_object* v___f_888_; lean_object* v___f_889_; lean_object* v___f_890_; lean_object* v___x_892_; 
v___f_883_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1));
v___f_884_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_876_);
v___f_885_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_885_, 0, v_toFunctor_876_);
v___f_886_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_886_, 0, v_toFunctor_876_);
v___x_887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_887_, 0, v___f_885_);
lean_ctor_set(v___x_887_, 1, v___f_886_);
v___f_888_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_888_, 0, v_toSeqRight_879_);
v___f_889_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_889_, 0, v_toSeqLeft_878_);
v___f_890_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_890_, 0, v_toSeq_877_);
if (v_isShared_882_ == 0)
{
lean_ctor_set(v___x_881_, 4, v___f_888_);
lean_ctor_set(v___x_881_, 3, v___f_889_);
lean_ctor_set(v___x_881_, 2, v___f_890_);
lean_ctor_set(v___x_881_, 1, v___f_883_);
lean_ctor_set(v___x_881_, 0, v___x_887_);
v___x_892_ = v___x_881_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v___x_887_);
lean_ctor_set(v_reuseFailAlloc_960_, 1, v___f_883_);
lean_ctor_set(v_reuseFailAlloc_960_, 2, v___f_890_);
lean_ctor_set(v_reuseFailAlloc_960_, 3, v___f_889_);
lean_ctor_set(v_reuseFailAlloc_960_, 4, v___f_888_);
v___x_892_ = v_reuseFailAlloc_960_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
lean_object* v___x_894_; 
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 1, v___f_884_);
lean_ctor_set(v___x_874_, 0, v___x_892_);
v___x_894_ = v___x_874_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v___x_892_);
lean_ctor_set(v_reuseFailAlloc_959_, 1, v___f_884_);
v___x_894_ = v_reuseFailAlloc_959_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
lean_object* v___x_895_; lean_object* v_toApplicative_896_; lean_object* v___x_898_; uint8_t v_isShared_899_; uint8_t v_isSharedCheck_957_; 
v___x_895_ = l_StateRefT_x27_instMonad___redArg(v___x_894_);
v_toApplicative_896_ = lean_ctor_get(v___x_895_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v___x_895_);
if (v_isSharedCheck_957_ == 0)
{
lean_object* v_unused_958_; 
v_unused_958_ = lean_ctor_get(v___x_895_, 1);
lean_dec(v_unused_958_);
v___x_898_ = v___x_895_;
v_isShared_899_ = v_isSharedCheck_957_;
goto v_resetjp_897_;
}
else
{
lean_inc(v_toApplicative_896_);
lean_dec(v___x_895_);
v___x_898_ = lean_box(0);
v_isShared_899_ = v_isSharedCheck_957_;
goto v_resetjp_897_;
}
v_resetjp_897_:
{
lean_object* v_toFunctor_900_; lean_object* v_toSeq_901_; lean_object* v_toSeqLeft_902_; lean_object* v_toSeqRight_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_955_; 
v_toFunctor_900_ = lean_ctor_get(v_toApplicative_896_, 0);
v_toSeq_901_ = lean_ctor_get(v_toApplicative_896_, 2);
v_toSeqLeft_902_ = lean_ctor_get(v_toApplicative_896_, 3);
v_toSeqRight_903_ = lean_ctor_get(v_toApplicative_896_, 4);
v_isSharedCheck_955_ = !lean_is_exclusive(v_toApplicative_896_);
if (v_isSharedCheck_955_ == 0)
{
lean_object* v_unused_956_; 
v_unused_956_ = lean_ctor_get(v_toApplicative_896_, 1);
lean_dec(v_unused_956_);
v___x_905_ = v_toApplicative_896_;
v_isShared_906_ = v_isSharedCheck_955_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_toSeqRight_903_);
lean_inc(v_toSeqLeft_902_);
lean_inc(v_toSeq_901_);
lean_inc(v_toFunctor_900_);
lean_dec(v_toApplicative_896_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_955_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___f_907_; lean_object* v___f_908_; lean_object* v___f_909_; lean_object* v___f_910_; lean_object* v___x_911_; lean_object* v___f_912_; lean_object* v___f_913_; lean_object* v___f_914_; lean_object* v___x_916_; 
v___f_907_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3));
v___f_908_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_900_);
v___f_909_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_909_, 0, v_toFunctor_900_);
v___f_910_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_910_, 0, v_toFunctor_900_);
v___x_911_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_911_, 0, v___f_909_);
lean_ctor_set(v___x_911_, 1, v___f_910_);
v___f_912_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_912_, 0, v_toSeqRight_903_);
v___f_913_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_913_, 0, v_toSeqLeft_902_);
v___f_914_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_914_, 0, v_toSeq_901_);
if (v_isShared_906_ == 0)
{
lean_ctor_set(v___x_905_, 4, v___f_912_);
lean_ctor_set(v___x_905_, 3, v___f_913_);
lean_ctor_set(v___x_905_, 2, v___f_914_);
lean_ctor_set(v___x_905_, 1, v___f_907_);
lean_ctor_set(v___x_905_, 0, v___x_911_);
v___x_916_ = v___x_905_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v___x_911_);
lean_ctor_set(v_reuseFailAlloc_954_, 1, v___f_907_);
lean_ctor_set(v_reuseFailAlloc_954_, 2, v___f_914_);
lean_ctor_set(v_reuseFailAlloc_954_, 3, v___f_913_);
lean_ctor_set(v_reuseFailAlloc_954_, 4, v___f_912_);
v___x_916_ = v_reuseFailAlloc_954_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
lean_object* v___x_918_; 
if (v_isShared_899_ == 0)
{
lean_ctor_set(v___x_898_, 1, v___f_908_);
lean_ctor_set(v___x_898_, 0, v___x_916_);
v___x_918_ = v___x_898_;
goto v_reusejp_917_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v___x_916_);
lean_ctor_set(v_reuseFailAlloc_953_, 1, v___f_908_);
v___x_918_ = v_reuseFailAlloc_953_;
goto v_reusejp_917_;
}
v_reusejp_917_:
{
lean_object* v___x_919_; lean_object* v_toApplicative_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_951_; 
v___x_919_ = l_StateRefT_x27_instMonad___redArg(v___x_918_);
v_toApplicative_920_ = lean_ctor_get(v___x_919_, 0);
v_isSharedCheck_951_ = !lean_is_exclusive(v___x_919_);
if (v_isSharedCheck_951_ == 0)
{
lean_object* v_unused_952_; 
v_unused_952_ = lean_ctor_get(v___x_919_, 1);
lean_dec(v_unused_952_);
v___x_922_ = v___x_919_;
v_isShared_923_ = v_isSharedCheck_951_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_toApplicative_920_);
lean_dec(v___x_919_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_951_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v_toFunctor_924_; lean_object* v_toSeq_925_; lean_object* v_toSeqLeft_926_; lean_object* v_toSeqRight_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_949_; 
v_toFunctor_924_ = lean_ctor_get(v_toApplicative_920_, 0);
v_toSeq_925_ = lean_ctor_get(v_toApplicative_920_, 2);
v_toSeqLeft_926_ = lean_ctor_get(v_toApplicative_920_, 3);
v_toSeqRight_927_ = lean_ctor_get(v_toApplicative_920_, 4);
v_isSharedCheck_949_ = !lean_is_exclusive(v_toApplicative_920_);
if (v_isSharedCheck_949_ == 0)
{
lean_object* v_unused_950_; 
v_unused_950_ = lean_ctor_get(v_toApplicative_920_, 1);
lean_dec(v_unused_950_);
v___x_929_ = v_toApplicative_920_;
v_isShared_930_ = v_isSharedCheck_949_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_toSeqRight_927_);
lean_inc(v_toSeqLeft_926_);
lean_inc(v_toSeq_925_);
lean_inc(v_toFunctor_924_);
lean_dec(v_toApplicative_920_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_949_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___f_931_; lean_object* v___f_932_; lean_object* v___f_933_; lean_object* v___f_934_; lean_object* v___x_935_; lean_object* v___f_936_; lean_object* v___f_937_; lean_object* v___f_938_; lean_object* v___x_940_; 
v___f_931_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__5));
v___f_932_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__6));
lean_inc_ref(v_toFunctor_924_);
v___f_933_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_933_, 0, v_toFunctor_924_);
v___f_934_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_934_, 0, v_toFunctor_924_);
v___x_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_935_, 0, v___f_933_);
lean_ctor_set(v___x_935_, 1, v___f_934_);
v___f_936_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_936_, 0, v_toSeqRight_927_);
v___f_937_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_937_, 0, v_toSeqLeft_926_);
v___f_938_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_938_, 0, v_toSeq_925_);
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 4, v___f_936_);
lean_ctor_set(v___x_929_, 3, v___f_937_);
lean_ctor_set(v___x_929_, 2, v___f_938_);
lean_ctor_set(v___x_929_, 1, v___f_931_);
lean_ctor_set(v___x_929_, 0, v___x_935_);
v___x_940_ = v___x_929_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v___x_935_);
lean_ctor_set(v_reuseFailAlloc_948_, 1, v___f_931_);
lean_ctor_set(v_reuseFailAlloc_948_, 2, v___f_938_);
lean_ctor_set(v_reuseFailAlloc_948_, 3, v___f_937_);
lean_ctor_set(v_reuseFailAlloc_948_, 4, v___f_936_);
v___x_940_ = v_reuseFailAlloc_948_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
lean_object* v___x_942_; 
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 1, v___f_932_);
lean_ctor_set(v___x_922_, 0, v___x_940_);
v___x_942_ = v___x_922_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v___x_940_);
lean_ctor_set(v_reuseFailAlloc_947_, 1, v___f_932_);
v___x_942_ = v_reuseFailAlloc_947_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_5492__overap_945_; lean_object* v___x_946_; 
v___x_943_ = lean_box(0);
v___x_944_ = l_instInhabitedOfMonad___redArg(v___x_942_, v___x_943_);
v___x_5492__overap_945_ = lean_panic_fn_borrowed(v___x_944_, v_msg_862_);
lean_dec(v___x_944_);
lean_inc(v___y_868_);
lean_inc_ref(v___y_867_);
lean_inc(v___y_866_);
lean_inc_ref(v___y_865_);
lean_inc(v___y_864_);
lean_inc_ref(v___y_863_);
v___x_946_ = lean_apply_7(v___x_5492__overap_945_, v___y_863_, v___y_864_, v___y_865_, v___y_866_, v___y_867_, v___y_868_, lean_box(0));
return v___x_946_;
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___boxed(lean_object* v_msg_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2(v_msg_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
return v_res_973_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0(void){
_start:
{
lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_974_ = lean_box(1);
v___x_975_ = l_Lean_MessageData_ofFormat(v___x_974_);
return v___x_975_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3(void){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__2));
v___x_980_ = l_Lean_MessageData_ofFormat(v___x_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6(lean_object* v_x_981_, lean_object* v_x_982_){
_start:
{
if (lean_obj_tag(v_x_982_) == 0)
{
return v_x_981_;
}
else
{
lean_object* v_head_983_; lean_object* v_tail_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_1006_; 
v_head_983_ = lean_ctor_get(v_x_982_, 0);
v_tail_984_ = lean_ctor_get(v_x_982_, 1);
v_isSharedCheck_1006_ = !lean_is_exclusive(v_x_982_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_986_ = v_x_982_;
v_isShared_987_ = v_isSharedCheck_1006_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_tail_984_);
lean_inc(v_head_983_);
lean_dec(v_x_982_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_1006_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v_before_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1004_; 
v_before_988_ = lean_ctor_get(v_head_983_, 0);
v_isSharedCheck_1004_ = !lean_is_exclusive(v_head_983_);
if (v_isSharedCheck_1004_ == 0)
{
lean_object* v_unused_1005_; 
v_unused_1005_ = lean_ctor_get(v_head_983_, 1);
lean_dec(v_unused_1005_);
v___x_990_ = v_head_983_;
v_isShared_991_ = v_isSharedCheck_1004_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_before_988_);
lean_dec(v_head_983_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1004_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___x_992_; lean_object* v___x_994_; 
v___x_992_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0);
if (v_isShared_991_ == 0)
{
lean_ctor_set_tag(v___x_990_, 7);
lean_ctor_set(v___x_990_, 1, v___x_992_);
lean_ctor_set(v___x_990_, 0, v_x_981_);
v___x_994_ = v___x_990_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_x_981_);
lean_ctor_set(v_reuseFailAlloc_1003_, 1, v___x_992_);
v___x_994_ = v_reuseFailAlloc_1003_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
lean_object* v___x_995_; lean_object* v___x_997_; 
v___x_995_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__3);
if (v_isShared_987_ == 0)
{
lean_ctor_set_tag(v___x_986_, 7);
lean_ctor_set(v___x_986_, 1, v___x_995_);
lean_ctor_set(v___x_986_, 0, v___x_994_);
v___x_997_ = v___x_986_;
goto v_reusejp_996_;
}
else
{
lean_object* v_reuseFailAlloc_1002_; 
v_reuseFailAlloc_1002_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1002_, 0, v___x_994_);
lean_ctor_set(v_reuseFailAlloc_1002_, 1, v___x_995_);
v___x_997_ = v_reuseFailAlloc_1002_;
goto v_reusejp_996_;
}
v_reusejp_996_:
{
lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
v___x_998_ = l_Lean_MessageData_ofSyntax(v_before_988_);
v___x_999_ = l_Lean_indentD(v___x_998_);
v___x_1000_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1000_, 0, v___x_997_);
lean_ctor_set(v___x_1000_, 1, v___x_999_);
v_x_981_ = v___x_1000_;
v_x_982_ = v_tail_984_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__1));
v___x_1011_ = l_Lean_MessageData_ofFormat(v___x_1010_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_1012_, lean_object* v_macroStack_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_options_1016_; lean_object* v___x_1017_; uint8_t v___x_1018_; 
v_options_1016_ = lean_ctor_get(v___y_1014_, 2);
v___x_1017_ = l_Lean_Elab_pp_macroStack;
v___x_1018_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(v_options_1016_, v___x_1017_);
if (v___x_1018_ == 0)
{
lean_object* v___x_1019_; 
lean_dec(v_macroStack_1013_);
v___x_1019_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1019_, 0, v_msgData_1012_);
return v___x_1019_;
}
else
{
if (lean_obj_tag(v_macroStack_1013_) == 0)
{
lean_object* v___x_1020_; 
v___x_1020_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1020_, 0, v_msgData_1012_);
return v___x_1020_;
}
else
{
lean_object* v_head_1021_; lean_object* v_after_1022_; lean_object* v___x_1024_; uint8_t v_isShared_1025_; uint8_t v_isSharedCheck_1037_; 
v_head_1021_ = lean_ctor_get(v_macroStack_1013_, 0);
lean_inc(v_head_1021_);
v_after_1022_ = lean_ctor_get(v_head_1021_, 1);
v_isSharedCheck_1037_ = !lean_is_exclusive(v_head_1021_);
if (v_isSharedCheck_1037_ == 0)
{
lean_object* v_unused_1038_; 
v_unused_1038_ = lean_ctor_get(v_head_1021_, 0);
lean_dec(v_unused_1038_);
v___x_1024_ = v_head_1021_;
v_isShared_1025_ = v_isSharedCheck_1037_;
goto v_resetjp_1023_;
}
else
{
lean_inc(v_after_1022_);
lean_dec(v_head_1021_);
v___x_1024_ = lean_box(0);
v_isShared_1025_ = v_isSharedCheck_1037_;
goto v_resetjp_1023_;
}
v_resetjp_1023_:
{
lean_object* v___x_1026_; lean_object* v___x_1028_; 
v___x_1026_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6___closed__0);
if (v_isShared_1025_ == 0)
{
lean_ctor_set_tag(v___x_1024_, 7);
lean_ctor_set(v___x_1024_, 1, v___x_1026_);
lean_ctor_set(v___x_1024_, 0, v_msgData_1012_);
v___x_1028_ = v___x_1024_;
goto v_reusejp_1027_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_msgData_1012_);
lean_ctor_set(v_reuseFailAlloc_1036_, 1, v___x_1026_);
v___x_1028_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1027_;
}
v_reusejp_1027_:
{
lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v_msgData_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1029_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___closed__2);
v___x_1030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1028_);
lean_ctor_set(v___x_1030_, 1, v___x_1029_);
v___x_1031_ = l_Lean_MessageData_ofSyntax(v_after_1022_);
v___x_1032_ = l_Lean_indentD(v___x_1031_);
v_msgData_1033_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1033_, 0, v___x_1030_);
lean_ctor_set(v_msgData_1033_, 1, v___x_1032_);
v___x_1034_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3_spec__6(v_msgData_1033_, v_macroStack_1013_);
v___x_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
return v___x_1035_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_1039_, lean_object* v_macroStack_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg(v_msgData_1039_, v_macroStack_1040_, v___y_1041_);
lean_dec_ref(v___y_1041_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(lean_object* v_msg_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v_ref_1052_; lean_object* v___x_1053_; lean_object* v_a_1054_; lean_object* v_macroStack_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v_a_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1066_; 
v_ref_1052_ = lean_ctor_get(v___y_1049_, 5);
v___x_1053_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(v_msg_1044_, v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_);
v_a_1054_ = lean_ctor_get(v___x_1053_, 0);
lean_inc(v_a_1054_);
lean_dec_ref(v___x_1053_);
v_macroStack_1055_ = lean_ctor_get(v___y_1045_, 1);
v___x_1056_ = l_Lean_Elab_getBetterRef(v_ref_1052_, v_macroStack_1055_);
lean_inc(v_macroStack_1055_);
v___x_1057_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg(v_a_1054_, v_macroStack_1055_, v___y_1049_);
v_a_1058_ = lean_ctor_get(v___x_1057_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_1057_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1060_ = v___x_1057_;
v_isShared_1061_ = v_isSharedCheck_1066_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_a_1058_);
lean_dec(v___x_1057_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1066_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v___x_1062_; lean_object* v___x_1064_; 
v___x_1062_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1056_);
lean_ctor_set(v___x_1062_, 1, v_a_1058_);
if (v_isShared_1061_ == 0)
{
lean_ctor_set_tag(v___x_1060_, 1);
lean_ctor_set(v___x_1060_, 0, v___x_1062_);
v___x_1064_ = v___x_1060_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v___x_1062_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg___boxed(lean_object* v_msg_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_){
_start:
{
lean_object* v_res_1075_; 
v_res_1075_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(v_msg_1067_, v___y_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_, v___y_1073_);
lean_dec(v___y_1073_);
lean_dec_ref(v___y_1072_);
lean_dec(v___y_1071_);
lean_dec_ref(v___y_1070_);
lean_dec(v___y_1069_);
lean_dec_ref(v___y_1068_);
return v_res_1075_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1077_; lean_object* v___x_1078_; 
v___x_1077_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__0));
v___x_1078_ = l_Lean_stringToMessageData(v___x_1077_);
return v___x_1078_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1080_; lean_object* v___x_1081_; 
v___x_1080_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__2));
v___x_1081_ = l_Lean_stringToMessageData(v___x_1080_);
return v___x_1081_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6(void){
_start:
{
lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
v___x_1084_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6));
v___x_1085_ = lean_unsigned_to_nat(11u);
v___x_1086_ = lean_unsigned_to_nat(115u);
v___x_1087_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__5));
v___x_1088_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__4));
v___x_1089_ = l_mkPanicMessageWithDecl(v___x_1088_, v___x_1087_, v___x_1086_, v___x_1085_, v___x_1084_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1(lean_object* v_constName_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
lean_object* v___x_1106_; lean_object* v_env_1107_; uint8_t v___x_1108_; lean_object* v___x_1109_; 
v___x_1106_ = lean_st_ref_get(v___y_1096_);
v_env_1107_ = lean_ctor_get(v___x_1106_, 0);
lean_inc_ref(v_env_1107_);
lean_dec(v___x_1106_);
v___x_1108_ = 0;
lean_inc(v_constName_1090_);
v___x_1109_ = l_Lean_Environment_findAsync_x3f(v_env_1107_, v_constName_1090_, v___x_1108_);
if (lean_obj_tag(v___x_1109_) == 1)
{
lean_object* v_val_1110_; uint8_t v_kind_1111_; 
v_val_1110_ = lean_ctor_get(v___x_1109_, 0);
lean_inc(v_val_1110_);
lean_dec_ref_known(v___x_1109_, 1);
v_kind_1111_ = lean_ctor_get_uint8(v_val_1110_, sizeof(void*)*3);
if (v_kind_1111_ == 0)
{
lean_object* v___x_1112_; 
v___x_1112_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_1110_);
if (lean_obj_tag(v___x_1112_) == 1)
{
lean_object* v_val_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1120_; 
lean_dec(v_constName_1090_);
v_val_1113_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1115_ = v___x_1112_;
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_val_1113_);
lean_dec(v___x_1112_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1118_; 
if (v_isShared_1116_ == 0)
{
lean_ctor_set_tag(v___x_1115_, 0);
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_val_1113_);
v___x_1118_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
return v___x_1118_;
}
}
}
else
{
lean_object* v___x_1121_; lean_object* v___x_1122_; 
lean_dec_ref(v___x_1112_);
v___x_1121_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6);
v___x_1122_ = lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2(v___x_1121_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
if (lean_obj_tag(v___x_1122_) == 0)
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1131_; 
v_a_1123_ = lean_ctor_get(v___x_1122_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1122_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1125_ = v___x_1122_;
v_isShared_1126_ = v_isSharedCheck_1131_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1122_);
v___x_1125_ = lean_box(0);
v_isShared_1126_ = v_isSharedCheck_1131_;
goto v_resetjp_1124_;
}
v_resetjp_1124_:
{
if (lean_obj_tag(v_a_1123_) == 0)
{
lean_del_object(v___x_1125_);
goto v___jp_1098_;
}
else
{
lean_object* v_val_1127_; lean_object* v___x_1129_; 
lean_dec(v_constName_1090_);
v_val_1127_ = lean_ctor_get(v_a_1123_, 0);
lean_inc(v_val_1127_);
lean_dec_ref_known(v_a_1123_, 1);
if (v_isShared_1126_ == 0)
{
lean_ctor_set(v___x_1125_, 0, v_val_1127_);
v___x_1129_ = v___x_1125_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_val_1127_);
v___x_1129_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
return v___x_1129_;
}
}
}
}
else
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1139_; 
lean_dec(v_constName_1090_);
v_a_1132_ = lean_ctor_get(v___x_1122_, 0);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_1122_);
if (v_isSharedCheck_1139_ == 0)
{
v___x_1134_ = v___x_1122_;
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_1122_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1137_; 
if (v_isShared_1135_ == 0)
{
v___x_1137_ = v___x_1134_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_a_1132_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
}
}
}
else
{
lean_dec(v_val_1110_);
goto v___jp_1098_;
}
}
else
{
lean_dec(v___x_1109_);
goto v___jp_1098_;
}
v___jp_1098_:
{
lean_object* v___x_1099_; uint8_t v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1099_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1);
v___x_1100_ = 0;
v___x_1101_ = l_Lean_MessageData_ofConstName(v_constName_1090_, v___x_1100_);
v___x_1102_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1099_);
lean_ctor_set(v___x_1102_, 1, v___x_1101_);
v___x_1103_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3);
v___x_1104_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1104_, 0, v___x_1102_);
lean_ctor_set(v___x_1104_, 1, v___x_1103_);
v___x_1105_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(v___x_1104_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
return v___x_1105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___boxed(lean_object* v_constName_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_){
_start:
{
lean_object* v_res_1148_; 
v_res_1148_ = lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1(v_constName_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_);
lean_dec(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec(v___y_1144_);
lean_dec_ref(v___y_1143_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
return v_res_1148_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = ((lean_object*)(lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__0));
v___x_1151_ = l_Lean_stringToMessageData(v___x_1150_);
return v___x_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0(lean_object* v___x_1152_, lean_object* v___x_1153_, lean_object* v_tk_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_){
_start:
{
lean_object* v___x_1162_; 
lean_inc(v___x_1152_);
v___x_1162_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v___x_1152_, v___x_1153_, v___y_1159_, v___y_1160_);
if (lean_obj_tag(v___x_1162_) == 0)
{
lean_object* v_a_1163_; lean_object* v___x_1164_; lean_object* v_env_1165_; uint8_t v___x_1166_; 
v_a_1163_ = lean_ctor_get(v___x_1162_, 0);
lean_inc(v_a_1163_);
lean_dec_ref_known(v___x_1162_, 1);
v___x_1164_ = lean_st_ref_get(v___y_1160_);
v_env_1165_ = lean_ctor_get(v___x_1164_, 0);
lean_inc_ref(v_env_1165_);
lean_dec(v___x_1164_);
v___x_1166_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_1165_, v_a_1163_);
if (v___x_1166_ == 0)
{
lean_object* v_fileName_1167_; lean_object* v_fileMap_1168_; lean_object* v_options_1169_; lean_object* v_currRecDepth_1170_; lean_object* v_maxRecDepth_1171_; lean_object* v_ref_1172_; lean_object* v_currNamespace_1173_; lean_object* v_openDecls_1174_; lean_object* v_initHeartbeats_1175_; lean_object* v_maxHeartbeats_1176_; lean_object* v_quotContext_1177_; lean_object* v_currMacroScope_1178_; uint8_t v_diag_1179_; lean_object* v_cancelTk_x3f_1180_; uint8_t v_suppressElabErrors_1181_; lean_object* v_inheritedTraceOptions_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1203_; 
v_fileName_1167_ = lean_ctor_get(v___y_1159_, 0);
v_fileMap_1168_ = lean_ctor_get(v___y_1159_, 1);
v_options_1169_ = lean_ctor_get(v___y_1159_, 2);
v_currRecDepth_1170_ = lean_ctor_get(v___y_1159_, 3);
v_maxRecDepth_1171_ = lean_ctor_get(v___y_1159_, 4);
v_ref_1172_ = lean_ctor_get(v___y_1159_, 5);
v_currNamespace_1173_ = lean_ctor_get(v___y_1159_, 6);
v_openDecls_1174_ = lean_ctor_get(v___y_1159_, 7);
v_initHeartbeats_1175_ = lean_ctor_get(v___y_1159_, 8);
v_maxHeartbeats_1176_ = lean_ctor_get(v___y_1159_, 9);
v_quotContext_1177_ = lean_ctor_get(v___y_1159_, 10);
v_currMacroScope_1178_ = lean_ctor_get(v___y_1159_, 11);
v_diag_1179_ = lean_ctor_get_uint8(v___y_1159_, sizeof(void*)*14);
v_cancelTk_x3f_1180_ = lean_ctor_get(v___y_1159_, 12);
v_suppressElabErrors_1181_ = lean_ctor_get_uint8(v___y_1159_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1182_ = lean_ctor_get(v___y_1159_, 13);
v_isSharedCheck_1203_ = !lean_is_exclusive(v___y_1159_);
if (v_isSharedCheck_1203_ == 0)
{
v___x_1184_ = v___y_1159_;
v_isShared_1185_ = v_isSharedCheck_1203_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_inheritedTraceOptions_1182_);
lean_inc(v_cancelTk_x3f_1180_);
lean_inc(v_currMacroScope_1178_);
lean_inc(v_quotContext_1177_);
lean_inc(v_maxHeartbeats_1176_);
lean_inc(v_initHeartbeats_1175_);
lean_inc(v_openDecls_1174_);
lean_inc(v_currNamespace_1173_);
lean_inc(v_ref_1172_);
lean_inc(v_maxRecDepth_1171_);
lean_inc(v_currRecDepth_1170_);
lean_inc(v_options_1169_);
lean_inc(v_fileMap_1168_);
lean_inc(v_fileName_1167_);
lean_dec(v___y_1159_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1203_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v_ref_1186_; lean_object* v___x_1188_; 
v_ref_1186_ = l_Lean_replaceRef(v___x_1152_, v_ref_1172_);
lean_dec(v___x_1152_);
lean_inc_ref(v_inheritedTraceOptions_1182_);
lean_inc(v_cancelTk_x3f_1180_);
lean_inc(v_currMacroScope_1178_);
lean_inc(v_quotContext_1177_);
lean_inc(v_maxHeartbeats_1176_);
lean_inc(v_initHeartbeats_1175_);
lean_inc(v_openDecls_1174_);
lean_inc(v_currNamespace_1173_);
lean_inc(v_maxRecDepth_1171_);
lean_inc(v_currRecDepth_1170_);
lean_inc_ref(v_options_1169_);
lean_inc_ref(v_fileMap_1168_);
lean_inc_ref(v_fileName_1167_);
if (v_isShared_1185_ == 0)
{
lean_ctor_set(v___x_1184_, 5, v_ref_1186_);
v___x_1188_ = v___x_1184_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1202_; 
v_reuseFailAlloc_1202_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_1202_, 0, v_fileName_1167_);
lean_ctor_set(v_reuseFailAlloc_1202_, 1, v_fileMap_1168_);
lean_ctor_set(v_reuseFailAlloc_1202_, 2, v_options_1169_);
lean_ctor_set(v_reuseFailAlloc_1202_, 3, v_currRecDepth_1170_);
lean_ctor_set(v_reuseFailAlloc_1202_, 4, v_maxRecDepth_1171_);
lean_ctor_set(v_reuseFailAlloc_1202_, 5, v_ref_1186_);
lean_ctor_set(v_reuseFailAlloc_1202_, 6, v_currNamespace_1173_);
lean_ctor_set(v_reuseFailAlloc_1202_, 7, v_openDecls_1174_);
lean_ctor_set(v_reuseFailAlloc_1202_, 8, v_initHeartbeats_1175_);
lean_ctor_set(v_reuseFailAlloc_1202_, 9, v_maxHeartbeats_1176_);
lean_ctor_set(v_reuseFailAlloc_1202_, 10, v_quotContext_1177_);
lean_ctor_set(v_reuseFailAlloc_1202_, 11, v_currMacroScope_1178_);
lean_ctor_set(v_reuseFailAlloc_1202_, 12, v_cancelTk_x3f_1180_);
lean_ctor_set(v_reuseFailAlloc_1202_, 13, v_inheritedTraceOptions_1182_);
lean_ctor_set_uint8(v_reuseFailAlloc_1202_, sizeof(void*)*14, v_diag_1179_);
lean_ctor_set_uint8(v_reuseFailAlloc_1202_, sizeof(void*)*14 + 1, v_suppressElabErrors_1181_);
v___x_1188_ = v_reuseFailAlloc_1202_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
lean_object* v___x_1189_; 
v___x_1189_ = lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1(v_a_1163_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_, v___x_1188_, v___y_1160_);
lean_dec_ref(v___x_1188_);
if (lean_obj_tag(v___x_1189_) == 0)
{
lean_object* v_a_1190_; lean_object* v_ref_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; 
v_a_1190_ = lean_ctor_get(v___x_1189_, 0);
lean_inc(v_a_1190_);
lean_dec_ref_known(v___x_1189_, 1);
v_ref_1191_ = l_Lean_replaceRef(v_tk_1154_, v_ref_1172_);
lean_dec(v_ref_1172_);
v___x_1192_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1192_, 0, v_fileName_1167_);
lean_ctor_set(v___x_1192_, 1, v_fileMap_1168_);
lean_ctor_set(v___x_1192_, 2, v_options_1169_);
lean_ctor_set(v___x_1192_, 3, v_currRecDepth_1170_);
lean_ctor_set(v___x_1192_, 4, v_maxRecDepth_1171_);
lean_ctor_set(v___x_1192_, 5, v_ref_1191_);
lean_ctor_set(v___x_1192_, 6, v_currNamespace_1173_);
lean_ctor_set(v___x_1192_, 7, v_openDecls_1174_);
lean_ctor_set(v___x_1192_, 8, v_initHeartbeats_1175_);
lean_ctor_set(v___x_1192_, 9, v_maxHeartbeats_1176_);
lean_ctor_set(v___x_1192_, 10, v_quotContext_1177_);
lean_ctor_set(v___x_1192_, 11, v_currMacroScope_1178_);
lean_ctor_set(v___x_1192_, 12, v_cancelTk_x3f_1180_);
lean_ctor_set(v___x_1192_, 13, v_inheritedTraceOptions_1182_);
lean_ctor_set_uint8(v___x_1192_, sizeof(void*)*14, v_diag_1179_);
lean_ctor_set_uint8(v___x_1192_, sizeof(void*)*14 + 1, v_suppressElabErrors_1181_);
v___x_1193_ = lp_mathlib_Mathlib_Util_compileDefn(v_a_1190_, v___y_1157_, v___y_1158_, v___x_1192_, v___y_1160_);
lean_dec_ref_known(v___x_1192_, 14);
return v___x_1193_;
}
else
{
lean_object* v_a_1194_; lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1201_; 
lean_dec_ref(v_inheritedTraceOptions_1182_);
lean_dec(v_cancelTk_x3f_1180_);
lean_dec(v_currMacroScope_1178_);
lean_dec(v_quotContext_1177_);
lean_dec(v_maxHeartbeats_1176_);
lean_dec(v_initHeartbeats_1175_);
lean_dec(v_openDecls_1174_);
lean_dec(v_currNamespace_1173_);
lean_dec(v_ref_1172_);
lean_dec(v_maxRecDepth_1171_);
lean_dec(v_currRecDepth_1170_);
lean_dec_ref(v_options_1169_);
lean_dec_ref(v_fileMap_1168_);
lean_dec_ref(v_fileName_1167_);
v_a_1194_ = lean_ctor_get(v___x_1189_, 0);
v_isSharedCheck_1201_ = !lean_is_exclusive(v___x_1189_);
if (v_isSharedCheck_1201_ == 0)
{
v___x_1196_ = v___x_1189_;
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
else
{
lean_inc(v_a_1194_);
lean_dec(v___x_1189_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1199_; 
if (v_isShared_1197_ == 0)
{
v___x_1199_ = v___x_1196_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v_a_1194_);
v___x_1199_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
return v___x_1199_;
}
}
}
}
}
}
else
{
lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
lean_dec(v___x_1152_);
v___x_1204_ = lean_obj_once(&lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1, &lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1);
v___x_1205_ = l_Lean_MessageData_ofName(v_a_1163_);
v___x_1206_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1206_, 0, v___x_1204_);
lean_ctor_set(v___x_1206_, 1, v___x_1205_);
v___x_1207_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2(v_tk_1154_, v___x_1206_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_);
lean_dec_ref(v___y_1159_);
if (lean_obj_tag(v___x_1207_) == 0)
{
lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1215_; 
v_isSharedCheck_1215_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1215_ == 0)
{
lean_object* v_unused_1216_; 
v_unused_1216_ = lean_ctor_get(v___x_1207_, 0);
lean_dec(v_unused_1216_);
v___x_1209_ = v___x_1207_;
v_isShared_1210_ = v_isSharedCheck_1215_;
goto v_resetjp_1208_;
}
else
{
lean_dec(v___x_1207_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1215_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1211_; lean_object* v___x_1213_; 
v___x_1211_ = lean_box(0);
if (v_isShared_1210_ == 0)
{
lean_ctor_set(v___x_1209_, 0, v___x_1211_);
v___x_1213_ = v___x_1209_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1214_; 
v_reuseFailAlloc_1214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1214_, 0, v___x_1211_);
v___x_1213_ = v_reuseFailAlloc_1214_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
return v___x_1213_;
}
}
}
else
{
return v___x_1207_;
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
lean_dec_ref(v___y_1159_);
lean_dec(v___x_1152_);
v_a_1217_ = lean_ctor_get(v___x_1162_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1162_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1162_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1162_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___boxed(lean_object* v___x_1225_, lean_object* v___x_1226_, lean_object* v_tk_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_){
_start:
{
lean_object* v_res_1235_; 
v_res_1235_ = lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0(v___x_1225_, v___x_1226_, v_tk_1227_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_, v___y_1232_, v___y_1233_);
lean_dec(v___y_1233_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
lean_dec(v_tk_1227_);
return v_res_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1(lean_object* v_x_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_){
_start:
{
lean_object* v___x_1240_; uint8_t v___x_1241_; 
v___x_1240_ = ((lean_object*)(lp_mathlib_Mathlib_Util_commandCompile__def_x25___00__closed__3));
lean_inc(v_x_1236_);
v___x_1241_ = l_Lean_Syntax_isOfKind(v_x_1236_, v___x_1240_);
if (v___x_1241_ == 0)
{
lean_object* v___x_1242_; 
lean_dec(v_x_1236_);
v___x_1242_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg();
return v___x_1242_;
}
else
{
lean_object* v___x_1243_; lean_object* v_tk_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___f_1248_; lean_object* v___x_1249_; 
v___x_1243_ = lean_unsigned_to_nat(0u);
v_tk_1244_ = l_Lean_Syntax_getArg(v_x_1236_, v___x_1243_);
v___x_1245_ = lean_unsigned_to_nat(1u);
v___x_1246_ = l_Lean_Syntax_getArg(v_x_1236_, v___x_1245_);
lean_dec(v_x_1236_);
v___x_1247_ = lean_box(0);
v___f_1248_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1248_, 0, v___x_1246_);
lean_closure_set(v___f_1248_, 1, v___x_1247_);
lean_closure_set(v___f_1248_, 2, v_tk_1244_);
v___x_1249_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_1248_, v_a_1237_, v_a_1238_);
return v___x_1249_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___boxed(lean_object* v_x_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_){
_start:
{
lean_object* v_res_1254_; 
v_res_1254_ = lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1(v_x_1250_, v_a_1251_, v_a_1252_);
lean_dec(v_a_1252_);
lean_dec_ref(v_a_1251_);
return v_res_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1(lean_object* v_00_u03b1_1255_, lean_object* v_msg_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_){
_start:
{
lean_object* v___x_1264_; 
v___x_1264_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(v_msg_1256_, v___y_1257_, v___y_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1265_, lean_object* v_msg_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1(v_00_u03b1_1265_, v_msg_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
lean_dec(v___y_1268_);
lean_dec_ref(v___y_1267_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4(lean_object* v_ref_1275_, lean_object* v_msgData_1276_, uint8_t v_severity_1277_, uint8_t v_isSilent_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v___x_1286_; 
v___x_1286_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg(v_ref_1275_, v_msgData_1276_, v_severity_1277_, v_isSilent_1278_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___boxed(lean_object* v_ref_1287_, lean_object* v_msgData_1288_, lean_object* v_severity_1289_, lean_object* v_isSilent_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_){
_start:
{
uint8_t v_severity_boxed_1298_; uint8_t v_isSilent_boxed_1299_; lean_object* v_res_1300_; 
v_severity_boxed_1298_ = lean_unbox(v_severity_1289_);
v_isSilent_boxed_1299_ = lean_unbox(v_isSilent_1290_);
v_res_1300_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4(v_ref_1287_, v_msgData_1288_, v_severity_boxed_1298_, v_isSilent_boxed_1299_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_);
lean_dec(v___y_1296_);
lean_dec_ref(v___y_1295_);
lean_dec(v___y_1294_);
lean_dec_ref(v___y_1293_);
lean_dec(v___y_1292_);
lean_dec_ref(v___y_1291_);
lean_dec(v_ref_1287_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3(lean_object* v_msgData_1301_, lean_object* v_macroStack_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_){
_start:
{
lean_object* v___x_1310_; 
v___x_1310_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___redArg(v_msgData_1301_, v_macroStack_1302_, v___y_1307_);
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_1311_, lean_object* v_macroStack_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_){
_start:
{
lean_object* v_res_1320_; 
v_res_1320_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__3(v_msgData_1311_, v_macroStack_1312_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
lean_dec(v___y_1314_);
lean_dec_ref(v___y_1313_);
return v_res_1320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1(lean_object* v_msg_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_){
_start:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v_toApplicative_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1390_; 
v___x_1327_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0, &lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0);
v___x_1328_ = l_StateRefT_x27_instMonad___redArg(v___x_1327_);
v_toApplicative_1329_ = lean_ctor_get(v___x_1328_, 0);
v_isSharedCheck_1390_ = !lean_is_exclusive(v___x_1328_);
if (v_isSharedCheck_1390_ == 0)
{
lean_object* v_unused_1391_; 
v_unused_1391_ = lean_ctor_get(v___x_1328_, 1);
lean_dec(v_unused_1391_);
v___x_1331_ = v___x_1328_;
v_isShared_1332_ = v_isSharedCheck_1390_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_toApplicative_1329_);
lean_dec(v___x_1328_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1390_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v_toFunctor_1333_; lean_object* v_toSeq_1334_; lean_object* v_toSeqLeft_1335_; lean_object* v_toSeqRight_1336_; lean_object* v___x_1338_; uint8_t v_isShared_1339_; uint8_t v_isSharedCheck_1388_; 
v_toFunctor_1333_ = lean_ctor_get(v_toApplicative_1329_, 0);
v_toSeq_1334_ = lean_ctor_get(v_toApplicative_1329_, 2);
v_toSeqLeft_1335_ = lean_ctor_get(v_toApplicative_1329_, 3);
v_toSeqRight_1336_ = lean_ctor_get(v_toApplicative_1329_, 4);
v_isSharedCheck_1388_ = !lean_is_exclusive(v_toApplicative_1329_);
if (v_isSharedCheck_1388_ == 0)
{
lean_object* v_unused_1389_; 
v_unused_1389_ = lean_ctor_get(v_toApplicative_1329_, 1);
lean_dec(v_unused_1389_);
v___x_1338_ = v_toApplicative_1329_;
v_isShared_1339_ = v_isSharedCheck_1388_;
goto v_resetjp_1337_;
}
else
{
lean_inc(v_toSeqRight_1336_);
lean_inc(v_toSeqLeft_1335_);
lean_inc(v_toSeq_1334_);
lean_inc(v_toFunctor_1333_);
lean_dec(v_toApplicative_1329_);
v___x_1338_ = lean_box(0);
v_isShared_1339_ = v_isSharedCheck_1388_;
goto v_resetjp_1337_;
}
v_resetjp_1337_:
{
lean_object* v___f_1340_; lean_object* v___f_1341_; lean_object* v___f_1342_; lean_object* v___f_1343_; lean_object* v___x_1344_; lean_object* v___f_1345_; lean_object* v___f_1346_; lean_object* v___f_1347_; lean_object* v___x_1349_; 
v___f_1340_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1));
v___f_1341_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_1333_);
v___f_1342_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1342_, 0, v_toFunctor_1333_);
v___f_1343_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1343_, 0, v_toFunctor_1333_);
v___x_1344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1344_, 0, v___f_1342_);
lean_ctor_set(v___x_1344_, 1, v___f_1343_);
v___f_1345_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1345_, 0, v_toSeqRight_1336_);
v___f_1346_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1346_, 0, v_toSeqLeft_1335_);
v___f_1347_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1347_, 0, v_toSeq_1334_);
if (v_isShared_1339_ == 0)
{
lean_ctor_set(v___x_1338_, 4, v___f_1345_);
lean_ctor_set(v___x_1338_, 3, v___f_1346_);
lean_ctor_set(v___x_1338_, 2, v___f_1347_);
lean_ctor_set(v___x_1338_, 1, v___f_1340_);
lean_ctor_set(v___x_1338_, 0, v___x_1344_);
v___x_1349_ = v___x_1338_;
goto v_reusejp_1348_;
}
else
{
lean_object* v_reuseFailAlloc_1387_; 
v_reuseFailAlloc_1387_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1387_, 0, v___x_1344_);
lean_ctor_set(v_reuseFailAlloc_1387_, 1, v___f_1340_);
lean_ctor_set(v_reuseFailAlloc_1387_, 2, v___f_1347_);
lean_ctor_set(v_reuseFailAlloc_1387_, 3, v___f_1346_);
lean_ctor_set(v_reuseFailAlloc_1387_, 4, v___f_1345_);
v___x_1349_ = v_reuseFailAlloc_1387_;
goto v_reusejp_1348_;
}
v_reusejp_1348_:
{
lean_object* v___x_1351_; 
if (v_isShared_1332_ == 0)
{
lean_ctor_set(v___x_1331_, 1, v___f_1341_);
lean_ctor_set(v___x_1331_, 0, v___x_1349_);
v___x_1351_ = v___x_1331_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1386_; 
v_reuseFailAlloc_1386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1386_, 0, v___x_1349_);
lean_ctor_set(v_reuseFailAlloc_1386_, 1, v___f_1341_);
v___x_1351_ = v_reuseFailAlloc_1386_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
lean_object* v___x_1352_; lean_object* v_toApplicative_1353_; lean_object* v___x_1355_; uint8_t v_isShared_1356_; uint8_t v_isSharedCheck_1384_; 
v___x_1352_ = l_StateRefT_x27_instMonad___redArg(v___x_1351_);
v_toApplicative_1353_ = lean_ctor_get(v___x_1352_, 0);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1352_);
if (v_isSharedCheck_1384_ == 0)
{
lean_object* v_unused_1385_; 
v_unused_1385_ = lean_ctor_get(v___x_1352_, 1);
lean_dec(v_unused_1385_);
v___x_1355_ = v___x_1352_;
v_isShared_1356_ = v_isSharedCheck_1384_;
goto v_resetjp_1354_;
}
else
{
lean_inc(v_toApplicative_1353_);
lean_dec(v___x_1352_);
v___x_1355_ = lean_box(0);
v_isShared_1356_ = v_isSharedCheck_1384_;
goto v_resetjp_1354_;
}
v_resetjp_1354_:
{
lean_object* v_toFunctor_1357_; lean_object* v_toSeq_1358_; lean_object* v_toSeqLeft_1359_; lean_object* v_toSeqRight_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1382_; 
v_toFunctor_1357_ = lean_ctor_get(v_toApplicative_1353_, 0);
v_toSeq_1358_ = lean_ctor_get(v_toApplicative_1353_, 2);
v_toSeqLeft_1359_ = lean_ctor_get(v_toApplicative_1353_, 3);
v_toSeqRight_1360_ = lean_ctor_get(v_toApplicative_1353_, 4);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_toApplicative_1353_);
if (v_isSharedCheck_1382_ == 0)
{
lean_object* v_unused_1383_; 
v_unused_1383_ = lean_ctor_get(v_toApplicative_1353_, 1);
lean_dec(v_unused_1383_);
v___x_1362_ = v_toApplicative_1353_;
v_isShared_1363_ = v_isSharedCheck_1382_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_toSeqRight_1360_);
lean_inc(v_toSeqLeft_1359_);
lean_inc(v_toSeq_1358_);
lean_inc(v_toFunctor_1357_);
lean_dec(v_toApplicative_1353_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1382_;
goto v_resetjp_1361_;
}
v_resetjp_1361_:
{
lean_object* v___f_1364_; lean_object* v___f_1365_; lean_object* v___f_1366_; lean_object* v___f_1367_; lean_object* v___x_1368_; lean_object* v___f_1369_; lean_object* v___f_1370_; lean_object* v___f_1371_; lean_object* v___x_1373_; 
v___f_1364_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3));
v___f_1365_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_1357_);
v___f_1366_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1366_, 0, v_toFunctor_1357_);
v___f_1367_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1367_, 0, v_toFunctor_1357_);
v___x_1368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1368_, 0, v___f_1366_);
lean_ctor_set(v___x_1368_, 1, v___f_1367_);
v___f_1369_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1369_, 0, v_toSeqRight_1360_);
v___f_1370_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1370_, 0, v_toSeqLeft_1359_);
v___f_1371_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1371_, 0, v_toSeq_1358_);
if (v_isShared_1363_ == 0)
{
lean_ctor_set(v___x_1362_, 4, v___f_1369_);
lean_ctor_set(v___x_1362_, 3, v___f_1370_);
lean_ctor_set(v___x_1362_, 2, v___f_1371_);
lean_ctor_set(v___x_1362_, 1, v___f_1364_);
lean_ctor_set(v___x_1362_, 0, v___x_1368_);
v___x_1373_ = v___x_1362_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v___x_1368_);
lean_ctor_set(v_reuseFailAlloc_1381_, 1, v___f_1364_);
lean_ctor_set(v_reuseFailAlloc_1381_, 2, v___f_1371_);
lean_ctor_set(v_reuseFailAlloc_1381_, 3, v___f_1370_);
lean_ctor_set(v_reuseFailAlloc_1381_, 4, v___f_1369_);
v___x_1373_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
lean_object* v___x_1375_; 
if (v_isShared_1356_ == 0)
{
lean_ctor_set(v___x_1355_, 1, v___f_1365_);
lean_ctor_set(v___x_1355_, 0, v___x_1373_);
v___x_1375_ = v___x_1355_;
goto v_reusejp_1374_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v___x_1373_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v___f_1365_);
v___x_1375_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1374_;
}
v_reusejp_1374_:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_2098__overap_1378_; lean_object* v___x_1379_; 
v___x_1376_ = lean_box(0);
v___x_1377_ = l_instInhabitedOfMonad___redArg(v___x_1375_, v___x_1376_);
v___x_2098__overap_1378_ = lean_panic_fn_borrowed(v___x_1377_, v_msg_1321_);
lean_dec(v___x_1377_);
lean_inc(v___y_1325_);
lean_inc_ref(v___y_1324_);
lean_inc(v___y_1323_);
lean_inc_ref(v___y_1322_);
v___x_1379_ = lean_apply_5(v___x_2098__overap_1378_, v___y_1322_, v___y_1323_, v___y_1324_, v___y_1325_, lean_box(0));
return v___x_1379_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1___boxed(lean_object* v_msg_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_){
_start:
{
lean_object* v_res_1398_; 
v_res_1398_ = lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1(v_msg_1392_, v___y_1393_, v___y_1394_, v___y_1395_, v___y_1396_);
lean_dec(v___y_1396_);
lean_dec_ref(v___y_1395_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
return v_res_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(lean_object* v_msg_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_){
_start:
{
lean_object* v_ref_1405_; lean_object* v___x_1406_; lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1415_; 
v_ref_1405_ = lean_ctor_get(v___y_1402_, 5);
v___x_1406_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(v_msg_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_);
v_a_1407_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1415_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1415_ == 0)
{
v___x_1409_ = v___x_1406_;
v_isShared_1410_ = v_isSharedCheck_1415_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___x_1406_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1415_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
lean_object* v___x_1411_; lean_object* v___x_1413_; 
lean_inc(v_ref_1405_);
v___x_1411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1411_, 0, v_ref_1405_);
lean_ctor_set(v___x_1411_, 1, v_a_1407_);
if (v_isShared_1410_ == 0)
{
lean_ctor_set_tag(v___x_1409_, 1);
lean_ctor_set(v___x_1409_, 0, v___x_1411_);
v___x_1413_ = v___x_1409_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v___x_1411_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg___boxed(lean_object* v_msg_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_){
_start:
{
lean_object* v_res_1422_; 
v_res_1422_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v_msg_1416_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_);
lean_dec(v___y_1420_);
lean_dec_ref(v___y_1419_);
lean_dec(v___y_1418_);
lean_dec_ref(v___y_1417_);
return v_res_1422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0(lean_object* v_constName_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_){
_start:
{
lean_object* v___x_1437_; lean_object* v_env_1438_; uint8_t v___x_1439_; lean_object* v___x_1440_; 
v___x_1437_ = lean_st_ref_get(v___y_1427_);
v_env_1438_ = lean_ctor_get(v___x_1437_, 0);
lean_inc_ref(v_env_1438_);
lean_dec(v___x_1437_);
v___x_1439_ = 0;
lean_inc(v_constName_1423_);
v___x_1440_ = l_Lean_Environment_findAsync_x3f(v_env_1438_, v_constName_1423_, v___x_1439_);
if (lean_obj_tag(v___x_1440_) == 1)
{
lean_object* v_val_1441_; uint8_t v_kind_1442_; 
v_val_1441_ = lean_ctor_get(v___x_1440_, 0);
lean_inc(v_val_1441_);
lean_dec_ref_known(v___x_1440_, 1);
v_kind_1442_ = lean_ctor_get_uint8(v_val_1441_, sizeof(void*)*3);
if (v_kind_1442_ == 0)
{
lean_object* v___x_1443_; 
v___x_1443_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_1441_);
if (lean_obj_tag(v___x_1443_) == 1)
{
lean_object* v_val_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1451_; 
lean_dec(v_constName_1423_);
v_val_1444_ = lean_ctor_get(v___x_1443_, 0);
v_isSharedCheck_1451_ = !lean_is_exclusive(v___x_1443_);
if (v_isSharedCheck_1451_ == 0)
{
v___x_1446_ = v___x_1443_;
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_val_1444_);
lean_dec(v___x_1443_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
lean_object* v___x_1449_; 
if (v_isShared_1447_ == 0)
{
lean_ctor_set_tag(v___x_1446_, 0);
v___x_1449_ = v___x_1446_;
goto v_reusejp_1448_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v_val_1444_);
v___x_1449_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1448_;
}
v_reusejp_1448_:
{
return v___x_1449_;
}
}
}
else
{
lean_object* v___x_1452_; lean_object* v___x_1453_; 
lean_dec_ref(v___x_1443_);
v___x_1452_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__6);
v___x_1453_ = lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__1(v___x_1452_, v___y_1424_, v___y_1425_, v___y_1426_, v___y_1427_);
if (lean_obj_tag(v___x_1453_) == 0)
{
lean_object* v_a_1454_; lean_object* v___x_1456_; uint8_t v_isShared_1457_; uint8_t v_isSharedCheck_1462_; 
v_a_1454_ = lean_ctor_get(v___x_1453_, 0);
v_isSharedCheck_1462_ = !lean_is_exclusive(v___x_1453_);
if (v_isSharedCheck_1462_ == 0)
{
v___x_1456_ = v___x_1453_;
v_isShared_1457_ = v_isSharedCheck_1462_;
goto v_resetjp_1455_;
}
else
{
lean_inc(v_a_1454_);
lean_dec(v___x_1453_);
v___x_1456_ = lean_box(0);
v_isShared_1457_ = v_isSharedCheck_1462_;
goto v_resetjp_1455_;
}
v_resetjp_1455_:
{
if (lean_obj_tag(v_a_1454_) == 0)
{
lean_del_object(v___x_1456_);
goto v___jp_1429_;
}
else
{
lean_object* v_val_1458_; lean_object* v___x_1460_; 
lean_dec(v_constName_1423_);
v_val_1458_ = lean_ctor_get(v_a_1454_, 0);
lean_inc(v_val_1458_);
lean_dec_ref_known(v_a_1454_, 1);
if (v_isShared_1457_ == 0)
{
lean_ctor_set(v___x_1456_, 0, v_val_1458_);
v___x_1460_ = v___x_1456_;
goto v_reusejp_1459_;
}
else
{
lean_object* v_reuseFailAlloc_1461_; 
v_reuseFailAlloc_1461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1461_, 0, v_val_1458_);
v___x_1460_ = v_reuseFailAlloc_1461_;
goto v_reusejp_1459_;
}
v_reusejp_1459_:
{
return v___x_1460_;
}
}
}
}
else
{
lean_object* v_a_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1470_; 
lean_dec(v_constName_1423_);
v_a_1463_ = lean_ctor_get(v___x_1453_, 0);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1453_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1465_ = v___x_1453_;
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_a_1463_);
lean_dec(v___x_1453_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v___x_1468_; 
if (v_isShared_1466_ == 0)
{
v___x_1468_ = v___x_1465_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1469_; 
v_reuseFailAlloc_1469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1469_, 0, v_a_1463_);
v___x_1468_ = v_reuseFailAlloc_1469_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
return v___x_1468_;
}
}
}
}
}
else
{
lean_dec(v_val_1441_);
goto v___jp_1429_;
}
}
else
{
lean_dec(v___x_1440_);
goto v___jp_1429_;
}
v___jp_1429_:
{
lean_object* v___x_1430_; uint8_t v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; 
v___x_1430_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1);
v___x_1431_ = 0;
v___x_1432_ = l_Lean_MessageData_ofConstName(v_constName_1423_, v___x_1431_);
v___x_1433_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1433_, 0, v___x_1430_);
lean_ctor_set(v___x_1433_, 1, v___x_1432_);
v___x_1434_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__3);
v___x_1435_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1435_, 0, v___x_1433_);
lean_ctor_set(v___x_1435_, 1, v___x_1434_);
v___x_1436_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v___x_1435_, v___y_1424_, v___y_1425_, v___y_1426_, v___y_1427_);
return v___x_1436_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0___boxed(lean_object* v_constName_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_){
_start:
{
lean_object* v_res_1477_; 
v_res_1477_ = lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0(v_constName_1471_, v___y_1472_, v___y_1473_, v___y_1474_, v___y_1475_);
lean_dec(v___y_1475_);
lean_dec_ref(v___y_1474_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
return v_res_1477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go(lean_object* v_iv_1478_, lean_object* v_rv_1479_, lean_object* v_value_1480_, lean_object* v_a_1481_, lean_object* v_a_1482_, lean_object* v_a_1483_, lean_object* v_a_1484_){
_start:
{
lean_object* v_toConstantVal_1486_; lean_object* v_all_1487_; lean_object* v_name_1488_; lean_object* v_levelParams_1489_; lean_object* v_type_1490_; lean_object* v___x_1492_; uint8_t v_isShared_1493_; uint8_t v_isSharedCheck_1571_; 
v_toConstantVal_1486_ = lean_ctor_get(v_rv_1479_, 0);
lean_inc_ref(v_toConstantVal_1486_);
v_all_1487_ = lean_ctor_get(v_rv_1479_, 1);
lean_inc(v_all_1487_);
lean_dec_ref(v_rv_1479_);
v_name_1488_ = lean_ctor_get(v_toConstantVal_1486_, 0);
v_levelParams_1489_ = lean_ctor_get(v_toConstantVal_1486_, 1);
v_type_1490_ = lean_ctor_get(v_toConstantVal_1486_, 2);
v_isSharedCheck_1571_ = !lean_is_exclusive(v_toConstantVal_1486_);
if (v_isSharedCheck_1571_ == 0)
{
v___x_1492_ = v_toConstantVal_1486_;
v_isShared_1493_ = v_isSharedCheck_1571_;
goto v_resetjp_1491_;
}
else
{
lean_inc(v_type_1490_);
lean_inc(v_levelParams_1489_);
lean_inc(v_name_1488_);
lean_dec(v_toConstantVal_1486_);
v___x_1492_ = lean_box(0);
v_isShared_1493_ = v_isSharedCheck_1571_;
goto v_resetjp_1491_;
}
v_resetjp_1491_:
{
lean_object* v___x_1494_; 
lean_inc(v_name_1488_);
v___x_1494_ = l_Lean_Core_mkFreshUserName(v_name_1488_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1494_) == 0)
{
lean_object* v_a_1495_; lean_object* v___x_1497_; 
v_a_1495_ = lean_ctor_get(v___x_1494_, 0);
lean_inc_n(v_a_1495_, 2);
lean_dec_ref_known(v___x_1494_, 1);
lean_inc(v_levelParams_1489_);
if (v_isShared_1493_ == 0)
{
lean_ctor_set(v___x_1492_, 0, v_a_1495_);
v___x_1497_ = v___x_1492_;
goto v_reusejp_1496_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_a_1495_);
lean_ctor_set(v_reuseFailAlloc_1562_, 1, v_levelParams_1489_);
lean_ctor_set(v_reuseFailAlloc_1562_, 2, v_type_1490_);
v___x_1497_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1496_;
}
v_reusejp_1496_:
{
lean_object* v___x_1498_; uint8_t v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; 
v___x_1498_ = lean_box(1);
v___x_1499_ = 1;
v___x_1500_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_1500_, 0, v___x_1497_);
lean_ctor_set(v___x_1500_, 1, v_value_1480_);
lean_ctor_set(v___x_1500_, 2, v___x_1498_);
lean_ctor_set(v___x_1500_, 3, v_all_1487_);
lean_ctor_set_uint8(v___x_1500_, sizeof(void*)*4, v___x_1499_);
v___x_1501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1501_, 0, v___x_1500_);
v___x_1502_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(v___x_1501_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1502_) == 0)
{
lean_object* v___x_1504_; uint8_t v_isShared_1505_; uint8_t v_isSharedCheck_1560_; 
v_isSharedCheck_1560_ = !lean_is_exclusive(v___x_1502_);
if (v_isSharedCheck_1560_ == 0)
{
lean_object* v_unused_1561_; 
v_unused_1561_ = lean_ctor_get(v___x_1502_, 0);
lean_dec(v_unused_1561_);
v___x_1504_ = v___x_1502_;
v_isShared_1505_ = v_isSharedCheck_1560_;
goto v_resetjp_1503_;
}
else
{
lean_dec(v___x_1502_);
v___x_1504_ = lean_box(0);
v_isShared_1505_ = v_isSharedCheck_1560_;
goto v_resetjp_1503_;
}
v_resetjp_1503_:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; 
v___x_1506_ = lean_box(0);
lean_inc(v_levelParams_1489_);
v___x_1507_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileDefn_spec__0(v_levelParams_1489_, v___x_1506_);
lean_inc(v___x_1507_);
lean_inc(v_name_1488_);
v___x_1508_ = l_Lean_Expr_const___override(v_name_1488_, v___x_1507_);
v___x_1509_ = ((lean_object*)(lp_mathlib_Mathlib_Util_compileDefn___closed__0));
v___x_1510_ = l_Lean_Name_str___override(v_name_1488_, v___x_1509_);
v___x_1511_ = l_Lean_Core_mkFreshUserName(v___x_1510_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1511_) == 0)
{
lean_object* v_a_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; 
v_a_1512_ = lean_ctor_get(v___x_1511_, 0);
lean_inc(v_a_1512_);
lean_dec_ref_known(v___x_1511_, 1);
lean_inc(v___x_1507_);
v___x_1513_ = l_Lean_Expr_const___override(v_a_1495_, v___x_1507_);
v___x_1514_ = l_Lean_Meta_mkEq(v___x_1508_, v___x_1513_, v_a_1481_, v_a_1482_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1514_) == 0)
{
lean_object* v_a_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1524_; 
v_a_1515_ = lean_ctor_get(v___x_1514_, 0);
lean_inc(v_a_1515_);
lean_dec_ref_known(v___x_1514_, 1);
lean_inc_n(v_a_1512_, 3);
v___x_1516_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1516_, 0, v_a_1512_);
lean_ctor_set(v___x_1516_, 1, v_levelParams_1489_);
lean_ctor_set(v___x_1516_, 2, v_a_1515_);
v___x_1517_ = l_Lean_Expr_const___override(v_a_1512_, v___x_1507_);
v___x_1518_ = lean_box(0);
v___x_1519_ = 2;
v___x_1520_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1520_, 0, v_a_1512_);
lean_ctor_set(v___x_1520_, 1, v___x_1506_);
v___x_1521_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_1521_, 0, v___x_1516_);
lean_ctor_set(v___x_1521_, 1, v___x_1517_);
lean_ctor_set(v___x_1521_, 2, v___x_1518_);
lean_ctor_set(v___x_1521_, 3, v___x_1520_);
lean_ctor_set_uint8(v___x_1521_, sizeof(void*)*4, v___x_1519_);
v___x_1522_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1522_, 0, v___x_1521_);
lean_ctor_set(v___x_1522_, 1, v___x_1506_);
if (v_isShared_1505_ == 0)
{
lean_ctor_set_tag(v___x_1504_, 5);
lean_ctor_set(v___x_1504_, 0, v___x_1522_);
v___x_1524_ = v___x_1504_;
goto v_reusejp_1523_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v___x_1522_);
v___x_1524_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1523_;
}
v_reusejp_1523_:
{
uint8_t v___x_1525_; lean_object* v___x_1526_; 
v___x_1525_ = 0;
v___x_1526_ = l_Lean_addDecl(v___x_1524_, v___x_1525_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1526_) == 0)
{
uint8_t v___x_1527_; lean_object* v___x_1528_; 
lean_dec_ref_known(v___x_1526_, 1);
v___x_1527_ = 0;
v___x_1528_ = l_Lean_Compiler_CSimp_add(v_a_1512_, v___x_1527_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1528_) == 0)
{
lean_object* v_toConstantVal_1529_; lean_object* v_name_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; 
lean_dec_ref_known(v___x_1528_, 1);
v_toConstantVal_1529_ = lean_ctor_get(v_iv_1478_, 0);
lean_inc_ref(v_toConstantVal_1529_);
lean_dec_ref(v_iv_1478_);
v_name_1530_ = lean_ctor_get(v_toConstantVal_1529_, 0);
lean_inc(v_name_1530_);
lean_dec_ref(v_toConstantVal_1529_);
v___x_1531_ = l_Lean_mkRecOnName(v_name_1530_);
v___x_1532_ = lp_mathlib_Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0(v___x_1531_, v_a_1481_, v_a_1482_, v_a_1483_, v_a_1484_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_object* v_a_1533_; lean_object* v___x_1534_; 
v_a_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_a_1533_);
lean_dec_ref_known(v___x_1532_, 1);
v___x_1534_ = lp_mathlib_Mathlib_Util_compileDefn(v_a_1533_, v_a_1481_, v_a_1482_, v_a_1483_, v_a_1484_);
return v___x_1534_;
}
else
{
lean_object* v_a_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1542_; 
v_a_1535_ = lean_ctor_get(v___x_1532_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1532_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1537_ = v___x_1532_;
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_a_1535_);
lean_dec(v___x_1532_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1542_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1540_; 
if (v_isShared_1538_ == 0)
{
v___x_1540_ = v___x_1537_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1541_; 
v_reuseFailAlloc_1541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1541_, 0, v_a_1535_);
v___x_1540_ = v_reuseFailAlloc_1541_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
return v___x_1540_;
}
}
}
}
else
{
lean_dec_ref(v_iv_1478_);
return v___x_1528_;
}
}
else
{
lean_dec(v_a_1512_);
lean_dec_ref(v_iv_1478_);
return v___x_1526_;
}
}
}
else
{
lean_object* v_a_1544_; lean_object* v___x_1546_; uint8_t v_isShared_1547_; uint8_t v_isSharedCheck_1551_; 
lean_dec(v_a_1512_);
lean_dec(v___x_1507_);
lean_del_object(v___x_1504_);
lean_dec(v_levelParams_1489_);
lean_dec_ref(v_iv_1478_);
v_a_1544_ = lean_ctor_get(v___x_1514_, 0);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1514_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1546_ = v___x_1514_;
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
else
{
lean_inc(v_a_1544_);
lean_dec(v___x_1514_);
v___x_1546_ = lean_box(0);
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
v_resetjp_1545_:
{
lean_object* v___x_1549_; 
if (v_isShared_1547_ == 0)
{
v___x_1549_ = v___x_1546_;
goto v_reusejp_1548_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_a_1544_);
v___x_1549_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1548_;
}
v_reusejp_1548_:
{
return v___x_1549_;
}
}
}
}
else
{
lean_object* v_a_1552_; lean_object* v___x_1554_; uint8_t v_isShared_1555_; uint8_t v_isSharedCheck_1559_; 
lean_dec_ref(v___x_1508_);
lean_dec(v___x_1507_);
lean_del_object(v___x_1504_);
lean_dec(v_a_1495_);
lean_dec(v_levelParams_1489_);
lean_dec_ref(v_iv_1478_);
v_a_1552_ = lean_ctor_get(v___x_1511_, 0);
v_isSharedCheck_1559_ = !lean_is_exclusive(v___x_1511_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1554_ = v___x_1511_;
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
else
{
lean_inc(v_a_1552_);
lean_dec(v___x_1511_);
v___x_1554_ = lean_box(0);
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
v_resetjp_1553_:
{
lean_object* v___x_1557_; 
if (v_isShared_1555_ == 0)
{
v___x_1557_ = v___x_1554_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v_a_1552_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
}
}
}
else
{
lean_dec(v_a_1495_);
lean_dec(v_levelParams_1489_);
lean_dec(v_name_1488_);
lean_dec_ref(v_iv_1478_);
return v___x_1502_;
}
}
}
else
{
lean_object* v_a_1563_; lean_object* v___x_1565_; uint8_t v_isShared_1566_; uint8_t v_isSharedCheck_1570_; 
lean_del_object(v___x_1492_);
lean_dec_ref(v_type_1490_);
lean_dec(v_levelParams_1489_);
lean_dec(v_name_1488_);
lean_dec(v_all_1487_);
lean_dec_ref(v_value_1480_);
lean_dec_ref(v_iv_1478_);
v_a_1563_ = lean_ctor_get(v___x_1494_, 0);
v_isSharedCheck_1570_ = !lean_is_exclusive(v___x_1494_);
if (v_isSharedCheck_1570_ == 0)
{
v___x_1565_ = v___x_1494_;
v_isShared_1566_ = v_isSharedCheck_1570_;
goto v_resetjp_1564_;
}
else
{
lean_inc(v_a_1563_);
lean_dec(v___x_1494_);
v___x_1565_ = lean_box(0);
v_isShared_1566_ = v_isSharedCheck_1570_;
goto v_resetjp_1564_;
}
v_resetjp_1564_:
{
lean_object* v___x_1568_; 
if (v_isShared_1566_ == 0)
{
v___x_1568_ = v___x_1565_;
goto v_reusejp_1567_;
}
else
{
lean_object* v_reuseFailAlloc_1569_; 
v_reuseFailAlloc_1569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1569_, 0, v_a_1563_);
v___x_1568_ = v_reuseFailAlloc_1569_;
goto v_reusejp_1567_;
}
v_reusejp_1567_:
{
return v___x_1568_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go___boxed(lean_object* v_iv_1572_, lean_object* v_rv_1573_, lean_object* v_value_1574_, lean_object* v_a_1575_, lean_object* v_a_1576_, lean_object* v_a_1577_, lean_object* v_a_1578_, lean_object* v_a_1579_){
_start:
{
lean_object* v_res_1580_; 
v_res_1580_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go(v_iv_1572_, v_rv_1573_, v_value_1574_, v_a_1575_, v_a_1576_, v_a_1577_, v_a_1578_);
lean_dec(v_a_1578_);
lean_dec_ref(v_a_1577_);
lean_dec(v_a_1576_);
lean_dec_ref(v_a_1575_);
return v_res_1580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0(lean_object* v_00_u03b1_1581_, lean_object* v_msg_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
lean_object* v___x_1588_; 
v___x_1588_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v_msg_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_);
return v___x_1588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1589_, lean_object* v_msg_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_){
_start:
{
lean_object* v_res_1596_; 
v_res_1596_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0(v_00_u03b1_1589_, v_msg_1590_, v___y_1591_, v___y_1592_, v___y_1593_, v___y_1594_);
lean_dec(v___y_1594_);
lean_dec_ref(v___y_1593_);
lean_dec(v___y_1592_);
lean_dec_ref(v___y_1591_);
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0(lean_object* v_k_1597_, lean_object* v_b_1598_, lean_object* v_c_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_){
_start:
{
lean_object* v___x_1605_; 
lean_inc(v___y_1603_);
lean_inc_ref(v___y_1602_);
lean_inc(v___y_1601_);
lean_inc_ref(v___y_1600_);
v___x_1605_ = lean_apply_7(v_k_1597_, v_b_1598_, v_c_1599_, v___y_1600_, v___y_1601_, v___y_1602_, v___y_1603_, lean_box(0));
return v___x_1605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0___boxed(lean_object* v_k_1606_, lean_object* v_b_1607_, lean_object* v_c_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_){
_start:
{
lean_object* v_res_1614_; 
v_res_1614_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0(v_k_1606_, v_b_1607_, v_c_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
return v_res_1614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(lean_object* v_type_1615_, lean_object* v_k_1616_, uint8_t v_cleanupAnnotations_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_){
_start:
{
lean_object* v___f_1623_; uint8_t v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; 
v___f_1623_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1623_, 0, v_k_1616_);
v___x_1624_ = 0;
v___x_1625_ = lean_box(0);
v___x_1626_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_1624_, v___x_1625_, v_type_1615_, v___f_1623_, v_cleanupAnnotations_1617_, v___x_1624_, v___y_1618_, v___y_1619_, v___y_1620_, v___y_1621_);
if (lean_obj_tag(v___x_1626_) == 0)
{
lean_object* v_a_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1634_; 
v_a_1627_ = lean_ctor_get(v___x_1626_, 0);
v_isSharedCheck_1634_ = !lean_is_exclusive(v___x_1626_);
if (v_isSharedCheck_1634_ == 0)
{
v___x_1629_ = v___x_1626_;
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_a_1627_);
lean_dec(v___x_1626_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___x_1632_; 
if (v_isShared_1630_ == 0)
{
v___x_1632_ = v___x_1629_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v_a_1627_);
v___x_1632_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
return v___x_1632_;
}
}
}
else
{
lean_object* v_a_1635_; lean_object* v___x_1637_; uint8_t v_isShared_1638_; uint8_t v_isSharedCheck_1642_; 
v_a_1635_ = lean_ctor_get(v___x_1626_, 0);
v_isSharedCheck_1642_ = !lean_is_exclusive(v___x_1626_);
if (v_isSharedCheck_1642_ == 0)
{
v___x_1637_ = v___x_1626_;
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
else
{
lean_inc(v_a_1635_);
lean_dec(v___x_1626_);
v___x_1637_ = lean_box(0);
v_isShared_1638_ = v_isSharedCheck_1642_;
goto v_resetjp_1636_;
}
v_resetjp_1636_:
{
lean_object* v___x_1640_; 
if (v_isShared_1638_ == 0)
{
v___x_1640_ = v___x_1637_;
goto v_reusejp_1639_;
}
else
{
lean_object* v_reuseFailAlloc_1641_; 
v_reuseFailAlloc_1641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1641_, 0, v_a_1635_);
v___x_1640_ = v_reuseFailAlloc_1641_;
goto v_reusejp_1639_;
}
v_reusejp_1639_:
{
return v___x_1640_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg___boxed(lean_object* v_type_1643_, lean_object* v_k_1644_, lean_object* v_cleanupAnnotations_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1651_; lean_object* v_res_1652_; 
v_cleanupAnnotations_boxed_1651_ = lean_unbox(v_cleanupAnnotations_1645_);
v_res_1652_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(v_type_1643_, v_k_1644_, v_cleanupAnnotations_boxed_1651_, v___y_1646_, v___y_1647_, v___y_1648_, v___y_1649_);
lean_dec(v___y_1649_);
lean_dec_ref(v___y_1648_);
lean_dec(v___y_1647_);
lean_dec_ref(v___y_1646_);
return v_res_1652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1(lean_object* v_00_u03b1_1653_, lean_object* v_type_1654_, lean_object* v_k_1655_, uint8_t v_cleanupAnnotations_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_){
_start:
{
lean_object* v___x_1662_; 
v___x_1662_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(v_type_1654_, v_k_1655_, v_cleanupAnnotations_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_);
return v___x_1662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___boxed(lean_object* v_00_u03b1_1663_, lean_object* v_type_1664_, lean_object* v_k_1665_, lean_object* v_cleanupAnnotations_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1672_; lean_object* v_res_1673_; 
v_cleanupAnnotations_boxed_1672_ = lean_unbox(v_cleanupAnnotations_1666_);
v_res_1673_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1(v_00_u03b1_1663_, v_type_1664_, v_k_1665_, v_cleanupAnnotations_boxed_1672_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_);
lean_dec(v___y_1670_);
lean_dec_ref(v___y_1669_);
lean_dec(v___y_1668_);
lean_dec_ref(v___y_1667_);
return v_res_1673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0(lean_object* v_iv_1674_, lean_object* v_rv_1675_, lean_object* v_xs_1676_, lean_object* v_a_1677_, lean_object* v_a_1678_){
_start:
{
if (lean_obj_tag(v_a_1677_) == 0)
{
lean_object* v___x_1679_; 
lean_dec_ref(v_iv_1674_);
v___x_1679_ = l_List_reverse___redArg(v_a_1678_);
return v___x_1679_;
}
else
{
lean_object* v_toConstantVal_1680_; lean_object* v_head_1681_; lean_object* v_tail_1682_; lean_object* v___x_1684_; uint8_t v_isShared_1685_; uint8_t v_isSharedCheck_1695_; 
v_toConstantVal_1680_ = lean_ctor_get(v_iv_1674_, 0);
v_head_1681_ = lean_ctor_get(v_a_1677_, 0);
v_tail_1682_ = lean_ctor_get(v_a_1677_, 1);
v_isSharedCheck_1695_ = !lean_is_exclusive(v_a_1677_);
if (v_isSharedCheck_1695_ == 0)
{
v___x_1684_ = v_a_1677_;
v_isShared_1685_ = v_isSharedCheck_1695_;
goto v_resetjp_1683_;
}
else
{
lean_inc(v_tail_1682_);
lean_inc(v_head_1681_);
lean_dec(v_a_1677_);
v___x_1684_ = lean_box(0);
v_isShared_1685_ = v_isSharedCheck_1695_;
goto v_resetjp_1683_;
}
v_resetjp_1683_:
{
lean_object* v_name_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1692_; 
v_name_1686_ = lean_ctor_get(v_toConstantVal_1680_, 0);
v___x_1687_ = l_Lean_instInhabitedExpr;
v___x_1688_ = l_Lean_RecursorVal_getMajorIdx(v_rv_1675_);
v___x_1689_ = lean_array_get_borrowed(v___x_1687_, v_xs_1676_, v___x_1688_);
lean_dec(v___x_1688_);
lean_inc(v___x_1689_);
lean_inc(v_name_1686_);
v___x_1690_ = l_Lean_Expr_proj___override(v_name_1686_, v_head_1681_, v___x_1689_);
if (v_isShared_1685_ == 0)
{
lean_ctor_set(v___x_1684_, 1, v_a_1678_);
lean_ctor_set(v___x_1684_, 0, v___x_1690_);
v___x_1692_ = v___x_1684_;
goto v_reusejp_1691_;
}
else
{
lean_object* v_reuseFailAlloc_1694_; 
v_reuseFailAlloc_1694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1694_, 0, v___x_1690_);
lean_ctor_set(v_reuseFailAlloc_1694_, 1, v_a_1678_);
v___x_1692_ = v_reuseFailAlloc_1694_;
goto v_reusejp_1691_;
}
v_reusejp_1691_:
{
v_a_1677_ = v_tail_1682_;
v_a_1678_ = v___x_1692_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0___boxed(lean_object* v_iv_1696_, lean_object* v_rv_1697_, lean_object* v_xs_1698_, lean_object* v_a_1699_, lean_object* v_a_1700_){
_start:
{
lean_object* v_res_1701_; 
v_res_1701_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0(v_iv_1696_, v_rv_1697_, v_xs_1698_, v_a_1699_, v_a_1700_);
lean_dec_ref(v_xs_1698_);
lean_dec_ref(v_rv_1697_);
return v_res_1701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0(lean_object* v___x_1702_, lean_object* v_rules_1703_, lean_object* v_rv_1704_, lean_object* v___x_1705_, lean_object* v_iv_1706_, lean_object* v_xs_1707_, lean_object* v_x_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_){
_start:
{
lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v_nfields_1716_; lean_object* v___x_1717_; lean_object* v_val_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v_val_1723_; uint8_t v___x_1724_; uint8_t v___x_1725_; uint8_t v___x_1726_; lean_object* v___x_1727_; 
v___x_1714_ = lean_unsigned_to_nat(0u);
v___x_1715_ = l_List_get_x21Internal___redArg(v___x_1702_, v_rules_1703_, v___x_1714_);
v_nfields_1716_ = lean_ctor_get(v___x_1715_, 1);
lean_inc(v_nfields_1716_);
lean_dec(v___x_1715_);
v___x_1717_ = l_Lean_RecursorVal_getFirstMinorIdx(v_rv_1704_);
v_val_1718_ = lean_array_get_borrowed(v___x_1705_, v_xs_1707_, v___x_1717_);
lean_dec(v___x_1717_);
v___x_1719_ = l_List_range(v_nfields_1716_);
v___x_1720_ = lean_box(0);
v___x_1721_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__0(v_iv_1706_, v_rv_1704_, v_xs_1707_, v___x_1719_, v___x_1720_);
v___x_1722_ = lean_array_mk(v___x_1721_);
lean_inc(v_val_1718_);
v_val_1723_ = l_Lean_mkAppN(v_val_1718_, v___x_1722_);
lean_dec_ref(v___x_1722_);
v___x_1724_ = 0;
v___x_1725_ = 1;
v___x_1726_ = 1;
v___x_1727_ = l_Lean_Meta_mkLambdaFVars(v_xs_1707_, v_val_1723_, v___x_1724_, v___x_1725_, v___x_1724_, v___x_1725_, v___x_1726_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0___boxed(lean_object* v___x_1728_, lean_object* v_rules_1729_, lean_object* v_rv_1730_, lean_object* v___x_1731_, lean_object* v_iv_1732_, lean_object* v_xs_1733_, lean_object* v_x_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_){
_start:
{
lean_object* v_res_1740_; 
v_res_1740_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0(v___x_1728_, v_rules_1729_, v_rv_1730_, v___x_1731_, v_iv_1732_, v_xs_1733_, v_x_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_);
lean_dec(v___y_1738_);
lean_dec_ref(v___y_1737_);
lean_dec(v___y_1736_);
lean_dec_ref(v___y_1735_);
lean_dec_ref(v_x_1734_);
lean_dec_ref(v_xs_1733_);
lean_dec_ref(v___x_1731_);
lean_dec_ref(v_rv_1730_);
lean_dec(v_rules_1729_);
lean_dec_ref(v___x_1728_);
return v_res_1740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly(lean_object* v_iv_1741_, lean_object* v_rv_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_){
_start:
{
lean_object* v_toConstantVal_1748_; lean_object* v_rules_1749_; lean_object* v_type_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___f_1753_; uint8_t v___x_1754_; lean_object* v___x_1755_; 
v_toConstantVal_1748_ = lean_ctor_get(v_rv_1742_, 0);
v_rules_1749_ = lean_ctor_get(v_rv_1742_, 6);
v_type_1750_ = lean_ctor_get(v_toConstantVal_1748_, 2);
v___x_1751_ = l_Lean_instInhabitedExpr;
v___x_1752_ = l_Lean_instInhabitedRecursorRule_default;
lean_inc_ref(v_iv_1741_);
lean_inc_ref(v_rv_1742_);
lean_inc(v_rules_1749_);
v___f_1753_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___lam__0___boxed), 12, 5);
lean_closure_set(v___f_1753_, 0, v___x_1752_);
lean_closure_set(v___f_1753_, 1, v_rules_1749_);
lean_closure_set(v___f_1753_, 2, v_rv_1742_);
lean_closure_set(v___f_1753_, 3, v___x_1751_);
lean_closure_set(v___f_1753_, 4, v_iv_1741_);
v___x_1754_ = 0;
lean_inc_ref(v_type_1750_);
v___x_1755_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(v_type_1750_, v___f_1753_, v___x_1754_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1755_) == 0)
{
lean_object* v_a_1756_; lean_object* v___x_1757_; 
v_a_1756_ = lean_ctor_get(v___x_1755_, 0);
lean_inc(v_a_1756_);
lean_dec_ref_known(v___x_1755_, 1);
v___x_1757_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go(v_iv_1741_, v_rv_1742_, v_a_1756_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
return v___x_1757_;
}
else
{
lean_object* v_a_1758_; lean_object* v___x_1760_; uint8_t v_isShared_1761_; uint8_t v_isSharedCheck_1765_; 
lean_dec_ref(v_rv_1742_);
lean_dec_ref(v_iv_1741_);
v_a_1758_ = lean_ctor_get(v___x_1755_, 0);
v_isSharedCheck_1765_ = !lean_is_exclusive(v___x_1755_);
if (v_isSharedCheck_1765_ == 0)
{
v___x_1760_ = v___x_1755_;
v_isShared_1761_ = v_isSharedCheck_1765_;
goto v_resetjp_1759_;
}
else
{
lean_inc(v_a_1758_);
lean_dec(v___x_1755_);
v___x_1760_ = lean_box(0);
v_isShared_1761_ = v_isSharedCheck_1765_;
goto v_resetjp_1759_;
}
v_resetjp_1759_:
{
lean_object* v___x_1763_; 
if (v_isShared_1761_ == 0)
{
v___x_1763_ = v___x_1760_;
goto v_reusejp_1762_;
}
else
{
lean_object* v_reuseFailAlloc_1764_; 
v_reuseFailAlloc_1764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1764_, 0, v_a_1758_);
v___x_1763_ = v_reuseFailAlloc_1764_;
goto v_reusejp_1762_;
}
v_reusejp_1762_:
{
return v___x_1763_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly___boxed(lean_object* v_iv_1766_, lean_object* v_rv_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_, lean_object* v_a_1770_, lean_object* v_a_1771_, lean_object* v_a_1772_){
_start:
{
lean_object* v_res_1773_; 
v_res_1773_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly(v_iv_1766_, v_rv_1767_, v_a_1768_, v_a_1769_, v_a_1770_, v_a_1771_);
lean_dec(v_a_1771_);
lean_dec_ref(v_a_1770_);
lean_dec(v_a_1769_);
lean_dec_ref(v_a_1768_);
return v_res_1773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg(lean_object* v___x_1774_, uint8_t v___y_1775_, lean_object* v_as_x27_1776_, lean_object* v_b_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_){
_start:
{
if (lean_obj_tag(v_as_x27_1776_) == 0)
{
lean_object* v___x_1783_; 
lean_dec(v___x_1774_);
v___x_1783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1783_, 0, v_b_1777_);
return v___x_1783_;
}
else
{
lean_object* v_head_1784_; lean_object* v_fst_1785_; lean_object* v_toConstantVal_1786_; lean_object* v_tail_1787_; lean_object* v_snd_1788_; lean_object* v_name_1789_; lean_object* v_levelParams_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; 
v_head_1784_ = lean_ctor_get(v_as_x27_1776_, 0);
v_fst_1785_ = lean_ctor_get(v_head_1784_, 0);
v_toConstantVal_1786_ = lean_ctor_get(v_fst_1785_, 0);
v_tail_1787_ = lean_ctor_get(v_as_x27_1776_, 1);
v_snd_1788_ = lean_ctor_get(v_head_1784_, 1);
v_name_1789_ = lean_ctor_get(v_toConstantVal_1786_, 0);
v_levelParams_1790_ = lean_ctor_get(v_toConstantVal_1786_, 1);
lean_inc(v___x_1774_);
lean_inc_n(v_name_1789_, 2);
v___x_1791_ = l_Lean_Expr_const___override(v_name_1789_, v___x_1774_);
v___x_1792_ = ((lean_object*)(lp_mathlib_Mathlib_Util_compileDefn___closed__0));
v___x_1793_ = l_Lean_Name_str___override(v_name_1789_, v___x_1792_);
v___x_1794_ = l_Lean_Core_mkFreshUserName(v___x_1793_, v___y_1780_, v___y_1781_);
if (lean_obj_tag(v___x_1794_) == 0)
{
lean_object* v_a_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; 
v_a_1795_ = lean_ctor_get(v___x_1794_, 0);
lean_inc(v_a_1795_);
lean_dec_ref_known(v___x_1794_, 1);
lean_inc(v___x_1774_);
lean_inc(v_snd_1788_);
v___x_1796_ = l_Lean_Expr_const___override(v_snd_1788_, v___x_1774_);
v___x_1797_ = l_Lean_Meta_mkEq(v___x_1791_, v___x_1796_, v___y_1778_, v___y_1779_, v___y_1780_, v___y_1781_);
if (lean_obj_tag(v___x_1797_) == 0)
{
lean_object* v_a_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; uint8_t v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; 
v_a_1798_ = lean_ctor_get(v___x_1797_, 0);
lean_inc(v_a_1798_);
lean_dec_ref_known(v___x_1797_, 1);
lean_inc(v_levelParams_1790_);
lean_inc_n(v_a_1795_, 3);
v___x_1799_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1799_, 0, v_a_1795_);
lean_ctor_set(v___x_1799_, 1, v_levelParams_1790_);
lean_ctor_set(v___x_1799_, 2, v_a_1798_);
lean_inc(v___x_1774_);
v___x_1800_ = l_Lean_Expr_const___override(v_a_1795_, v___x_1774_);
v___x_1801_ = lean_box(0);
v___x_1802_ = 2;
v___x_1803_ = lean_box(0);
v___x_1804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1804_, 0, v_a_1795_);
lean_ctor_set(v___x_1804_, 1, v___x_1803_);
v___x_1805_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_1805_, 0, v___x_1799_);
lean_ctor_set(v___x_1805_, 1, v___x_1800_);
lean_ctor_set(v___x_1805_, 2, v___x_1801_);
lean_ctor_set(v___x_1805_, 3, v___x_1804_);
lean_ctor_set_uint8(v___x_1805_, sizeof(void*)*4, v___x_1802_);
v___x_1806_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1806_, 0, v___x_1805_);
lean_ctor_set(v___x_1806_, 1, v___x_1803_);
v___x_1807_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v___x_1807_, 0, v___x_1806_);
v___x_1808_ = l_Lean_addDecl(v___x_1807_, v___y_1775_, v___y_1780_, v___y_1781_);
if (lean_obj_tag(v___x_1808_) == 0)
{
uint8_t v___x_1809_; lean_object* v___x_1810_; 
lean_dec_ref_known(v___x_1808_, 1);
v___x_1809_ = 0;
v___x_1810_ = l_Lean_Compiler_CSimp_add(v_a_1795_, v___x_1809_, v___y_1780_, v___y_1781_);
if (lean_obj_tag(v___x_1810_) == 0)
{
lean_object* v___x_1811_; 
lean_dec_ref_known(v___x_1810_, 1);
v___x_1811_ = lean_box(0);
v_as_x27_1776_ = v_tail_1787_;
v_b_1777_ = v___x_1811_;
goto _start;
}
else
{
lean_dec(v___x_1774_);
return v___x_1810_;
}
}
else
{
lean_dec(v_a_1795_);
lean_dec(v___x_1774_);
return v___x_1808_;
}
}
else
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
lean_dec(v_a_1795_);
lean_dec(v___x_1774_);
v_a_1813_ = lean_ctor_get(v___x_1797_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1797_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1797_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1797_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
return v___x_1818_;
}
}
}
}
else
{
lean_object* v_a_1821_; lean_object* v___x_1823_; uint8_t v_isShared_1824_; uint8_t v_isSharedCheck_1828_; 
lean_dec_ref(v___x_1791_);
lean_dec(v___x_1774_);
v_a_1821_ = lean_ctor_get(v___x_1794_, 0);
v_isSharedCheck_1828_ = !lean_is_exclusive(v___x_1794_);
if (v_isSharedCheck_1828_ == 0)
{
v___x_1823_ = v___x_1794_;
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
else
{
lean_inc(v_a_1821_);
lean_dec(v___x_1794_);
v___x_1823_ = lean_box(0);
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
v_resetjp_1822_:
{
lean_object* v___x_1826_; 
if (v_isShared_1824_ == 0)
{
v___x_1826_ = v___x_1823_;
goto v_reusejp_1825_;
}
else
{
lean_object* v_reuseFailAlloc_1827_; 
v_reuseFailAlloc_1827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1827_, 0, v_a_1821_);
v___x_1826_ = v_reuseFailAlloc_1827_;
goto v_reusejp_1825_;
}
v_reusejp_1825_:
{
return v___x_1826_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg___boxed(lean_object* v___x_1829_, lean_object* v___y_1830_, lean_object* v_as_x27_1831_, lean_object* v_b_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_, lean_object* v___y_1837_){
_start:
{
uint8_t v___y_15665__boxed_1838_; lean_object* v_res_1839_; 
v___y_15665__boxed_1838_ = lean_unbox(v___y_1830_);
v_res_1839_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg(v___x_1829_, v___y_15665__boxed_1838_, v_as_x27_1831_, v_b_1832_, v___y_1833_, v___y_1834_, v___y_1835_, v___y_1836_);
lean_dec(v___y_1836_);
lean_dec_ref(v___y_1835_);
lean_dec(v___y_1834_);
lean_dec_ref(v___y_1833_);
lean_dec(v_as_x27_1831_);
return v_res_1839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3(lean_object* v_msg_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_){
_start:
{
lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v_toApplicative_1848_; lean_object* v___x_1850_; uint8_t v_isShared_1851_; uint8_t v_isSharedCheck_1909_; 
v___x_1846_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0, &lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__0);
v___x_1847_ = l_StateRefT_x27_instMonad___redArg(v___x_1846_);
v_toApplicative_1848_ = lean_ctor_get(v___x_1847_, 0);
v_isSharedCheck_1909_ = !lean_is_exclusive(v___x_1847_);
if (v_isSharedCheck_1909_ == 0)
{
lean_object* v_unused_1910_; 
v_unused_1910_ = lean_ctor_get(v___x_1847_, 1);
lean_dec(v_unused_1910_);
v___x_1850_ = v___x_1847_;
v_isShared_1851_ = v_isSharedCheck_1909_;
goto v_resetjp_1849_;
}
else
{
lean_inc(v_toApplicative_1848_);
lean_dec(v___x_1847_);
v___x_1850_ = lean_box(0);
v_isShared_1851_ = v_isSharedCheck_1909_;
goto v_resetjp_1849_;
}
v_resetjp_1849_:
{
lean_object* v_toFunctor_1852_; lean_object* v_toSeq_1853_; lean_object* v_toSeqLeft_1854_; lean_object* v_toSeqRight_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1907_; 
v_toFunctor_1852_ = lean_ctor_get(v_toApplicative_1848_, 0);
v_toSeq_1853_ = lean_ctor_get(v_toApplicative_1848_, 2);
v_toSeqLeft_1854_ = lean_ctor_get(v_toApplicative_1848_, 3);
v_toSeqRight_1855_ = lean_ctor_get(v_toApplicative_1848_, 4);
v_isSharedCheck_1907_ = !lean_is_exclusive(v_toApplicative_1848_);
if (v_isSharedCheck_1907_ == 0)
{
lean_object* v_unused_1908_; 
v_unused_1908_ = lean_ctor_get(v_toApplicative_1848_, 1);
lean_dec(v_unused_1908_);
v___x_1857_ = v_toApplicative_1848_;
v_isShared_1858_ = v_isSharedCheck_1907_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_toSeqRight_1855_);
lean_inc(v_toSeqLeft_1854_);
lean_inc(v_toSeq_1853_);
lean_inc(v_toFunctor_1852_);
lean_dec(v_toApplicative_1848_);
v___x_1857_ = lean_box(0);
v_isShared_1858_ = v_isSharedCheck_1907_;
goto v_resetjp_1856_;
}
v_resetjp_1856_:
{
lean_object* v___f_1859_; lean_object* v___f_1860_; lean_object* v___f_1861_; lean_object* v___f_1862_; lean_object* v___x_1863_; lean_object* v___f_1864_; lean_object* v___f_1865_; lean_object* v___f_1866_; lean_object* v___x_1868_; 
v___f_1859_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__1));
v___f_1860_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__2));
lean_inc_ref(v_toFunctor_1852_);
v___f_1861_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1861_, 0, v_toFunctor_1852_);
v___f_1862_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1862_, 0, v_toFunctor_1852_);
v___x_1863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1863_, 0, v___f_1861_);
lean_ctor_set(v___x_1863_, 1, v___f_1862_);
v___f_1864_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1864_, 0, v_toSeqRight_1855_);
v___f_1865_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1865_, 0, v_toSeqLeft_1854_);
v___f_1866_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1866_, 0, v_toSeq_1853_);
if (v_isShared_1858_ == 0)
{
lean_ctor_set(v___x_1857_, 4, v___f_1864_);
lean_ctor_set(v___x_1857_, 3, v___f_1865_);
lean_ctor_set(v___x_1857_, 2, v___f_1866_);
lean_ctor_set(v___x_1857_, 1, v___f_1859_);
lean_ctor_set(v___x_1857_, 0, v___x_1863_);
v___x_1868_ = v___x_1857_;
goto v_reusejp_1867_;
}
else
{
lean_object* v_reuseFailAlloc_1906_; 
v_reuseFailAlloc_1906_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1906_, 0, v___x_1863_);
lean_ctor_set(v_reuseFailAlloc_1906_, 1, v___f_1859_);
lean_ctor_set(v_reuseFailAlloc_1906_, 2, v___f_1866_);
lean_ctor_set(v_reuseFailAlloc_1906_, 3, v___f_1865_);
lean_ctor_set(v_reuseFailAlloc_1906_, 4, v___f_1864_);
v___x_1868_ = v_reuseFailAlloc_1906_;
goto v_reusejp_1867_;
}
v_reusejp_1867_:
{
lean_object* v___x_1870_; 
if (v_isShared_1851_ == 0)
{
lean_ctor_set(v___x_1850_, 1, v___f_1860_);
lean_ctor_set(v___x_1850_, 0, v___x_1868_);
v___x_1870_ = v___x_1850_;
goto v_reusejp_1869_;
}
else
{
lean_object* v_reuseFailAlloc_1905_; 
v_reuseFailAlloc_1905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1905_, 0, v___x_1868_);
lean_ctor_set(v_reuseFailAlloc_1905_, 1, v___f_1860_);
v___x_1870_ = v_reuseFailAlloc_1905_;
goto v_reusejp_1869_;
}
v_reusejp_1869_:
{
lean_object* v___x_1871_; lean_object* v_toApplicative_1872_; lean_object* v___x_1874_; uint8_t v_isShared_1875_; uint8_t v_isSharedCheck_1903_; 
v___x_1871_ = l_StateRefT_x27_instMonad___redArg(v___x_1870_);
v_toApplicative_1872_ = lean_ctor_get(v___x_1871_, 0);
v_isSharedCheck_1903_ = !lean_is_exclusive(v___x_1871_);
if (v_isSharedCheck_1903_ == 0)
{
lean_object* v_unused_1904_; 
v_unused_1904_ = lean_ctor_get(v___x_1871_, 1);
lean_dec(v_unused_1904_);
v___x_1874_ = v___x_1871_;
v_isShared_1875_ = v_isSharedCheck_1903_;
goto v_resetjp_1873_;
}
else
{
lean_inc(v_toApplicative_1872_);
lean_dec(v___x_1871_);
v___x_1874_ = lean_box(0);
v_isShared_1875_ = v_isSharedCheck_1903_;
goto v_resetjp_1873_;
}
v_resetjp_1873_:
{
lean_object* v_toFunctor_1876_; lean_object* v_toSeq_1877_; lean_object* v_toSeqLeft_1878_; lean_object* v_toSeqRight_1879_; lean_object* v___x_1881_; uint8_t v_isShared_1882_; uint8_t v_isSharedCheck_1901_; 
v_toFunctor_1876_ = lean_ctor_get(v_toApplicative_1872_, 0);
v_toSeq_1877_ = lean_ctor_get(v_toApplicative_1872_, 2);
v_toSeqLeft_1878_ = lean_ctor_get(v_toApplicative_1872_, 3);
v_toSeqRight_1879_ = lean_ctor_get(v_toApplicative_1872_, 4);
v_isSharedCheck_1901_ = !lean_is_exclusive(v_toApplicative_1872_);
if (v_isSharedCheck_1901_ == 0)
{
lean_object* v_unused_1902_; 
v_unused_1902_ = lean_ctor_get(v_toApplicative_1872_, 1);
lean_dec(v_unused_1902_);
v___x_1881_ = v_toApplicative_1872_;
v_isShared_1882_ = v_isSharedCheck_1901_;
goto v_resetjp_1880_;
}
else
{
lean_inc(v_toSeqRight_1879_);
lean_inc(v_toSeqLeft_1878_);
lean_inc(v_toSeq_1877_);
lean_inc(v_toFunctor_1876_);
lean_dec(v_toApplicative_1872_);
v___x_1881_ = lean_box(0);
v_isShared_1882_ = v_isSharedCheck_1901_;
goto v_resetjp_1880_;
}
v_resetjp_1880_:
{
lean_object* v___f_1883_; lean_object* v___f_1884_; lean_object* v___f_1885_; lean_object* v___f_1886_; lean_object* v___x_1887_; lean_object* v___f_1888_; lean_object* v___f_1889_; lean_object* v___f_1890_; lean_object* v___x_1892_; 
v___f_1883_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__3));
v___f_1884_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__2___closed__4));
lean_inc_ref(v_toFunctor_1876_);
v___f_1885_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1885_, 0, v_toFunctor_1876_);
v___f_1886_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1886_, 0, v_toFunctor_1876_);
v___x_1887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1887_, 0, v___f_1885_);
lean_ctor_set(v___x_1887_, 1, v___f_1886_);
v___f_1888_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1888_, 0, v_toSeqRight_1879_);
v___f_1889_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1889_, 0, v_toSeqLeft_1878_);
v___f_1890_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1890_, 0, v_toSeq_1877_);
if (v_isShared_1882_ == 0)
{
lean_ctor_set(v___x_1881_, 4, v___f_1888_);
lean_ctor_set(v___x_1881_, 3, v___f_1889_);
lean_ctor_set(v___x_1881_, 2, v___f_1890_);
lean_ctor_set(v___x_1881_, 1, v___f_1883_);
lean_ctor_set(v___x_1881_, 0, v___x_1887_);
v___x_1892_ = v___x_1881_;
goto v_reusejp_1891_;
}
else
{
lean_object* v_reuseFailAlloc_1900_; 
v_reuseFailAlloc_1900_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1900_, 0, v___x_1887_);
lean_ctor_set(v_reuseFailAlloc_1900_, 1, v___f_1883_);
lean_ctor_set(v_reuseFailAlloc_1900_, 2, v___f_1890_);
lean_ctor_set(v_reuseFailAlloc_1900_, 3, v___f_1889_);
lean_ctor_set(v_reuseFailAlloc_1900_, 4, v___f_1888_);
v___x_1892_ = v_reuseFailAlloc_1900_;
goto v_reusejp_1891_;
}
v_reusejp_1891_:
{
lean_object* v___x_1894_; 
if (v_isShared_1875_ == 0)
{
lean_ctor_set(v___x_1874_, 1, v___f_1884_);
lean_ctor_set(v___x_1874_, 0, v___x_1892_);
v___x_1894_ = v___x_1874_;
goto v_reusejp_1893_;
}
else
{
lean_object* v_reuseFailAlloc_1899_; 
v_reuseFailAlloc_1899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1899_, 0, v___x_1892_);
lean_ctor_set(v_reuseFailAlloc_1899_, 1, v___f_1884_);
v___x_1894_ = v_reuseFailAlloc_1899_;
goto v_reusejp_1893_;
}
v_reusejp_1893_:
{
lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_12528__overap_1897_; lean_object* v___x_1898_; 
v___x_1895_ = lean_box(0);
v___x_1896_ = l_instInhabitedOfMonad___redArg(v___x_1894_, v___x_1895_);
v___x_12528__overap_1897_ = lean_panic_fn_borrowed(v___x_1896_, v_msg_1840_);
lean_dec(v___x_1896_);
lean_inc(v___y_1844_);
lean_inc_ref(v___y_1843_);
lean_inc(v___y_1842_);
lean_inc_ref(v___y_1841_);
v___x_1898_ = lean_apply_5(v___x_12528__overap_1897_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_, lean_box(0));
return v___x_1898_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3___boxed(lean_object* v_msg_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
lean_object* v_res_1917_; 
v_res_1917_ = lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3(v_msg_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
lean_dec(v___y_1913_);
lean_dec_ref(v___y_1912_);
return v_res_1917_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1919_; lean_object* v___x_1920_; 
v___x_1919_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__0));
v___x_1920_ = l_Lean_stringToMessageData(v___x_1919_);
return v___x_1920_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3(void){
_start:
{
lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; 
v___x_1922_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27___closed__6));
v___x_1923_ = lean_unsigned_to_nat(11u);
v___x_1924_ = lean_unsigned_to_nat(129u);
v___x_1925_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__2));
v___x_1926_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__4));
v___x_1927_ = l_mkPanicMessageWithDecl(v___x_1926_, v___x_1925_, v___x_1924_, v___x_1923_, v___x_1922_);
return v___x_1927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(lean_object* v_constName_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_){
_start:
{
lean_object* v___x_1942_; lean_object* v_env_1943_; uint8_t v___x_1944_; lean_object* v___x_1945_; 
v___x_1942_ = lean_st_ref_get(v___y_1932_);
v_env_1943_ = lean_ctor_get(v___x_1942_, 0);
lean_inc_ref(v_env_1943_);
lean_dec(v___x_1942_);
v___x_1944_ = 0;
lean_inc(v_constName_1928_);
v___x_1945_ = l_Lean_Environment_findAsync_x3f(v_env_1943_, v_constName_1928_, v___x_1944_);
if (lean_obj_tag(v___x_1945_) == 1)
{
lean_object* v_val_1946_; uint8_t v_kind_1947_; 
v_val_1946_ = lean_ctor_get(v___x_1945_, 0);
lean_inc(v_val_1946_);
lean_dec_ref_known(v___x_1945_, 1);
v_kind_1947_ = lean_ctor_get_uint8(v_val_1946_, sizeof(void*)*3);
if (v_kind_1947_ == 7)
{
lean_object* v___x_1948_; 
v___x_1948_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_1946_);
if (lean_obj_tag(v___x_1948_) == 7)
{
lean_object* v_val_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1956_; 
lean_dec(v_constName_1928_);
v_val_1949_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1956_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1956_ == 0)
{
v___x_1951_ = v___x_1948_;
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_val_1949_);
lean_dec(v___x_1948_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v___x_1954_; 
if (v_isShared_1952_ == 0)
{
lean_ctor_set_tag(v___x_1951_, 0);
v___x_1954_ = v___x_1951_;
goto v_reusejp_1953_;
}
else
{
lean_object* v_reuseFailAlloc_1955_; 
v_reuseFailAlloc_1955_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1955_, 0, v_val_1949_);
v___x_1954_ = v_reuseFailAlloc_1955_;
goto v_reusejp_1953_;
}
v_reusejp_1953_:
{
return v___x_1954_;
}
}
}
else
{
lean_object* v___x_1957_; lean_object* v___x_1958_; 
lean_dec_ref(v___x_1948_);
v___x_1957_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3, &lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3_once, _init_lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__3);
v___x_1958_ = lp_mathlib_panic___at___00Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3_spec__3(v___x_1957_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
if (lean_obj_tag(v___x_1958_) == 0)
{
lean_object* v_a_1959_; lean_object* v___x_1961_; uint8_t v_isShared_1962_; uint8_t v_isSharedCheck_1967_; 
v_a_1959_ = lean_ctor_get(v___x_1958_, 0);
v_isSharedCheck_1967_ = !lean_is_exclusive(v___x_1958_);
if (v_isSharedCheck_1967_ == 0)
{
v___x_1961_ = v___x_1958_;
v_isShared_1962_ = v_isSharedCheck_1967_;
goto v_resetjp_1960_;
}
else
{
lean_inc(v_a_1959_);
lean_dec(v___x_1958_);
v___x_1961_ = lean_box(0);
v_isShared_1962_ = v_isSharedCheck_1967_;
goto v_resetjp_1960_;
}
v_resetjp_1960_:
{
if (lean_obj_tag(v_a_1959_) == 0)
{
lean_del_object(v___x_1961_);
goto v___jp_1934_;
}
else
{
lean_object* v_val_1963_; lean_object* v___x_1965_; 
lean_dec(v_constName_1928_);
v_val_1963_ = lean_ctor_get(v_a_1959_, 0);
lean_inc(v_val_1963_);
lean_dec_ref_known(v_a_1959_, 1);
if (v_isShared_1962_ == 0)
{
lean_ctor_set(v___x_1961_, 0, v_val_1963_);
v___x_1965_ = v___x_1961_;
goto v_reusejp_1964_;
}
else
{
lean_object* v_reuseFailAlloc_1966_; 
v_reuseFailAlloc_1966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1966_, 0, v_val_1963_);
v___x_1965_ = v_reuseFailAlloc_1966_;
goto v_reusejp_1964_;
}
v_reusejp_1964_:
{
return v___x_1965_;
}
}
}
}
else
{
lean_object* v_a_1968_; lean_object* v___x_1970_; uint8_t v_isShared_1971_; uint8_t v_isSharedCheck_1975_; 
lean_dec(v_constName_1928_);
v_a_1968_ = lean_ctor_get(v___x_1958_, 0);
v_isSharedCheck_1975_ = !lean_is_exclusive(v___x_1958_);
if (v_isSharedCheck_1975_ == 0)
{
v___x_1970_ = v___x_1958_;
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
else
{
lean_inc(v_a_1968_);
lean_dec(v___x_1958_);
v___x_1970_ = lean_box(0);
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
v_resetjp_1969_:
{
lean_object* v___x_1973_; 
if (v_isShared_1971_ == 0)
{
v___x_1973_ = v___x_1970_;
goto v_reusejp_1972_;
}
else
{
lean_object* v_reuseFailAlloc_1974_; 
v_reuseFailAlloc_1974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1974_, 0, v_a_1968_);
v___x_1973_ = v_reuseFailAlloc_1974_;
goto v_reusejp_1972_;
}
v_reusejp_1972_:
{
return v___x_1973_;
}
}
}
}
}
else
{
lean_dec(v_val_1946_);
goto v___jp_1934_;
}
}
else
{
lean_dec(v___x_1945_);
goto v___jp_1934_;
}
v___jp_1934_:
{
lean_object* v___x_1935_; uint8_t v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; 
v___x_1935_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1);
v___x_1936_ = 0;
v___x_1937_ = l_Lean_MessageData_ofConstName(v_constName_1928_, v___x_1936_);
v___x_1938_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1938_, 0, v___x_1935_);
lean_ctor_set(v___x_1938_, 1, v___x_1937_);
v___x_1939_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1, &lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1_once, _init_lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___closed__1);
v___x_1940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1940_, 0, v___x_1938_);
lean_ctor_set(v___x_1940_, 1, v___x_1939_);
v___x_1941_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v___x_1940_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
return v___x_1941_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3___boxed(lean_object* v_constName_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
lean_object* v_res_1982_; 
v_res_1982_ = lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(v_constName_1976_, v___y_1977_, v___y_1978_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec(v___y_1978_);
lean_dec_ref(v___y_1977_);
return v_res_1982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11(lean_object* v_x_1983_, lean_object* v_x_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_){
_start:
{
if (lean_obj_tag(v_x_1983_) == 0)
{
lean_object* v___x_1990_; lean_object* v___x_1991_; 
v___x_1990_ = l_List_reverse___redArg(v_x_1984_);
v___x_1991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1991_, 0, v___x_1990_);
return v___x_1991_;
}
else
{
lean_object* v_head_1992_; lean_object* v_tail_1993_; lean_object* v___x_1995_; uint8_t v_isShared_1996_; uint8_t v_isSharedCheck_2011_; 
v_head_1992_ = lean_ctor_get(v_x_1983_, 0);
v_tail_1993_ = lean_ctor_get(v_x_1983_, 1);
v_isSharedCheck_2011_ = !lean_is_exclusive(v_x_1983_);
if (v_isSharedCheck_2011_ == 0)
{
v___x_1995_ = v_x_1983_;
v_isShared_1996_ = v_isSharedCheck_2011_;
goto v_resetjp_1994_;
}
else
{
lean_inc(v_tail_1993_);
lean_inc(v_head_1992_);
lean_dec(v_x_1983_);
v___x_1995_ = lean_box(0);
v_isShared_1996_ = v_isSharedCheck_2011_;
goto v_resetjp_1994_;
}
v_resetjp_1994_:
{
lean_object* v___x_1997_; 
v___x_1997_ = lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(v_head_1992_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_1997_) == 0)
{
lean_object* v_a_1998_; lean_object* v___x_2000_; 
v_a_1998_ = lean_ctor_get(v___x_1997_, 0);
lean_inc(v_a_1998_);
lean_dec_ref_known(v___x_1997_, 1);
if (v_isShared_1996_ == 0)
{
lean_ctor_set(v___x_1995_, 1, v_x_1984_);
lean_ctor_set(v___x_1995_, 0, v_a_1998_);
v___x_2000_ = v___x_1995_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_a_1998_);
lean_ctor_set(v_reuseFailAlloc_2002_, 1, v_x_1984_);
v___x_2000_ = v_reuseFailAlloc_2002_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
v_x_1983_ = v_tail_1993_;
v_x_1984_ = v___x_2000_;
goto _start;
}
}
else
{
lean_object* v_a_2003_; lean_object* v___x_2005_; uint8_t v_isShared_2006_; uint8_t v_isSharedCheck_2010_; 
lean_del_object(v___x_1995_);
lean_dec(v_tail_1993_);
lean_dec(v_x_1984_);
v_a_2003_ = lean_ctor_get(v___x_1997_, 0);
v_isSharedCheck_2010_ = !lean_is_exclusive(v___x_1997_);
if (v_isSharedCheck_2010_ == 0)
{
v___x_2005_ = v___x_1997_;
v_isShared_2006_ = v_isSharedCheck_2010_;
goto v_resetjp_2004_;
}
else
{
lean_inc(v_a_2003_);
lean_dec(v___x_1997_);
v___x_2005_ = lean_box(0);
v_isShared_2006_ = v_isSharedCheck_2010_;
goto v_resetjp_2004_;
}
v_resetjp_2004_:
{
lean_object* v___x_2008_; 
if (v_isShared_2006_ == 0)
{
v___x_2008_ = v___x_2005_;
goto v_reusejp_2007_;
}
else
{
lean_object* v_reuseFailAlloc_2009_; 
v_reuseFailAlloc_2009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2009_, 0, v_a_2003_);
v___x_2008_ = v_reuseFailAlloc_2009_;
goto v_reusejp_2007_;
}
v_reusejp_2007_:
{
return v___x_2008_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11___boxed(lean_object* v_x_2012_, lean_object* v_x_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_){
_start:
{
lean_object* v_res_2019_; 
v_res_2019_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11(v_x_2012_, v_x_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_);
lean_dec(v___y_2017_);
lean_dec_ref(v___y_2016_);
lean_dec(v___y_2015_);
lean_dec_ref(v___y_2014_);
return v_res_2019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg(uint8_t v___y_2020_, lean_object* v_as_x27_2021_, lean_object* v_b_2022_, lean_object* v___y_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_){
_start:
{
if (lean_obj_tag(v_as_x27_2021_) == 0)
{
lean_object* v___x_2028_; 
v___x_2028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2028_, 0, v_b_2022_);
return v___x_2028_;
}
else
{
lean_object* v_head_2029_; lean_object* v_tail_2030_; lean_object* v___x_2031_; lean_object* v_env_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; 
v_head_2029_ = lean_ctor_get(v_as_x27_2021_, 0);
v_tail_2030_ = lean_ctor_get(v_as_x27_2021_, 1);
v___x_2031_ = lean_st_ref_get(v___y_2026_);
v_env_2032_ = lean_ctor_get(v___x_2031_, 0);
lean_inc_ref(v_env_2032_);
lean_dec(v___x_2031_);
v___x_2033_ = lean_box(0);
lean_inc(v_head_2029_);
v___x_2034_ = l_Lean_Environment_find_x3f(v_env_2032_, v_head_2029_, v___y_2020_);
if (lean_obj_tag(v___x_2034_) == 1)
{
lean_object* v_val_2035_; 
v_val_2035_ = lean_ctor_get(v___x_2034_, 0);
lean_inc(v_val_2035_);
lean_dec_ref_known(v___x_2034_, 1);
if (lean_obj_tag(v_val_2035_) == 1)
{
lean_object* v_val_2036_; lean_object* v___x_2037_; 
v_val_2036_ = lean_ctor_get(v_val_2035_, 0);
lean_inc_ref(v_val_2036_);
lean_dec_ref_known(v_val_2035_, 1);
v___x_2037_ = lp_mathlib_Mathlib_Util_compileDefn(v_val_2036_, v___y_2023_, v___y_2024_, v___y_2025_, v___y_2026_);
if (lean_obj_tag(v___x_2037_) == 0)
{
lean_dec_ref_known(v___x_2037_, 1);
v_as_x27_2021_ = v_tail_2030_;
v_b_2022_ = v___x_2033_;
goto _start;
}
else
{
return v___x_2037_;
}
}
else
{
lean_dec(v_val_2035_);
v_as_x27_2021_ = v_tail_2030_;
v_b_2022_ = v___x_2033_;
goto _start;
}
}
else
{
lean_dec(v___x_2034_);
v_as_x27_2021_ = v_tail_2030_;
v_b_2022_ = v___x_2033_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg___boxed(lean_object* v___y_2041_, lean_object* v_as_x27_2042_, lean_object* v_b_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_){
_start:
{
uint8_t v___y_16100__boxed_2049_; lean_object* v_res_2050_; 
v___y_16100__boxed_2049_ = lean_unbox(v___y_2041_);
v_res_2050_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg(v___y_16100__boxed_2049_, v_as_x27_2042_, v_b_2043_, v___y_2044_, v___y_2045_, v___y_2046_, v___y_2047_);
lean_dec(v___y_2047_);
lean_dec_ref(v___y_2046_);
lean_dec(v___y_2045_);
lean_dec_ref(v___y_2044_);
lean_dec(v_as_x27_2042_);
return v_res_2050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg(uint8_t v___y_2052_, lean_object* v_as_x27_2053_, lean_object* v_b_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_){
_start:
{
if (lean_obj_tag(v_as_x27_2053_) == 0)
{
lean_object* v___x_2060_; 
v___x_2060_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2060_, 0, v_b_2054_);
return v___x_2060_;
}
else
{
lean_object* v_head_2061_; lean_object* v_tail_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; 
v_head_2061_ = lean_ctor_get(v_as_x27_2053_, 0);
v_tail_2062_ = lean_ctor_get(v_as_x27_2053_, 1);
v___x_2063_ = lean_box(0);
lean_inc_n(v_head_2061_, 2);
v___x_2064_ = l_Lean_mkRecOnName(v_head_2061_);
v___x_2065_ = l_Lean_mkBRecOnName(v_head_2061_);
v___x_2066_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___closed__0));
lean_inc(v___x_2065_);
v___x_2067_ = l_Lean_Name_str___override(v___x_2065_, v___x_2066_);
v___x_2068_ = lean_box(0);
v___x_2069_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2069_, 0, v___x_2065_);
lean_ctor_set(v___x_2069_, 1, v___x_2068_);
v___x_2070_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2070_, 0, v___x_2067_);
lean_ctor_set(v___x_2070_, 1, v___x_2069_);
v___x_2071_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2071_, 0, v___x_2064_);
lean_ctor_set(v___x_2071_, 1, v___x_2070_);
v___x_2072_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg(v___y_2052_, v___x_2071_, v___x_2063_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_);
lean_dec_ref_known(v___x_2071_, 2);
if (lean_obj_tag(v___x_2072_) == 0)
{
lean_dec_ref_known(v___x_2072_, 1);
v_as_x27_2053_ = v_tail_2062_;
v_b_2054_ = v___x_2063_;
goto _start;
}
else
{
return v___x_2072_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg___boxed(lean_object* v___y_2074_, lean_object* v_as_x27_2075_, lean_object* v_b_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_){
_start:
{
uint8_t v___y_16146__boxed_2082_; lean_object* v_res_2083_; 
v___y_16146__boxed_2082_ = lean_unbox(v___y_2074_);
v_res_2083_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg(v___y_16146__boxed_2082_, v_as_x27_2075_, v_b_2076_, v___y_2077_, v___y_2078_, v___y_2079_, v___y_2080_);
lean_dec(v___y_2080_);
lean_dec_ref(v___y_2079_);
lean_dec(v___y_2078_);
lean_dec_ref(v___y_2077_);
lean_dec(v_as_x27_2075_);
return v_res_2083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Util_compileInductiveOnly_spec__1(lean_object* v_x_2084_, lean_object* v_x_2085_){
_start:
{
if (lean_obj_tag(v_x_2085_) == 0)
{
return v_x_2084_;
}
else
{
lean_object* v_head_2086_; lean_object* v_fst_2087_; lean_object* v_toConstantVal_2088_; lean_object* v_tail_2089_; lean_object* v_snd_2090_; lean_object* v_name_2091_; lean_object* v___x_2093_; uint8_t v_isShared_2094_; uint8_t v_isSharedCheck_2099_; 
v_head_2086_ = lean_ctor_get(v_x_2085_, 0);
lean_inc(v_head_2086_);
v_fst_2087_ = lean_ctor_get(v_head_2086_, 0);
v_toConstantVal_2088_ = lean_ctor_get(v_fst_2087_, 0);
lean_inc_ref(v_toConstantVal_2088_);
v_tail_2089_ = lean_ctor_get(v_x_2085_, 1);
lean_inc(v_tail_2089_);
lean_dec_ref_known(v_x_2085_, 2);
v_snd_2090_ = lean_ctor_get(v_head_2086_, 1);
lean_inc(v_snd_2090_);
lean_dec(v_head_2086_);
v_name_2091_ = lean_ctor_get(v_toConstantVal_2088_, 0);
v_isSharedCheck_2099_ = !lean_is_exclusive(v_toConstantVal_2088_);
if (v_isSharedCheck_2099_ == 0)
{
lean_object* v_unused_2100_; lean_object* v_unused_2101_; 
v_unused_2100_ = lean_ctor_get(v_toConstantVal_2088_, 2);
lean_dec(v_unused_2100_);
v_unused_2101_ = lean_ctor_get(v_toConstantVal_2088_, 1);
lean_dec(v_unused_2101_);
v___x_2093_ = v_toConstantVal_2088_;
v_isShared_2094_ = v_isSharedCheck_2099_;
goto v_resetjp_2092_;
}
else
{
lean_inc(v_name_2091_);
lean_dec(v_toConstantVal_2088_);
v___x_2093_ = lean_box(0);
v_isShared_2094_ = v_isSharedCheck_2099_;
goto v_resetjp_2092_;
}
v_resetjp_2092_:
{
lean_object* v___x_2096_; 
if (v_isShared_2094_ == 0)
{
lean_ctor_set_tag(v___x_2093_, 1);
lean_ctor_set(v___x_2093_, 2, v_x_2084_);
lean_ctor_set(v___x_2093_, 1, v_snd_2090_);
v___x_2096_ = v___x_2093_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2098_; 
v_reuseFailAlloc_2098_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2098_, 0, v_name_2091_);
lean_ctor_set(v_reuseFailAlloc_2098_, 1, v_snd_2090_);
lean_ctor_set(v_reuseFailAlloc_2098_, 2, v_x_2084_);
v___x_2096_ = v_reuseFailAlloc_2098_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
v_x_2084_ = v___x_2096_;
v_x_2085_ = v_tail_2089_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg(lean_object* v_x_2102_, lean_object* v_x_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_){
_start:
{
if (lean_obj_tag(v_x_2102_) == 0)
{
lean_object* v___x_2107_; lean_object* v___x_2108_; 
v___x_2107_ = l_List_reverse___redArg(v_x_2103_);
v___x_2108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2108_, 0, v___x_2107_);
return v___x_2108_;
}
else
{
lean_object* v_head_2109_; lean_object* v_toConstantVal_2110_; lean_object* v_tail_2111_; lean_object* v___x_2113_; uint8_t v_isShared_2114_; uint8_t v_isSharedCheck_2131_; 
v_head_2109_ = lean_ctor_get(v_x_2102_, 0);
lean_inc(v_head_2109_);
v_toConstantVal_2110_ = lean_ctor_get(v_head_2109_, 0);
v_tail_2111_ = lean_ctor_get(v_x_2102_, 1);
v_isSharedCheck_2131_ = !lean_is_exclusive(v_x_2102_);
if (v_isSharedCheck_2131_ == 0)
{
lean_object* v_unused_2132_; 
v_unused_2132_ = lean_ctor_get(v_x_2102_, 0);
lean_dec(v_unused_2132_);
v___x_2113_ = v_x_2102_;
v_isShared_2114_ = v_isSharedCheck_2131_;
goto v_resetjp_2112_;
}
else
{
lean_inc(v_tail_2111_);
lean_dec(v_x_2102_);
v___x_2113_ = lean_box(0);
v_isShared_2114_ = v_isSharedCheck_2131_;
goto v_resetjp_2112_;
}
v_resetjp_2112_:
{
lean_object* v_name_2115_; lean_object* v___x_2116_; 
v_name_2115_ = lean_ctor_get(v_toConstantVal_2110_, 0);
lean_inc(v_name_2115_);
v___x_2116_ = l_Lean_Core_mkFreshUserName(v_name_2115_, v___y_2104_, v___y_2105_);
if (lean_obj_tag(v___x_2116_) == 0)
{
lean_object* v_a_2117_; lean_object* v___x_2118_; lean_object* v___x_2120_; 
v_a_2117_ = lean_ctor_get(v___x_2116_, 0);
lean_inc(v_a_2117_);
lean_dec_ref_known(v___x_2116_, 1);
v___x_2118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2118_, 0, v_head_2109_);
lean_ctor_set(v___x_2118_, 1, v_a_2117_);
if (v_isShared_2114_ == 0)
{
lean_ctor_set(v___x_2113_, 1, v_x_2103_);
lean_ctor_set(v___x_2113_, 0, v___x_2118_);
v___x_2120_ = v___x_2113_;
goto v_reusejp_2119_;
}
else
{
lean_object* v_reuseFailAlloc_2122_; 
v_reuseFailAlloc_2122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2122_, 0, v___x_2118_);
lean_ctor_set(v_reuseFailAlloc_2122_, 1, v_x_2103_);
v___x_2120_ = v_reuseFailAlloc_2122_;
goto v_reusejp_2119_;
}
v_reusejp_2119_:
{
v_x_2102_ = v_tail_2111_;
v_x_2103_ = v___x_2120_;
goto _start;
}
}
else
{
lean_object* v_a_2123_; lean_object* v___x_2125_; uint8_t v_isShared_2126_; uint8_t v_isSharedCheck_2130_; 
lean_del_object(v___x_2113_);
lean_dec(v_tail_2111_);
lean_dec(v_head_2109_);
lean_dec(v_x_2103_);
v_a_2123_ = lean_ctor_get(v___x_2116_, 0);
v_isSharedCheck_2130_ = !lean_is_exclusive(v___x_2116_);
if (v_isSharedCheck_2130_ == 0)
{
v___x_2125_ = v___x_2116_;
v_isShared_2126_ = v_isSharedCheck_2130_;
goto v_resetjp_2124_;
}
else
{
lean_inc(v_a_2123_);
lean_dec(v___x_2116_);
v___x_2125_ = lean_box(0);
v_isShared_2126_ = v_isSharedCheck_2130_;
goto v_resetjp_2124_;
}
v_resetjp_2124_:
{
lean_object* v___x_2128_; 
if (v_isShared_2126_ == 0)
{
v___x_2128_ = v___x_2125_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2129_; 
v_reuseFailAlloc_2129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2129_, 0, v_a_2123_);
v___x_2128_ = v_reuseFailAlloc_2129_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
return v___x_2128_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg___boxed(lean_object* v_x_2133_, lean_object* v_x_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_){
_start:
{
lean_object* v_res_2138_; 
v_res_2138_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg(v_x_2133_, v_x_2134_, v___y_2135_, v___y_2136_);
lean_dec(v___y_2136_);
lean_dec_ref(v___y_2135_);
return v_res_2138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15(lean_object* v_ref_2139_, lean_object* v_msgData_2140_, uint8_t v_severity_2141_, uint8_t v_isSilent_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_){
_start:
{
lean_object* v___y_2149_; lean_object* v___y_2150_; lean_object* v___y_2151_; lean_object* v___y_2152_; lean_object* v___y_2153_; uint8_t v___y_2154_; uint8_t v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2185_; lean_object* v___y_2186_; lean_object* v___y_2187_; lean_object* v___y_2188_; uint8_t v___y_2189_; uint8_t v___y_2190_; uint8_t v___y_2191_; lean_object* v___y_2192_; lean_object* v___y_2210_; lean_object* v___y_2211_; lean_object* v___y_2212_; lean_object* v___y_2213_; uint8_t v___y_2214_; uint8_t v___y_2215_; uint8_t v___y_2216_; lean_object* v___y_2217_; lean_object* v___y_2221_; lean_object* v___y_2222_; lean_object* v___y_2223_; lean_object* v___y_2224_; uint8_t v___y_2225_; uint8_t v___y_2226_; uint8_t v___y_2227_; uint8_t v___x_2232_; lean_object* v___y_2234_; lean_object* v___y_2235_; lean_object* v___y_2236_; lean_object* v___y_2237_; uint8_t v___y_2238_; uint8_t v___y_2239_; uint8_t v___y_2240_; uint8_t v___y_2242_; uint8_t v___x_2257_; 
v___x_2232_ = 2;
v___x_2257_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2141_, v___x_2232_);
if (v___x_2257_ == 0)
{
v___y_2242_ = v___x_2257_;
goto v___jp_2241_;
}
else
{
uint8_t v___x_2258_; 
lean_inc_ref(v_msgData_2140_);
v___x_2258_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2140_);
v___y_2242_ = v___x_2258_;
goto v___jp_2241_;
}
v___jp_2148_:
{
lean_object* v___x_2158_; lean_object* v_currNamespace_2159_; lean_object* v_openDecls_2160_; lean_object* v_env_2161_; lean_object* v_nextMacroScope_2162_; lean_object* v_ngen_2163_; lean_object* v_auxDeclNGen_2164_; lean_object* v_traceState_2165_; lean_object* v_cache_2166_; lean_object* v_messages_2167_; lean_object* v_infoState_2168_; lean_object* v_snapshotTasks_2169_; lean_object* v___x_2171_; uint8_t v_isShared_2172_; uint8_t v_isSharedCheck_2183_; 
v___x_2158_ = lean_st_ref_take(v___y_2157_);
v_currNamespace_2159_ = lean_ctor_get(v___y_2156_, 6);
v_openDecls_2160_ = lean_ctor_get(v___y_2156_, 7);
v_env_2161_ = lean_ctor_get(v___x_2158_, 0);
v_nextMacroScope_2162_ = lean_ctor_get(v___x_2158_, 1);
v_ngen_2163_ = lean_ctor_get(v___x_2158_, 2);
v_auxDeclNGen_2164_ = lean_ctor_get(v___x_2158_, 3);
v_traceState_2165_ = lean_ctor_get(v___x_2158_, 4);
v_cache_2166_ = lean_ctor_get(v___x_2158_, 5);
v_messages_2167_ = lean_ctor_get(v___x_2158_, 6);
v_infoState_2168_ = lean_ctor_get(v___x_2158_, 7);
v_snapshotTasks_2169_ = lean_ctor_get(v___x_2158_, 8);
v_isSharedCheck_2183_ = !lean_is_exclusive(v___x_2158_);
if (v_isSharedCheck_2183_ == 0)
{
v___x_2171_ = v___x_2158_;
v_isShared_2172_ = v_isSharedCheck_2183_;
goto v_resetjp_2170_;
}
else
{
lean_inc(v_snapshotTasks_2169_);
lean_inc(v_infoState_2168_);
lean_inc(v_messages_2167_);
lean_inc(v_cache_2166_);
lean_inc(v_traceState_2165_);
lean_inc(v_auxDeclNGen_2164_);
lean_inc(v_ngen_2163_);
lean_inc(v_nextMacroScope_2162_);
lean_inc(v_env_2161_);
lean_dec(v___x_2158_);
v___x_2171_ = lean_box(0);
v_isShared_2172_ = v_isSharedCheck_2183_;
goto v_resetjp_2170_;
}
v_resetjp_2170_:
{
lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2178_; 
lean_inc(v_openDecls_2160_);
lean_inc(v_currNamespace_2159_);
v___x_2173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2173_, 0, v_currNamespace_2159_);
lean_ctor_set(v___x_2173_, 1, v_openDecls_2160_);
v___x_2174_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2174_, 0, v___x_2173_);
lean_ctor_set(v___x_2174_, 1, v___y_2149_);
lean_inc_ref(v___y_2152_);
lean_inc_ref(v___y_2151_);
v___x_2175_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2175_, 0, v___y_2151_);
lean_ctor_set(v___x_2175_, 1, v___y_2153_);
lean_ctor_set(v___x_2175_, 2, v___y_2150_);
lean_ctor_set(v___x_2175_, 3, v___y_2152_);
lean_ctor_set(v___x_2175_, 4, v___x_2174_);
lean_ctor_set_uint8(v___x_2175_, sizeof(void*)*5, v___y_2155_);
lean_ctor_set_uint8(v___x_2175_, sizeof(void*)*5 + 1, v___y_2154_);
lean_ctor_set_uint8(v___x_2175_, sizeof(void*)*5 + 2, v_isSilent_2142_);
v___x_2176_ = l_Lean_MessageLog_add(v___x_2175_, v_messages_2167_);
if (v_isShared_2172_ == 0)
{
lean_ctor_set(v___x_2171_, 6, v___x_2176_);
v___x_2178_ = v___x_2171_;
goto v_reusejp_2177_;
}
else
{
lean_object* v_reuseFailAlloc_2182_; 
v_reuseFailAlloc_2182_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2182_, 0, v_env_2161_);
lean_ctor_set(v_reuseFailAlloc_2182_, 1, v_nextMacroScope_2162_);
lean_ctor_set(v_reuseFailAlloc_2182_, 2, v_ngen_2163_);
lean_ctor_set(v_reuseFailAlloc_2182_, 3, v_auxDeclNGen_2164_);
lean_ctor_set(v_reuseFailAlloc_2182_, 4, v_traceState_2165_);
lean_ctor_set(v_reuseFailAlloc_2182_, 5, v_cache_2166_);
lean_ctor_set(v_reuseFailAlloc_2182_, 6, v___x_2176_);
lean_ctor_set(v_reuseFailAlloc_2182_, 7, v_infoState_2168_);
lean_ctor_set(v_reuseFailAlloc_2182_, 8, v_snapshotTasks_2169_);
v___x_2178_ = v_reuseFailAlloc_2182_;
goto v_reusejp_2177_;
}
v_reusejp_2177_:
{
lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; 
v___x_2179_ = lean_st_ref_set(v___y_2157_, v___x_2178_);
v___x_2180_ = lean_box(0);
v___x_2181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2181_, 0, v___x_2180_);
return v___x_2181_;
}
}
}
v___jp_2184_:
{
lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v_a_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2208_; 
v___x_2193_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2140_);
v___x_2194_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1_spec__2(v___x_2193_, v___y_2143_, v___y_2144_, v___y_2145_, v___y_2146_);
v_a_2195_ = lean_ctor_get(v___x_2194_, 0);
v_isSharedCheck_2208_ = !lean_is_exclusive(v___x_2194_);
if (v_isSharedCheck_2208_ == 0)
{
v___x_2197_ = v___x_2194_;
v_isShared_2198_ = v_isSharedCheck_2208_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_a_2195_);
lean_dec(v___x_2194_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2208_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; 
lean_inc_ref_n(v___y_2186_, 2);
v___x_2199_ = l_Lean_FileMap_toPosition(v___y_2186_, v___y_2188_);
lean_dec(v___y_2188_);
v___x_2200_ = l_Lean_FileMap_toPosition(v___y_2186_, v___y_2192_);
lean_dec(v___y_2192_);
v___x_2201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2201_, 0, v___x_2200_);
v___x_2202_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___closed__0));
if (v___y_2189_ == 0)
{
lean_del_object(v___x_2197_);
lean_dec_ref(v___y_2185_);
v___y_2149_ = v_a_2195_;
v___y_2150_ = v___x_2201_;
v___y_2151_ = v___y_2187_;
v___y_2152_ = v___x_2202_;
v___y_2153_ = v___x_2199_;
v___y_2154_ = v___y_2190_;
v___y_2155_ = v___y_2191_;
v___y_2156_ = v___y_2145_;
v___y_2157_ = v___y_2146_;
goto v___jp_2148_;
}
else
{
uint8_t v___x_2203_; 
lean_inc(v_a_2195_);
v___x_2203_ = l_Lean_MessageData_hasTag(v___y_2185_, v_a_2195_);
if (v___x_2203_ == 0)
{
lean_object* v___x_2204_; lean_object* v___x_2206_; 
lean_dec_ref_known(v___x_2201_, 1);
lean_dec_ref(v___x_2199_);
lean_dec(v_a_2195_);
v___x_2204_ = lean_box(0);
if (v_isShared_2198_ == 0)
{
lean_ctor_set(v___x_2197_, 0, v___x_2204_);
v___x_2206_ = v___x_2197_;
goto v_reusejp_2205_;
}
else
{
lean_object* v_reuseFailAlloc_2207_; 
v_reuseFailAlloc_2207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2207_, 0, v___x_2204_);
v___x_2206_ = v_reuseFailAlloc_2207_;
goto v_reusejp_2205_;
}
v_reusejp_2205_:
{
return v___x_2206_;
}
}
else
{
lean_del_object(v___x_2197_);
v___y_2149_ = v_a_2195_;
v___y_2150_ = v___x_2201_;
v___y_2151_ = v___y_2187_;
v___y_2152_ = v___x_2202_;
v___y_2153_ = v___x_2199_;
v___y_2154_ = v___y_2190_;
v___y_2155_ = v___y_2191_;
v___y_2156_ = v___y_2145_;
v___y_2157_ = v___y_2146_;
goto v___jp_2148_;
}
}
}
}
v___jp_2209_:
{
lean_object* v___x_2218_; 
v___x_2218_ = l_Lean_Syntax_getTailPos_x3f(v___y_2212_, v___y_2216_);
lean_dec(v___y_2212_);
if (lean_obj_tag(v___x_2218_) == 0)
{
lean_inc(v___y_2217_);
v___y_2185_ = v___y_2210_;
v___y_2186_ = v___y_2211_;
v___y_2187_ = v___y_2213_;
v___y_2188_ = v___y_2217_;
v___y_2189_ = v___y_2214_;
v___y_2190_ = v___y_2215_;
v___y_2191_ = v___y_2216_;
v___y_2192_ = v___y_2217_;
goto v___jp_2184_;
}
else
{
lean_object* v_val_2219_; 
v_val_2219_ = lean_ctor_get(v___x_2218_, 0);
lean_inc(v_val_2219_);
lean_dec_ref_known(v___x_2218_, 1);
v___y_2185_ = v___y_2210_;
v___y_2186_ = v___y_2211_;
v___y_2187_ = v___y_2213_;
v___y_2188_ = v___y_2217_;
v___y_2189_ = v___y_2214_;
v___y_2190_ = v___y_2215_;
v___y_2191_ = v___y_2216_;
v___y_2192_ = v_val_2219_;
goto v___jp_2184_;
}
}
v___jp_2220_:
{
lean_object* v_ref_2228_; lean_object* v___x_2229_; 
v_ref_2228_ = l_Lean_replaceRef(v_ref_2139_, v___y_2223_);
v___x_2229_ = l_Lean_Syntax_getPos_x3f(v_ref_2228_, v___y_2226_);
if (lean_obj_tag(v___x_2229_) == 0)
{
lean_object* v___x_2230_; 
v___x_2230_ = lean_unsigned_to_nat(0u);
v___y_2210_ = v___y_2221_;
v___y_2211_ = v___y_2222_;
v___y_2212_ = v_ref_2228_;
v___y_2213_ = v___y_2224_;
v___y_2214_ = v___y_2225_;
v___y_2215_ = v___y_2227_;
v___y_2216_ = v___y_2226_;
v___y_2217_ = v___x_2230_;
goto v___jp_2209_;
}
else
{
lean_object* v_val_2231_; 
v_val_2231_ = lean_ctor_get(v___x_2229_, 0);
lean_inc(v_val_2231_);
lean_dec_ref_known(v___x_2229_, 1);
v___y_2210_ = v___y_2221_;
v___y_2211_ = v___y_2222_;
v___y_2212_ = v_ref_2228_;
v___y_2213_ = v___y_2224_;
v___y_2214_ = v___y_2225_;
v___y_2215_ = v___y_2227_;
v___y_2216_ = v___y_2226_;
v___y_2217_ = v_val_2231_;
goto v___jp_2209_;
}
}
v___jp_2233_:
{
if (v___y_2240_ == 0)
{
v___y_2221_ = v___y_2237_;
v___y_2222_ = v___y_2234_;
v___y_2223_ = v___y_2235_;
v___y_2224_ = v___y_2236_;
v___y_2225_ = v___y_2238_;
v___y_2226_ = v___y_2239_;
v___y_2227_ = v_severity_2141_;
goto v___jp_2220_;
}
else
{
v___y_2221_ = v___y_2237_;
v___y_2222_ = v___y_2234_;
v___y_2223_ = v___y_2235_;
v___y_2224_ = v___y_2236_;
v___y_2225_ = v___y_2238_;
v___y_2226_ = v___y_2239_;
v___y_2227_ = v___x_2232_;
goto v___jp_2220_;
}
}
v___jp_2241_:
{
if (v___y_2242_ == 0)
{
lean_object* v_fileName_2243_; lean_object* v_fileMap_2244_; lean_object* v_options_2245_; lean_object* v_ref_2246_; uint8_t v_suppressElabErrors_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___f_2250_; uint8_t v___x_2251_; uint8_t v___x_2252_; 
v_fileName_2243_ = lean_ctor_get(v___y_2145_, 0);
v_fileMap_2244_ = lean_ctor_get(v___y_2145_, 1);
v_options_2245_ = lean_ctor_get(v___y_2145_, 2);
v_ref_2246_ = lean_ctor_get(v___y_2145_, 5);
v_suppressElabErrors_2247_ = lean_ctor_get_uint8(v___y_2145_, sizeof(void*)*14 + 1);
v___x_2248_ = lean_box(v___y_2242_);
v___x_2249_ = lean_box(v_suppressElabErrors_2247_);
v___f_2250_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2250_, 0, v___x_2248_);
lean_closure_set(v___f_2250_, 1, v___x_2249_);
v___x_2251_ = 1;
v___x_2252_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2141_, v___x_2251_);
if (v___x_2252_ == 0)
{
v___y_2234_ = v_fileMap_2244_;
v___y_2235_ = v_ref_2246_;
v___y_2236_ = v_fileName_2243_;
v___y_2237_ = v___f_2250_;
v___y_2238_ = v_suppressElabErrors_2247_;
v___y_2239_ = v___y_2242_;
v___y_2240_ = v___x_2252_;
goto v___jp_2233_;
}
else
{
lean_object* v___x_2253_; uint8_t v___x_2254_; 
v___x_2253_ = l_Lean_warningAsError;
v___x_2254_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__2_spec__4_spec__7(v_options_2245_, v___x_2253_);
v___y_2234_ = v_fileMap_2244_;
v___y_2235_ = v_ref_2246_;
v___y_2236_ = v_fileName_2243_;
v___y_2237_ = v___f_2250_;
v___y_2238_ = v_suppressElabErrors_2247_;
v___y_2239_ = v___y_2242_;
v___y_2240_ = v___x_2254_;
goto v___jp_2233_;
}
}
else
{
lean_object* v___x_2255_; lean_object* v___x_2256_; 
lean_dec_ref(v_msgData_2140_);
v___x_2255_ = lean_box(0);
v___x_2256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2256_, 0, v___x_2255_);
return v___x_2256_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15___boxed(lean_object* v_ref_2259_, lean_object* v_msgData_2260_, lean_object* v_severity_2261_, lean_object* v_isSilent_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_){
_start:
{
uint8_t v_severity_boxed_2268_; uint8_t v_isSilent_boxed_2269_; lean_object* v_res_2270_; 
v_severity_boxed_2268_ = lean_unbox(v_severity_2261_);
v_isSilent_boxed_2269_ = lean_unbox(v_isSilent_2262_);
v_res_2270_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15(v_ref_2259_, v_msgData_2260_, v_severity_boxed_2268_, v_isSilent_boxed_2269_, v___y_2263_, v___y_2264_, v___y_2265_, v___y_2266_);
lean_dec(v___y_2266_);
lean_dec_ref(v___y_2265_);
lean_dec(v___y_2264_);
lean_dec_ref(v___y_2263_);
lean_dec(v_ref_2259_);
return v_res_2270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14(lean_object* v_msgData_2271_, uint8_t v_severity_2272_, uint8_t v_isSilent_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_){
_start:
{
lean_object* v_ref_2279_; lean_object* v___x_2280_; 
v_ref_2279_ = lean_ctor_get(v___y_2276_, 5);
v___x_2280_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14_spec__15(v_ref_2279_, v_msgData_2271_, v_severity_2272_, v_isSilent_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_);
return v___x_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14___boxed(lean_object* v_msgData_2281_, lean_object* v_severity_2282_, lean_object* v_isSilent_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_){
_start:
{
uint8_t v_severity_boxed_2289_; uint8_t v_isSilent_boxed_2290_; lean_object* v_res_2291_; 
v_severity_boxed_2289_ = lean_unbox(v_severity_2282_);
v_isSilent_boxed_2290_ = lean_unbox(v_isSilent_2283_);
v_res_2291_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14(v_msgData_2281_, v_severity_boxed_2289_, v_isSilent_boxed_2290_, v___y_2284_, v___y_2285_, v___y_2286_, v___y_2287_);
lean_dec(v___y_2287_);
lean_dec_ref(v___y_2286_);
lean_dec(v___y_2285_);
lean_dec_ref(v___y_2284_);
return v_res_2291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12(lean_object* v_msgData_2292_, lean_object* v___y_2293_, lean_object* v___y_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_){
_start:
{
uint8_t v___x_2298_; uint8_t v___x_2299_; lean_object* v___x_2300_; 
v___x_2298_ = 1;
v___x_2299_ = 0;
v___x_2300_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12_spec__14(v_msgData_2292_, v___x_2298_, v___x_2299_, v___y_2293_, v___y_2294_, v___y_2295_, v___y_2296_);
return v___x_2300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12___boxed(lean_object* v_msgData_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_){
_start:
{
lean_object* v_res_2307_; 
v_res_2307_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12(v_msgData_2301_, v___y_2302_, v___y_2303_, v___y_2304_, v___y_2305_);
lean_dec(v___y_2305_);
lean_dec_ref(v___y_2304_);
lean_dec(v___y_2303_);
lean_dec_ref(v___y_2302_);
return v_res_2307_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1(void){
_start:
{
lean_object* v___x_2309_; lean_object* v___x_2310_; 
v___x_2309_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__0));
v___x_2310_ = l_Lean_stringToMessageData(v___x_2309_);
return v___x_2310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2(lean_object* v_constName_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_){
_start:
{
lean_object* v___x_2317_; lean_object* v_env_2318_; lean_object* v___x_2319_; 
v___x_2317_ = lean_st_ref_get(v___y_2315_);
v_env_2318_ = lean_ctor_get(v___x_2317_, 0);
lean_inc_ref(v_env_2318_);
lean_dec(v___x_2317_);
lean_inc(v_constName_2311_);
v___x_2319_ = l_Lean_isInductiveCore_x3f(v_env_2318_, v_constName_2311_);
if (lean_obj_tag(v___x_2319_) == 0)
{
lean_object* v___x_2320_; uint8_t v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; 
v___x_2320_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1);
v___x_2321_ = 0;
v___x_2322_ = l_Lean_MessageData_ofConstName(v_constName_2311_, v___x_2321_);
v___x_2323_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2323_, 0, v___x_2320_);
lean_ctor_set(v___x_2323_, 1, v___x_2322_);
v___x_2324_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1, &lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1_once, _init_lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1);
v___x_2325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2325_, 0, v___x_2323_);
lean_ctor_set(v___x_2325_, 1, v___x_2324_);
v___x_2326_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v___x_2325_, v___y_2312_, v___y_2313_, v___y_2314_, v___y_2315_);
return v___x_2326_;
}
else
{
lean_object* v_val_2327_; lean_object* v___x_2329_; uint8_t v_isShared_2330_; uint8_t v_isSharedCheck_2334_; 
lean_dec(v_constName_2311_);
v_val_2327_ = lean_ctor_get(v___x_2319_, 0);
v_isSharedCheck_2334_ = !lean_is_exclusive(v___x_2319_);
if (v_isSharedCheck_2334_ == 0)
{
v___x_2329_ = v___x_2319_;
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
else
{
lean_inc(v_val_2327_);
lean_dec(v___x_2319_);
v___x_2329_ = lean_box(0);
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
v_resetjp_2328_:
{
lean_object* v___x_2332_; 
if (v_isShared_2330_ == 0)
{
lean_ctor_set_tag(v___x_2329_, 0);
v___x_2332_ = v___x_2329_;
goto v_reusejp_2331_;
}
else
{
lean_object* v_reuseFailAlloc_2333_; 
v_reuseFailAlloc_2333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2333_, 0, v_val_2327_);
v___x_2332_ = v_reuseFailAlloc_2333_;
goto v_reusejp_2331_;
}
v_reusejp_2331_:
{
return v___x_2332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___boxed(lean_object* v_constName_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_){
_start:
{
lean_object* v_res_2341_; 
v_res_2341_ = lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2(v_constName_2335_, v___y_2336_, v___y_2337_, v___y_2338_, v___y_2339_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v___y_2337_);
lean_dec_ref(v___y_2336_);
return v_res_2341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4(lean_object* v___x_2342_, lean_object* v_xs_2343_, lean_object* v___x_2344_, size_t v_sz_2345_, size_t v_i_2346_, lean_object* v_bs_2347_){
_start:
{
uint8_t v___x_2348_; 
v___x_2348_ = lean_usize_dec_lt(v_i_2346_, v_sz_2345_);
if (v___x_2348_ == 0)
{
lean_dec(v___x_2344_);
lean_dec_ref(v_xs_2343_);
lean_dec(v___x_2342_);
return v_bs_2347_;
}
else
{
lean_object* v_v_2349_; lean_object* v_rhs_2350_; lean_object* v___x_2351_; lean_object* v_bs_x27_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; size_t v___x_2357_; size_t v___x_2358_; lean_object* v___x_2359_; 
v_v_2349_ = lean_array_uget_borrowed(v_bs_2347_, v_i_2346_);
v_rhs_2350_ = lean_ctor_get(v_v_2349_, 2);
lean_inc_ref(v_rhs_2350_);
v___x_2351_ = lean_unsigned_to_nat(0u);
v_bs_x27_2352_ = lean_array_uset(v_bs_2347_, v_i_2346_, v___x_2351_);
lean_inc(v___x_2342_);
v___x_2353_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(v___x_2342_, v_rhs_2350_);
lean_dec_ref(v_rhs_2350_);
lean_inc(v___x_2344_);
lean_inc_ref(v_xs_2343_);
v___x_2354_ = l_Array_toSubarray___redArg(v_xs_2343_, v___x_2351_, v___x_2344_);
v___x_2355_ = l_Subarray_copy___redArg(v___x_2354_);
v___x_2356_ = l_Lean_Expr_beta(v___x_2353_, v___x_2355_);
v___x_2357_ = ((size_t)1ULL);
v___x_2358_ = lean_usize_add(v_i_2346_, v___x_2357_);
v___x_2359_ = lean_array_uset(v_bs_x27_2352_, v_i_2346_, v___x_2356_);
v_i_2346_ = v___x_2358_;
v_bs_2347_ = v___x_2359_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4___boxed(lean_object* v___x_2361_, lean_object* v_xs_2362_, lean_object* v___x_2363_, lean_object* v_sz_2364_, lean_object* v_i_2365_, lean_object* v_bs_2366_){
_start:
{
size_t v_sz_boxed_2367_; size_t v_i_boxed_2368_; lean_object* v_res_2369_; 
v_sz_boxed_2367_ = lean_unbox_usize(v_sz_2364_);
lean_dec(v_sz_2364_);
v_i_boxed_2368_ = lean_unbox_usize(v_i_2365_);
lean_dec(v_i_2365_);
v_res_2369_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4(v___x_2361_, v_xs_2362_, v___x_2363_, v_sz_boxed_2367_, v_i_boxed_2368_, v_bs_2366_);
return v_res_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileInductiveOnly_spec__5(lean_object* v___x_2370_, lean_object* v___x_2371_, lean_object* v_a_2372_, lean_object* v_a_2373_){
_start:
{
if (lean_obj_tag(v_a_2372_) == 0)
{
lean_object* v___x_2374_; 
lean_dec_ref(v___x_2371_);
lean_dec(v___x_2370_);
v___x_2374_ = l_List_reverse___redArg(v_a_2373_);
return v___x_2374_;
}
else
{
lean_object* v_head_2375_; lean_object* v_tail_2376_; lean_object* v___x_2378_; uint8_t v_isShared_2379_; uint8_t v_isSharedCheck_2385_; 
v_head_2375_ = lean_ctor_get(v_a_2372_, 0);
v_tail_2376_ = lean_ctor_get(v_a_2372_, 1);
v_isSharedCheck_2385_ = !lean_is_exclusive(v_a_2372_);
if (v_isSharedCheck_2385_ == 0)
{
v___x_2378_ = v_a_2372_;
v_isShared_2379_ = v_isSharedCheck_2385_;
goto v_resetjp_2377_;
}
else
{
lean_inc(v_tail_2376_);
lean_inc(v_head_2375_);
lean_dec(v_a_2372_);
v___x_2378_ = lean_box(0);
v_isShared_2379_ = v_isSharedCheck_2385_;
goto v_resetjp_2377_;
}
v_resetjp_2377_:
{
lean_object* v___x_2380_; lean_object* v___x_2382_; 
lean_inc_ref(v___x_2371_);
lean_inc(v___x_2370_);
v___x_2380_ = l_Lean_Expr_proj___override(v___x_2370_, v_head_2375_, v___x_2371_);
if (v_isShared_2379_ == 0)
{
lean_ctor_set(v___x_2378_, 1, v_a_2373_);
lean_ctor_set(v___x_2378_, 0, v___x_2380_);
v___x_2382_ = v___x_2378_;
goto v_reusejp_2381_;
}
else
{
lean_object* v_reuseFailAlloc_2384_; 
v_reuseFailAlloc_2384_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2384_, 0, v___x_2380_);
lean_ctor_set(v_reuseFailAlloc_2384_, 1, v_a_2373_);
v___x_2382_ = v_reuseFailAlloc_2384_;
goto v_reusejp_2381_;
}
v_reusejp_2381_:
{
v_a_2372_ = v_tail_2376_;
v_a_2373_ = v___x_2382_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1(void){
_start:
{
lean_object* v___x_2387_; lean_object* v___x_2388_; 
v___x_2387_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__0));
v___x_2388_ = l_Lean_stringToMessageData(v___x_2387_);
return v___x_2388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7(lean_object* v_fst_2389_, lean_object* v_xs_2390_, lean_object* v_body_2391_, lean_object* v___x_2392_, lean_object* v___x_2393_, lean_object* v___x_2394_, uint8_t v___y_2395_, lean_object* v_x_2396_, lean_object* v_x_2397_, lean_object* v_x_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_){
_start:
{
if (lean_obj_tag(v_x_2396_) == 5)
{
lean_object* v_fn_2404_; lean_object* v_arg_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
v_fn_2404_ = lean_ctor_get(v_x_2396_, 0);
lean_inc_ref(v_fn_2404_);
v_arg_2405_ = lean_ctor_get(v_x_2396_, 1);
lean_inc_ref(v_arg_2405_);
lean_dec_ref_known(v_x_2396_, 2);
v___x_2406_ = lean_array_set(v_x_2397_, v_x_2398_, v_arg_2405_);
v___x_2407_ = lean_unsigned_to_nat(1u);
v___x_2408_ = lean_nat_sub(v_x_2398_, v___x_2407_);
lean_dec(v_x_2398_);
v_x_2396_ = v_fn_2404_;
v_x_2397_ = v___x_2406_;
v_x_2398_ = v___x_2408_;
goto _start;
}
else
{
lean_dec(v_x_2398_);
if (lean_obj_tag(v_x_2396_) == 4)
{
lean_object* v_declName_2410_; lean_object* v_us_2411_; lean_object* v___x_2412_; 
v_declName_2410_ = lean_ctor_get(v_x_2396_, 0);
lean_inc(v_declName_2410_);
v_us_2411_ = lean_ctor_get(v_x_2396_, 1);
lean_inc(v_us_2411_);
lean_dec_ref_known(v_x_2396_, 2);
v___x_2412_ = lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2(v_declName_2410_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
if (lean_obj_tag(v___x_2412_) == 0)
{
lean_object* v_a_2413_; lean_object* v_toConstantVal_2414_; lean_object* v_numIndices_2415_; uint8_t v_isRec_2416_; lean_object* v_name_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; 
v_a_2413_ = lean_ctor_get(v___x_2412_, 0);
lean_inc(v_a_2413_);
lean_dec_ref_known(v___x_2412_, 1);
v_toConstantVal_2414_ = lean_ctor_get(v_a_2413_, 0);
v_numIndices_2415_ = lean_ctor_get(v_a_2413_, 2);
lean_inc(v_numIndices_2415_);
v_isRec_2416_ = lean_ctor_get_uint8(v_a_2413_, sizeof(void*)*6);
v_name_2417_ = lean_ctor_get(v_toConstantVal_2414_, 0);
lean_inc_n(v_name_2417_, 2);
v___x_2418_ = l_Lean_mkRecName(v_name_2417_);
v___x_2419_ = lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(v___x_2418_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
if (lean_obj_tag(v___x_2419_) == 0)
{
lean_object* v_a_2420_; uint8_t v___x_2421_; uint8_t v___y_2423_; 
v_a_2420_ = lean_ctor_get(v___x_2419_, 0);
lean_inc(v_a_2420_);
lean_dec_ref_known(v___x_2419_, 1);
v___x_2421_ = 1;
if (v_isRec_2416_ == 0)
{
goto v___jp_2452_;
}
else
{
if (v___y_2395_ == 0)
{
lean_dec(v_numIndices_2415_);
lean_dec(v_a_2413_);
lean_dec_ref(v___x_2394_);
v___y_2423_ = v___y_2395_;
goto v___jp_2422_;
}
else
{
goto v___jp_2452_;
}
}
v___jp_2422_:
{
lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; uint8_t v___x_2428_; lean_object* v___x_2429_; 
v___x_2424_ = l_Lean_RecursorVal_getFirstIndexIdx(v_fst_2389_);
v___x_2425_ = lean_array_get_size(v_xs_2390_);
lean_inc(v___x_2424_);
lean_inc_ref(v_xs_2390_);
v___x_2426_ = l_Array_toSubarray___redArg(v_xs_2390_, v___x_2424_, v___x_2425_);
v___x_2427_ = l_Subarray_copy___redArg(v___x_2426_);
v___x_2428_ = 1;
v___x_2429_ = l_Lean_Meta_mkLambdaFVars(v___x_2427_, v_body_2391_, v___y_2423_, v___x_2421_, v___y_2423_, v___x_2421_, v___x_2428_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
if (lean_obj_tag(v___x_2429_) == 0)
{
lean_object* v_a_2430_; lean_object* v_levelParams_2431_; lean_object* v_numParams_2432_; lean_object* v_rules_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; size_t v_sz_2447_; size_t v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; 
v_a_2430_ = lean_ctor_get(v___x_2429_, 0);
lean_inc(v_a_2430_);
lean_dec_ref_known(v___x_2429_, 1);
v_levelParams_2431_ = lean_ctor_get(v___x_2392_, 1);
v_numParams_2432_ = lean_ctor_get(v_a_2420_, 2);
lean_inc(v_numParams_2432_);
lean_dec(v_a_2420_);
v_rules_2433_ = lean_ctor_get(v_fst_2389_, 6);
lean_inc(v_rules_2433_);
lean_dec_ref(v_fst_2389_);
v___x_2434_ = lean_box(0);
v___x_2435_ = l_Lean_mkCasesOnName(v_name_2417_);
v___x_2436_ = l_List_head_x21___redArg(v___x_2434_, v_levelParams_2431_);
v___x_2437_ = l_Lean_Level_param___override(v___x_2436_);
v___x_2438_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2438_, 0, v___x_2437_);
lean_ctor_set(v___x_2438_, 1, v_us_2411_);
v___x_2439_ = l_Lean_Expr_const___override(v___x_2435_, v___x_2438_);
v___x_2440_ = lean_unsigned_to_nat(0u);
v___x_2441_ = l_Array_toSubarray___redArg(v_x_2397_, v___x_2440_, v_numParams_2432_);
v___x_2442_ = l_Subarray_copy___redArg(v___x_2441_);
v___x_2443_ = l_Lean_mkAppN(v___x_2439_, v___x_2442_);
lean_dec_ref(v___x_2442_);
v___x_2444_ = l_Lean_Expr_app___override(v___x_2443_, v_a_2430_);
v___x_2445_ = l_Lean_mkAppN(v___x_2444_, v___x_2427_);
lean_dec_ref(v___x_2427_);
v___x_2446_ = lean_array_mk(v_rules_2433_);
v_sz_2447_ = lean_array_size(v___x_2446_);
v___x_2448_ = ((size_t)0ULL);
lean_inc_ref(v_xs_2390_);
v___x_2449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4(v___x_2393_, v_xs_2390_, v___x_2424_, v_sz_2447_, v___x_2448_, v___x_2446_);
v___x_2450_ = l_Lean_mkAppN(v___x_2445_, v___x_2449_);
lean_dec_ref(v___x_2449_);
v___x_2451_ = l_Lean_Meta_mkLambdaFVars(v_xs_2390_, v___x_2450_, v___y_2423_, v___x_2421_, v___y_2423_, v___x_2421_, v___x_2428_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
lean_dec_ref(v_xs_2390_);
return v___x_2451_;
}
else
{
lean_dec_ref(v___x_2427_);
lean_dec(v___x_2424_);
lean_dec(v_a_2420_);
lean_dec(v_name_2417_);
lean_dec(v_us_2411_);
lean_dec_ref(v_x_2397_);
lean_dec(v___x_2393_);
lean_dec_ref(v_xs_2390_);
lean_dec_ref(v_fst_2389_);
return v___x_2429_;
}
}
v___jp_2452_:
{
lean_object* v_numMotives_2453_; lean_object* v___x_2454_; uint8_t v___x_2455_; 
v_numMotives_2453_ = lean_ctor_get(v_a_2420_, 4);
v___x_2454_ = lean_unsigned_to_nat(1u);
v___x_2455_ = lean_nat_dec_eq(v_numMotives_2453_, v___x_2454_);
if (v___x_2455_ == 0)
{
lean_dec(v_numIndices_2415_);
lean_dec(v_a_2413_);
lean_dec_ref(v___x_2394_);
v___y_2423_ = v___x_2455_;
goto v___jp_2422_;
}
else
{
lean_object* v___x_2456_; uint8_t v___x_2457_; 
v___x_2456_ = l_Lean_InductiveVal_numCtors(v_a_2413_);
lean_dec(v_a_2413_);
v___x_2457_ = lean_nat_dec_eq(v___x_2456_, v___x_2454_);
lean_dec(v___x_2456_);
if (v___x_2457_ == 0)
{
lean_dec(v_numIndices_2415_);
lean_dec_ref(v___x_2394_);
v___y_2423_ = v___x_2457_;
goto v___jp_2422_;
}
else
{
lean_object* v___x_2458_; uint8_t v___x_2459_; 
v___x_2458_ = lean_unsigned_to_nat(0u);
v___x_2459_ = lean_nat_dec_eq(v_numIndices_2415_, v___x_2458_);
lean_dec(v_numIndices_2415_);
if (v___x_2459_ == 0)
{
lean_dec_ref(v___x_2394_);
v___y_2423_ = v___x_2459_;
goto v___jp_2422_;
}
else
{
lean_object* v_rules_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v_nfields_2463_; lean_object* v_rhs_2464_; lean_object* v___x_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; lean_object* v___x_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; uint8_t v___x_2475_; lean_object* v___x_2476_; 
lean_dec(v_a_2420_);
lean_dec(v_us_2411_);
lean_dec_ref(v_x_2397_);
lean_dec_ref(v_body_2391_);
v_rules_2460_ = lean_ctor_get(v_fst_2389_, 6);
v___x_2461_ = l_Lean_instInhabitedRecursorRule_default;
v___x_2462_ = l_List_get_x21Internal___redArg(v___x_2461_, v_rules_2460_, v___x_2458_);
v_nfields_2463_ = lean_ctor_get(v___x_2462_, 1);
lean_inc(v_nfields_2463_);
v_rhs_2464_ = lean_ctor_get(v___x_2462_, 2);
lean_inc_ref(v_rhs_2464_);
lean_dec(v___x_2462_);
v___x_2465_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(v___x_2393_, v_rhs_2464_);
lean_dec_ref(v_rhs_2464_);
v___x_2466_ = l_Lean_RecursorVal_getFirstIndexIdx(v_fst_2389_);
lean_dec_ref(v_fst_2389_);
lean_inc_ref(v_xs_2390_);
v___x_2467_ = l_Array_toSubarray___redArg(v_xs_2390_, v___x_2458_, v___x_2466_);
v___x_2468_ = l_Subarray_copy___redArg(v___x_2467_);
v___x_2469_ = l_Lean_Expr_beta(v___x_2465_, v___x_2468_);
v___x_2470_ = l_List_range(v_nfields_2463_);
v___x_2471_ = lean_box(0);
v___x_2472_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileInductiveOnly_spec__5(v_name_2417_, v___x_2394_, v___x_2470_, v___x_2471_);
v___x_2473_ = lean_array_mk(v___x_2472_);
v___x_2474_ = l_Lean_Expr_beta(v___x_2469_, v___x_2473_);
v___x_2475_ = 1;
v___x_2476_ = l_Lean_Meta_mkLambdaFVars(v_xs_2390_, v___x_2474_, v___y_2395_, v___x_2421_, v___y_2395_, v___x_2421_, v___x_2475_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
lean_dec_ref(v_xs_2390_);
return v___x_2476_;
}
}
}
}
}
else
{
lean_object* v_a_2477_; lean_object* v___x_2479_; uint8_t v_isShared_2480_; uint8_t v_isSharedCheck_2484_; 
lean_dec(v_name_2417_);
lean_dec(v_numIndices_2415_);
lean_dec(v_a_2413_);
lean_dec(v_us_2411_);
lean_dec_ref(v_x_2397_);
lean_dec_ref(v___x_2394_);
lean_dec(v___x_2393_);
lean_dec_ref(v_body_2391_);
lean_dec_ref(v_xs_2390_);
lean_dec_ref(v_fst_2389_);
v_a_2477_ = lean_ctor_get(v___x_2419_, 0);
v_isSharedCheck_2484_ = !lean_is_exclusive(v___x_2419_);
if (v_isSharedCheck_2484_ == 0)
{
v___x_2479_ = v___x_2419_;
v_isShared_2480_ = v_isSharedCheck_2484_;
goto v_resetjp_2478_;
}
else
{
lean_inc(v_a_2477_);
lean_dec(v___x_2419_);
v___x_2479_ = lean_box(0);
v_isShared_2480_ = v_isSharedCheck_2484_;
goto v_resetjp_2478_;
}
v_resetjp_2478_:
{
lean_object* v___x_2482_; 
if (v_isShared_2480_ == 0)
{
v___x_2482_ = v___x_2479_;
goto v_reusejp_2481_;
}
else
{
lean_object* v_reuseFailAlloc_2483_; 
v_reuseFailAlloc_2483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2483_, 0, v_a_2477_);
v___x_2482_ = v_reuseFailAlloc_2483_;
goto v_reusejp_2481_;
}
v_reusejp_2481_:
{
return v___x_2482_;
}
}
}
}
else
{
lean_object* v_a_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2492_; 
lean_dec(v_us_2411_);
lean_dec_ref(v_x_2397_);
lean_dec_ref(v___x_2394_);
lean_dec(v___x_2393_);
lean_dec_ref(v_body_2391_);
lean_dec_ref(v_xs_2390_);
lean_dec_ref(v_fst_2389_);
v_a_2485_ = lean_ctor_get(v___x_2412_, 0);
v_isSharedCheck_2492_ = !lean_is_exclusive(v___x_2412_);
if (v_isSharedCheck_2492_ == 0)
{
v___x_2487_ = v___x_2412_;
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_a_2485_);
lean_dec(v___x_2412_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
lean_object* v___x_2490_; 
if (v_isShared_2488_ == 0)
{
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
return v___x_2490_;
}
}
}
}
else
{
lean_object* v___x_2493_; lean_object* v___x_2494_; 
lean_dec_ref(v_x_2397_);
lean_dec_ref(v_x_2396_);
lean_dec_ref(v___x_2394_);
lean_dec(v___x_2393_);
lean_dec_ref(v_body_2391_);
lean_dec_ref(v_xs_2390_);
lean_dec_ref(v_fst_2389_);
v___x_2493_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1);
v___x_2494_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v___x_2493_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
return v___x_2494_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___boxed(lean_object* v_fst_2495_, lean_object* v_xs_2496_, lean_object* v_body_2497_, lean_object* v___x_2498_, lean_object* v___x_2499_, lean_object* v___x_2500_, lean_object* v___y_2501_, lean_object* v_x_2502_, lean_object* v_x_2503_, lean_object* v_x_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_){
_start:
{
uint8_t v___y_16670__boxed_2510_; lean_object* v_res_2511_; 
v___y_16670__boxed_2510_ = lean_unbox(v___y_2501_);
v_res_2511_ = lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7(v_fst_2495_, v_xs_2496_, v_body_2497_, v___x_2498_, v___x_2499_, v___x_2500_, v___y_16670__boxed_2510_, v_x_2502_, v_x_2503_, v_x_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
lean_dec(v___y_2508_);
lean_dec_ref(v___y_2507_);
lean_dec(v___y_2506_);
lean_dec_ref(v___y_2505_);
lean_dec_ref(v___x_2498_);
return v_res_2511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6(lean_object* v___x_2512_, lean_object* v_fst_2513_, lean_object* v_xs_2514_, lean_object* v_body_2515_, lean_object* v___x_2516_, lean_object* v___x_2517_, uint8_t v___y_2518_, lean_object* v_x_2519_, lean_object* v_x_2520_, lean_object* v_x_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_){
_start:
{
if (lean_obj_tag(v_x_2519_) == 5)
{
lean_object* v_fn_2527_; lean_object* v_arg_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; 
v_fn_2527_ = lean_ctor_get(v_x_2519_, 0);
lean_inc_ref(v_fn_2527_);
v_arg_2528_ = lean_ctor_get(v_x_2519_, 1);
lean_inc_ref(v_arg_2528_);
lean_dec_ref_known(v_x_2519_, 2);
v___x_2529_ = lean_array_set(v_x_2520_, v_x_2521_, v_arg_2528_);
v___x_2530_ = lean_unsigned_to_nat(1u);
v___x_2531_ = lean_nat_sub(v_x_2521_, v___x_2530_);
v___x_2532_ = lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7(v_fst_2513_, v_xs_2514_, v_body_2515_, v___x_2516_, v___x_2517_, v___x_2512_, v___y_2518_, v_fn_2527_, v___x_2529_, v___x_2531_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
return v___x_2532_;
}
else
{
if (lean_obj_tag(v_x_2519_) == 4)
{
lean_object* v_declName_2533_; lean_object* v_us_2534_; lean_object* v___x_2535_; 
v_declName_2533_ = lean_ctor_get(v_x_2519_, 0);
lean_inc(v_declName_2533_);
v_us_2534_ = lean_ctor_get(v_x_2519_, 1);
lean_inc(v_us_2534_);
lean_dec_ref_known(v_x_2519_, 2);
v___x_2535_ = lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2(v_declName_2533_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
if (lean_obj_tag(v___x_2535_) == 0)
{
lean_object* v_a_2536_; lean_object* v_toConstantVal_2537_; lean_object* v_numIndices_2538_; uint8_t v_isRec_2539_; lean_object* v_name_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; 
v_a_2536_ = lean_ctor_get(v___x_2535_, 0);
lean_inc(v_a_2536_);
lean_dec_ref_known(v___x_2535_, 1);
v_toConstantVal_2537_ = lean_ctor_get(v_a_2536_, 0);
v_numIndices_2538_ = lean_ctor_get(v_a_2536_, 2);
lean_inc(v_numIndices_2538_);
v_isRec_2539_ = lean_ctor_get_uint8(v_a_2536_, sizeof(void*)*6);
v_name_2540_ = lean_ctor_get(v_toConstantVal_2537_, 0);
lean_inc_n(v_name_2540_, 2);
v___x_2541_ = l_Lean_mkRecName(v_name_2540_);
v___x_2542_ = lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(v___x_2541_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
if (lean_obj_tag(v___x_2542_) == 0)
{
lean_object* v_a_2543_; uint8_t v___x_2544_; uint8_t v___y_2546_; 
v_a_2543_ = lean_ctor_get(v___x_2542_, 0);
lean_inc(v_a_2543_);
lean_dec_ref_known(v___x_2542_, 1);
v___x_2544_ = 1;
if (v_isRec_2539_ == 0)
{
goto v___jp_2575_;
}
else
{
if (v___y_2518_ == 0)
{
lean_dec(v_numIndices_2538_);
lean_dec(v_a_2536_);
lean_dec_ref(v___x_2512_);
v___y_2546_ = v___y_2518_;
goto v___jp_2545_;
}
else
{
goto v___jp_2575_;
}
}
v___jp_2545_:
{
lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; uint8_t v___x_2551_; lean_object* v___x_2552_; 
v___x_2547_ = l_Lean_RecursorVal_getFirstIndexIdx(v_fst_2513_);
v___x_2548_ = lean_array_get_size(v_xs_2514_);
lean_inc(v___x_2547_);
lean_inc_ref(v_xs_2514_);
v___x_2549_ = l_Array_toSubarray___redArg(v_xs_2514_, v___x_2547_, v___x_2548_);
v___x_2550_ = l_Subarray_copy___redArg(v___x_2549_);
v___x_2551_ = 1;
v___x_2552_ = l_Lean_Meta_mkLambdaFVars(v___x_2550_, v_body_2515_, v___y_2546_, v___x_2544_, v___y_2546_, v___x_2544_, v___x_2551_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
if (lean_obj_tag(v___x_2552_) == 0)
{
lean_object* v_a_2553_; lean_object* v_levelParams_2554_; lean_object* v_numParams_2555_; lean_object* v_rules_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; size_t v_sz_2570_; size_t v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; 
v_a_2553_ = lean_ctor_get(v___x_2552_, 0);
lean_inc(v_a_2553_);
lean_dec_ref_known(v___x_2552_, 1);
v_levelParams_2554_ = lean_ctor_get(v___x_2516_, 1);
v_numParams_2555_ = lean_ctor_get(v_a_2543_, 2);
lean_inc(v_numParams_2555_);
lean_dec(v_a_2543_);
v_rules_2556_ = lean_ctor_get(v_fst_2513_, 6);
lean_inc(v_rules_2556_);
lean_dec_ref(v_fst_2513_);
v___x_2557_ = lean_box(0);
v___x_2558_ = l_Lean_mkCasesOnName(v_name_2540_);
v___x_2559_ = l_List_head_x21___redArg(v___x_2557_, v_levelParams_2554_);
v___x_2560_ = l_Lean_Level_param___override(v___x_2559_);
v___x_2561_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2561_, 0, v___x_2560_);
lean_ctor_set(v___x_2561_, 1, v_us_2534_);
v___x_2562_ = l_Lean_Expr_const___override(v___x_2558_, v___x_2561_);
v___x_2563_ = lean_unsigned_to_nat(0u);
v___x_2564_ = l_Array_toSubarray___redArg(v_x_2520_, v___x_2563_, v_numParams_2555_);
v___x_2565_ = l_Subarray_copy___redArg(v___x_2564_);
v___x_2566_ = l_Lean_mkAppN(v___x_2562_, v___x_2565_);
lean_dec_ref(v___x_2565_);
v___x_2567_ = l_Lean_Expr_app___override(v___x_2566_, v_a_2553_);
v___x_2568_ = l_Lean_mkAppN(v___x_2567_, v___x_2550_);
lean_dec_ref(v___x_2550_);
v___x_2569_ = lean_array_mk(v_rules_2556_);
v_sz_2570_ = lean_array_size(v___x_2569_);
v___x_2571_ = ((size_t)0ULL);
lean_inc_ref(v_xs_2514_);
v___x_2572_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Util_compileInductiveOnly_spec__4(v___x_2517_, v_xs_2514_, v___x_2547_, v_sz_2570_, v___x_2571_, v___x_2569_);
v___x_2573_ = l_Lean_mkAppN(v___x_2568_, v___x_2572_);
lean_dec_ref(v___x_2572_);
v___x_2574_ = l_Lean_Meta_mkLambdaFVars(v_xs_2514_, v___x_2573_, v___y_2546_, v___x_2544_, v___y_2546_, v___x_2544_, v___x_2551_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
lean_dec_ref(v_xs_2514_);
return v___x_2574_;
}
else
{
lean_dec_ref(v___x_2550_);
lean_dec(v___x_2547_);
lean_dec(v_a_2543_);
lean_dec(v_name_2540_);
lean_dec(v_us_2534_);
lean_dec_ref(v_x_2520_);
lean_dec(v___x_2517_);
lean_dec_ref(v_xs_2514_);
lean_dec_ref(v_fst_2513_);
return v___x_2552_;
}
}
v___jp_2575_:
{
lean_object* v_numMotives_2576_; lean_object* v___x_2577_; uint8_t v___x_2578_; 
v_numMotives_2576_ = lean_ctor_get(v_a_2543_, 4);
v___x_2577_ = lean_unsigned_to_nat(1u);
v___x_2578_ = lean_nat_dec_eq(v_numMotives_2576_, v___x_2577_);
if (v___x_2578_ == 0)
{
lean_dec(v_numIndices_2538_);
lean_dec(v_a_2536_);
lean_dec_ref(v___x_2512_);
v___y_2546_ = v___x_2578_;
goto v___jp_2545_;
}
else
{
lean_object* v___x_2579_; uint8_t v___x_2580_; 
v___x_2579_ = l_Lean_InductiveVal_numCtors(v_a_2536_);
lean_dec(v_a_2536_);
v___x_2580_ = lean_nat_dec_eq(v___x_2579_, v___x_2577_);
lean_dec(v___x_2579_);
if (v___x_2580_ == 0)
{
lean_dec(v_numIndices_2538_);
lean_dec_ref(v___x_2512_);
v___y_2546_ = v___x_2580_;
goto v___jp_2545_;
}
else
{
lean_object* v___x_2581_; uint8_t v___x_2582_; 
v___x_2581_ = lean_unsigned_to_nat(0u);
v___x_2582_ = lean_nat_dec_eq(v_numIndices_2538_, v___x_2581_);
lean_dec(v_numIndices_2538_);
if (v___x_2582_ == 0)
{
lean_dec_ref(v___x_2512_);
v___y_2546_ = v___x_2582_;
goto v___jp_2545_;
}
else
{
lean_object* v_rules_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v_nfields_2586_; lean_object* v_rhs_2587_; lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; uint8_t v___x_2598_; lean_object* v___x_2599_; 
lean_dec(v_a_2543_);
lean_dec(v_us_2534_);
lean_dec_ref(v_x_2520_);
lean_dec_ref(v_body_2515_);
v_rules_2583_ = lean_ctor_get(v_fst_2513_, 6);
v___x_2584_ = l_Lean_instInhabitedRecursorRule_default;
v___x_2585_ = l_List_get_x21Internal___redArg(v___x_2584_, v_rules_2583_, v___x_2581_);
v_nfields_2586_ = lean_ctor_get(v___x_2585_, 1);
lean_inc(v_nfields_2586_);
v_rhs_2587_ = lean_ctor_get(v___x_2585_, 2);
lean_inc_ref(v_rhs_2587_);
lean_dec(v___x_2585_);
v___x_2588_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_replaceConst(v___x_2517_, v_rhs_2587_);
lean_dec_ref(v_rhs_2587_);
v___x_2589_ = l_Lean_RecursorVal_getFirstIndexIdx(v_fst_2513_);
lean_dec_ref(v_fst_2513_);
lean_inc_ref(v_xs_2514_);
v___x_2590_ = l_Array_toSubarray___redArg(v_xs_2514_, v___x_2581_, v___x_2589_);
v___x_2591_ = l_Subarray_copy___redArg(v___x_2590_);
v___x_2592_ = l_Lean_Expr_beta(v___x_2588_, v___x_2591_);
v___x_2593_ = l_List_range(v_nfields_2586_);
v___x_2594_ = lean_box(0);
v___x_2595_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileInductiveOnly_spec__5(v_name_2540_, v___x_2512_, v___x_2593_, v___x_2594_);
v___x_2596_ = lean_array_mk(v___x_2595_);
v___x_2597_ = l_Lean_Expr_beta(v___x_2592_, v___x_2596_);
v___x_2598_ = 1;
v___x_2599_ = l_Lean_Meta_mkLambdaFVars(v_xs_2514_, v___x_2597_, v___y_2518_, v___x_2544_, v___y_2518_, v___x_2544_, v___x_2598_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
lean_dec_ref(v_xs_2514_);
return v___x_2599_;
}
}
}
}
}
else
{
lean_object* v_a_2600_; lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2607_; 
lean_dec(v_name_2540_);
lean_dec(v_numIndices_2538_);
lean_dec(v_a_2536_);
lean_dec(v_us_2534_);
lean_dec_ref(v_x_2520_);
lean_dec(v___x_2517_);
lean_dec_ref(v_body_2515_);
lean_dec_ref(v_xs_2514_);
lean_dec_ref(v_fst_2513_);
lean_dec_ref(v___x_2512_);
v_a_2600_ = lean_ctor_get(v___x_2542_, 0);
v_isSharedCheck_2607_ = !lean_is_exclusive(v___x_2542_);
if (v_isSharedCheck_2607_ == 0)
{
v___x_2602_ = v___x_2542_;
v_isShared_2603_ = v_isSharedCheck_2607_;
goto v_resetjp_2601_;
}
else
{
lean_inc(v_a_2600_);
lean_dec(v___x_2542_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2607_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v___x_2605_; 
if (v_isShared_2603_ == 0)
{
v___x_2605_ = v___x_2602_;
goto v_reusejp_2604_;
}
else
{
lean_object* v_reuseFailAlloc_2606_; 
v_reuseFailAlloc_2606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2606_, 0, v_a_2600_);
v___x_2605_ = v_reuseFailAlloc_2606_;
goto v_reusejp_2604_;
}
v_reusejp_2604_:
{
return v___x_2605_;
}
}
}
}
else
{
lean_object* v_a_2608_; lean_object* v___x_2610_; uint8_t v_isShared_2611_; uint8_t v_isSharedCheck_2615_; 
lean_dec(v_us_2534_);
lean_dec_ref(v_x_2520_);
lean_dec(v___x_2517_);
lean_dec_ref(v_body_2515_);
lean_dec_ref(v_xs_2514_);
lean_dec_ref(v_fst_2513_);
lean_dec_ref(v___x_2512_);
v_a_2608_ = lean_ctor_get(v___x_2535_, 0);
v_isSharedCheck_2615_ = !lean_is_exclusive(v___x_2535_);
if (v_isSharedCheck_2615_ == 0)
{
v___x_2610_ = v___x_2535_;
v_isShared_2611_ = v_isSharedCheck_2615_;
goto v_resetjp_2609_;
}
else
{
lean_inc(v_a_2608_);
lean_dec(v___x_2535_);
v___x_2610_ = lean_box(0);
v_isShared_2611_ = v_isSharedCheck_2615_;
goto v_resetjp_2609_;
}
v_resetjp_2609_:
{
lean_object* v___x_2613_; 
if (v_isShared_2611_ == 0)
{
v___x_2613_ = v___x_2610_;
goto v_reusejp_2612_;
}
else
{
lean_object* v_reuseFailAlloc_2614_; 
v_reuseFailAlloc_2614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2614_, 0, v_a_2608_);
v___x_2613_ = v_reuseFailAlloc_2614_;
goto v_reusejp_2612_;
}
v_reusejp_2612_:
{
return v___x_2613_;
}
}
}
}
else
{
lean_object* v___x_2616_; lean_object* v___x_2617_; 
lean_dec_ref(v_x_2520_);
lean_dec_ref(v_x_2519_);
lean_dec(v___x_2517_);
lean_dec_ref(v_body_2515_);
lean_dec_ref(v_xs_2514_);
lean_dec_ref(v_fst_2513_);
lean_dec_ref(v___x_2512_);
v___x_2616_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6_spec__7___closed__1);
v___x_2617_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_go_spec__0_spec__0___redArg(v___x_2616_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
return v___x_2617_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6___boxed(lean_object* v___x_2618_, lean_object* v_fst_2619_, lean_object* v_xs_2620_, lean_object* v_body_2621_, lean_object* v___x_2622_, lean_object* v___x_2623_, lean_object* v___y_2624_, lean_object* v_x_2625_, lean_object* v_x_2626_, lean_object* v_x_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_, lean_object* v___y_2632_){
_start:
{
uint8_t v___y_16874__boxed_2633_; lean_object* v_res_2634_; 
v___y_16874__boxed_2633_ = lean_unbox(v___y_2624_);
v_res_2634_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6(v___x_2618_, v_fst_2619_, v_xs_2620_, v_body_2621_, v___x_2622_, v___x_2623_, v___y_16874__boxed_2633_, v_x_2625_, v_x_2626_, v_x_2627_, v___y_2628_, v___y_2629_, v___y_2630_, v___y_2631_);
lean_dec(v___y_2631_);
lean_dec_ref(v___y_2630_);
lean_dec(v___y_2629_);
lean_dec_ref(v___y_2628_);
lean_dec(v_x_2627_);
lean_dec_ref(v___x_2622_);
return v_res_2634_;
}
}
static lean_object* _init_lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2635_; lean_object* v_dummy_2636_; 
v___x_2635_ = lean_box(0);
v_dummy_2636_ = l_Lean_Expr_sort___override(v___x_2635_);
return v_dummy_2636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0(lean_object* v_fst_2637_, lean_object* v_toConstantVal_2638_, lean_object* v___x_2639_, uint8_t v___y_2640_, lean_object* v_xs_2641_, lean_object* v_body_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_, lean_object* v___y_2646_){
_start:
{
lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; 
v___x_2648_ = l_Lean_instInhabitedExpr;
v___x_2649_ = l_Lean_RecursorVal_getMajorIdx(v_fst_2637_);
v___x_2650_ = lean_array_get(v___x_2648_, v_xs_2641_, v___x_2649_);
lean_dec(v___x_2649_);
lean_inc(v___y_2646_);
lean_inc_ref(v___y_2645_);
lean_inc(v___y_2644_);
lean_inc_ref(v___y_2643_);
lean_inc(v___x_2650_);
v___x_2651_ = lean_infer_type(v___x_2650_, v___y_2643_, v___y_2644_, v___y_2645_, v___y_2646_);
if (lean_obj_tag(v___x_2651_) == 0)
{
lean_object* v_a_2652_; lean_object* v___x_2653_; 
v_a_2652_ = lean_ctor_get(v___x_2651_, 0);
lean_inc(v_a_2652_);
lean_dec_ref_known(v___x_2651_, 1);
v___x_2653_ = l_Lean_Meta_whnfD(v_a_2652_, v___y_2643_, v___y_2644_, v___y_2645_, v___y_2646_);
if (lean_obj_tag(v___x_2653_) == 0)
{
lean_object* v_a_2654_; lean_object* v_dummy_2655_; lean_object* v_nargs_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; lean_object* v___x_2660_; 
v_a_2654_ = lean_ctor_get(v___x_2653_, 0);
lean_inc(v_a_2654_);
lean_dec_ref_known(v___x_2653_, 1);
v_dummy_2655_ = lean_obj_once(&lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0, &lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0_once, _init_lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___closed__0);
v_nargs_2656_ = l_Lean_Expr_getAppNumArgs(v_a_2654_);
lean_inc(v_nargs_2656_);
v___x_2657_ = lean_mk_array(v_nargs_2656_, v_dummy_2655_);
v___x_2658_ = lean_unsigned_to_nat(1u);
v___x_2659_ = lean_nat_sub(v_nargs_2656_, v___x_2658_);
lean_dec(v_nargs_2656_);
v___x_2660_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Util_compileInductiveOnly_spec__6(v___x_2650_, v_fst_2637_, v_xs_2641_, v_body_2642_, v_toConstantVal_2638_, v___x_2639_, v___y_2640_, v_a_2654_, v___x_2657_, v___x_2659_, v___y_2643_, v___y_2644_, v___y_2645_, v___y_2646_);
lean_dec(v___x_2659_);
return v___x_2660_;
}
else
{
lean_dec(v___x_2650_);
lean_dec_ref(v_body_2642_);
lean_dec_ref(v_xs_2641_);
lean_dec(v___x_2639_);
lean_dec_ref(v_fst_2637_);
return v___x_2653_;
}
}
else
{
lean_dec(v___x_2650_);
lean_dec_ref(v_body_2642_);
lean_dec_ref(v_xs_2641_);
lean_dec(v___x_2639_);
lean_dec_ref(v_fst_2637_);
return v___x_2651_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___boxed(lean_object* v_fst_2661_, lean_object* v_toConstantVal_2662_, lean_object* v___x_2663_, lean_object* v___y_2664_, lean_object* v_xs_2665_, lean_object* v_body_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_, lean_object* v___y_2670_, lean_object* v___y_2671_){
_start:
{
uint8_t v___y_17073__boxed_2672_; lean_object* v_res_2673_; 
v___y_17073__boxed_2672_ = lean_unbox(v___y_2664_);
v_res_2673_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0(v_fst_2661_, v_toConstantVal_2662_, v___x_2663_, v___y_17073__boxed_2672_, v_xs_2665_, v_body_2666_, v___y_2667_, v___y_2668_, v___y_2669_, v___y_2670_);
lean_dec(v___y_2670_);
lean_dec_ref(v___y_2669_);
lean_dec(v___y_2668_);
lean_dec_ref(v___y_2667_);
lean_dec_ref(v_toConstantVal_2662_);
return v_res_2673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7(lean_object* v___x_2674_, uint8_t v___y_2675_, lean_object* v_x_2676_, lean_object* v_x_2677_, lean_object* v___y_2678_, lean_object* v___y_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_){
_start:
{
if (lean_obj_tag(v_x_2676_) == 0)
{
lean_object* v___x_2683_; lean_object* v___x_2684_; 
lean_dec(v___x_2674_);
v___x_2683_ = l_List_reverse___redArg(v_x_2677_);
v___x_2684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2684_, 0, v___x_2683_);
return v___x_2684_;
}
else
{
lean_object* v_head_2685_; lean_object* v_fst_2686_; lean_object* v_toConstantVal_2687_; lean_object* v_tail_2688_; lean_object* v___x_2690_; uint8_t v_isShared_2691_; uint8_t v_isSharedCheck_2716_; 
v_head_2685_ = lean_ctor_get(v_x_2676_, 0);
lean_inc(v_head_2685_);
v_fst_2686_ = lean_ctor_get(v_head_2685_, 0);
lean_inc(v_fst_2686_);
v_toConstantVal_2687_ = lean_ctor_get(v_fst_2686_, 0);
lean_inc_ref(v_toConstantVal_2687_);
v_tail_2688_ = lean_ctor_get(v_x_2676_, 1);
v_isSharedCheck_2716_ = !lean_is_exclusive(v_x_2676_);
if (v_isSharedCheck_2716_ == 0)
{
lean_object* v_unused_2717_; 
v_unused_2717_ = lean_ctor_get(v_x_2676_, 0);
lean_dec(v_unused_2717_);
v___x_2690_ = v_x_2676_;
v_isShared_2691_ = v_isSharedCheck_2716_;
goto v_resetjp_2689_;
}
else
{
lean_inc(v_tail_2688_);
lean_dec(v_x_2676_);
v___x_2690_ = lean_box(0);
v_isShared_2691_ = v_isSharedCheck_2716_;
goto v_resetjp_2689_;
}
v_resetjp_2689_:
{
lean_object* v_snd_2692_; lean_object* v_all_2693_; lean_object* v_levelParams_2694_; lean_object* v_type_2695_; lean_object* v___x_2696_; lean_object* v___f_2697_; lean_object* v___x_2698_; 
v_snd_2692_ = lean_ctor_get(v_head_2685_, 1);
lean_inc(v_snd_2692_);
lean_dec(v_head_2685_);
v_all_2693_ = lean_ctor_get(v_fst_2686_, 1);
lean_inc(v_all_2693_);
v_levelParams_2694_ = lean_ctor_get(v_toConstantVal_2687_, 1);
lean_inc(v_levelParams_2694_);
v_type_2695_ = lean_ctor_get(v_toConstantVal_2687_, 2);
lean_inc_ref_n(v_type_2695_, 2);
v___x_2696_ = lean_box(v___y_2675_);
lean_inc(v___x_2674_);
v___f_2697_ = lean_alloc_closure((void*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___lam__0___boxed), 11, 4);
lean_closure_set(v___f_2697_, 0, v_fst_2686_);
lean_closure_set(v___f_2697_, 1, v_toConstantVal_2687_);
lean_closure_set(v___f_2697_, 2, v___x_2674_);
lean_closure_set(v___f_2697_, 3, v___x_2696_);
v___x_2698_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly_spec__1___redArg(v_type_2695_, v___f_2697_, v___y_2675_, v___y_2678_, v___y_2679_, v___y_2680_, v___y_2681_);
if (lean_obj_tag(v___x_2698_) == 0)
{
lean_object* v_a_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; uint8_t v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2705_; 
v_a_2699_ = lean_ctor_get(v___x_2698_, 0);
lean_inc(v_a_2699_);
lean_dec_ref_known(v___x_2698_, 1);
v___x_2700_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2700_, 0, v_snd_2692_);
lean_ctor_set(v___x_2700_, 1, v_levelParams_2694_);
lean_ctor_set(v___x_2700_, 2, v_type_2695_);
v___x_2701_ = lean_box(0);
v___x_2702_ = 2;
v___x_2703_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_2703_, 0, v___x_2700_);
lean_ctor_set(v___x_2703_, 1, v_a_2699_);
lean_ctor_set(v___x_2703_, 2, v___x_2701_);
lean_ctor_set(v___x_2703_, 3, v_all_2693_);
lean_ctor_set_uint8(v___x_2703_, sizeof(void*)*4, v___x_2702_);
if (v_isShared_2691_ == 0)
{
lean_ctor_set(v___x_2690_, 1, v_x_2677_);
lean_ctor_set(v___x_2690_, 0, v___x_2703_);
v___x_2705_ = v___x_2690_;
goto v_reusejp_2704_;
}
else
{
lean_object* v_reuseFailAlloc_2707_; 
v_reuseFailAlloc_2707_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2707_, 0, v___x_2703_);
lean_ctor_set(v_reuseFailAlloc_2707_, 1, v_x_2677_);
v___x_2705_ = v_reuseFailAlloc_2707_;
goto v_reusejp_2704_;
}
v_reusejp_2704_:
{
v_x_2676_ = v_tail_2688_;
v_x_2677_ = v___x_2705_;
goto _start;
}
}
else
{
lean_object* v_a_2708_; lean_object* v___x_2710_; uint8_t v_isShared_2711_; uint8_t v_isSharedCheck_2715_; 
lean_dec_ref(v_type_2695_);
lean_dec(v_levelParams_2694_);
lean_dec(v_all_2693_);
lean_dec(v_snd_2692_);
lean_del_object(v___x_2690_);
lean_dec(v_tail_2688_);
lean_dec(v_x_2677_);
lean_dec(v___x_2674_);
v_a_2708_ = lean_ctor_get(v___x_2698_, 0);
v_isSharedCheck_2715_ = !lean_is_exclusive(v___x_2698_);
if (v_isSharedCheck_2715_ == 0)
{
v___x_2710_ = v___x_2698_;
v_isShared_2711_ = v_isSharedCheck_2715_;
goto v_resetjp_2709_;
}
else
{
lean_inc(v_a_2708_);
lean_dec(v___x_2698_);
v___x_2710_ = lean_box(0);
v_isShared_2711_ = v_isSharedCheck_2715_;
goto v_resetjp_2709_;
}
v_resetjp_2709_:
{
lean_object* v___x_2713_; 
if (v_isShared_2711_ == 0)
{
v___x_2713_ = v___x_2710_;
goto v_reusejp_2712_;
}
else
{
lean_object* v_reuseFailAlloc_2714_; 
v_reuseFailAlloc_2714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2714_, 0, v_a_2708_);
v___x_2713_ = v_reuseFailAlloc_2714_;
goto v_reusejp_2712_;
}
v_reusejp_2712_:
{
return v___x_2713_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7___boxed(lean_object* v___x_2718_, lean_object* v___y_2719_, lean_object* v_x_2720_, lean_object* v_x_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_){
_start:
{
uint8_t v___y_17121__boxed_2727_; lean_object* v_res_2728_; 
v___y_17121__boxed_2727_ = lean_unbox(v___y_2719_);
v_res_2728_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7(v___x_2718_, v___y_17121__boxed_2727_, v_x_2720_, v_x_2721_, v___y_2722_, v___y_2723_, v___y_2724_, v___y_2725_);
lean_dec(v___y_2725_);
lean_dec_ref(v___y_2724_);
lean_dec(v___y_2723_);
lean_dec_ref(v___y_2722_);
return v_res_2728_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1(void){
_start:
{
lean_object* v___x_2730_; lean_object* v___x_2731_; 
v___x_2730_ = ((lean_object*)(lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__0));
v___x_2731_ = l_Lean_stringToMessageData(v___x_2730_);
return v___x_2731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly(lean_object* v_iv_2732_, lean_object* v_rv_2733_, uint8_t v_warn_2734_, lean_object* v_a_2735_, lean_object* v_a_2736_, lean_object* v_a_2737_, lean_object* v_a_2738_){
_start:
{
lean_object* v_toConstantVal_2743_; lean_object* v_numMotives_2744_; lean_object* v_name_2745_; lean_object* v_levelParams_2746_; lean_object* v_type_2747_; lean_object* v___x_2748_; 
v_toConstantVal_2743_ = lean_ctor_get(v_rv_2733_, 0);
v_numMotives_2744_ = lean_ctor_get(v_rv_2733_, 4);
v_name_2745_ = lean_ctor_get(v_toConstantVal_2743_, 0);
v_levelParams_2746_ = lean_ctor_get(v_toConstantVal_2743_, 1);
v_type_2747_ = lean_ctor_get(v_toConstantVal_2743_, 2);
lean_inc_ref(v_type_2747_);
v___x_2748_ = l_Lean_Meta_isProp(v_type_2747_, v_a_2735_, v_a_2736_, v_a_2737_, v_a_2738_);
if (lean_obj_tag(v___x_2748_) == 0)
{
lean_object* v_a_2749_; uint8_t v___x_2750_; 
v_a_2749_ = lean_ctor_get(v___x_2748_, 0);
lean_inc(v_a_2749_);
lean_dec_ref_known(v___x_2748_, 1);
v___x_2750_ = lean_unbox(v_a_2749_);
if (v___x_2750_ == 0)
{
lean_object* v_numIndices_2751_; lean_object* v_all_2752_; uint8_t v_isRec_2753_; uint8_t v___y_2755_; lean_object* v___y_2756_; lean_object* v_rvs_2757_; lean_object* v___y_2758_; lean_object* v___y_2759_; lean_object* v___y_2760_; lean_object* v___y_2761_; uint8_t v___y_2799_; 
v_numIndices_2751_ = lean_ctor_get(v_iv_2732_, 2);
v_all_2752_ = lean_ctor_get(v_iv_2732_, 3);
v_isRec_2753_ = lean_ctor_get_uint8(v_iv_2732_, sizeof(void*)*6);
if (v_isRec_2753_ == 0)
{
lean_object* v___x_2816_; uint8_t v___x_2817_; 
lean_dec(v_a_2749_);
v___x_2816_ = lean_unsigned_to_nat(1u);
v___x_2817_ = lean_nat_dec_eq(v_numMotives_2744_, v___x_2816_);
if (v___x_2817_ == 0)
{
lean_inc(v_all_2752_);
lean_dec_ref(v_iv_2732_);
v___y_2799_ = v___x_2817_;
goto v___jp_2798_;
}
else
{
lean_object* v___x_2818_; uint8_t v___x_2819_; 
v___x_2818_ = l_Lean_InductiveVal_numCtors(v_iv_2732_);
v___x_2819_ = lean_nat_dec_eq(v___x_2818_, v___x_2816_);
lean_dec(v___x_2818_);
if (v___x_2819_ == 0)
{
lean_inc(v_all_2752_);
lean_dec_ref(v_iv_2732_);
v___y_2799_ = v___x_2819_;
goto v___jp_2798_;
}
else
{
lean_object* v___x_2820_; uint8_t v___x_2821_; 
v___x_2820_ = lean_unsigned_to_nat(0u);
v___x_2821_ = lean_nat_dec_eq(v_numIndices_2751_, v___x_2820_);
if (v___x_2821_ == 0)
{
lean_inc(v_all_2752_);
lean_dec_ref(v_iv_2732_);
v___y_2799_ = v___x_2821_;
goto v___jp_2798_;
}
else
{
lean_object* v___x_2822_; 
v___x_2822_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_compileStructOnly(v_iv_2732_, v_rv_2733_, v_a_2735_, v_a_2736_, v_a_2737_, v_a_2738_);
if (lean_obj_tag(v___x_2822_) == 0)
{
lean_object* v___x_2824_; uint8_t v_isShared_2825_; uint8_t v_isSharedCheck_2830_; 
v_isSharedCheck_2830_ = !lean_is_exclusive(v___x_2822_);
if (v_isSharedCheck_2830_ == 0)
{
lean_object* v_unused_2831_; 
v_unused_2831_ = lean_ctor_get(v___x_2822_, 0);
lean_dec(v_unused_2831_);
v___x_2824_ = v___x_2822_;
v_isShared_2825_ = v_isSharedCheck_2830_;
goto v_resetjp_2823_;
}
else
{
lean_dec(v___x_2822_);
v___x_2824_ = lean_box(0);
v_isShared_2825_ = v_isSharedCheck_2830_;
goto v_resetjp_2823_;
}
v_resetjp_2823_:
{
lean_object* v___x_2826_; lean_object* v___x_2828_; 
v___x_2826_ = lean_box(0);
if (v_isShared_2825_ == 0)
{
lean_ctor_set(v___x_2824_, 0, v___x_2826_);
v___x_2828_ = v___x_2824_;
goto v_reusejp_2827_;
}
else
{
lean_object* v_reuseFailAlloc_2829_; 
v_reuseFailAlloc_2829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2829_, 0, v___x_2826_);
v___x_2828_ = v_reuseFailAlloc_2829_;
goto v_reusejp_2827_;
}
v_reusejp_2827_:
{
return v___x_2828_;
}
}
}
else
{
return v___x_2822_;
}
}
}
}
}
else
{
uint8_t v___x_2832_; 
lean_inc(v_all_2752_);
lean_dec_ref(v_iv_2732_);
v___x_2832_ = lean_unbox(v_a_2749_);
lean_dec(v_a_2749_);
v___y_2799_ = v___x_2832_;
goto v___jp_2798_;
}
v___jp_2754_:
{
lean_object* v___x_2762_; lean_object* v___x_2763_; 
v___x_2762_ = lean_box(0);
v___x_2763_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg(v_rvs_2757_, v___x_2762_, v___y_2760_, v___y_2761_);
if (lean_obj_tag(v___x_2763_) == 0)
{
lean_object* v_a_2764_; lean_object* v___x_2765_; lean_object* v___x_2766_; lean_object* v___x_2767_; 
v_a_2764_ = lean_ctor_get(v___x_2763_, 0);
lean_inc_n(v_a_2764_, 3);
lean_dec_ref_known(v___x_2763_, 1);
v___x_2765_ = lean_box(0);
v___x_2766_ = lp_mathlib_List_foldl___at___00Mathlib_Util_compileInductiveOnly_spec__1(v___x_2765_, v_a_2764_);
v___x_2767_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__7(v___x_2766_, v___y_2755_, v_a_2764_, v___x_2762_, v___y_2758_, v___y_2759_, v___y_2760_, v___y_2761_);
if (lean_obj_tag(v___x_2767_) == 0)
{
lean_object* v_a_2768_; lean_object* v___x_2769_; lean_object* v___x_2770_; 
v_a_2768_ = lean_ctor_get(v___x_2767_, 0);
lean_inc(v_a_2768_);
lean_dec_ref_known(v___x_2767_, 1);
v___x_2769_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v___x_2769_, 0, v_a_2768_);
v___x_2770_ = lp_mathlib___private_Mathlib_Util_CompileInductive_0__Mathlib_Util_addAndCompile_x27(v___x_2769_, v___y_2760_, v___y_2761_);
if (lean_obj_tag(v___x_2770_) == 0)
{
lean_object* v___x_2771_; lean_object* v___x_2772_; 
lean_dec_ref_known(v___x_2770_, 1);
v___x_2771_ = lean_box(0);
v___x_2772_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg(v___y_2756_, v___y_2755_, v_a_2764_, v___x_2771_, v___y_2758_, v___y_2759_, v___y_2760_, v___y_2761_);
lean_dec(v_a_2764_);
if (lean_obj_tag(v___x_2772_) == 0)
{
lean_object* v___x_2773_; 
lean_dec_ref_known(v___x_2772_, 1);
v___x_2773_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg(v___y_2755_, v_all_2752_, v___x_2771_, v___y_2758_, v___y_2759_, v___y_2760_, v___y_2761_);
lean_dec(v_all_2752_);
if (lean_obj_tag(v___x_2773_) == 0)
{
lean_object* v___x_2775_; uint8_t v_isShared_2776_; uint8_t v_isSharedCheck_2780_; 
v_isSharedCheck_2780_ = !lean_is_exclusive(v___x_2773_);
if (v_isSharedCheck_2780_ == 0)
{
lean_object* v_unused_2781_; 
v_unused_2781_ = lean_ctor_get(v___x_2773_, 0);
lean_dec(v_unused_2781_);
v___x_2775_ = v___x_2773_;
v_isShared_2776_ = v_isSharedCheck_2780_;
goto v_resetjp_2774_;
}
else
{
lean_dec(v___x_2773_);
v___x_2775_ = lean_box(0);
v_isShared_2776_ = v_isSharedCheck_2780_;
goto v_resetjp_2774_;
}
v_resetjp_2774_:
{
lean_object* v___x_2778_; 
if (v_isShared_2776_ == 0)
{
lean_ctor_set(v___x_2775_, 0, v___x_2771_);
v___x_2778_ = v___x_2775_;
goto v_reusejp_2777_;
}
else
{
lean_object* v_reuseFailAlloc_2779_; 
v_reuseFailAlloc_2779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2779_, 0, v___x_2771_);
v___x_2778_ = v_reuseFailAlloc_2779_;
goto v_reusejp_2777_;
}
v_reusejp_2777_:
{
return v___x_2778_;
}
}
}
else
{
return v___x_2773_;
}
}
else
{
lean_dec(v_all_2752_);
return v___x_2772_;
}
}
else
{
lean_dec(v_a_2764_);
lean_dec(v___y_2756_);
lean_dec(v_all_2752_);
return v___x_2770_;
}
}
else
{
lean_object* v_a_2782_; lean_object* v___x_2784_; uint8_t v_isShared_2785_; uint8_t v_isSharedCheck_2789_; 
lean_dec(v_a_2764_);
lean_dec(v___y_2756_);
lean_dec(v_all_2752_);
v_a_2782_ = lean_ctor_get(v___x_2767_, 0);
v_isSharedCheck_2789_ = !lean_is_exclusive(v___x_2767_);
if (v_isSharedCheck_2789_ == 0)
{
v___x_2784_ = v___x_2767_;
v_isShared_2785_ = v_isSharedCheck_2789_;
goto v_resetjp_2783_;
}
else
{
lean_inc(v_a_2782_);
lean_dec(v___x_2767_);
v___x_2784_ = lean_box(0);
v_isShared_2785_ = v_isSharedCheck_2789_;
goto v_resetjp_2783_;
}
v_resetjp_2783_:
{
lean_object* v___x_2787_; 
if (v_isShared_2785_ == 0)
{
v___x_2787_ = v___x_2784_;
goto v_reusejp_2786_;
}
else
{
lean_object* v_reuseFailAlloc_2788_; 
v_reuseFailAlloc_2788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2788_, 0, v_a_2782_);
v___x_2787_ = v_reuseFailAlloc_2788_;
goto v_reusejp_2786_;
}
v_reusejp_2786_:
{
return v___x_2787_;
}
}
}
}
else
{
lean_object* v_a_2790_; lean_object* v___x_2792_; uint8_t v_isShared_2793_; uint8_t v_isSharedCheck_2797_; 
lean_dec(v___y_2756_);
lean_dec(v_all_2752_);
v_a_2790_ = lean_ctor_get(v___x_2763_, 0);
v_isSharedCheck_2797_ = !lean_is_exclusive(v___x_2763_);
if (v_isSharedCheck_2797_ == 0)
{
v___x_2792_ = v___x_2763_;
v_isShared_2793_ = v_isSharedCheck_2797_;
goto v_resetjp_2791_;
}
else
{
lean_inc(v_a_2790_);
lean_dec(v___x_2763_);
v___x_2792_ = lean_box(0);
v_isShared_2793_ = v_isSharedCheck_2797_;
goto v_resetjp_2791_;
}
v_resetjp_2791_:
{
lean_object* v___x_2795_; 
if (v_isShared_2793_ == 0)
{
v___x_2795_ = v___x_2792_;
goto v_reusejp_2794_;
}
else
{
lean_object* v_reuseFailAlloc_2796_; 
v_reuseFailAlloc_2796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2796_, 0, v_a_2790_);
v___x_2795_ = v_reuseFailAlloc_2796_;
goto v_reusejp_2794_;
}
v_reusejp_2794_:
{
return v___x_2795_;
}
}
}
}
v___jp_2798_:
{
lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; uint8_t v___x_2803_; 
v___x_2800_ = lean_box(0);
lean_inc(v_levelParams_2746_);
v___x_2801_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Util_compileDefn_spec__0(v_levelParams_2746_, v___x_2800_);
v___x_2802_ = lean_unsigned_to_nat(1u);
v___x_2803_ = lean_nat_dec_eq(v_numMotives_2744_, v___x_2802_);
if (v___x_2803_ == 0)
{
lean_object* v___x_2804_; lean_object* v___x_2805_; 
lean_inc(v_numMotives_2744_);
lean_dec_ref(v_rv_2733_);
lean_inc(v_all_2752_);
v___x_2804_ = lp_mathlib_Mathlib_Util_mkRecNames(v_all_2752_, v_numMotives_2744_);
lean_dec(v_numMotives_2744_);
v___x_2805_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__11(v___x_2804_, v___x_2800_, v_a_2735_, v_a_2736_, v_a_2737_, v_a_2738_);
if (lean_obj_tag(v___x_2805_) == 0)
{
lean_object* v_a_2806_; 
v_a_2806_ = lean_ctor_get(v___x_2805_, 0);
lean_inc(v_a_2806_);
lean_dec_ref_known(v___x_2805_, 1);
v___y_2755_ = v___y_2799_;
v___y_2756_ = v___x_2801_;
v_rvs_2757_ = v_a_2806_;
v___y_2758_ = v_a_2735_;
v___y_2759_ = v_a_2736_;
v___y_2760_ = v_a_2737_;
v___y_2761_ = v_a_2738_;
goto v___jp_2754_;
}
else
{
lean_object* v_a_2807_; lean_object* v___x_2809_; uint8_t v_isShared_2810_; uint8_t v_isSharedCheck_2814_; 
lean_dec(v___x_2801_);
lean_dec(v_all_2752_);
v_a_2807_ = lean_ctor_get(v___x_2805_, 0);
v_isSharedCheck_2814_ = !lean_is_exclusive(v___x_2805_);
if (v_isSharedCheck_2814_ == 0)
{
v___x_2809_ = v___x_2805_;
v_isShared_2810_ = v_isSharedCheck_2814_;
goto v_resetjp_2808_;
}
else
{
lean_inc(v_a_2807_);
lean_dec(v___x_2805_);
v___x_2809_ = lean_box(0);
v_isShared_2810_ = v_isSharedCheck_2814_;
goto v_resetjp_2808_;
}
v_resetjp_2808_:
{
lean_object* v___x_2812_; 
if (v_isShared_2810_ == 0)
{
v___x_2812_ = v___x_2809_;
goto v_reusejp_2811_;
}
else
{
lean_object* v_reuseFailAlloc_2813_; 
v_reuseFailAlloc_2813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2813_, 0, v_a_2807_);
v___x_2812_ = v_reuseFailAlloc_2813_;
goto v_reusejp_2811_;
}
v_reusejp_2811_:
{
return v___x_2812_;
}
}
}
}
else
{
lean_object* v___x_2815_; 
v___x_2815_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2815_, 0, v_rv_2733_);
lean_ctor_set(v___x_2815_, 1, v___x_2800_);
v___y_2755_ = v___y_2799_;
v___y_2756_ = v___x_2801_;
v_rvs_2757_ = v___x_2815_;
v___y_2758_ = v_a_2735_;
v___y_2759_ = v_a_2736_;
v___y_2760_ = v_a_2737_;
v___y_2761_ = v_a_2738_;
goto v___jp_2754_;
}
}
}
else
{
lean_inc(v_name_2745_);
lean_dec(v_a_2749_);
lean_dec_ref(v_rv_2733_);
lean_dec_ref(v_iv_2732_);
if (v_warn_2734_ == 0)
{
lean_dec(v_name_2745_);
goto v___jp_2740_;
}
else
{
lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; 
v___x_2833_ = lean_obj_once(&lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1, &lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1_once, _init_lp_mathlib_Mathlib_Util_compileInductiveOnly___closed__1);
v___x_2834_ = l_Lean_MessageData_ofName(v_name_2745_);
v___x_2835_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2835_, 0, v___x_2833_);
lean_ctor_set(v___x_2835_, 1, v___x_2834_);
v___x_2836_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12(v___x_2835_, v_a_2735_, v_a_2736_, v_a_2737_, v_a_2738_);
if (lean_obj_tag(v___x_2836_) == 0)
{
lean_dec_ref_known(v___x_2836_, 1);
goto v___jp_2740_;
}
else
{
return v___x_2836_;
}
}
}
}
else
{
lean_object* v_a_2837_; lean_object* v___x_2839_; uint8_t v_isShared_2840_; uint8_t v_isSharedCheck_2844_; 
lean_dec_ref(v_rv_2733_);
lean_dec_ref(v_iv_2732_);
v_a_2837_ = lean_ctor_get(v___x_2748_, 0);
v_isSharedCheck_2844_ = !lean_is_exclusive(v___x_2748_);
if (v_isSharedCheck_2844_ == 0)
{
v___x_2839_ = v___x_2748_;
v_isShared_2840_ = v_isSharedCheck_2844_;
goto v_resetjp_2838_;
}
else
{
lean_inc(v_a_2837_);
lean_dec(v___x_2748_);
v___x_2839_ = lean_box(0);
v_isShared_2840_ = v_isSharedCheck_2844_;
goto v_resetjp_2838_;
}
v_resetjp_2838_:
{
lean_object* v___x_2842_; 
if (v_isShared_2840_ == 0)
{
v___x_2842_ = v___x_2839_;
goto v_reusejp_2841_;
}
else
{
lean_object* v_reuseFailAlloc_2843_; 
v_reuseFailAlloc_2843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2843_, 0, v_a_2837_);
v___x_2842_ = v_reuseFailAlloc_2843_;
goto v_reusejp_2841_;
}
v_reusejp_2841_:
{
return v___x_2842_;
}
}
}
v___jp_2740_:
{
lean_object* v___x_2741_; lean_object* v___x_2742_; 
v___x_2741_ = lean_box(0);
v___x_2742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2742_, 0, v___x_2741_);
return v___x_2742_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductiveOnly___boxed(lean_object* v_iv_2845_, lean_object* v_rv_2846_, lean_object* v_warn_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_, lean_object* v_a_2852_){
_start:
{
uint8_t v_warn_boxed_2853_; lean_object* v_res_2854_; 
v_warn_boxed_2853_ = lean_unbox(v_warn_2847_);
v_res_2854_ = lp_mathlib_Mathlib_Util_compileInductiveOnly(v_iv_2845_, v_rv_2846_, v_warn_boxed_2853_, v_a_2848_, v_a_2849_, v_a_2850_, v_a_2851_);
lean_dec(v_a_2851_);
lean_dec_ref(v_a_2850_);
lean_dec(v_a_2849_);
lean_dec_ref(v_a_2848_);
return v_res_2854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0(lean_object* v_x_2855_, lean_object* v_x_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_, lean_object* v___y_2859_, lean_object* v___y_2860_){
_start:
{
lean_object* v___x_2862_; 
v___x_2862_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___redArg(v_x_2855_, v_x_2856_, v___y_2859_, v___y_2860_);
return v___x_2862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0___boxed(lean_object* v_x_2863_, lean_object* v_x_2864_, lean_object* v___y_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_){
_start:
{
lean_object* v_res_2870_; 
v_res_2870_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Util_compileInductiveOnly_spec__0(v_x_2863_, v_x_2864_, v___y_2865_, v___y_2866_, v___y_2867_, v___y_2868_);
lean_dec(v___y_2868_);
lean_dec_ref(v___y_2867_);
lean_dec(v___y_2866_);
lean_dec_ref(v___y_2865_);
return v_res_2870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8(uint8_t v___y_2871_, lean_object* v_as_2872_, lean_object* v_as_x27_2873_, lean_object* v_b_2874_, lean_object* v_a_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_){
_start:
{
lean_object* v___x_2881_; 
v___x_2881_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___redArg(v___y_2871_, v_as_x27_2873_, v_b_2874_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
return v___x_2881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8___boxed(lean_object* v___y_2882_, lean_object* v_as_2883_, lean_object* v_as_x27_2884_, lean_object* v_b_2885_, lean_object* v_a_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_){
_start:
{
uint8_t v___y_17446__boxed_2892_; lean_object* v_res_2893_; 
v___y_17446__boxed_2892_ = lean_unbox(v___y_2882_);
v_res_2893_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__8(v___y_17446__boxed_2892_, v_as_2883_, v_as_x27_2884_, v_b_2885_, v_a_2886_, v___y_2887_, v___y_2888_, v___y_2889_, v___y_2890_);
lean_dec(v___y_2890_);
lean_dec_ref(v___y_2889_);
lean_dec(v___y_2888_);
lean_dec_ref(v___y_2887_);
lean_dec(v_as_x27_2884_);
lean_dec(v_as_2883_);
return v_res_2893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9(lean_object* v___x_2894_, uint8_t v___y_2895_, lean_object* v_as_2896_, lean_object* v_as_x27_2897_, lean_object* v_b_2898_, lean_object* v_a_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_, lean_object* v___y_2903_){
_start:
{
lean_object* v___x_2905_; 
v___x_2905_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___redArg(v___x_2894_, v___y_2895_, v_as_x27_2897_, v_b_2898_, v___y_2900_, v___y_2901_, v___y_2902_, v___y_2903_);
return v___x_2905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9___boxed(lean_object* v___x_2906_, lean_object* v___y_2907_, lean_object* v_as_2908_, lean_object* v_as_x27_2909_, lean_object* v_b_2910_, lean_object* v_a_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_, lean_object* v___y_2914_, lean_object* v___y_2915_, lean_object* v___y_2916_){
_start:
{
uint8_t v___y_17470__boxed_2917_; lean_object* v_res_2918_; 
v___y_17470__boxed_2917_ = lean_unbox(v___y_2907_);
v_res_2918_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__9(v___x_2906_, v___y_17470__boxed_2917_, v_as_2908_, v_as_x27_2909_, v_b_2910_, v_a_2911_, v___y_2912_, v___y_2913_, v___y_2914_, v___y_2915_);
lean_dec(v___y_2915_);
lean_dec_ref(v___y_2914_);
lean_dec(v___y_2913_);
lean_dec_ref(v___y_2912_);
lean_dec(v_as_x27_2909_);
lean_dec(v_as_2908_);
return v_res_2918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10(uint8_t v___y_2919_, lean_object* v_as_2920_, lean_object* v_as_x27_2921_, lean_object* v_b_2922_, lean_object* v_a_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_){
_start:
{
lean_object* v___x_2929_; 
v___x_2929_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___redArg(v___y_2919_, v_as_x27_2921_, v_b_2922_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_);
return v___x_2929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10___boxed(lean_object* v___y_2930_, lean_object* v_as_2931_, lean_object* v_as_x27_2932_, lean_object* v_b_2933_, lean_object* v_a_2934_, lean_object* v___y_2935_, lean_object* v___y_2936_, lean_object* v___y_2937_, lean_object* v___y_2938_, lean_object* v___y_2939_){
_start:
{
uint8_t v___y_17495__boxed_2940_; lean_object* v_res_2941_; 
v___y_17495__boxed_2940_ = lean_unbox(v___y_2930_);
v_res_2941_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileInductiveOnly_spec__10(v___y_17495__boxed_2940_, v_as_2931_, v_as_x27_2932_, v_b_2933_, v_a_2934_, v___y_2935_, v___y_2936_, v___y_2937_, v___y_2938_);
lean_dec(v___y_2938_);
lean_dec_ref(v___y_2937_);
lean_dec(v___y_2936_);
lean_dec_ref(v___y_2935_);
lean_dec(v_as_x27_2932_);
lean_dec(v_as_2931_);
return v_res_2941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0(lean_object* v_c_2943_, lean_object* v_arr_2944_){
_start:
{
if (lean_obj_tag(v_c_2943_) == 1)
{
lean_object* v_pre_2945_; lean_object* v_str_2946_; lean_object* v___x_2947_; uint8_t v___x_2948_; 
v_pre_2945_ = lean_ctor_get(v_c_2943_, 0);
lean_inc(v_pre_2945_);
v_str_2946_ = lean_ctor_get(v_c_2943_, 1);
lean_inc_ref(v_str_2946_);
lean_dec_ref_known(v_c_2943_, 2);
v___x_2947_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0));
v___x_2948_ = lean_string_dec_eq(v_str_2946_, v___x_2947_);
lean_dec_ref(v_str_2946_);
if (v___x_2948_ == 0)
{
lean_dec(v_pre_2945_);
return v_arr_2944_;
}
else
{
lean_object* v___x_2949_; 
v___x_2949_ = l_Lean_NameSet_insert(v_arr_2944_, v_pre_2945_);
return v___x_2949_;
}
}
else
{
lean_dec(v_c_2943_);
return v_arr_2944_;
}
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_2952_; lean_object* v___x_2953_; lean_object* v___x_2954_; 
v___x_2952_ = lean_box(0);
v___x_2953_ = lean_unsigned_to_nat(16u);
v___x_2954_ = lean_mk_array(v___x_2953_, v___x_2952_);
return v___x_2954_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_2955_; lean_object* v___x_2956_; lean_object* v___x_2957_; 
v___x_2955_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__2);
v___x_2956_ = lean_unsigned_to_nat(0u);
v___x_2957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2957_, 0, v___x_2956_);
lean_ctor_set(v___x_2957_, 1, v___x_2955_);
return v___x_2957_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_2958_; lean_object* v___x_2959_; 
v___x_2958_ = lean_unsigned_to_nat(64u);
v___x_2959_ = l_Lean_mkPtrSet___redArg(v___x_2958_);
return v___x_2959_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4(void){
_start:
{
lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; 
v___x_2960_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__3);
v___x_2961_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__1);
v___x_2962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2962_, 0, v___x_2961_);
lean_ctor_set(v___x_2962_, 1, v___x_2960_);
return v___x_2962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductive(lean_object* v_iv_2963_, uint8_t v_warn_2964_, lean_object* v_a_2965_, lean_object* v_a_2966_, lean_object* v_a_2967_, lean_object* v_a_2968_){
_start:
{
lean_object* v_toConstantVal_2973_; lean_object* v_name_2974_; lean_object* v___x_2975_; lean_object* v___x_2976_; 
v_toConstantVal_2973_ = lean_ctor_get(v_iv_2963_, 0);
v_name_2974_ = lean_ctor_get(v_toConstantVal_2973_, 0);
lean_inc(v_name_2974_);
v___x_2975_ = l_Lean_mkRecName(v_name_2974_);
v___x_2976_ = lp_mathlib_Lean_getConstInfoRec___at___00Mathlib_Util_compileInductiveOnly_spec__3(v___x_2975_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_);
if (lean_obj_tag(v___x_2976_) == 0)
{
lean_object* v_a_2977_; lean_object* v___x_2978_; lean_object* v_toConstantVal_2979_; lean_object* v_env_2980_; lean_object* v_name_2981_; uint8_t v___x_2982_; 
v_a_2977_ = lean_ctor_get(v___x_2976_, 0);
lean_inc(v_a_2977_);
lean_dec_ref_known(v___x_2976_, 1);
v___x_2978_ = lean_st_ref_get(v_a_2968_);
v_toConstantVal_2979_ = lean_ctor_get(v_a_2977_, 0);
v_env_2980_ = lean_ctor_get(v___x_2978_, 0);
lean_inc_ref(v_env_2980_);
lean_dec(v___x_2978_);
v_name_2981_ = lean_ctor_get(v_toConstantVal_2979_, 0);
v___x_2982_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_2980_, v_name_2981_);
if (v___x_2982_ == 0)
{
lean_object* v___x_2983_; 
lean_inc(v_a_2977_);
lean_inc_ref(v_iv_2963_);
v___x_2983_ = lp_mathlib_Mathlib_Util_compileInductiveOnly(v_iv_2963_, v_a_2977_, v_warn_2964_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_);
if (lean_obj_tag(v___x_2983_) == 0)
{
lean_object* v___x_2984_; 
lean_dec_ref_known(v___x_2983_, 1);
v___x_2984_ = lp_mathlib_Mathlib_Util_compileSizeOf(v_iv_2963_, v_a_2977_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_);
lean_dec(v_a_2977_);
lean_dec_ref(v_iv_2963_);
return v___x_2984_;
}
else
{
lean_dec(v_a_2977_);
lean_dec_ref(v_iv_2963_);
return v___x_2983_;
}
}
else
{
lean_inc(v_name_2981_);
lean_dec(v_a_2977_);
lean_dec_ref(v_iv_2963_);
if (v_warn_2964_ == 0)
{
lean_dec(v_name_2981_);
goto v___jp_2970_;
}
else
{
lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; lean_object* v___x_2988_; 
v___x_2985_ = lean_obj_once(&lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1, &lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1___lam__0___closed__1);
v___x_2986_ = l_Lean_MessageData_ofName(v_name_2981_);
v___x_2987_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2987_, 0, v___x_2985_);
lean_ctor_set(v___x_2987_, 1, v___x_2986_);
v___x_2988_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Util_compileInductiveOnly_spec__12(v___x_2987_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_);
if (lean_obj_tag(v___x_2988_) == 0)
{
lean_dec_ref_known(v___x_2988_, 1);
goto v___jp_2970_;
}
else
{
return v___x_2988_;
}
}
}
}
else
{
lean_object* v_a_2989_; lean_object* v___x_2991_; uint8_t v_isShared_2992_; uint8_t v_isSharedCheck_2996_; 
lean_dec_ref(v_iv_2963_);
v_a_2989_ = lean_ctor_get(v___x_2976_, 0);
v_isSharedCheck_2996_ = !lean_is_exclusive(v___x_2976_);
if (v_isSharedCheck_2996_ == 0)
{
v___x_2991_ = v___x_2976_;
v_isShared_2992_ = v_isSharedCheck_2996_;
goto v_resetjp_2990_;
}
else
{
lean_inc(v_a_2989_);
lean_dec(v___x_2976_);
v___x_2991_ = lean_box(0);
v_isShared_2992_ = v_isSharedCheck_2996_;
goto v_resetjp_2990_;
}
v_resetjp_2990_:
{
lean_object* v___x_2994_; 
if (v_isShared_2992_ == 0)
{
v___x_2994_ = v___x_2991_;
goto v_reusejp_2993_;
}
else
{
lean_object* v_reuseFailAlloc_2995_; 
v_reuseFailAlloc_2995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2995_, 0, v_a_2989_);
v___x_2994_ = v_reuseFailAlloc_2995_;
goto v_reusejp_2993_;
}
v_reusejp_2993_:
{
return v___x_2994_;
}
}
}
v___jp_2970_:
{
lean_object* v___x_2971_; lean_object* v___x_2972_; 
v___x_2971_ = lean_box(0);
v___x_2972_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2972_, 0, v___x_2971_);
return v___x_2972_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(uint8_t v___x_2997_, lean_object* v_init_2998_, lean_object* v_x_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_){
_start:
{
if (lean_obj_tag(v_x_2999_) == 0)
{
lean_object* v_k_3005_; lean_object* v_l_3006_; lean_object* v_r_3007_; lean_object* v___x_3008_; 
v_k_3005_ = lean_ctor_get(v_x_2999_, 1);
lean_inc(v_k_3005_);
v_l_3006_ = lean_ctor_get(v_x_2999_, 3);
lean_inc(v_l_3006_);
v_r_3007_ = lean_ctor_get(v_x_2999_, 4);
lean_inc(v_r_3007_);
lean_dec_ref_known(v_x_2999_, 5);
v___x_3008_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_2997_, v_init_2998_, v_l_3006_, v___y_3000_, v___y_3001_, v___y_3002_, v___y_3003_);
if (lean_obj_tag(v___x_3008_) == 0)
{
lean_object* v___x_3009_; lean_object* v_env_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; 
lean_dec_ref_known(v___x_3008_, 1);
v___x_3009_ = lean_st_ref_get(v___y_3003_);
v_env_3010_ = lean_ctor_get(v___x_3009_, 0);
lean_inc_ref(v_env_3010_);
lean_dec(v___x_3009_);
v___x_3011_ = lean_box(0);
v___x_3012_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_3010_, v_k_3005_);
lean_dec_ref(v_env_3010_);
if (lean_obj_tag(v___x_3012_) == 0)
{
lean_dec(v_k_3005_);
v_init_2998_ = v___x_3011_;
v_x_2999_ = v_r_3007_;
goto _start;
}
else
{
lean_object* v___x_3014_; lean_object* v_env_3015_; lean_object* v___x_3016_; 
lean_dec_ref_known(v___x_3012_, 1);
v___x_3014_ = lean_st_ref_get(v___y_3003_);
v_env_3015_ = lean_ctor_get(v___x_3014_, 0);
lean_inc_ref(v_env_3015_);
lean_dec(v___x_3014_);
v___x_3016_ = l_Lean_Environment_find_x3f(v_env_3015_, v_k_3005_, v___x_2997_);
if (lean_obj_tag(v___x_3016_) == 1)
{
lean_object* v_val_3017_; 
v_val_3017_ = lean_ctor_get(v___x_3016_, 0);
lean_inc(v_val_3017_);
lean_dec_ref_known(v___x_3016_, 1);
if (lean_obj_tag(v_val_3017_) == 5)
{
lean_object* v_val_3018_; lean_object* v___x_3019_; 
v_val_3018_ = lean_ctor_get(v_val_3017_, 0);
lean_inc_ref(v_val_3018_);
lean_dec_ref_known(v_val_3017_, 1);
v___x_3019_ = lp_mathlib_Mathlib_Util_compileInductive(v_val_3018_, v___x_2997_, v___y_3000_, v___y_3001_, v___y_3002_, v___y_3003_);
if (lean_obj_tag(v___x_3019_) == 0)
{
lean_dec_ref_known(v___x_3019_, 1);
v_init_2998_ = v___x_3011_;
v_x_2999_ = v_r_3007_;
goto _start;
}
else
{
lean_object* v_a_3021_; lean_object* v___x_3023_; uint8_t v_isShared_3024_; uint8_t v_isSharedCheck_3028_; 
lean_dec(v_r_3007_);
v_a_3021_ = lean_ctor_get(v___x_3019_, 0);
v_isSharedCheck_3028_ = !lean_is_exclusive(v___x_3019_);
if (v_isSharedCheck_3028_ == 0)
{
v___x_3023_ = v___x_3019_;
v_isShared_3024_ = v_isSharedCheck_3028_;
goto v_resetjp_3022_;
}
else
{
lean_inc(v_a_3021_);
lean_dec(v___x_3019_);
v___x_3023_ = lean_box(0);
v_isShared_3024_ = v_isSharedCheck_3028_;
goto v_resetjp_3022_;
}
v_resetjp_3022_:
{
lean_object* v___x_3026_; 
if (v_isShared_3024_ == 0)
{
v___x_3026_ = v___x_3023_;
goto v_reusejp_3025_;
}
else
{
lean_object* v_reuseFailAlloc_3027_; 
v_reuseFailAlloc_3027_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3027_, 0, v_a_3021_);
v___x_3026_ = v_reuseFailAlloc_3027_;
goto v_reusejp_3025_;
}
v_reusejp_3025_:
{
return v___x_3026_;
}
}
}
}
else
{
lean_dec(v_val_3017_);
v_init_2998_ = v___x_3011_;
v_x_2999_ = v_r_3007_;
goto _start;
}
}
else
{
lean_dec(v___x_3016_);
v_init_2998_ = v___x_3011_;
v_x_2999_ = v_r_3007_;
goto _start;
}
}
}
else
{
lean_dec(v_r_3007_);
lean_dec(v_k_3005_);
return v___x_3008_;
}
}
else
{
lean_object* v___x_3031_; lean_object* v___x_3032_; 
v___x_3031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3031_, 0, v_init_2998_);
v___x_3032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3032_, 0, v___x_3031_);
return v___x_3032_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg(lean_object* v_a_3033_, lean_object* v_range_3034_, lean_object* v_b_3035_, lean_object* v_i_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_){
_start:
{
lean_object* v_stop_3042_; lean_object* v_step_3043_; lean_object* v_a_3045_; uint8_t v___x_3048_; 
v_stop_3042_ = lean_ctor_get(v_range_3034_, 1);
v_step_3043_ = lean_ctor_get(v_range_3034_, 2);
v___x_3048_ = lean_nat_dec_lt(v_i_3036_, v_stop_3042_);
if (v___x_3048_ == 0)
{
lean_object* v___x_3049_; 
lean_dec(v_i_3036_);
lean_dec(v_a_3033_);
v___x_3049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3049_, 0, v_b_3035_);
return v___x_3049_;
}
else
{
lean_object* v___x_3050_; lean_object* v_env_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; uint8_t v___x_3059_; lean_object* v___x_3060_; 
v___x_3050_ = lean_st_ref_get(v___y_3040_);
v_env_3051_ = lean_ctor_get(v___x_3050_, 0);
lean_inc_ref(v_env_3051_);
lean_dec(v___x_3050_);
v___x_3052_ = lean_unsigned_to_nat(1u);
v___x_3053_ = lean_box(0);
v___x_3054_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___closed__0));
v___x_3055_ = lean_nat_add(v_i_3036_, v___x_3052_);
v___x_3056_ = l_Nat_reprFast(v___x_3055_);
v___x_3057_ = lean_string_append(v___x_3054_, v___x_3056_);
lean_dec_ref(v___x_3056_);
lean_inc(v_a_3033_);
v___x_3058_ = l_Lean_Name_str___override(v_a_3033_, v___x_3057_);
v___x_3059_ = 0;
lean_inc(v___x_3058_);
v___x_3060_ = l_Lean_Environment_find_x3f(v_env_3051_, v___x_3058_, v___x_3059_);
if (lean_obj_tag(v___x_3060_) == 1)
{
lean_object* v_val_3061_; 
v_val_3061_ = lean_ctor_get(v___x_3060_, 0);
lean_inc(v_val_3061_);
lean_dec_ref_known(v___x_3060_, 1);
if (lean_obj_tag(v_val_3061_) == 1)
{
lean_object* v_val_3062_; lean_object* v___x_3063_; lean_object* v_env_3064_; uint8_t v___x_3065_; 
v_val_3062_ = lean_ctor_get(v_val_3061_, 0);
lean_inc_ref(v_val_3062_);
lean_dec_ref_known(v_val_3061_, 1);
v___x_3063_ = lean_st_ref_get(v___y_3040_);
v_env_3064_ = lean_ctor_get(v___x_3063_, 0);
lean_inc_ref(v_env_3064_);
lean_dec(v___x_3063_);
v___x_3065_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_3064_, v___x_3058_);
lean_dec(v___x_3058_);
if (v___x_3065_ == 0)
{
lean_object* v_value_3066_; lean_object* v___f_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3070_; lean_object* v_fst_3071_; lean_object* v___x_3072_; 
v_value_3066_ = lean_ctor_get(v_val_3062_, 1);
v___f_3067_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0));
v___x_3068_ = l_Lean_NameSet_empty;
v___x_3069_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4);
lean_inc_ref(v_value_3066_);
v___x_3070_ = l_Lean_Expr_FoldConstsImpl_fold___redArg(v___f_3067_, v_value_3066_, v___x_3068_, v___x_3069_);
v_fst_3071_ = lean_ctor_get(v___x_3070_, 0);
lean_inc(v_fst_3071_);
lean_dec_ref(v___x_3070_);
v___x_3072_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_3065_, v___x_3053_, v_fst_3071_, v___y_3037_, v___y_3038_, v___y_3039_, v___y_3040_);
if (lean_obj_tag(v___x_3072_) == 0)
{
lean_object* v___x_3073_; 
lean_dec_ref_known(v___x_3072_, 1);
v___x_3073_ = lp_mathlib_Mathlib_Util_compileDefn(v_val_3062_, v___y_3037_, v___y_3038_, v___y_3039_, v___y_3040_);
if (lean_obj_tag(v___x_3073_) == 0)
{
lean_dec_ref_known(v___x_3073_, 1);
v_a_3045_ = v___x_3053_;
goto v___jp_3044_;
}
else
{
lean_dec(v_i_3036_);
lean_dec(v_a_3033_);
return v___x_3073_;
}
}
else
{
lean_dec_ref(v_val_3062_);
if (lean_obj_tag(v___x_3072_) == 0)
{
lean_object* v_a_3074_; lean_object* v_a_3075_; 
v_a_3074_ = lean_ctor_get(v___x_3072_, 0);
lean_inc(v_a_3074_);
lean_dec_ref_known(v___x_3072_, 1);
v_a_3075_ = lean_ctor_get(v_a_3074_, 0);
lean_inc(v_a_3075_);
lean_dec(v_a_3074_);
v_a_3045_ = v_a_3075_;
goto v___jp_3044_;
}
else
{
lean_object* v_a_3076_; lean_object* v___x_3078_; uint8_t v_isShared_3079_; uint8_t v_isSharedCheck_3083_; 
lean_dec(v_i_3036_);
lean_dec(v_a_3033_);
v_a_3076_ = lean_ctor_get(v___x_3072_, 0);
v_isSharedCheck_3083_ = !lean_is_exclusive(v___x_3072_);
if (v_isSharedCheck_3083_ == 0)
{
v___x_3078_ = v___x_3072_;
v_isShared_3079_ = v_isSharedCheck_3083_;
goto v_resetjp_3077_;
}
else
{
lean_inc(v_a_3076_);
lean_dec(v___x_3072_);
v___x_3078_ = lean_box(0);
v_isShared_3079_ = v_isSharedCheck_3083_;
goto v_resetjp_3077_;
}
v_resetjp_3077_:
{
lean_object* v___x_3081_; 
if (v_isShared_3079_ == 0)
{
v___x_3081_ = v___x_3078_;
goto v_reusejp_3080_;
}
else
{
lean_object* v_reuseFailAlloc_3082_; 
v_reuseFailAlloc_3082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3082_, 0, v_a_3076_);
v___x_3081_ = v_reuseFailAlloc_3082_;
goto v_reusejp_3080_;
}
v_reusejp_3080_:
{
return v___x_3081_;
}
}
}
}
}
else
{
lean_dec_ref(v_val_3062_);
v_a_3045_ = v___x_3053_;
goto v___jp_3044_;
}
}
else
{
lean_dec(v_val_3061_);
lean_dec(v___x_3058_);
v_a_3045_ = v___x_3053_;
goto v___jp_3044_;
}
}
else
{
lean_dec(v___x_3060_);
lean_dec(v___x_3058_);
v_a_3045_ = v___x_3053_;
goto v___jp_3044_;
}
}
v___jp_3044_:
{
lean_object* v___x_3046_; 
v___x_3046_ = lean_nat_add(v_i_3036_, v_step_3043_);
lean_dec(v_i_3036_);
v_b_3035_ = v_a_3045_;
v_i_3036_ = v___x_3046_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(lean_object* v_a_3084_, lean_object* v_range_3085_, lean_object* v_b_3086_, lean_object* v_i_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_, lean_object* v___y_3091_){
_start:
{
lean_object* v_stop_3093_; lean_object* v_step_3094_; lean_object* v_a_3096_; uint8_t v___x_3099_; 
v_stop_3093_ = lean_ctor_get(v_range_3085_, 1);
v_step_3094_ = lean_ctor_get(v_range_3085_, 2);
v___x_3099_ = lean_nat_dec_lt(v_i_3087_, v_stop_3093_);
if (v___x_3099_ == 0)
{
lean_object* v___x_3100_; 
lean_dec(v_a_3084_);
v___x_3100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3100_, 0, v_b_3086_);
return v___x_3100_;
}
else
{
lean_object* v___x_3101_; lean_object* v_env_3102_; lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; uint8_t v___x_3110_; lean_object* v___x_3111_; 
v___x_3101_ = lean_st_ref_get(v___y_3091_);
v_env_3102_ = lean_ctor_get(v___x_3101_, 0);
lean_inc_ref(v_env_3102_);
lean_dec(v___x_3101_);
v___x_3103_ = lean_unsigned_to_nat(1u);
v___x_3104_ = lean_box(0);
v___x_3105_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___closed__0));
v___x_3106_ = lean_nat_add(v_i_3087_, v___x_3103_);
v___x_3107_ = l_Nat_reprFast(v___x_3106_);
v___x_3108_ = lean_string_append(v___x_3105_, v___x_3107_);
lean_dec_ref(v___x_3107_);
lean_inc(v_a_3084_);
v___x_3109_ = l_Lean_Name_str___override(v_a_3084_, v___x_3108_);
v___x_3110_ = 0;
lean_inc(v___x_3109_);
v___x_3111_ = l_Lean_Environment_find_x3f(v_env_3102_, v___x_3109_, v___x_3110_);
if (lean_obj_tag(v___x_3111_) == 1)
{
lean_object* v_val_3112_; 
v_val_3112_ = lean_ctor_get(v___x_3111_, 0);
lean_inc(v_val_3112_);
lean_dec_ref_known(v___x_3111_, 1);
if (lean_obj_tag(v_val_3112_) == 1)
{
lean_object* v_val_3113_; lean_object* v___x_3114_; lean_object* v_env_3115_; uint8_t v___x_3116_; 
v_val_3113_ = lean_ctor_get(v_val_3112_, 0);
lean_inc_ref(v_val_3113_);
lean_dec_ref_known(v_val_3112_, 1);
v___x_3114_ = lean_st_ref_get(v___y_3091_);
v_env_3115_ = lean_ctor_get(v___x_3114_, 0);
lean_inc_ref(v_env_3115_);
lean_dec(v___x_3114_);
v___x_3116_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_3115_, v___x_3109_);
lean_dec(v___x_3109_);
if (v___x_3116_ == 0)
{
lean_object* v_value_3117_; lean_object* v___f_3118_; lean_object* v___x_3119_; lean_object* v___x_3120_; lean_object* v___x_3121_; lean_object* v_fst_3122_; lean_object* v___x_3123_; 
v_value_3117_ = lean_ctor_get(v_val_3113_, 1);
v___f_3118_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0));
v___x_3119_ = l_Lean_NameSet_empty;
v___x_3120_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4);
lean_inc_ref(v_value_3117_);
v___x_3121_ = l_Lean_Expr_FoldConstsImpl_fold___redArg(v___f_3118_, v_value_3117_, v___x_3119_, v___x_3120_);
v_fst_3122_ = lean_ctor_get(v___x_3121_, 0);
lean_inc(v_fst_3122_);
lean_dec_ref(v___x_3121_);
v___x_3123_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_3116_, v___x_3104_, v_fst_3122_, v___y_3088_, v___y_3089_, v___y_3090_, v___y_3091_);
if (lean_obj_tag(v___x_3123_) == 0)
{
lean_object* v___x_3124_; 
lean_dec_ref_known(v___x_3123_, 1);
v___x_3124_ = lp_mathlib_Mathlib_Util_compileDefn(v_val_3113_, v___y_3088_, v___y_3089_, v___y_3090_, v___y_3091_);
if (lean_obj_tag(v___x_3124_) == 0)
{
lean_dec_ref_known(v___x_3124_, 1);
v_a_3096_ = v___x_3104_;
goto v___jp_3095_;
}
else
{
lean_dec(v_a_3084_);
return v___x_3124_;
}
}
else
{
lean_dec_ref(v_val_3113_);
if (lean_obj_tag(v___x_3123_) == 0)
{
lean_object* v_a_3125_; lean_object* v_a_3126_; 
v_a_3125_ = lean_ctor_get(v___x_3123_, 0);
lean_inc(v_a_3125_);
lean_dec_ref_known(v___x_3123_, 1);
v_a_3126_ = lean_ctor_get(v_a_3125_, 0);
lean_inc(v_a_3126_);
lean_dec(v_a_3125_);
v_a_3096_ = v_a_3126_;
goto v___jp_3095_;
}
else
{
lean_object* v_a_3127_; lean_object* v___x_3129_; uint8_t v_isShared_3130_; uint8_t v_isSharedCheck_3134_; 
lean_dec(v_a_3084_);
v_a_3127_ = lean_ctor_get(v___x_3123_, 0);
v_isSharedCheck_3134_ = !lean_is_exclusive(v___x_3123_);
if (v_isSharedCheck_3134_ == 0)
{
v___x_3129_ = v___x_3123_;
v_isShared_3130_ = v_isSharedCheck_3134_;
goto v_resetjp_3128_;
}
else
{
lean_inc(v_a_3127_);
lean_dec(v___x_3123_);
v___x_3129_ = lean_box(0);
v_isShared_3130_ = v_isSharedCheck_3134_;
goto v_resetjp_3128_;
}
v_resetjp_3128_:
{
lean_object* v___x_3132_; 
if (v_isShared_3130_ == 0)
{
v___x_3132_ = v___x_3129_;
goto v_reusejp_3131_;
}
else
{
lean_object* v_reuseFailAlloc_3133_; 
v_reuseFailAlloc_3133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3133_, 0, v_a_3127_);
v___x_3132_ = v_reuseFailAlloc_3133_;
goto v_reusejp_3131_;
}
v_reusejp_3131_:
{
return v___x_3132_;
}
}
}
}
}
else
{
lean_dec_ref(v_val_3113_);
v_a_3096_ = v___x_3104_;
goto v___jp_3095_;
}
}
else
{
lean_dec(v_val_3112_);
lean_dec(v___x_3109_);
v_a_3096_ = v___x_3104_;
goto v___jp_3095_;
}
}
else
{
lean_dec(v___x_3111_);
lean_dec(v___x_3109_);
v_a_3096_ = v___x_3104_;
goto v___jp_3095_;
}
}
v___jp_3095_:
{
lean_object* v___x_3097_; lean_object* v___x_3098_; 
v___x_3097_ = lean_nat_add(v_i_3087_, v_step_3094_);
v___x_3098_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg(v_a_3084_, v_range_3085_, v_a_3096_, v___x_3097_, v___y_3088_, v___y_3089_, v___y_3090_, v___y_3091_);
return v___x_3098_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(lean_object* v_rv_3135_, lean_object* v_as_x27_3136_, lean_object* v_b_3137_, lean_object* v___y_3138_, lean_object* v___y_3139_, lean_object* v___y_3140_, lean_object* v___y_3141_){
_start:
{
if (lean_obj_tag(v_as_x27_3136_) == 0)
{
lean_object* v___x_3143_; 
v___x_3143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3143_, 0, v_b_3137_);
return v___x_3143_;
}
else
{
lean_object* v_head_3144_; lean_object* v_tail_3145_; lean_object* v_numMotives_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v___x_3151_; 
v_head_3144_ = lean_ctor_get(v_as_x27_3136_, 0);
v_tail_3145_ = lean_ctor_get(v_as_x27_3136_, 1);
v_numMotives_3146_ = lean_ctor_get(v_rv_3135_, 4);
v___x_3147_ = lean_box(0);
v___x_3148_ = lean_unsigned_to_nat(0u);
v___x_3149_ = lean_unsigned_to_nat(1u);
lean_inc(v_numMotives_3146_);
v___x_3150_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3150_, 0, v___x_3148_);
lean_ctor_set(v___x_3150_, 1, v_numMotives_3146_);
lean_ctor_set(v___x_3150_, 2, v___x_3149_);
lean_inc(v_head_3144_);
v___x_3151_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(v_head_3144_, v___x_3150_, v___x_3147_, v___x_3148_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
lean_dec_ref_known(v___x_3150_, 3);
if (lean_obj_tag(v___x_3151_) == 0)
{
lean_object* v___x_3152_; lean_object* v_env_3153_; lean_object* v___x_3154_; lean_object* v___x_3155_; uint8_t v___x_3156_; lean_object* v___x_3157_; 
lean_dec_ref_known(v___x_3151_, 1);
v___x_3152_ = lean_st_ref_get(v___y_3141_);
v_env_3153_ = lean_ctor_get(v___x_3152_, 0);
lean_inc_ref(v_env_3153_);
lean_dec(v___x_3152_);
v___x_3154_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0));
lean_inc(v_head_3144_);
v___x_3155_ = l_Lean_Name_str___override(v_head_3144_, v___x_3154_);
v___x_3156_ = 0;
lean_inc(v___x_3155_);
v___x_3157_ = l_Lean_Environment_find_x3f(v_env_3153_, v___x_3155_, v___x_3156_);
if (lean_obj_tag(v___x_3157_) == 1)
{
lean_object* v_val_3158_; 
v_val_3158_ = lean_ctor_get(v___x_3157_, 0);
lean_inc(v_val_3158_);
lean_dec_ref_known(v___x_3157_, 1);
if (lean_obj_tag(v_val_3158_) == 1)
{
lean_object* v_val_3159_; lean_object* v___x_3160_; lean_object* v_env_3161_; uint8_t v___x_3162_; 
v_val_3159_ = lean_ctor_get(v_val_3158_, 0);
lean_inc_ref(v_val_3159_);
lean_dec_ref_known(v_val_3158_, 1);
v___x_3160_ = lean_st_ref_get(v___y_3141_);
v_env_3161_ = lean_ctor_get(v___x_3160_, 0);
lean_inc_ref(v_env_3161_);
lean_dec(v___x_3160_);
v___x_3162_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_3161_, v___x_3155_);
lean_dec(v___x_3155_);
if (v___x_3162_ == 0)
{
lean_object* v_value_3163_; lean_object* v___f_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v_fst_3168_; lean_object* v___x_3169_; 
v_value_3163_ = lean_ctor_get(v_val_3159_, 1);
v___f_3164_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0));
v___x_3165_ = l_Lean_NameSet_empty;
v___x_3166_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4);
lean_inc_ref(v_value_3163_);
v___x_3167_ = l_Lean_Expr_FoldConstsImpl_fold___redArg(v___f_3164_, v_value_3163_, v___x_3165_, v___x_3166_);
v_fst_3168_ = lean_ctor_get(v___x_3167_, 0);
lean_inc(v_fst_3168_);
lean_dec_ref(v___x_3167_);
v___x_3169_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_3162_, v___x_3147_, v_fst_3168_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
if (lean_obj_tag(v___x_3169_) == 0)
{
lean_object* v___x_3170_; 
lean_dec_ref_known(v___x_3169_, 1);
v___x_3170_ = lp_mathlib_Mathlib_Util_compileDefn(v_val_3159_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
if (lean_obj_tag(v___x_3170_) == 0)
{
lean_dec_ref_known(v___x_3170_, 1);
v_as_x27_3136_ = v_tail_3145_;
v_b_3137_ = v___x_3147_;
goto _start;
}
else
{
return v___x_3170_;
}
}
else
{
lean_dec_ref(v_val_3159_);
if (lean_obj_tag(v___x_3169_) == 0)
{
lean_object* v_a_3172_; lean_object* v_a_3173_; 
v_a_3172_ = lean_ctor_get(v___x_3169_, 0);
lean_inc(v_a_3172_);
lean_dec_ref_known(v___x_3169_, 1);
v_a_3173_ = lean_ctor_get(v_a_3172_, 0);
lean_inc(v_a_3173_);
lean_dec(v_a_3172_);
v_as_x27_3136_ = v_tail_3145_;
v_b_3137_ = v_a_3173_;
goto _start;
}
else
{
lean_object* v_a_3175_; lean_object* v___x_3177_; uint8_t v_isShared_3178_; uint8_t v_isSharedCheck_3182_; 
v_a_3175_ = lean_ctor_get(v___x_3169_, 0);
v_isSharedCheck_3182_ = !lean_is_exclusive(v___x_3169_);
if (v_isSharedCheck_3182_ == 0)
{
v___x_3177_ = v___x_3169_;
v_isShared_3178_ = v_isSharedCheck_3182_;
goto v_resetjp_3176_;
}
else
{
lean_inc(v_a_3175_);
lean_dec(v___x_3169_);
v___x_3177_ = lean_box(0);
v_isShared_3178_ = v_isSharedCheck_3182_;
goto v_resetjp_3176_;
}
v_resetjp_3176_:
{
lean_object* v___x_3180_; 
if (v_isShared_3178_ == 0)
{
v___x_3180_ = v___x_3177_;
goto v_reusejp_3179_;
}
else
{
lean_object* v_reuseFailAlloc_3181_; 
v_reuseFailAlloc_3181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3181_, 0, v_a_3175_);
v___x_3180_ = v_reuseFailAlloc_3181_;
goto v_reusejp_3179_;
}
v_reusejp_3179_:
{
return v___x_3180_;
}
}
}
}
}
else
{
lean_dec_ref(v_val_3159_);
v_as_x27_3136_ = v_tail_3145_;
v_b_3137_ = v___x_3147_;
goto _start;
}
}
else
{
lean_dec(v_val_3158_);
lean_dec(v___x_3155_);
v_as_x27_3136_ = v_tail_3145_;
v_b_3137_ = v___x_3147_;
goto _start;
}
}
else
{
lean_dec(v___x_3157_);
lean_dec(v___x_3155_);
v_as_x27_3136_ = v_tail_3145_;
v_b_3137_ = v___x_3147_;
goto _start;
}
}
else
{
return v___x_3151_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg(lean_object* v_rv_3186_, lean_object* v_as_3187_, lean_object* v_as_x27_3188_, lean_object* v_b_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_){
_start:
{
if (lean_obj_tag(v_as_x27_3188_) == 0)
{
lean_object* v___x_3195_; 
v___x_3195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3195_, 0, v_b_3189_);
return v___x_3195_;
}
else
{
lean_object* v_head_3196_; lean_object* v_tail_3197_; lean_object* v_numMotives_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; lean_object* v___x_3202_; lean_object* v___x_3203_; 
v_head_3196_ = lean_ctor_get(v_as_x27_3188_, 0);
v_tail_3197_ = lean_ctor_get(v_as_x27_3188_, 1);
v_numMotives_3198_ = lean_ctor_get(v_rv_3186_, 4);
v___x_3199_ = lean_box(0);
v___x_3200_ = lean_unsigned_to_nat(0u);
v___x_3201_ = lean_unsigned_to_nat(1u);
lean_inc(v_numMotives_3198_);
v___x_3202_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3202_, 0, v___x_3200_);
lean_ctor_set(v___x_3202_, 1, v_numMotives_3198_);
lean_ctor_set(v___x_3202_, 2, v___x_3201_);
lean_inc(v_head_3196_);
v___x_3203_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(v_head_3196_, v___x_3202_, v___x_3199_, v___x_3200_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
lean_dec_ref_known(v___x_3202_, 3);
if (lean_obj_tag(v___x_3203_) == 0)
{
lean_object* v___x_3204_; lean_object* v_env_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; uint8_t v___x_3208_; lean_object* v___x_3209_; 
lean_dec_ref_known(v___x_3203_, 1);
v___x_3204_ = lean_st_ref_get(v___y_3193_);
v_env_3205_ = lean_ctor_get(v___x_3204_, 0);
lean_inc_ref(v_env_3205_);
lean_dec(v___x_3204_);
v___x_3206_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___lam__0___closed__0));
lean_inc(v_head_3196_);
v___x_3207_ = l_Lean_Name_str___override(v_head_3196_, v___x_3206_);
v___x_3208_ = 0;
lean_inc(v___x_3207_);
v___x_3209_ = l_Lean_Environment_find_x3f(v_env_3205_, v___x_3207_, v___x_3208_);
if (lean_obj_tag(v___x_3209_) == 1)
{
lean_object* v_val_3210_; 
v_val_3210_ = lean_ctor_get(v___x_3209_, 0);
lean_inc(v_val_3210_);
lean_dec_ref_known(v___x_3209_, 1);
if (lean_obj_tag(v_val_3210_) == 1)
{
lean_object* v_val_3211_; lean_object* v___x_3212_; lean_object* v_env_3213_; uint8_t v___x_3214_; 
v_val_3211_ = lean_ctor_get(v_val_3210_, 0);
lean_inc_ref(v_val_3211_);
lean_dec_ref_known(v_val_3210_, 1);
v___x_3212_ = lean_st_ref_get(v___y_3193_);
v_env_3213_ = lean_ctor_get(v___x_3212_, 0);
lean_inc_ref(v_env_3213_);
lean_dec(v___x_3212_);
v___x_3214_ = lp_mathlib_Mathlib_Util_hasCSimpLemma(v_env_3213_, v___x_3207_);
lean_dec(v___x_3207_);
if (v___x_3214_ == 0)
{
lean_object* v_value_3215_; lean_object* v___f_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v_fst_3220_; lean_object* v___x_3221_; 
v_value_3215_ = lean_ctor_get(v_val_3211_, 1);
v___f_3216_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__0));
v___x_3217_ = l_Lean_NameSet_empty;
v___x_3218_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4, &lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4_once, _init_lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___closed__4);
lean_inc_ref(v_value_3215_);
v___x_3219_ = l_Lean_Expr_FoldConstsImpl_fold___redArg(v___f_3216_, v_value_3215_, v___x_3217_, v___x_3218_);
v_fst_3220_ = lean_ctor_get(v___x_3219_, 0);
lean_inc(v_fst_3220_);
lean_dec_ref(v___x_3219_);
v___x_3221_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_3214_, v___x_3199_, v_fst_3220_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
if (lean_obj_tag(v___x_3221_) == 0)
{
lean_object* v___x_3222_; 
lean_dec_ref_known(v___x_3221_, 1);
v___x_3222_ = lp_mathlib_Mathlib_Util_compileDefn(v_val_3211_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
if (lean_obj_tag(v___x_3222_) == 0)
{
lean_object* v___x_3223_; 
lean_dec_ref_known(v___x_3222_, 1);
v___x_3223_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3186_, v_tail_3197_, v___x_3199_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3223_;
}
else
{
return v___x_3222_;
}
}
else
{
lean_dec_ref(v_val_3211_);
if (lean_obj_tag(v___x_3221_) == 0)
{
lean_object* v_a_3224_; lean_object* v_a_3225_; lean_object* v___x_3226_; 
v_a_3224_ = lean_ctor_get(v___x_3221_, 0);
lean_inc(v_a_3224_);
lean_dec_ref_known(v___x_3221_, 1);
v_a_3225_ = lean_ctor_get(v_a_3224_, 0);
lean_inc(v_a_3225_);
lean_dec(v_a_3224_);
v___x_3226_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3186_, v_tail_3197_, v_a_3225_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3226_;
}
else
{
lean_object* v_a_3227_; lean_object* v___x_3229_; uint8_t v_isShared_3230_; uint8_t v_isSharedCheck_3234_; 
v_a_3227_ = lean_ctor_get(v___x_3221_, 0);
v_isSharedCheck_3234_ = !lean_is_exclusive(v___x_3221_);
if (v_isSharedCheck_3234_ == 0)
{
v___x_3229_ = v___x_3221_;
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
else
{
lean_inc(v_a_3227_);
lean_dec(v___x_3221_);
v___x_3229_ = lean_box(0);
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
v_resetjp_3228_:
{
lean_object* v___x_3232_; 
if (v_isShared_3230_ == 0)
{
v___x_3232_ = v___x_3229_;
goto v_reusejp_3231_;
}
else
{
lean_object* v_reuseFailAlloc_3233_; 
v_reuseFailAlloc_3233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3233_, 0, v_a_3227_);
v___x_3232_ = v_reuseFailAlloc_3233_;
goto v_reusejp_3231_;
}
v_reusejp_3231_:
{
return v___x_3232_;
}
}
}
}
}
else
{
lean_object* v___x_3235_; 
lean_dec_ref(v_val_3211_);
v___x_3235_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3186_, v_tail_3197_, v___x_3199_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3235_;
}
}
else
{
lean_object* v___x_3236_; 
lean_dec(v_val_3210_);
lean_dec(v___x_3207_);
v___x_3236_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3186_, v_tail_3197_, v___x_3199_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3236_;
}
}
else
{
lean_object* v___x_3237_; 
lean_dec(v___x_3209_);
lean_dec(v___x_3207_);
v___x_3237_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3186_, v_tail_3197_, v___x_3199_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3237_;
}
}
else
{
return v___x_3203_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileSizeOf(lean_object* v_iv_3238_, lean_object* v_rv_3239_, lean_object* v_a_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_){
_start:
{
lean_object* v_all_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; 
v_all_3245_ = lean_ctor_get(v_iv_3238_, 3);
v___x_3246_ = lean_box(0);
v___x_3247_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg(v_rv_3239_, v_all_3245_, v_all_3245_, v___x_3246_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_);
if (lean_obj_tag(v___x_3247_) == 0)
{
lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3254_; 
v_isSharedCheck_3254_ = !lean_is_exclusive(v___x_3247_);
if (v_isSharedCheck_3254_ == 0)
{
lean_object* v_unused_3255_; 
v_unused_3255_ = lean_ctor_get(v___x_3247_, 0);
lean_dec(v_unused_3255_);
v___x_3249_ = v___x_3247_;
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
else
{
lean_dec(v___x_3247_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
lean_object* v___x_3252_; 
if (v_isShared_3250_ == 0)
{
lean_ctor_set(v___x_3249_, 0, v___x_3246_);
v___x_3252_ = v___x_3249_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3253_; 
v_reuseFailAlloc_3253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3253_, 0, v___x_3246_);
v___x_3252_ = v_reuseFailAlloc_3253_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
return v___x_3252_;
}
}
}
else
{
return v___x_3247_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileSizeOf___boxed(lean_object* v_iv_3256_, lean_object* v_rv_3257_, lean_object* v_a_3258_, lean_object* v_a_3259_, lean_object* v_a_3260_, lean_object* v_a_3261_, lean_object* v_a_3262_){
_start:
{
lean_object* v_res_3263_; 
v_res_3263_ = lp_mathlib_Mathlib_Util_compileSizeOf(v_iv_3256_, v_rv_3257_, v_a_3258_, v_a_3259_, v_a_3260_, v_a_3261_);
lean_dec(v_a_3261_);
lean_dec_ref(v_a_3260_);
lean_dec(v_a_3259_);
lean_dec_ref(v_a_3258_);
lean_dec_ref(v_rv_3257_);
lean_dec_ref(v_iv_3256_);
return v_res_3263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_compileInductive___boxed(lean_object* v_iv_3264_, lean_object* v_warn_3265_, lean_object* v_a_3266_, lean_object* v_a_3267_, lean_object* v_a_3268_, lean_object* v_a_3269_, lean_object* v_a_3270_){
_start:
{
uint8_t v_warn_boxed_3271_; lean_object* v_res_3272_; 
v_warn_boxed_3271_ = lean_unbox(v_warn_3265_);
v_res_3272_ = lp_mathlib_Mathlib_Util_compileInductive(v_iv_3264_, v_warn_boxed_3271_, v_a_3266_, v_a_3267_, v_a_3268_, v_a_3269_);
lean_dec(v_a_3269_);
lean_dec_ref(v_a_3268_);
lean_dec(v_a_3267_);
lean_dec_ref(v_a_3266_);
return v_res_3272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1___boxed(lean_object* v___x_3273_, lean_object* v_init_3274_, lean_object* v_x_3275_, lean_object* v___y_3276_, lean_object* v___y_3277_, lean_object* v___y_3278_, lean_object* v___y_3279_, lean_object* v___y_3280_){
_start:
{
uint8_t v___x_6407__boxed_3281_; lean_object* v_res_3282_; 
v___x_6407__boxed_3281_ = lean_unbox(v___x_3273_);
v_res_3282_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Util_compileSizeOf_spec__1(v___x_6407__boxed_3281_, v_init_3274_, v_x_3275_, v___y_3276_, v___y_3277_, v___y_3278_, v___y_3279_);
lean_dec(v___y_3279_);
lean_dec_ref(v___y_3278_);
lean_dec(v___y_3277_);
lean_dec_ref(v___y_3276_);
return v_res_3282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg___boxed(lean_object* v_rv_3283_, lean_object* v_as_x27_3284_, lean_object* v_b_3285_, lean_object* v___y_3286_, lean_object* v___y_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_, lean_object* v___y_3290_){
_start:
{
lean_object* v_res_3291_; 
v_res_3291_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3283_, v_as_x27_3284_, v_b_3285_, v___y_3286_, v___y_3287_, v___y_3288_, v___y_3289_);
lean_dec(v___y_3289_);
lean_dec_ref(v___y_3288_);
lean_dec(v___y_3287_);
lean_dec_ref(v___y_3286_);
lean_dec(v_as_x27_3284_);
lean_dec_ref(v_rv_3283_);
return v_res_3291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg___boxed(lean_object* v_rv_3292_, lean_object* v_as_3293_, lean_object* v_as_x27_3294_, lean_object* v_b_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_){
_start:
{
lean_object* v_res_3301_; 
v_res_3301_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg(v_rv_3292_, v_as_3293_, v_as_x27_3294_, v_b_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_);
lean_dec(v___y_3299_);
lean_dec_ref(v___y_3298_);
lean_dec(v___y_3297_);
lean_dec_ref(v___y_3296_);
lean_dec(v_as_x27_3294_);
lean_dec(v_as_3293_);
lean_dec_ref(v_rv_3292_);
return v_res_3301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg___boxed(lean_object* v_a_3302_, lean_object* v_range_3303_, lean_object* v_b_3304_, lean_object* v_i_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_){
_start:
{
lean_object* v_res_3311_; 
v_res_3311_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(v_a_3302_, v_range_3303_, v_b_3304_, v_i_3305_, v___y_3306_, v___y_3307_, v___y_3308_, v___y_3309_);
lean_dec(v___y_3309_);
lean_dec_ref(v___y_3308_);
lean_dec(v___y_3307_);
lean_dec_ref(v___y_3306_);
lean_dec(v_i_3305_);
lean_dec_ref(v_range_3303_);
return v_res_3311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg___boxed(lean_object* v_a_3312_, lean_object* v_range_3313_, lean_object* v_b_3314_, lean_object* v_i_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_, lean_object* v___y_3319_, lean_object* v___y_3320_){
_start:
{
lean_object* v_res_3321_; 
v_res_3321_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg(v_a_3312_, v_range_3313_, v_b_3314_, v_i_3315_, v___y_3316_, v___y_3317_, v___y_3318_, v___y_3319_);
lean_dec(v___y_3319_);
lean_dec_ref(v___y_3318_);
lean_dec(v___y_3317_);
lean_dec_ref(v___y_3316_);
lean_dec_ref(v_range_3313_);
return v_res_3321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2(lean_object* v_a_3322_, lean_object* v_range_3323_, lean_object* v_b_3324_, lean_object* v_i_3325_, lean_object* v_hs_3326_, lean_object* v_hl_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_){
_start:
{
lean_object* v___x_3333_; 
v___x_3333_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___redArg(v_a_3322_, v_range_3323_, v_b_3324_, v_i_3325_, v___y_3328_, v___y_3329_, v___y_3330_, v___y_3331_);
return v___x_3333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2___boxed(lean_object* v_a_3334_, lean_object* v_range_3335_, lean_object* v_b_3336_, lean_object* v_i_3337_, lean_object* v_hs_3338_, lean_object* v_hl_3339_, lean_object* v___y_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_){
_start:
{
lean_object* v_res_3345_; 
v_res_3345_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2(v_a_3334_, v_range_3335_, v_b_3336_, v_i_3337_, v_hs_3338_, v_hl_3339_, v___y_3340_, v___y_3341_, v___y_3342_, v___y_3343_);
lean_dec(v___y_3343_);
lean_dec_ref(v___y_3342_);
lean_dec(v___y_3341_);
lean_dec_ref(v___y_3340_);
lean_dec(v_i_3337_);
lean_dec_ref(v_range_3335_);
return v_res_3345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3(lean_object* v_rv_3346_, lean_object* v_as_3347_, lean_object* v_as_x27_3348_, lean_object* v_b_3349_, lean_object* v_a_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_){
_start:
{
lean_object* v___x_3356_; 
v___x_3356_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___redArg(v_rv_3346_, v_as_3347_, v_as_x27_3348_, v_b_3349_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_);
return v___x_3356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3___boxed(lean_object* v_rv_3357_, lean_object* v_as_3358_, lean_object* v_as_x27_3359_, lean_object* v_b_3360_, lean_object* v_a_3361_, lean_object* v___y_3362_, lean_object* v___y_3363_, lean_object* v___y_3364_, lean_object* v___y_3365_, lean_object* v___y_3366_){
_start:
{
lean_object* v_res_3367_; 
v_res_3367_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3(v_rv_3357_, v_as_3358_, v_as_x27_3359_, v_b_3360_, v_a_3361_, v___y_3362_, v___y_3363_, v___y_3364_, v___y_3365_);
lean_dec(v___y_3365_);
lean_dec_ref(v___y_3364_);
lean_dec(v___y_3363_);
lean_dec_ref(v___y_3362_);
lean_dec(v_as_x27_3359_);
lean_dec(v_as_3358_);
lean_dec_ref(v_rv_3357_);
return v_res_3367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2(lean_object* v_a_3368_, lean_object* v_range_3369_, lean_object* v_b_3370_, lean_object* v_i_3371_, lean_object* v_hs_3372_, lean_object* v_hl_3373_, lean_object* v___y_3374_, lean_object* v___y_3375_, lean_object* v___y_3376_, lean_object* v___y_3377_){
_start:
{
lean_object* v___x_3379_; 
v___x_3379_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___redArg(v_a_3368_, v_range_3369_, v_b_3370_, v_i_3371_, v___y_3374_, v___y_3375_, v___y_3376_, v___y_3377_);
return v___x_3379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2___boxed(lean_object* v_a_3380_, lean_object* v_range_3381_, lean_object* v_b_3382_, lean_object* v_i_3383_, lean_object* v_hs_3384_, lean_object* v_hl_3385_, lean_object* v___y_3386_, lean_object* v___y_3387_, lean_object* v___y_3388_, lean_object* v___y_3389_, lean_object* v___y_3390_){
_start:
{
lean_object* v_res_3391_; 
v_res_3391_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__2_spec__2(v_a_3380_, v_range_3381_, v_b_3382_, v_i_3383_, v_hs_3384_, v_hl_3385_, v___y_3386_, v___y_3387_, v___y_3388_, v___y_3389_);
lean_dec(v___y_3389_);
lean_dec_ref(v___y_3388_);
lean_dec(v___y_3387_);
lean_dec_ref(v___y_3386_);
lean_dec_ref(v_range_3381_);
return v_res_3391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4(lean_object* v_rv_3392_, lean_object* v_as_3393_, lean_object* v_as_x27_3394_, lean_object* v_b_3395_, lean_object* v_a_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_){
_start:
{
lean_object* v___x_3402_; 
v___x_3402_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___redArg(v_rv_3392_, v_as_x27_3394_, v_b_3395_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_);
return v___x_3402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4___boxed(lean_object* v_rv_3403_, lean_object* v_as_3404_, lean_object* v_as_x27_3405_, lean_object* v_b_3406_, lean_object* v_a_3407_, lean_object* v___y_3408_, lean_object* v___y_3409_, lean_object* v___y_3410_, lean_object* v___y_3411_, lean_object* v___y_3412_){
_start:
{
lean_object* v_res_3413_; 
v_res_3413_ = lp_mathlib_List_forIn_x27_loop___at___00List_forIn_x27_loop___at___00Mathlib_Util_compileSizeOf_spec__3_spec__4(v_rv_3403_, v_as_3404_, v_as_x27_3405_, v_b_3406_, v_a_3407_, v___y_3408_, v___y_3409_, v___y_3410_, v___y_3411_);
lean_dec(v___y_3411_);
lean_dec_ref(v___y_3410_);
lean_dec(v___y_3409_);
lean_dec_ref(v___y_3408_);
lean_dec(v_as_x27_3405_);
lean_dec(v_as_3404_);
lean_dec_ref(v_rv_3403_);
return v_res_3413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0(lean_object* v_constName_3431_, lean_object* v___y_3432_, lean_object* v___y_3433_, lean_object* v___y_3434_, lean_object* v___y_3435_, lean_object* v___y_3436_, lean_object* v___y_3437_){
_start:
{
lean_object* v___x_3439_; lean_object* v_env_3440_; lean_object* v___x_3441_; 
v___x_3439_ = lean_st_ref_get(v___y_3437_);
v_env_3440_ = lean_ctor_get(v___x_3439_, 0);
lean_inc_ref(v_env_3440_);
lean_dec(v___x_3439_);
lean_inc(v_constName_3431_);
v___x_3441_ = l_Lean_isInductiveCore_x3f(v_env_3440_, v_constName_3431_);
if (lean_obj_tag(v___x_3441_) == 0)
{
lean_object* v___x_3442_; uint8_t v___x_3443_; lean_object* v___x_3444_; lean_object* v___x_3445_; lean_object* v___x_3446_; lean_object* v___x_3447_; lean_object* v___x_3448_; 
v___x_3442_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1, &lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1___closed__1);
v___x_3443_ = 0;
v___x_3444_ = l_Lean_MessageData_ofConstName(v_constName_3431_, v___x_3443_);
v___x_3445_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3445_, 0, v___x_3442_);
lean_ctor_set(v___x_3445_, 1, v___x_3444_);
v___x_3446_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1, &lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1_once, _init_lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util_compileInductiveOnly_spec__2___closed__1);
v___x_3447_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3447_, 0, v___x_3445_);
lean_ctor_set(v___x_3447_, 1, v___x_3446_);
v___x_3448_ = lp_mathlib_Lean_throwError___at___00Lean_getConstInfoDefn___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__1_spec__1___redArg(v___x_3447_, v___y_3432_, v___y_3433_, v___y_3434_, v___y_3435_, v___y_3436_, v___y_3437_);
return v___x_3448_;
}
else
{
lean_object* v_val_3449_; lean_object* v___x_3451_; uint8_t v_isShared_3452_; uint8_t v_isSharedCheck_3456_; 
lean_dec(v_constName_3431_);
v_val_3449_ = lean_ctor_get(v___x_3441_, 0);
v_isSharedCheck_3456_ = !lean_is_exclusive(v___x_3441_);
if (v_isSharedCheck_3456_ == 0)
{
v___x_3451_ = v___x_3441_;
v_isShared_3452_ = v_isSharedCheck_3456_;
goto v_resetjp_3450_;
}
else
{
lean_inc(v_val_3449_);
lean_dec(v___x_3441_);
v___x_3451_ = lean_box(0);
v_isShared_3452_ = v_isSharedCheck_3456_;
goto v_resetjp_3450_;
}
v_resetjp_3450_:
{
lean_object* v___x_3454_; 
if (v_isShared_3452_ == 0)
{
lean_ctor_set_tag(v___x_3451_, 0);
v___x_3454_ = v___x_3451_;
goto v_reusejp_3453_;
}
else
{
lean_object* v_reuseFailAlloc_3455_; 
v_reuseFailAlloc_3455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3455_, 0, v_val_3449_);
v___x_3454_ = v_reuseFailAlloc_3455_;
goto v_reusejp_3453_;
}
v_reusejp_3453_:
{
return v___x_3454_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0___boxed(lean_object* v_constName_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_, lean_object* v___y_3461_, lean_object* v___y_3462_, lean_object* v___y_3463_, lean_object* v___y_3464_){
_start:
{
lean_object* v_res_3465_; 
v_res_3465_ = lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0(v_constName_3457_, v___y_3458_, v___y_3459_, v___y_3460_, v___y_3461_, v___y_3462_, v___y_3463_);
lean_dec(v___y_3463_);
lean_dec_ref(v___y_3462_);
lean_dec(v___y_3461_);
lean_dec_ref(v___y_3460_);
lean_dec(v___y_3459_);
lean_dec_ref(v___y_3458_);
return v_res_3465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0(lean_object* v___x_3466_, lean_object* v___x_3467_, lean_object* v_tk_3468_, uint8_t v___x_3469_, lean_object* v___y_3470_, lean_object* v___y_3471_, lean_object* v___y_3472_, lean_object* v___y_3473_, lean_object* v___y_3474_, lean_object* v___y_3475_){
_start:
{
lean_object* v___x_3477_; 
lean_inc(v___x_3466_);
v___x_3477_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v___x_3466_, v___x_3467_, v___y_3474_, v___y_3475_);
if (lean_obj_tag(v___x_3477_) == 0)
{
lean_object* v_a_3478_; lean_object* v_fileName_3479_; lean_object* v_fileMap_3480_; lean_object* v_options_3481_; lean_object* v_currRecDepth_3482_; lean_object* v_maxRecDepth_3483_; lean_object* v_ref_3484_; lean_object* v_currNamespace_3485_; lean_object* v_openDecls_3486_; lean_object* v_initHeartbeats_3487_; lean_object* v_maxHeartbeats_3488_; lean_object* v_quotContext_3489_; lean_object* v_currMacroScope_3490_; uint8_t v_diag_3491_; lean_object* v_cancelTk_x3f_3492_; uint8_t v_suppressElabErrors_3493_; lean_object* v_inheritedTraceOptions_3494_; lean_object* v___x_3496_; uint8_t v_isShared_3497_; uint8_t v_isSharedCheck_3515_; 
v_a_3478_ = lean_ctor_get(v___x_3477_, 0);
lean_inc(v_a_3478_);
lean_dec_ref_known(v___x_3477_, 1);
v_fileName_3479_ = lean_ctor_get(v___y_3474_, 0);
v_fileMap_3480_ = lean_ctor_get(v___y_3474_, 1);
v_options_3481_ = lean_ctor_get(v___y_3474_, 2);
v_currRecDepth_3482_ = lean_ctor_get(v___y_3474_, 3);
v_maxRecDepth_3483_ = lean_ctor_get(v___y_3474_, 4);
v_ref_3484_ = lean_ctor_get(v___y_3474_, 5);
v_currNamespace_3485_ = lean_ctor_get(v___y_3474_, 6);
v_openDecls_3486_ = lean_ctor_get(v___y_3474_, 7);
v_initHeartbeats_3487_ = lean_ctor_get(v___y_3474_, 8);
v_maxHeartbeats_3488_ = lean_ctor_get(v___y_3474_, 9);
v_quotContext_3489_ = lean_ctor_get(v___y_3474_, 10);
v_currMacroScope_3490_ = lean_ctor_get(v___y_3474_, 11);
v_diag_3491_ = lean_ctor_get_uint8(v___y_3474_, sizeof(void*)*14);
v_cancelTk_x3f_3492_ = lean_ctor_get(v___y_3474_, 12);
v_suppressElabErrors_3493_ = lean_ctor_get_uint8(v___y_3474_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3494_ = lean_ctor_get(v___y_3474_, 13);
v_isSharedCheck_3515_ = !lean_is_exclusive(v___y_3474_);
if (v_isSharedCheck_3515_ == 0)
{
v___x_3496_ = v___y_3474_;
v_isShared_3497_ = v_isSharedCheck_3515_;
goto v_resetjp_3495_;
}
else
{
lean_inc(v_inheritedTraceOptions_3494_);
lean_inc(v_cancelTk_x3f_3492_);
lean_inc(v_currMacroScope_3490_);
lean_inc(v_quotContext_3489_);
lean_inc(v_maxHeartbeats_3488_);
lean_inc(v_initHeartbeats_3487_);
lean_inc(v_openDecls_3486_);
lean_inc(v_currNamespace_3485_);
lean_inc(v_ref_3484_);
lean_inc(v_maxRecDepth_3483_);
lean_inc(v_currRecDepth_3482_);
lean_inc(v_options_3481_);
lean_inc(v_fileMap_3480_);
lean_inc(v_fileName_3479_);
lean_dec(v___y_3474_);
v___x_3496_ = lean_box(0);
v_isShared_3497_ = v_isSharedCheck_3515_;
goto v_resetjp_3495_;
}
v_resetjp_3495_:
{
lean_object* v_ref_3498_; lean_object* v___x_3500_; 
v_ref_3498_ = l_Lean_replaceRef(v___x_3466_, v_ref_3484_);
lean_dec(v___x_3466_);
lean_inc_ref(v_inheritedTraceOptions_3494_);
lean_inc(v_cancelTk_x3f_3492_);
lean_inc(v_currMacroScope_3490_);
lean_inc(v_quotContext_3489_);
lean_inc(v_maxHeartbeats_3488_);
lean_inc(v_initHeartbeats_3487_);
lean_inc(v_openDecls_3486_);
lean_inc(v_currNamespace_3485_);
lean_inc(v_maxRecDepth_3483_);
lean_inc(v_currRecDepth_3482_);
lean_inc_ref(v_options_3481_);
lean_inc_ref(v_fileMap_3480_);
lean_inc_ref(v_fileName_3479_);
if (v_isShared_3497_ == 0)
{
lean_ctor_set(v___x_3496_, 5, v_ref_3498_);
v___x_3500_ = v___x_3496_;
goto v_reusejp_3499_;
}
else
{
lean_object* v_reuseFailAlloc_3514_; 
v_reuseFailAlloc_3514_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_3514_, 0, v_fileName_3479_);
lean_ctor_set(v_reuseFailAlloc_3514_, 1, v_fileMap_3480_);
lean_ctor_set(v_reuseFailAlloc_3514_, 2, v_options_3481_);
lean_ctor_set(v_reuseFailAlloc_3514_, 3, v_currRecDepth_3482_);
lean_ctor_set(v_reuseFailAlloc_3514_, 4, v_maxRecDepth_3483_);
lean_ctor_set(v_reuseFailAlloc_3514_, 5, v_ref_3498_);
lean_ctor_set(v_reuseFailAlloc_3514_, 6, v_currNamespace_3485_);
lean_ctor_set(v_reuseFailAlloc_3514_, 7, v_openDecls_3486_);
lean_ctor_set(v_reuseFailAlloc_3514_, 8, v_initHeartbeats_3487_);
lean_ctor_set(v_reuseFailAlloc_3514_, 9, v_maxHeartbeats_3488_);
lean_ctor_set(v_reuseFailAlloc_3514_, 10, v_quotContext_3489_);
lean_ctor_set(v_reuseFailAlloc_3514_, 11, v_currMacroScope_3490_);
lean_ctor_set(v_reuseFailAlloc_3514_, 12, v_cancelTk_x3f_3492_);
lean_ctor_set(v_reuseFailAlloc_3514_, 13, v_inheritedTraceOptions_3494_);
lean_ctor_set_uint8(v_reuseFailAlloc_3514_, sizeof(void*)*14, v_diag_3491_);
lean_ctor_set_uint8(v_reuseFailAlloc_3514_, sizeof(void*)*14 + 1, v_suppressElabErrors_3493_);
v___x_3500_ = v_reuseFailAlloc_3514_;
goto v_reusejp_3499_;
}
v_reusejp_3499_:
{
lean_object* v___x_3501_; 
v___x_3501_ = lp_mathlib_Lean_getConstInfoInduct___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1_spec__0(v_a_3478_, v___y_3470_, v___y_3471_, v___y_3472_, v___y_3473_, v___x_3500_, v___y_3475_);
lean_dec_ref(v___x_3500_);
if (lean_obj_tag(v___x_3501_) == 0)
{
lean_object* v_a_3502_; lean_object* v_ref_3503_; lean_object* v___x_3504_; lean_object* v___x_3505_; 
v_a_3502_ = lean_ctor_get(v___x_3501_, 0);
lean_inc(v_a_3502_);
lean_dec_ref_known(v___x_3501_, 1);
v_ref_3503_ = l_Lean_replaceRef(v_tk_3468_, v_ref_3484_);
lean_dec(v_ref_3484_);
v___x_3504_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3504_, 0, v_fileName_3479_);
lean_ctor_set(v___x_3504_, 1, v_fileMap_3480_);
lean_ctor_set(v___x_3504_, 2, v_options_3481_);
lean_ctor_set(v___x_3504_, 3, v_currRecDepth_3482_);
lean_ctor_set(v___x_3504_, 4, v_maxRecDepth_3483_);
lean_ctor_set(v___x_3504_, 5, v_ref_3503_);
lean_ctor_set(v___x_3504_, 6, v_currNamespace_3485_);
lean_ctor_set(v___x_3504_, 7, v_openDecls_3486_);
lean_ctor_set(v___x_3504_, 8, v_initHeartbeats_3487_);
lean_ctor_set(v___x_3504_, 9, v_maxHeartbeats_3488_);
lean_ctor_set(v___x_3504_, 10, v_quotContext_3489_);
lean_ctor_set(v___x_3504_, 11, v_currMacroScope_3490_);
lean_ctor_set(v___x_3504_, 12, v_cancelTk_x3f_3492_);
lean_ctor_set(v___x_3504_, 13, v_inheritedTraceOptions_3494_);
lean_ctor_set_uint8(v___x_3504_, sizeof(void*)*14, v_diag_3491_);
lean_ctor_set_uint8(v___x_3504_, sizeof(void*)*14 + 1, v_suppressElabErrors_3493_);
v___x_3505_ = lp_mathlib_Mathlib_Util_compileInductive(v_a_3502_, v___x_3469_, v___y_3472_, v___y_3473_, v___x_3504_, v___y_3475_);
lean_dec_ref_known(v___x_3504_, 14);
return v___x_3505_;
}
else
{
lean_object* v_a_3506_; lean_object* v___x_3508_; uint8_t v_isShared_3509_; uint8_t v_isSharedCheck_3513_; 
lean_dec_ref(v_inheritedTraceOptions_3494_);
lean_dec(v_cancelTk_x3f_3492_);
lean_dec(v_currMacroScope_3490_);
lean_dec(v_quotContext_3489_);
lean_dec(v_maxHeartbeats_3488_);
lean_dec(v_initHeartbeats_3487_);
lean_dec(v_openDecls_3486_);
lean_dec(v_currNamespace_3485_);
lean_dec(v_ref_3484_);
lean_dec(v_maxRecDepth_3483_);
lean_dec(v_currRecDepth_3482_);
lean_dec_ref(v_options_3481_);
lean_dec_ref(v_fileMap_3480_);
lean_dec_ref(v_fileName_3479_);
v_a_3506_ = lean_ctor_get(v___x_3501_, 0);
v_isSharedCheck_3513_ = !lean_is_exclusive(v___x_3501_);
if (v_isSharedCheck_3513_ == 0)
{
v___x_3508_ = v___x_3501_;
v_isShared_3509_ = v_isSharedCheck_3513_;
goto v_resetjp_3507_;
}
else
{
lean_inc(v_a_3506_);
lean_dec(v___x_3501_);
v___x_3508_ = lean_box(0);
v_isShared_3509_ = v_isSharedCheck_3513_;
goto v_resetjp_3507_;
}
v_resetjp_3507_:
{
lean_object* v___x_3511_; 
if (v_isShared_3509_ == 0)
{
v___x_3511_ = v___x_3508_;
goto v_reusejp_3510_;
}
else
{
lean_object* v_reuseFailAlloc_3512_; 
v_reuseFailAlloc_3512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3512_, 0, v_a_3506_);
v___x_3511_ = v_reuseFailAlloc_3512_;
goto v_reusejp_3510_;
}
v_reusejp_3510_:
{
return v___x_3511_;
}
}
}
}
}
}
else
{
lean_object* v_a_3516_; lean_object* v___x_3518_; uint8_t v_isShared_3519_; uint8_t v_isSharedCheck_3523_; 
lean_dec_ref(v___y_3474_);
lean_dec(v___x_3466_);
v_a_3516_ = lean_ctor_get(v___x_3477_, 0);
v_isSharedCheck_3523_ = !lean_is_exclusive(v___x_3477_);
if (v_isSharedCheck_3523_ == 0)
{
v___x_3518_ = v___x_3477_;
v_isShared_3519_ = v_isSharedCheck_3523_;
goto v_resetjp_3517_;
}
else
{
lean_inc(v_a_3516_);
lean_dec(v___x_3477_);
v___x_3518_ = lean_box(0);
v_isShared_3519_ = v_isSharedCheck_3523_;
goto v_resetjp_3517_;
}
v_resetjp_3517_:
{
lean_object* v___x_3521_; 
if (v_isShared_3519_ == 0)
{
v___x_3521_ = v___x_3518_;
goto v_reusejp_3520_;
}
else
{
lean_object* v_reuseFailAlloc_3522_; 
v_reuseFailAlloc_3522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3522_, 0, v_a_3516_);
v___x_3521_ = v_reuseFailAlloc_3522_;
goto v_reusejp_3520_;
}
v_reusejp_3520_:
{
return v___x_3521_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0___boxed(lean_object* v___x_3524_, lean_object* v___x_3525_, lean_object* v_tk_3526_, lean_object* v___x_3527_, lean_object* v___y_3528_, lean_object* v___y_3529_, lean_object* v___y_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_){
_start:
{
uint8_t v___x_3625__boxed_3535_; lean_object* v_res_3536_; 
v___x_3625__boxed_3535_ = lean_unbox(v___x_3527_);
v_res_3536_ = lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0(v___x_3524_, v___x_3525_, v_tk_3526_, v___x_3625__boxed_3535_, v___y_3528_, v___y_3529_, v___y_3530_, v___y_3531_, v___y_3532_, v___y_3533_);
lean_dec(v___y_3533_);
lean_dec(v___y_3531_);
lean_dec_ref(v___y_3530_);
lean_dec(v___y_3529_);
lean_dec_ref(v___y_3528_);
lean_dec(v_tk_3526_);
return v_res_3536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1(lean_object* v_x_3537_, lean_object* v_a_3538_, lean_object* v_a_3539_){
_start:
{
lean_object* v___x_3541_; uint8_t v___x_3542_; 
v___x_3541_ = ((lean_object*)(lp_mathlib_Mathlib_Util_commandCompile__inductive_x25___00__closed__1));
lean_inc(v_x_3537_);
v___x_3542_ = l_Lean_Syntax_isOfKind(v_x_3537_, v___x_3541_);
if (v___x_3542_ == 0)
{
lean_object* v___x_3543_; 
lean_dec(v_x_3537_);
v___x_3543_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__def_x25____1_spec__0___redArg();
return v___x_3543_;
}
else
{
lean_object* v___x_3544_; lean_object* v_tk_3545_; lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; lean_object* v___f_3550_; lean_object* v___x_3551_; 
v___x_3544_ = lean_unsigned_to_nat(0u);
v_tk_3545_ = l_Lean_Syntax_getArg(v_x_3537_, v___x_3544_);
v___x_3546_ = lean_unsigned_to_nat(1u);
v___x_3547_ = l_Lean_Syntax_getArg(v_x_3537_, v___x_3546_);
lean_dec(v_x_3537_);
v___x_3548_ = lean_box(0);
v___x_3549_ = lean_box(v___x_3542_);
v___f_3550_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___lam__0___boxed), 11, 4);
lean_closure_set(v___f_3550_, 0, v___x_3547_);
lean_closure_set(v___f_3550_, 1, v___x_3548_);
lean_closure_set(v___f_3550_, 2, v_tk_3545_);
lean_closure_set(v___f_3550_, 3, v___x_3549_);
v___x_3551_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_3550_, v_a_3538_, v_a_3539_);
return v___x_3551_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1___boxed(lean_object* v_x_3552_, lean_object* v_a_3553_, lean_object* v_a_3554_, lean_object* v_a_3555_){
_start:
{
lean_object* v_res_3556_; 
v_res_3556_ = lp_mathlib_Mathlib_Util___aux__Mathlib__Util__CompileInductive______elabRules__Mathlib__Util__commandCompile__inductive_x25____1(v_x_3552_, v_a_3553_, v_a_3554_);
lean_dec(v_a_3554_);
lean_dec_ref(v_a_3553_);
return v_res_3556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(lean_object* v_t_3557_, lean_object* v_zero_3558_, lean_object* v_succ_3559_){
_start:
{
lean_object* v___x_3560_; 
v___x_3560_ = l_Nat_recCompiled___redArg(v_zero_3558_, v_succ_3559_, v_t_3557_);
return v___x_3560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3____boxed(lean_object* v_t_3561_, lean_object* v_zero_3562_, lean_object* v_succ_3563_){
_start:
{
lean_object* v_res_3564_; 
v_res_3564_ = lp_mathlib_Nat_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(v_t_3561_, v_zero_3562_, v_succ_3563_);
lean_dec(v_zero_3562_);
lean_dec(v_t_3561_);
return v_res_3564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(lean_object* v_motive_3565_, lean_object* v_t_3566_, lean_object* v_zero_3567_, lean_object* v_succ_3568_){
_start:
{
lean_object* v___x_3569_; 
v___x_3569_ = l_Nat_recCompiled___redArg(v_zero_3567_, v_succ_3568_, v_t_3566_);
return v___x_3569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOn_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3____boxed(lean_object* v_motive_3570_, lean_object* v_t_3571_, lean_object* v_zero_3572_, lean_object* v_succ_3573_){
_start:
{
lean_object* v_res_3574_; 
v_res_3574_ = lp_mathlib_Nat_recOn_00___x40_Mathlib_Util_CompileInductive_2970574227____hygCtx___hyg_3_(v_motive_3570_, v_t_3571_, v_zero_3572_, v_succ_3573_);
lean_dec(v_zero_3572_);
lean_dec(v_t_3571_);
return v_res_3574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object* v_F__1_3575_, lean_object* v_n_3576_, lean_object* v_n__ih_3577_){
_start:
{
lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; 
v___x_3578_ = lean_unsigned_to_nat(1u);
v___x_3579_ = lean_nat_add(v_n_3576_, v___x_3578_);
lean_inc_ref(v_n__ih_3577_);
v___x_3580_ = lean_apply_2(v_F__1_3575_, v___x_3579_, v_n__ih_3577_);
v___x_3581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3581_, 0, v___x_3580_);
lean_ctor_set(v___x_3581_, 1, v_n__ih_3577_);
return v___x_3581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object* v_F__1_3582_, lean_object* v_n_3583_, lean_object* v_n__ih_3584_){
_start:
{
lean_object* v_res_3585_; 
v_res_3585_ = lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(v_F__1_3582_, v_n_3583_, v_n__ih_3584_);
lean_dec(v_n_3583_);
return v_res_3585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object* v_t_3586_, lean_object* v_F__1_3587_){
_start:
{
lean_object* v___f_3588_; lean_object* v___x_3589_; lean_object* v___x_3590_; lean_object* v___x_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; 
lean_inc(v_F__1_3587_);
v___f_3588_ = lean_alloc_closure((void*)(lp_mathlib_Nat_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed), 3, 1);
lean_closure_set(v___f_3588_, 0, v_F__1_3587_);
v___x_3589_ = lean_unsigned_to_nat(0u);
v___x_3590_ = lean_box(0);
v___x_3591_ = lean_apply_2(v_F__1_3587_, v___x_3589_, v___x_3590_);
v___x_3592_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3592_, 0, v___x_3591_);
lean_ctor_set(v___x_3592_, 1, v___x_3590_);
v___x_3593_ = l_Nat_recCompiled___redArg(v___x_3592_, v___f_3588_, v_t_3586_);
lean_dec_ref_known(v___x_3592_, 2);
return v___x_3593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object* v_t_3594_, lean_object* v_F__1_3595_){
_start:
{
lean_object* v_res_3596_; 
v_res_3596_ = lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(v_t_3594_, v_F__1_3595_);
lean_dec(v_t_3594_);
return v_res_3596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(lean_object* v_motive_3597_, lean_object* v_t_3598_, lean_object* v_F__1_3599_){
_start:
{
lean_object* v___x_3600_; 
v___x_3600_ = lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(v_t_3598_, v_F__1_3599_);
return v___x_3600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3____boxed(lean_object* v_motive_3601_, lean_object* v_t_3602_, lean_object* v_F__1_3603_){
_start:
{
lean_object* v_res_3604_; 
v_res_3604_ = lp_mathlib_Nat_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(v_motive_3601_, v_t_3602_, v_F__1_3603_);
lean_dec(v_t_3602_);
return v_res_3604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(lean_object* v_t_3605_, lean_object* v_F__1_3606_){
_start:
{
lean_object* v___x_3607_; lean_object* v_fst_3608_; 
v___x_3607_ = lp_mathlib_Nat_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1571041013____hygCtx___hyg_3_(v_t_3605_, v_F__1_3606_);
v_fst_3608_ = lean_ctor_get(v___x_3607_, 0);
lean_inc(v_fst_3608_);
lean_dec_ref(v___x_3607_);
return v_fst_3608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3____boxed(lean_object* v_t_3609_, lean_object* v_F__1_3610_){
_start:
{
lean_object* v_res_3611_; 
v_res_3611_ = lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(v_t_3609_, v_F__1_3610_);
lean_dec(v_t_3609_);
return v_res_3611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(lean_object* v_motive_3612_, lean_object* v_t_3613_, lean_object* v_F__1_3614_){
_start:
{
lean_object* v___x_3615_; 
v___x_3615_ = lp_mathlib_Nat_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(v_t_3613_, v_F__1_3614_);
return v___x_3615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_brecOn_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3____boxed(lean_object* v_motive_3616_, lean_object* v_t_3617_, lean_object* v_F__1_3618_){
_start:
{
lean_object* v_res_3619_; 
v_res_3619_ = lp_mathlib_Nat_brecOn_00___x40_Mathlib_Util_CompileInductive_2486050327____hygCtx___hyg_3_(v_motive_3616_, v_t_3617_, v_F__1_3618_);
lean_dec(v_t_3617_);
return v_res_3619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(lean_object* v_mk_3620_, lean_object* v_t_3621_){
_start:
{
lean_object* v_fst_3622_; lean_object* v_snd_3623_; lean_object* v___x_3624_; 
v_fst_3622_ = lean_ctor_get(v_t_3621_, 0);
lean_inc(v_fst_3622_);
v_snd_3623_ = lean_ctor_get(v_t_3621_, 1);
lean_inc(v_snd_3623_);
lean_dec_ref(v_t_3621_);
v___x_3624_ = lean_apply_2(v_mk_3620_, v_fst_3622_, v_snd_3623_);
return v___x_3624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_rec_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(lean_object* v_00_u03b1_3625_, lean_object* v_00_u03b2_3626_, lean_object* v_motive_3627_, lean_object* v_mk_3628_, lean_object* v_t_3629_){
_start:
{
lean_object* v___x_3630_; 
v___x_3630_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v_mk_3628_, v_t_3629_);
return v___x_3630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_5_(lean_object* v_t_3631_, lean_object* v_mk_3632_){
_start:
{
lean_object* v___x_3633_; 
v___x_3633_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v_mk_3632_, v_t_3631_);
return v___x_3633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_recOn_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_5_(lean_object* v_00_u03b1_3634_, lean_object* v_00_u03b2_3635_, lean_object* v_motive_3636_, lean_object* v_t_3637_, lean_object* v_mk_3638_){
_start:
{
lean_object* v___x_3639_; 
v___x_3639_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v_mk_3638_, v_t_3637_);
return v___x_3639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object* v_inst_3640_, lean_object* v_inst_3641_, lean_object* v_fst_3642_, lean_object* v_snd_3643_){
_start:
{
lean_object* v___x_3644_; lean_object* v___x_3645_; lean_object* v___x_3646_; lean_object* v___x_3647_; lean_object* v___x_3648_; 
v___x_3644_ = lean_unsigned_to_nat(1u);
v___x_3645_ = lean_apply_1(v_inst_3640_, v_fst_3642_);
v___x_3646_ = lean_nat_add(v___x_3644_, v___x_3645_);
lean_dec(v___x_3645_);
v___x_3647_ = lean_apply_1(v_inst_3641_, v_snd_3643_);
v___x_3648_ = lean_nat_add(v___x_3646_, v___x_3647_);
lean_dec(v___x_3647_);
lean_dec(v___x_3646_);
return v___x_3648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object* v_inst_3649_, lean_object* v_inst_3650_, lean_object* v_t_3651_){
_start:
{
lean_object* v___f_3652_; lean_object* v___x_3653_; 
v___f_3652_ = lean_alloc_closure((void*)(lp_mathlib_Prod___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_), 4, 2);
lean_closure_set(v___f_3652_, 0, v_inst_3649_);
lean_closure_set(v___f_3652_, 1, v_inst_3650_);
v___x_3653_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v___f_3652_, v_t_3651_);
return v___x_3653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(lean_object* v_00_u03b1_3654_, lean_object* v_00_u03b2_3655_, lean_object* v_inst_3656_, lean_object* v_inst_3657_, lean_object* v_t_3658_){
_start:
{
lean_object* v___x_3659_; 
v___x_3659_ = lp_mathlib_Prod___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(v_inst_3656_, v_inst_3657_, v_t_3658_);
return v___x_3659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object* v_inst_3660_, lean_object* v_inst_3661_, lean_object* v_m_3662_){
_start:
{
lean_object* v___x_3663_; 
v___x_3663_ = lp_mathlib_Prod___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_7_(v_inst_3660_, v_inst_3661_, v_m_3662_);
return v___x_3663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object* v_inst_3664_, lean_object* v_inst_3665_){
_start:
{
lean_object* v___f_3666_; 
v___f_3666_ = lean_alloc_closure((void*)(lp_mathlib_Prod___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3666_, 0, v_inst_3664_);
lean_closure_set(v___f_3666_, 1, v_inst_3665_);
return v___f_3666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_(lean_object* v_00_u03b1_3667_, lean_object* v_00_u03b2_3668_, lean_object* v_inst_3669_, lean_object* v_inst_3670_){
_start:
{
lean_object* v___f_3671_; 
v___f_3671_ = lean_alloc_closure((void*)(lp_mathlib_Prod___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3671_, 0, v_inst_3669_);
lean_closure_set(v___f_3671_, 1, v_inst_3670_);
return v___f_3671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object* v_nil_3672_, lean_object* v_cons_3673_, lean_object* v_t_3674_){
_start:
{
if (lean_obj_tag(v_t_3674_) == 0)
{
lean_dec(v_cons_3673_);
lean_inc(v_nil_3672_);
return v_nil_3672_;
}
else
{
lean_object* v_head_3675_; lean_object* v_tail_3676_; lean_object* v___x_3677_; lean_object* v___x_3678_; 
v_head_3675_ = lean_ctor_get(v_t_3674_, 0);
lean_inc(v_head_3675_);
v_tail_3676_ = lean_ctor_get(v_t_3674_, 1);
lean_inc_n(v_tail_3676_, 2);
lean_dec_ref_known(v_t_3674_, 2);
lean_inc(v_cons_3673_);
v___x_3677_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_nil_3672_, v_cons_3673_, v_tail_3676_);
v___x_3678_ = lean_apply_3(v_cons_3673_, v_head_3675_, v_tail_3676_, v___x_3677_);
return v___x_3678_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3____boxed(lean_object* v_nil_3679_, lean_object* v_cons_3680_, lean_object* v_t_3681_){
_start:
{
lean_object* v_res_3682_; 
v_res_3682_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_nil_3679_, v_cons_3680_, v_t_3681_);
lean_dec(v_nil_3679_);
return v_res_3682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_rec_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object* v_00_u03b1_3683_, lean_object* v_motive_3684_, lean_object* v_nil_3685_, lean_object* v_cons_3686_, lean_object* v_t_3687_){
_start:
{
lean_object* v___x_3688_; 
v___x_3688_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_nil_3685_, v_cons_3686_, v_t_3687_);
return v___x_3688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_rec_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3____boxed(lean_object* v_00_u03b1_3689_, lean_object* v_motive_3690_, lean_object* v_nil_3691_, lean_object* v_cons_3692_, lean_object* v_t_3693_){
_start:
{
lean_object* v_res_3694_; 
v_res_3694_ = lp_mathlib_List_rec_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_00_u03b1_3689_, v_motive_3690_, v_nil_3691_, v_cons_3692_, v_t_3693_);
lean_dec(v_nil_3691_);
return v_res_3694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(lean_object* v_t_3695_, lean_object* v_nil_3696_, lean_object* v_cons_3697_){
_start:
{
lean_object* v___x_3698_; 
v___x_3698_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_nil_3696_, v_cons_3697_, v_t_3695_);
return v___x_3698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5____boxed(lean_object* v_t_3699_, lean_object* v_nil_3700_, lean_object* v_cons_3701_){
_start:
{
lean_object* v_res_3702_; 
v_res_3702_ = lp_mathlib_List_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(v_t_3699_, v_nil_3700_, v_cons_3701_);
lean_dec(v_nil_3700_);
return v_res_3702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(lean_object* v_00_u03b1_3703_, lean_object* v_motive_3704_, lean_object* v_t_3705_, lean_object* v_nil_3706_, lean_object* v_cons_3707_){
_start:
{
lean_object* v___x_3708_; 
v___x_3708_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_nil_3706_, v_cons_3707_, v_t_3705_);
return v___x_3708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_recOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5____boxed(lean_object* v_00_u03b1_3709_, lean_object* v_motive_3710_, lean_object* v_t_3711_, lean_object* v_nil_3712_, lean_object* v_cons_3713_){
_start:
{
lean_object* v_res_3714_; 
v_res_3714_ = lp_mathlib_List_recOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_5_(v_00_u03b1_3709_, v_motive_3710_, v_t_3711_, v_nil_3712_, v_cons_3713_);
lean_dec(v_nil_3712_);
return v_res_3714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object* v_F__1_3715_, lean_object* v_head_3716_, lean_object* v_tail_3717_, lean_object* v_tail__ih_3718_){
_start:
{
lean_object* v___x_3719_; lean_object* v___x_3720_; lean_object* v___x_3721_; 
v___x_3719_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3719_, 0, v_head_3716_);
lean_ctor_set(v___x_3719_, 1, v_tail_3717_);
lean_inc_ref(v_tail__ih_3718_);
v___x_3720_ = lean_apply_2(v_F__1_3715_, v___x_3719_, v_tail__ih_3718_);
v___x_3721_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3721_, 0, v___x_3720_);
lean_ctor_set(v___x_3721_, 1, v_tail__ih_3718_);
return v___x_3721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object* v_t_3722_, lean_object* v_F__1_3723_){
_start:
{
lean_object* v___f_3724_; lean_object* v___x_3725_; lean_object* v___x_3726_; lean_object* v___x_3727_; lean_object* v___x_3728_; lean_object* v___x_3729_; 
lean_inc(v_F__1_3723_);
v___f_3724_ = lean_alloc_closure((void*)(lp_mathlib_List_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_), 4, 1);
lean_closure_set(v___f_3724_, 0, v_F__1_3723_);
v___x_3725_ = lean_box(0);
v___x_3726_ = lean_box(0);
v___x_3727_ = lean_apply_2(v_F__1_3723_, v___x_3725_, v___x_3726_);
v___x_3728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3728_, 0, v___x_3727_);
lean_ctor_set(v___x_3728_, 1, v___x_3726_);
v___x_3729_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v___x_3728_, v___f_3724_, v_t_3722_);
lean_dec_ref_known(v___x_3728_, 2);
return v___x_3729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_go_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(lean_object* v_00_u03b1_3730_, lean_object* v_motive_3731_, lean_object* v_t_3732_, lean_object* v_F__1_3733_){
_start:
{
lean_object* v___x_3734_; 
v___x_3734_ = lp_mathlib_List_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(v_t_3732_, v_F__1_3733_);
return v___x_3734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_9_(lean_object* v_t_3735_, lean_object* v_F__1_3736_){
_start:
{
lean_object* v___x_3737_; lean_object* v_fst_3738_; 
v___x_3737_ = lp_mathlib_List_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_7_(v_t_3735_, v_F__1_3736_);
v_fst_3738_ = lean_ctor_get(v___x_3737_, 0);
lean_inc(v_fst_3738_);
lean_dec_ref(v___x_3737_);
return v_fst_3738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_brecOn_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_9_(lean_object* v_00_u03b1_3739_, lean_object* v_motive_3740_, lean_object* v_t_3741_, lean_object* v_F__1_3742_){
_start:
{
lean_object* v___x_3743_; 
v___x_3743_ = lp_mathlib_List_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_9_(v_t_3741_, v_F__1_3742_);
return v___x_3743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object* v_inst_3744_, lean_object* v___x_3745_, lean_object* v_head_3746_, lean_object* v_tail_3747_, lean_object* v_tail__ih_3748_){
_start:
{
lean_object* v___x_3749_; lean_object* v___x_3750_; lean_object* v___x_3751_; 
v___x_3749_ = lean_apply_1(v_inst_3744_, v_head_3746_);
v___x_3750_ = lean_nat_add(v___x_3745_, v___x_3749_);
lean_dec(v___x_3749_);
v___x_3751_ = lean_nat_add(v___x_3750_, v_tail__ih_3748_);
lean_dec(v___x_3750_);
return v___x_3751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____boxed(lean_object* v_inst_3752_, lean_object* v___x_3753_, lean_object* v_head_3754_, lean_object* v_tail_3755_, lean_object* v_tail__ih_3756_){
_start:
{
lean_object* v_res_3757_; 
v_res_3757_ = lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_3752_, v___x_3753_, v_head_3754_, v_tail_3755_, v_tail__ih_3756_);
lean_dec(v_tail__ih_3756_);
lean_dec(v_tail_3755_);
lean_dec(v___x_3753_);
return v_res_3757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object* v_inst_3758_, lean_object* v_t_3759_){
_start:
{
lean_object* v___x_3760_; lean_object* v___f_3761_; lean_object* v___x_3762_; 
v___x_3760_ = lean_unsigned_to_nat(1u);
v___f_3761_ = lean_alloc_closure((void*)(lp_mathlib_List___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____boxed), 5, 2);
lean_closure_set(v___f_3761_, 0, v_inst_3758_);
lean_closure_set(v___f_3761_, 1, v___x_3760_);
v___x_3762_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v___x_3760_, v___f_3761_, v_t_3759_);
return v___x_3762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object* v_00_u03b1_3763_, lean_object* v_inst_3764_, lean_object* v_t_3765_){
_start:
{
lean_object* v___x_3766_; 
v___x_3766_ = lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_3764_, v_t_3765_);
return v___x_3766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object* v_inst_3767_, lean_object* v_m_3768_){
_start:
{
lean_object* v___x_3769_; 
v___x_3769_ = lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_3767_, v_m_3768_);
return v___x_3769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object* v_inst_3770_){
_start:
{
lean_object* v___f_3771_; 
v___f_3771_ = lean_alloc_closure((void*)(lp_mathlib_List___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_), 2, 1);
lean_closure_set(v___f_3771_, 0, v_inst_3770_);
return v___f_3771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_(lean_object* v_00_u03b1_3772_, lean_object* v_inst_3773_){
_start:
{
lean_object* v___f_3774_; 
v___f_3774_ = lean_alloc_closure((void*)(lp_mathlib_List___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_13_), 2, 1);
lean_closure_set(v___f_3774_, 0, v_inst_3773_);
return v___f_3774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(lean_object* v_unit_3775_){
_start:
{
lean_inc(v_unit_3775_);
return v_unit_3775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3____boxed(lean_object* v_unit_3776_){
_start:
{
lean_object* v_res_3777_; 
v_res_3777_ = lp_mathlib_PUnit_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(v_unit_3776_);
lean_dec(v_unit_3776_);
return v_res_3777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(lean_object* v_motive_3778_, lean_object* v_unit_3779_, lean_object* v_t_3780_){
_start:
{
lean_inc(v_unit_3779_);
return v_unit_3779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_rec_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3____boxed(lean_object* v_motive_3781_, lean_object* v_unit_3782_, lean_object* v_t_3783_){
_start:
{
lean_object* v_res_3784_; 
v_res_3784_ = lp_mathlib_PUnit_rec_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_3_(v_motive_3781_, v_unit_3782_, v_t_3783_);
lean_dec(v_unit_3782_);
return v_res_3784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(lean_object* v_unit_3785_){
_start:
{
lean_inc(v_unit_3785_);
return v_unit_3785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5____boxed(lean_object* v_unit_3786_){
_start:
{
lean_object* v_res_3787_; 
v_res_3787_ = lp_mathlib_PUnit_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(v_unit_3786_);
lean_dec(v_unit_3786_);
return v_res_3787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(lean_object* v_motive_3788_, lean_object* v_t_3789_, lean_object* v_unit_3790_){
_start:
{
lean_inc(v_unit_3790_);
return v_unit_3790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_recOn_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5____boxed(lean_object* v_motive_3791_, lean_object* v_t_3792_, lean_object* v_unit_3793_){
_start:
{
lean_object* v_res_3794_; 
v_res_3794_ = lp_mathlib_PUnit_recOn_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_5_(v_motive_3791_, v_t_3792_, v_unit_3793_);
lean_dec(v_unit_3793_);
return v_res_3794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_7_(lean_object* v_t_3795_){
_start:
{
lean_object* v___x_3796_; 
v___x_3796_ = lean_unsigned_to_nat(1u);
return v___x_3796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_2121875758____hygCtx___hyg_9_(lean_object* v_m_3797_){
_start:
{
lean_object* v___x_3798_; 
v___x_3798_ = lean_unsigned_to_nat(1u);
return v___x_3798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_rec_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_3_(lean_object* v_motive_3801_, uint8_t v_t_3802_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_rec_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_3____boxed(lean_object* v_motive_3803_, lean_object* v_t_3804_){
_start:
{
uint8_t v_t_boxed_3805_; lean_object* v_res_3806_; 
v_t_boxed_3805_ = lean_unbox(v_t_3804_);
v_res_3806_ = lp_mathlib_PEmpty_rec_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_3_(v_motive_3803_, v_t_boxed_3805_);
return v_res_3806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_recOn_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_5_(lean_object* v_motive_3807_, uint8_t v_t_3808_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_PEmpty_recOn_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_5____boxed(lean_object* v_motive_3809_, lean_object* v_t_3810_){
_start:
{
uint8_t v_t_boxed_3811_; lean_object* v_res_3812_; 
v_t_boxed_3811_ = lean_unbox(v_t_3810_);
v_res_3812_ = lp_mathlib_PEmpty_recOn_00___x40_Mathlib_Util_CompileInductive_1077078790____hygCtx___hyg_5_(v_motive_3809_, v_t_boxed_3811_);
return v_res_3812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(lean_object* v_inl_3813_, lean_object* v_inr_3814_, lean_object* v_t_3815_){
_start:
{
if (lean_obj_tag(v_t_3815_) == 0)
{
lean_object* v_val_3816_; lean_object* v___x_3817_; 
lean_dec(v_inr_3814_);
v_val_3816_ = lean_ctor_get(v_t_3815_, 0);
lean_inc(v_val_3816_);
lean_dec_ref_known(v_t_3815_, 1);
v___x_3817_ = lean_apply_1(v_inl_3813_, v_val_3816_);
return v___x_3817_;
}
else
{
lean_object* v_val_3818_; lean_object* v___x_3819_; 
lean_dec(v_inl_3813_);
v_val_3818_ = lean_ctor_get(v_t_3815_, 0);
lean_inc(v_val_3818_);
lean_dec_ref_known(v_t_3815_, 1);
v___x_3819_ = lean_apply_1(v_inr_3814_, v_val_3818_);
return v___x_3819_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_rec_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(lean_object* v_00_u03b1_3820_, lean_object* v_00_u03b2_3821_, lean_object* v_motive_3822_, lean_object* v_inl_3823_, lean_object* v_inr_3824_, lean_object* v_t_3825_){
_start:
{
lean_object* v___x_3826_; 
v___x_3826_ = lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(v_inl_3823_, v_inr_3824_, v_t_3825_);
return v___x_3826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_5_(lean_object* v_t_3827_, lean_object* v_inl_3828_, lean_object* v_inr_3829_){
_start:
{
lean_object* v___x_3830_; 
v___x_3830_ = lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(v_inl_3828_, v_inr_3829_, v_t_3827_);
return v___x_3830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_recOn_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_5_(lean_object* v_00_u03b1_3831_, lean_object* v_00_u03b2_3832_, lean_object* v_motive_3833_, lean_object* v_t_3834_, lean_object* v_inl_3835_, lean_object* v_inr_3836_){
_start:
{
lean_object* v___x_3837_; 
v___x_3837_ = lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(v_inl_3835_, v_inr_3836_, v_t_3834_);
return v___x_3837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object* v_inst_3838_, lean_object* v_val_3839_){
_start:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; 
v___x_3840_ = lean_unsigned_to_nat(1u);
v___x_3841_ = lean_apply_1(v_inst_3838_, v_val_3839_);
v___x_3842_ = lean_nat_add(v___x_3840_, v___x_3841_);
lean_dec(v___x_3841_);
return v___x_3842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object* v_inst_3843_, lean_object* v_inst_3844_, lean_object* v_t_3845_){
_start:
{
lean_object* v___f_3846_; lean_object* v___f_3847_; lean_object* v___x_3848_; 
v___f_3846_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_), 2, 1);
lean_closure_set(v___f_3846_, 0, v_inst_3843_);
v___f_3847_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_), 2, 1);
lean_closure_set(v___f_3847_, 0, v_inst_3844_);
v___x_3848_ = lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(v___f_3846_, v___f_3847_, v_t_3845_);
return v___x_3848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(lean_object* v_00_u03b1_3849_, lean_object* v_00_u03b2_3850_, lean_object* v_inst_3851_, lean_object* v_inst_3852_, lean_object* v_t_3853_){
_start:
{
lean_object* v___x_3854_; 
v___x_3854_ = lp_mathlib_Sum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(v_inst_3851_, v_inst_3852_, v_t_3853_);
return v___x_3854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object* v_inst_3855_, lean_object* v_inst_3856_, lean_object* v_m_3857_){
_start:
{
lean_object* v___x_3858_; 
v___x_3858_ = lp_mathlib_Sum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_(v_inst_3855_, v_inst_3856_, v_m_3857_);
return v___x_3858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object* v_inst_3859_, lean_object* v_inst_3860_){
_start:
{
lean_object* v___f_3861_; 
v___f_3861_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3861_, 0, v_inst_3859_);
lean_closure_set(v___f_3861_, 1, v_inst_3860_);
return v___f_3861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_(lean_object* v_00_u03b1_3862_, lean_object* v_00_u03b2_3863_, lean_object* v_inst_3864_, lean_object* v_inst_3865_){
_start:
{
lean_object* v___f_3866_; 
v___f_3866_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3866_, 0, v_inst_3864_);
lean_closure_set(v___f_3866_, 1, v_inst_3865_);
return v___f_3866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(lean_object* v_inl_3867_, lean_object* v_inr_3868_, lean_object* v_t_3869_){
_start:
{
if (lean_obj_tag(v_t_3869_) == 0)
{
lean_object* v_val_3870_; lean_object* v___x_3871_; 
lean_dec(v_inr_3868_);
v_val_3870_ = lean_ctor_get(v_t_3869_, 0);
lean_inc(v_val_3870_);
lean_dec_ref_known(v_t_3869_, 1);
v___x_3871_ = lean_apply_1(v_inl_3867_, v_val_3870_);
return v___x_3871_;
}
else
{
lean_object* v_val_3872_; lean_object* v___x_3873_; 
lean_dec(v_inl_3867_);
v_val_3872_ = lean_ctor_get(v_t_3869_, 0);
lean_inc(v_val_3872_);
lean_dec_ref_known(v_t_3869_, 1);
v___x_3873_ = lean_apply_1(v_inr_3868_, v_val_3872_);
return v___x_3873_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum_rec_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(lean_object* v_00_u03b1_3874_, lean_object* v_00_u03b2_3875_, lean_object* v_motive_3876_, lean_object* v_inl_3877_, lean_object* v_inr_3878_, lean_object* v_t_3879_){
_start:
{
lean_object* v___x_3880_; 
v___x_3880_ = lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(v_inl_3877_, v_inr_3878_, v_t_3879_);
return v___x_3880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_5_(lean_object* v_t_3881_, lean_object* v_inl_3882_, lean_object* v_inr_3883_){
_start:
{
lean_object* v___x_3884_; 
v___x_3884_ = lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(v_inl_3882_, v_inr_3883_, v_t_3881_);
return v___x_3884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum_recOn_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_5_(lean_object* v_00_u03b1_3885_, lean_object* v_00_u03b2_3886_, lean_object* v_motive_3887_, lean_object* v_t_3888_, lean_object* v_inl_3889_, lean_object* v_inr_3890_){
_start:
{
lean_object* v___x_3891_; 
v___x_3891_ = lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(v_inl_3889_, v_inr_3890_, v_t_3888_);
return v___x_3891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(lean_object* v_inst_3892_, lean_object* v_inst_3893_, lean_object* v_t_3894_){
_start:
{
lean_object* v___f_3895_; lean_object* v___f_3896_; lean_object* v___x_3897_; 
v___f_3895_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_), 2, 1);
lean_closure_set(v___f_3895_, 0, v_inst_3892_);
v___f_3896_ = lean_alloc_closure((void*)(lp_mathlib_Sum___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_7_), 2, 1);
lean_closure_set(v___f_3896_, 0, v_inst_3893_);
v___x_3897_ = lp_mathlib_PSum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_3_(v___f_3895_, v___f_3896_, v_t_3894_);
return v___x_3897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(lean_object* v_00_u03b1_3898_, lean_object* v_00_u03b2_3899_, lean_object* v_inst_3900_, lean_object* v_inst_3901_, lean_object* v_t_3902_){
_start:
{
lean_object* v___x_3903_; 
v___x_3903_ = lp_mathlib_PSum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(v_inst_3900_, v_inst_3901_, v_t_3902_);
return v___x_3903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object* v_inst_3904_, lean_object* v_inst_3905_, lean_object* v_m_3906_){
_start:
{
lean_object* v___x_3907_; 
v___x_3907_ = lp_mathlib_PSum___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_7_(v_inst_3904_, v_inst_3905_, v_m_3906_);
return v___x_3907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object* v_inst_3908_, lean_object* v_inst_3909_){
_start:
{
lean_object* v___f_3910_; 
v___f_3910_ = lean_alloc_closure((void*)(lp_mathlib_PSum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3910_, 0, v_inst_3908_);
lean_closure_set(v___f_3910_, 1, v_inst_3909_);
return v___f_3910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSum___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_(lean_object* v_00_u03b1_3911_, lean_object* v_00_u03b2_3912_, lean_object* v_inst_3913_, lean_object* v_inst_3914_){
_start:
{
lean_object* v___f_3915_; 
v___f_3915_ = lean_alloc_closure((void*)(lp_mathlib_PSum___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_2377832586____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_3915_, 0, v_inst_3913_);
lean_closure_set(v___f_3915_, 1, v_inst_3914_);
return v___f_3915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_And_rec___redArg_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_3_(lean_object* v_intro_3916_){
_start:
{
lean_object* v___x_3917_; 
v___x_3917_ = lean_apply_2(v_intro_3916_, lean_box(0), lean_box(0));
return v___x_3917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_And_rec_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_3_(lean_object* v_a_3918_, lean_object* v_b_3919_, lean_object* v_motive_3920_, lean_object* v_intro_3921_, lean_object* v_t_3922_){
_start:
{
lean_object* v___x_3923_; 
v___x_3923_ = lean_apply_2(v_intro_3921_, lean_box(0), lean_box(0));
return v___x_3923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_And_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_5_(lean_object* v_intro_3924_){
_start:
{
lean_object* v___x_3925_; 
v___x_3925_ = lean_apply_2(v_intro_3924_, lean_box(0), lean_box(0));
return v___x_3925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_And_recOn_00___x40_Mathlib_Util_CompileInductive_318578898____hygCtx___hyg_5_(lean_object* v_a_3926_, lean_object* v_b_3927_, lean_object* v_motive_3928_, lean_object* v_t_3929_, lean_object* v_intro_3930_){
_start:
{
lean_object* v___x_3931_; 
v___x_3931_ = lean_apply_2(v_intro_3930_, lean_box(0), lean_box(0));
return v___x_3931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(lean_object* v_false_3932_, lean_object* v_true_3933_, uint8_t v_t_3934_){
_start:
{
if (v_t_3934_ == 0)
{
lean_inc(v_false_3932_);
return v_false_3932_;
}
else
{
lean_inc(v_true_3933_);
return v_true_3933_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3____boxed(lean_object* v_false_3935_, lean_object* v_true_3936_, lean_object* v_t_3937_){
_start:
{
uint8_t v_t_boxed_3938_; lean_object* v_res_3939_; 
v_t_boxed_3938_ = lean_unbox(v_t_3937_);
v_res_3939_ = lp_mathlib_Bool_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(v_false_3935_, v_true_3936_, v_t_boxed_3938_);
lean_dec(v_true_3936_);
lean_dec(v_false_3935_);
return v_res_3939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(lean_object* v_motive_3940_, lean_object* v_false_3941_, lean_object* v_true_3942_, uint8_t v_t_3943_){
_start:
{
if (v_t_3943_ == 0)
{
lean_inc(v_false_3941_);
return v_false_3941_;
}
else
{
lean_inc(v_true_3942_);
return v_true_3942_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_rec_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3____boxed(lean_object* v_motive_3944_, lean_object* v_false_3945_, lean_object* v_true_3946_, lean_object* v_t_3947_){
_start:
{
uint8_t v_t_boxed_3948_; lean_object* v_res_3949_; 
v_t_boxed_3948_ = lean_unbox(v_t_3947_);
v_res_3949_ = lp_mathlib_Bool_rec_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_3_(v_motive_3944_, v_false_3945_, v_true_3946_, v_t_boxed_3948_);
lean_dec(v_true_3946_);
lean_dec(v_false_3945_);
return v_res_3949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(uint8_t v_t_3950_, lean_object* v_false_3951_, lean_object* v_true_3952_){
_start:
{
if (v_t_3950_ == 0)
{
lean_inc(v_false_3951_);
return v_false_3951_;
}
else
{
lean_inc(v_true_3952_);
return v_true_3952_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5____boxed(lean_object* v_t_3953_, lean_object* v_false_3954_, lean_object* v_true_3955_){
_start:
{
uint8_t v_t_boxed_3956_; lean_object* v_res_3957_; 
v_t_boxed_3956_ = lean_unbox(v_t_3953_);
v_res_3957_ = lp_mathlib_Bool_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(v_t_boxed_3956_, v_false_3954_, v_true_3955_);
lean_dec(v_true_3955_);
lean_dec(v_false_3954_);
return v_res_3957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(lean_object* v_motive_3958_, uint8_t v_t_3959_, lean_object* v_false_3960_, lean_object* v_true_3961_){
_start:
{
if (v_t_3959_ == 0)
{
lean_inc(v_false_3960_);
return v_false_3960_;
}
else
{
lean_inc(v_true_3961_);
return v_true_3961_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_recOn_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5____boxed(lean_object* v_motive_3962_, lean_object* v_t_3963_, lean_object* v_false_3964_, lean_object* v_true_3965_){
_start:
{
uint8_t v_t_boxed_3966_; lean_object* v_res_3967_; 
v_t_boxed_3966_ = lean_unbox(v_t_3963_);
v_res_3967_ = lp_mathlib_Bool_recOn_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_5_(v_motive_3962_, v_t_boxed_3966_, v_false_3964_, v_true_3965_);
lean_dec(v_true_3965_);
lean_dec(v_false_3964_);
return v_res_3967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_7_(uint8_t v_t_3968_){
_start:
{
lean_object* v___x_3969_; 
v___x_3969_ = lean_unsigned_to_nat(1u);
return v___x_3969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_7____boxed(lean_object* v_t_3970_){
_start:
{
uint8_t v_t_boxed_3971_; lean_object* v_res_3972_; 
v_t_boxed_3971_ = lean_unbox(v_t_3970_);
v_res_3972_ = lp_mathlib_Bool___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_7_(v_t_boxed_3971_);
return v_res_3972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9_(uint8_t v_m_3973_){
_start:
{
lean_object* v___x_3974_; 
v___x_3974_ = lean_unsigned_to_nat(1u);
return v___x_3974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9____boxed(lean_object* v_m_3975_){
_start:
{
uint8_t v_m_boxed_3976_; lean_object* v_res_3977_; 
v_m_boxed_3976_ = lean_unbox(v_m_3975_);
v_res_3977_ = lp_mathlib_Bool___sizeOf__inst___lam__0_00___x40_Mathlib_Util_CompileInductive_3618634379____hygCtx___hyg_9_(v_m_boxed_3976_);
return v_res_3977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(lean_object* v_mk_3980_, lean_object* v_t_3981_){
_start:
{
lean_object* v_fst_3982_; lean_object* v_snd_3983_; lean_object* v___x_3984_; 
v_fst_3982_ = lean_ctor_get(v_t_3981_, 0);
lean_inc(v_fst_3982_);
v_snd_3983_ = lean_ctor_get(v_t_3981_, 1);
lean_inc(v_snd_3983_);
lean_dec_ref(v_t_3981_);
v___x_3984_ = lean_apply_2(v_mk_3980_, v_fst_3982_, v_snd_3983_);
return v___x_3984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_rec_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(lean_object* v_00_u03b1_3985_, lean_object* v_00_u03b2_3986_, lean_object* v_motive_3987_, lean_object* v_mk_3988_, lean_object* v_t_3989_){
_start:
{
lean_object* v___x_3990_; 
v___x_3990_ = lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(v_mk_3988_, v_t_3989_);
return v___x_3990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_5_(lean_object* v_t_3991_, lean_object* v_mk_3992_){
_start:
{
lean_object* v___x_3993_; 
v___x_3993_ = lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(v_mk_3992_, v_t_3991_);
return v___x_3993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_recOn_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_5_(lean_object* v_00_u03b1_3994_, lean_object* v_00_u03b2_3995_, lean_object* v_motive_3996_, lean_object* v_t_3997_, lean_object* v_mk_3998_){
_start:
{
lean_object* v___x_3999_; 
v___x_3999_ = lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(v_mk_3998_, v_t_3997_);
return v___x_3999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object* v_inst_4000_, lean_object* v_inst_4001_, lean_object* v_fst_4002_, lean_object* v_snd_4003_){
_start:
{
lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; 
v___x_4004_ = lean_unsigned_to_nat(1u);
lean_inc(v_fst_4002_);
v___x_4005_ = lean_apply_1(v_inst_4000_, v_fst_4002_);
v___x_4006_ = lean_nat_add(v___x_4004_, v___x_4005_);
lean_dec(v___x_4005_);
v___x_4007_ = lean_apply_2(v_inst_4001_, v_fst_4002_, v_snd_4003_);
v___x_4008_ = lean_nat_add(v___x_4006_, v___x_4007_);
lean_dec(v___x_4007_);
lean_dec(v___x_4006_);
return v___x_4008_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object* v_inst_4009_, lean_object* v_inst_4010_, lean_object* v_t_4011_){
_start:
{
lean_object* v___f_4012_; lean_object* v___x_4013_; 
v___f_4012_ = lean_alloc_closure((void*)(lp_mathlib_Sigma___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_), 4, 2);
lean_closure_set(v___f_4012_, 0, v_inst_4009_);
lean_closure_set(v___f_4012_, 1, v_inst_4010_);
v___x_4013_ = lp_mathlib_Sigma_rec___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_3_(v___f_4012_, v_t_4011_);
return v___x_4013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(lean_object* v_00_u03b1_4014_, lean_object* v_00_u03b2_4015_, lean_object* v_inst_4016_, lean_object* v_inst_4017_, lean_object* v_t_4018_){
_start:
{
lean_object* v___x_4019_; 
v___x_4019_ = lp_mathlib_Sigma___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(v_inst_4016_, v_inst_4017_, v_t_4018_);
return v___x_4019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object* v_inst_4020_, lean_object* v_inst_4021_, lean_object* v_m_4022_){
_start:
{
lean_object* v___x_4023_; 
v___x_4023_ = lp_mathlib_Sigma___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_7_(v_inst_4020_, v_inst_4021_, v_m_4022_);
return v___x_4023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object* v_inst_4024_, lean_object* v_inst_4025_){
_start:
{
lean_object* v___f_4026_; 
v___f_4026_ = lean_alloc_closure((void*)(lp_mathlib_Sigma___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_4026_, 0, v_inst_4024_);
lean_closure_set(v___f_4026_, 1, v_inst_4025_);
return v___f_4026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_(lean_object* v_00_u03b1_4027_, lean_object* v_00_u03b2_4028_, lean_object* v_inst_4029_, lean_object* v_inst_4030_){
_start:
{
lean_object* v___f_4031_; 
v___f_4031_ = lean_alloc_closure((void*)(lp_mathlib_Sigma___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_4222055393____hygCtx___hyg_9_), 3, 2);
lean_closure_set(v___f_4031_, 0, v_inst_4029_);
lean_closure_set(v___f_4031_, 1, v_inst_4030_);
return v___f_4031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(lean_object* v_none_4032_, lean_object* v_some_4033_, lean_object* v_t_4034_){
_start:
{
if (lean_obj_tag(v_t_4034_) == 0)
{
lean_dec(v_some_4033_);
lean_inc(v_none_4032_);
return v_none_4032_;
}
else
{
lean_object* v_val_4035_; lean_object* v___x_4036_; 
v_val_4035_ = lean_ctor_get(v_t_4034_, 0);
lean_inc(v_val_4035_);
lean_dec_ref_known(v_t_4034_, 1);
v___x_4036_ = lean_apply_1(v_some_4033_, v_val_4035_);
return v___x_4036_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3____boxed(lean_object* v_none_4037_, lean_object* v_some_4038_, lean_object* v_t_4039_){
_start:
{
lean_object* v_res_4040_; 
v_res_4040_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_none_4037_, v_some_4038_, v_t_4039_);
lean_dec(v_none_4037_);
return v_res_4040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_rec_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(lean_object* v_00_u03b1_4041_, lean_object* v_motive_4042_, lean_object* v_none_4043_, lean_object* v_some_4044_, lean_object* v_t_4045_){
_start:
{
lean_object* v___x_4046_; 
v___x_4046_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_none_4043_, v_some_4044_, v_t_4045_);
return v___x_4046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_rec_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3____boxed(lean_object* v_00_u03b1_4047_, lean_object* v_motive_4048_, lean_object* v_none_4049_, lean_object* v_some_4050_, lean_object* v_t_4051_){
_start:
{
lean_object* v_res_4052_; 
v_res_4052_ = lp_mathlib_Option_rec_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_00_u03b1_4047_, v_motive_4048_, v_none_4049_, v_some_4050_, v_t_4051_);
lean_dec(v_none_4049_);
return v_res_4052_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(lean_object* v_t_4053_, lean_object* v_none_4054_, lean_object* v_some_4055_){
_start:
{
lean_object* v___x_4056_; 
v___x_4056_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_none_4054_, v_some_4055_, v_t_4053_);
return v___x_4056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5____boxed(lean_object* v_t_4057_, lean_object* v_none_4058_, lean_object* v_some_4059_){
_start:
{
lean_object* v_res_4060_; 
v_res_4060_ = lp_mathlib_Option_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(v_t_4057_, v_none_4058_, v_some_4059_);
lean_dec(v_none_4058_);
return v_res_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(lean_object* v_00_u03b1_4061_, lean_object* v_motive_4062_, lean_object* v_t_4063_, lean_object* v_none_4064_, lean_object* v_some_4065_){
_start:
{
lean_object* v___x_4066_; 
v___x_4066_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_none_4064_, v_some_4065_, v_t_4063_);
return v___x_4066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_recOn_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5____boxed(lean_object* v_00_u03b1_4067_, lean_object* v_motive_4068_, lean_object* v_t_4069_, lean_object* v_none_4070_, lean_object* v_some_4071_){
_start:
{
lean_object* v_res_4072_; 
v_res_4072_ = lp_mathlib_Option_recOn_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_5_(v_00_u03b1_4067_, v_motive_4068_, v_t_4069_, v_none_4070_, v_some_4071_);
lean_dec(v_none_4070_);
return v_res_4072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object* v_inst_4073_, lean_object* v___x_4074_, lean_object* v_val_4075_){
_start:
{
lean_object* v___x_4076_; lean_object* v___x_4077_; 
v___x_4076_ = lean_apply_1(v_inst_4073_, v_val_4075_);
v___x_4077_ = lean_nat_add(v___x_4074_, v___x_4076_);
lean_dec(v___x_4076_);
return v___x_4077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7____boxed(lean_object* v_inst_4078_, lean_object* v___x_4079_, lean_object* v_val_4080_){
_start:
{
lean_object* v_res_4081_; 
v_res_4081_ = lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(v_inst_4078_, v___x_4079_, v_val_4080_);
lean_dec(v___x_4079_);
return v_res_4081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object* v_inst_4082_, lean_object* v_t_4083_){
_start:
{
lean_object* v___x_4084_; lean_object* v___f_4085_; lean_object* v___x_4086_; 
v___x_4084_ = lean_unsigned_to_nat(1u);
v___f_4085_ = lean_alloc_closure((void*)(lp_mathlib_Option___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7____boxed), 3, 2);
lean_closure_set(v___f_4085_, 0, v_inst_4082_);
lean_closure_set(v___f_4085_, 1, v___x_4084_);
v___x_4086_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v___x_4084_, v___f_4085_, v_t_4083_);
return v___x_4086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(lean_object* v_00_u03b1_4087_, lean_object* v_inst_4088_, lean_object* v_t_4089_){
_start:
{
lean_object* v___x_4090_; 
v___x_4090_ = lp_mathlib_Option___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(v_inst_4088_, v_t_4089_);
return v___x_4090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object* v_inst_4091_, lean_object* v_m_4092_){
_start:
{
lean_object* v___x_4093_; 
v___x_4093_ = lp_mathlib_Option___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_7_(v_inst_4091_, v_m_4092_);
return v___x_4093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object* v_inst_4094_){
_start:
{
lean_object* v___f_4095_; 
v___f_4095_ = lean_alloc_closure((void*)(lp_mathlib_Option___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_), 2, 1);
lean_closure_set(v___f_4095_, 0, v_inst_4094_);
return v___f_4095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_(lean_object* v_00_u03b1_4096_, lean_object* v_inst_4097_){
_start:
{
lean_object* v___f_4098_; 
v___f_4098_ = lean_alloc_closure((void*)(lp_mathlib_Option___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_9_), 2, 1);
lean_closure_set(v___f_4098_, 0, v_inst_4097_);
return v___f_4098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_False_recOn_00___x40_Mathlib_Util_CompileInductive_2706509475____hygCtx___hyg_3_(lean_object* v_motive_4099_, lean_object* v_t_4100_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Empty_recOn_00___x40_Mathlib_Util_CompileInductive_2909641248____hygCtx___hyg_3_(lean_object* v_motive_4101_, uint8_t v_t_4102_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Empty_recOn_00___x40_Mathlib_Util_CompileInductive_2909641248____hygCtx___hyg_3____boxed(lean_object* v_motive_4103_, lean_object* v_t_4104_){
_start:
{
uint8_t v_t_boxed_4105_; lean_object* v_res_4106_; 
v_t_boxed_4105_ = lean_unbox(v_t_4104_);
v_res_4106_ = lp_mathlib_Empty_recOn_00___x40_Mathlib_Util_CompileInductive_2909641248____hygCtx___hyg_3_(v_motive_4103_, v_t_boxed_4105_);
return v_res_4106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(lean_object* v_H_4107_, uint8_t v_t_4108_){
_start:
{
lean_object* v___x_4109_; lean_object* v___x_4110_; 
v___x_4109_ = lean_uint8_to_nat(v_t_4108_);
v___x_4110_ = lean_apply_1(v_H_4107_, v___x_4109_);
return v___x_4110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219____boxed(lean_object* v_H_4111_, lean_object* v_t_4112_){
_start:
{
uint8_t v_t_12__boxed_4113_; lean_object* v_res_4114_; 
v_t_12__boxed_4113_ = lean_unbox(v_t_4112_);
v_res_4114_ = lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v_H_4111_, v_t_12__boxed_4113_);
return v_res_4114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(lean_object* v_motive_4115_, lean_object* v_H_4116_, uint8_t v_t_4117_){
_start:
{
lean_object* v___x_4118_; 
v___x_4118_ = lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v_H_4116_, v_t_4117_);
return v___x_4118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219____boxed(lean_object* v_motive_4119_, lean_object* v_H_4120_, lean_object* v_t_4121_){
_start:
{
uint8_t v_t_23__boxed_4122_; lean_object* v_res_4123_; 
v_t_23__boxed_4122_ = lean_unbox(v_t_4121_);
v_res_4123_ = lp_mathlib_UInt8_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v_motive_4119_, v_H_4120_, v_t_23__boxed_4122_);
return v_res_4123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(uint8_t v_t_4124_, lean_object* v_ofBitVec_4125_){
_start:
{
lean_object* v___x_4126_; 
v___x_4126_ = lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v_ofBitVec_4125_, v_t_4124_);
return v___x_4126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221____boxed(lean_object* v_t_4127_, lean_object* v_ofBitVec_4128_){
_start:
{
uint8_t v_t_boxed_4129_; lean_object* v_res_4130_; 
v_t_boxed_4129_ = lean_unbox(v_t_4127_);
v_res_4130_ = lp_mathlib_UInt8_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(v_t_boxed_4129_, v_ofBitVec_4128_);
return v_res_4130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(lean_object* v_motive_4131_, uint8_t v_t_4132_, lean_object* v_ofBitVec_4133_){
_start:
{
lean_object* v___x_4134_; 
v___x_4134_ = lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v_ofBitVec_4133_, v_t_4132_);
return v___x_4134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221____boxed(lean_object* v_motive_4135_, lean_object* v_t_4136_, lean_object* v_ofBitVec_4137_){
_start:
{
uint8_t v_t_boxed_4138_; lean_object* v_res_4139_; 
v_t_boxed_4138_ = lean_unbox(v_t_4136_);
v_res_4139_ = lp_mathlib_UInt8_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_221_(v_motive_4135_, v_t_boxed_4138_, v_ofBitVec_4137_);
return v_res_4139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223_(lean_object* v_ofFin_4140_, lean_object* v_t_4141_){
_start:
{
lean_object* v___x_4142_; 
v___x_4142_ = lean_apply_1(v_ofFin_4140_, v_t_4141_);
return v___x_4142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223_(lean_object* v_w_4143_, lean_object* v_motive_4144_, lean_object* v_ofFin_4145_, lean_object* v_t_4146_){
_start:
{
lean_object* v___x_4147_; 
v___x_4147_ = lean_apply_1(v_ofFin_4145_, v_t_4146_);
return v___x_4147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223____boxed(lean_object* v_w_4148_, lean_object* v_motive_4149_, lean_object* v_ofFin_4150_, lean_object* v_t_4151_){
_start:
{
lean_object* v_res_4152_; 
v_res_4152_ = lp_mathlib_BitVec_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_223_(v_w_4148_, v_motive_4149_, v_ofFin_4150_, v_t_4151_);
lean_dec(v_w_4148_);
return v_res_4152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225_(lean_object* v_t_4153_, lean_object* v_ofFin_4154_){
_start:
{
lean_object* v___x_4155_; 
v___x_4155_ = lean_apply_1(v_ofFin_4154_, v_t_4153_);
return v___x_4155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225_(lean_object* v_w_4156_, lean_object* v_motive_4157_, lean_object* v_t_4158_, lean_object* v_ofFin_4159_){
_start:
{
lean_object* v___x_4160_; 
v___x_4160_ = lean_apply_1(v_ofFin_4159_, v_t_4158_);
return v___x_4160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225____boxed(lean_object* v_w_4161_, lean_object* v_motive_4162_, lean_object* v_t_4163_, lean_object* v_ofFin_4164_){
_start:
{
lean_object* v_res_4165_; 
v_res_4165_ = lp_mathlib_BitVec_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_225_(v_w_4161_, v_motive_4162_, v_t_4163_, v_ofFin_4164_);
lean_dec(v_w_4161_);
return v_res_4165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227_(lean_object* v_mk_4166_, lean_object* v_t_4167_){
_start:
{
lean_object* v___x_4168_; 
v___x_4168_ = lean_apply_2(v_mk_4166_, v_t_4167_, lean_box(0));
return v___x_4168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227_(lean_object* v_n_4169_, lean_object* v_motive_4170_, lean_object* v_mk_4171_, lean_object* v_t_4172_){
_start:
{
lean_object* v___x_4173_; 
v___x_4173_ = lean_apply_2(v_mk_4171_, v_t_4172_, lean_box(0));
return v___x_4173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227____boxed(lean_object* v_n_4174_, lean_object* v_motive_4175_, lean_object* v_mk_4176_, lean_object* v_t_4177_){
_start:
{
lean_object* v_res_4178_; 
v_res_4178_ = lp_mathlib_Fin_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_227_(v_n_4174_, v_motive_4175_, v_mk_4176_, v_t_4177_);
lean_dec(v_n_4174_);
return v_res_4178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229_(lean_object* v_t_4179_, lean_object* v_mk_4180_){
_start:
{
lean_object* v___x_4181_; 
v___x_4181_ = lean_apply_2(v_mk_4180_, v_t_4179_, lean_box(0));
return v___x_4181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229_(lean_object* v_n_4182_, lean_object* v_motive_4183_, lean_object* v_t_4184_, lean_object* v_mk_4185_){
_start:
{
lean_object* v___x_4186_; 
v___x_4186_ = lean_apply_2(v_mk_4185_, v_t_4184_, lean_box(0));
return v___x_4186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229____boxed(lean_object* v_n_4187_, lean_object* v_motive_4188_, lean_object* v_t_4189_, lean_object* v_mk_4190_){
_start:
{
lean_object* v_res_4191_; 
v_res_4191_ = lp_mathlib_Fin_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_229_(v_n_4187_, v_motive_4188_, v_t_4189_, v_mk_4190_);
lean_dec(v_n_4187_);
return v_res_4191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(lean_object* v_t_4192_){
_start:
{
lean_object* v___x_4193_; lean_object* v___x_4194_; 
v___x_4193_ = lean_unsigned_to_nat(1u);
v___x_4194_ = lean_nat_add(v___x_4193_, v_t_4192_);
return v___x_4194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231____boxed(lean_object* v_t_4195_){
_start:
{
lean_object* v_res_4196_; 
v_res_4196_ = lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(v_t_4195_);
lean_dec(v_t_4195_);
return v_res_4196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(lean_object* v_n_4197_, lean_object* v_t_4198_){
_start:
{
lean_object* v___x_4199_; 
v___x_4199_ = lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(v_t_4198_);
return v___x_4199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231____boxed(lean_object* v_n_4200_, lean_object* v_t_4201_){
_start:
{
lean_object* v_res_4202_; 
v_res_4202_ = lp_mathlib_Fin___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(v_n_4200_, v_t_4201_);
lean_dec(v_t_4201_);
lean_dec(v_n_4200_);
return v_res_4202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233_(lean_object* v_n_4204_){
_start:
{
lean_object* v___f_4205_; 
v___f_4205_ = ((lean_object*)(lp_mathlib_Fin___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233_));
return v___f_4205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233____boxed(lean_object* v_n_4206_){
_start:
{
lean_object* v_res_4207_; 
v_res_4207_ = lp_mathlib_Fin___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_233_(v_n_4206_);
lean_dec(v_n_4206_);
return v_res_4207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(lean_object* v_t_4208_){
_start:
{
lean_object* v___x_4209_; lean_object* v___x_4210_; lean_object* v___x_4211_; 
v___x_4209_ = lean_unsigned_to_nat(1u);
v___x_4210_ = lp_mathlib_Fin___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_231_(v_t_4208_);
v___x_4211_ = lean_nat_add(v___x_4209_, v___x_4210_);
lean_dec(v___x_4210_);
return v___x_4211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235____boxed(lean_object* v_t_4212_){
_start:
{
lean_object* v_res_4213_; 
v_res_4213_ = lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(v_t_4212_);
lean_dec(v_t_4212_);
return v_res_4213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(lean_object* v_w_4214_, lean_object* v_t_4215_){
_start:
{
lean_object* v___x_4216_; 
v___x_4216_ = lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(v_t_4215_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235____boxed(lean_object* v_w_4217_, lean_object* v_t_4218_){
_start:
{
lean_object* v_res_4219_; 
v_res_4219_ = lp_mathlib_BitVec___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(v_w_4217_, v_t_4218_);
lean_dec(v_t_4218_);
lean_dec(v_w_4217_);
return v_res_4219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237_(lean_object* v_w_4221_){
_start:
{
lean_object* v___f_4222_; 
v___f_4222_ = ((lean_object*)(lp_mathlib_BitVec___sizeOf__inst___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237_));
return v___f_4222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BitVec___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237____boxed(lean_object* v_w_4223_){
_start:
{
lean_object* v_res_4224_; 
v_res_4224_ = lp_mathlib_BitVec___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_237_(v_w_4223_);
lean_dec(v_w_4223_);
return v_res_4224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(lean_object* v_toBitVec_4225_){
_start:
{
lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v___x_4228_; 
v___x_4226_ = lean_unsigned_to_nat(1u);
v___x_4227_ = lp_mathlib_BitVec___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_235_(v_toBitVec_4225_);
v___x_4228_ = lean_nat_add(v___x_4226_, v___x_4227_);
lean_dec(v___x_4227_);
return v___x_4228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed(lean_object* v_toBitVec_4229_){
_start:
{
lean_object* v_res_4230_; 
v_res_4230_ = lp_mathlib_UInt8___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(v_toBitVec_4229_);
lean_dec(v_toBitVec_4229_);
return v_res_4230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(uint8_t v_t_4232_){
_start:
{
lean_object* v___f_4233_; lean_object* v___x_4234_; 
v___f_4233_ = ((lean_object*)(lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_));
v___x_4234_ = lp_mathlib_UInt8_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_219_(v___f_4233_, v_t_4232_);
return v___x_4234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239____boxed(lean_object* v_t_4235_){
_start:
{
uint8_t v_t_boxed_4236_; lean_object* v_res_4237_; 
v_t_boxed_4236_ = lean_unbox(v_t_4235_);
v_res_4237_ = lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(v_t_boxed_4236_);
return v_res_4237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(lean_object* v_H_4240_, uint16_t v_t_4241_){
_start:
{
lean_object* v___x_4242_; lean_object* v___x_4243_; 
v___x_4242_ = lean_uint16_to_nat(v_t_4241_);
v___x_4243_ = lean_apply_1(v_H_4240_, v___x_4242_);
return v___x_4243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250____boxed(lean_object* v_H_4244_, lean_object* v_t_4245_){
_start:
{
uint16_t v_t_12__boxed_4246_; lean_object* v_res_4247_; 
v_t_12__boxed_4246_ = lean_unbox(v_t_4245_);
v_res_4247_ = lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v_H_4244_, v_t_12__boxed_4246_);
return v_res_4247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(lean_object* v_motive_4248_, lean_object* v_H_4249_, uint16_t v_t_4250_){
_start:
{
lean_object* v___x_4251_; 
v___x_4251_ = lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v_H_4249_, v_t_4250_);
return v___x_4251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250____boxed(lean_object* v_motive_4252_, lean_object* v_H_4253_, lean_object* v_t_4254_){
_start:
{
uint16_t v_t_23__boxed_4255_; lean_object* v_res_4256_; 
v_t_23__boxed_4255_ = lean_unbox(v_t_4254_);
v_res_4256_ = lp_mathlib_UInt16_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v_motive_4252_, v_H_4253_, v_t_23__boxed_4255_);
return v_res_4256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(uint16_t v_t_4257_, lean_object* v_ofBitVec_4258_){
_start:
{
lean_object* v___x_4259_; 
v___x_4259_ = lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v_ofBitVec_4258_, v_t_4257_);
return v___x_4259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252____boxed(lean_object* v_t_4260_, lean_object* v_ofBitVec_4261_){
_start:
{
uint16_t v_t_boxed_4262_; lean_object* v_res_4263_; 
v_t_boxed_4262_ = lean_unbox(v_t_4260_);
v_res_4263_ = lp_mathlib_UInt16_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(v_t_boxed_4262_, v_ofBitVec_4261_);
return v_res_4263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(lean_object* v_motive_4264_, uint16_t v_t_4265_, lean_object* v_ofBitVec_4266_){
_start:
{
lean_object* v___x_4267_; 
v___x_4267_ = lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v_ofBitVec_4266_, v_t_4265_);
return v___x_4267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252____boxed(lean_object* v_motive_4268_, lean_object* v_t_4269_, lean_object* v_ofBitVec_4270_){
_start:
{
uint16_t v_t_boxed_4271_; lean_object* v_res_4272_; 
v_t_boxed_4271_ = lean_unbox(v_t_4269_);
v_res_4272_ = lp_mathlib_UInt16_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_252_(v_motive_4268_, v_t_boxed_4271_, v_ofBitVec_4270_);
return v_res_4272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254_(uint16_t v_t_4273_){
_start:
{
lean_object* v___f_4274_; lean_object* v___x_4275_; 
v___f_4274_ = ((lean_object*)(lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_));
v___x_4275_ = lp_mathlib_UInt16_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_250_(v___f_4274_, v_t_4273_);
return v___x_4275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254____boxed(lean_object* v_t_4276_){
_start:
{
uint16_t v_t_boxed_4277_; lean_object* v_res_4278_; 
v_t_boxed_4277_ = lean_unbox(v_t_4276_);
v_res_4278_ = lp_mathlib_UInt16___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_254_(v_t_boxed_4277_);
return v_res_4278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(lean_object* v_H_4281_, uint32_t v_t_4282_){
_start:
{
lean_object* v___x_4283_; lean_object* v___x_4284_; 
v___x_4283_ = lean_uint32_to_nat(v_t_4282_);
v___x_4284_ = lean_apply_1(v_H_4281_, v___x_4283_);
return v___x_4284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265____boxed(lean_object* v_H_4285_, lean_object* v_t_4286_){
_start:
{
uint32_t v_t_12__boxed_4287_; lean_object* v_res_4288_; 
v_t_12__boxed_4287_ = lean_unbox_uint32(v_t_4286_);
lean_dec(v_t_4286_);
v_res_4288_ = lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v_H_4285_, v_t_12__boxed_4287_);
return v_res_4288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(lean_object* v_motive_4289_, lean_object* v_H_4290_, uint32_t v_t_4291_){
_start:
{
lean_object* v___x_4292_; 
v___x_4292_ = lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v_H_4290_, v_t_4291_);
return v___x_4292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265____boxed(lean_object* v_motive_4293_, lean_object* v_H_4294_, lean_object* v_t_4295_){
_start:
{
uint32_t v_t_23__boxed_4296_; lean_object* v_res_4297_; 
v_t_23__boxed_4296_ = lean_unbox_uint32(v_t_4295_);
lean_dec(v_t_4295_);
v_res_4297_ = lp_mathlib_UInt32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v_motive_4293_, v_H_4294_, v_t_23__boxed_4296_);
return v_res_4297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(uint32_t v_t_4298_, lean_object* v_ofBitVec_4299_){
_start:
{
lean_object* v___x_4300_; 
v___x_4300_ = lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v_ofBitVec_4299_, v_t_4298_);
return v___x_4300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267____boxed(lean_object* v_t_4301_, lean_object* v_ofBitVec_4302_){
_start:
{
uint32_t v_t_boxed_4303_; lean_object* v_res_4304_; 
v_t_boxed_4303_ = lean_unbox_uint32(v_t_4301_);
lean_dec(v_t_4301_);
v_res_4304_ = lp_mathlib_UInt32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(v_t_boxed_4303_, v_ofBitVec_4302_);
return v_res_4304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(lean_object* v_motive_4305_, uint32_t v_t_4306_, lean_object* v_ofBitVec_4307_){
_start:
{
lean_object* v___x_4308_; 
v___x_4308_ = lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v_ofBitVec_4307_, v_t_4306_);
return v___x_4308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267____boxed(lean_object* v_motive_4309_, lean_object* v_t_4310_, lean_object* v_ofBitVec_4311_){
_start:
{
uint32_t v_t_boxed_4312_; lean_object* v_res_4313_; 
v_t_boxed_4312_ = lean_unbox_uint32(v_t_4310_);
lean_dec(v_t_4310_);
v_res_4313_ = lp_mathlib_UInt32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_267_(v_motive_4309_, v_t_boxed_4312_, v_ofBitVec_4311_);
return v_res_4313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269_(uint32_t v_t_4314_){
_start:
{
lean_object* v___f_4315_; lean_object* v___x_4316_; 
v___f_4315_ = ((lean_object*)(lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_));
v___x_4316_ = lp_mathlib_UInt32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_265_(v___f_4315_, v_t_4314_);
return v___x_4316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269____boxed(lean_object* v_t_4317_){
_start:
{
uint32_t v_t_boxed_4318_; lean_object* v_res_4319_; 
v_t_boxed_4318_ = lean_unbox_uint32(v_t_4317_);
lean_dec(v_t_4317_);
v_res_4319_ = lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269_(v_t_boxed_4318_);
return v_res_4319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(lean_object* v_H_4322_, uint64_t v_t_4323_){
_start:
{
lean_object* v___x_4324_; lean_object* v___x_4325_; 
v___x_4324_ = lean_uint64_to_nat(v_t_4323_);
v___x_4325_ = lean_apply_1(v_H_4322_, v___x_4324_);
return v___x_4325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280____boxed(lean_object* v_H_4326_, lean_object* v_t_4327_){
_start:
{
uint64_t v_t_12__boxed_4328_; lean_object* v_res_4329_; 
v_t_12__boxed_4328_ = lean_unbox_uint64(v_t_4327_);
lean_dec_ref(v_t_4327_);
v_res_4329_ = lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v_H_4326_, v_t_12__boxed_4328_);
return v_res_4329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(lean_object* v_motive_4330_, lean_object* v_H_4331_, uint64_t v_t_4332_){
_start:
{
lean_object* v___x_4333_; 
v___x_4333_ = lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v_H_4331_, v_t_4332_);
return v___x_4333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280____boxed(lean_object* v_motive_4334_, lean_object* v_H_4335_, lean_object* v_t_4336_){
_start:
{
uint64_t v_t_23__boxed_4337_; lean_object* v_res_4338_; 
v_t_23__boxed_4337_ = lean_unbox_uint64(v_t_4336_);
lean_dec_ref(v_t_4336_);
v_res_4338_ = lp_mathlib_UInt64_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v_motive_4334_, v_H_4335_, v_t_23__boxed_4337_);
return v_res_4338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(uint64_t v_t_4339_, lean_object* v_ofBitVec_4340_){
_start:
{
lean_object* v___x_4341_; 
v___x_4341_ = lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v_ofBitVec_4340_, v_t_4339_);
return v___x_4341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282____boxed(lean_object* v_t_4342_, lean_object* v_ofBitVec_4343_){
_start:
{
uint64_t v_t_boxed_4344_; lean_object* v_res_4345_; 
v_t_boxed_4344_ = lean_unbox_uint64(v_t_4342_);
lean_dec_ref(v_t_4342_);
v_res_4345_ = lp_mathlib_UInt64_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(v_t_boxed_4344_, v_ofBitVec_4343_);
return v_res_4345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(lean_object* v_motive_4346_, uint64_t v_t_4347_, lean_object* v_ofBitVec_4348_){
_start:
{
lean_object* v___x_4349_; 
v___x_4349_ = lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v_ofBitVec_4348_, v_t_4347_);
return v___x_4349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282____boxed(lean_object* v_motive_4350_, lean_object* v_t_4351_, lean_object* v_ofBitVec_4352_){
_start:
{
uint64_t v_t_boxed_4353_; lean_object* v_res_4354_; 
v_t_boxed_4353_ = lean_unbox_uint64(v_t_4351_);
lean_dec_ref(v_t_4351_);
v_res_4354_ = lp_mathlib_UInt64_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_282_(v_motive_4350_, v_t_boxed_4353_, v_ofBitVec_4352_);
return v_res_4354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284_(uint64_t v_t_4355_){
_start:
{
lean_object* v___f_4356_; lean_object* v___x_4357_; 
v___f_4356_ = ((lean_object*)(lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_));
v___x_4357_ = lp_mathlib_UInt64_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_280_(v___f_4356_, v_t_4355_);
return v___x_4357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284____boxed(lean_object* v_t_4358_){
_start:
{
uint64_t v_t_boxed_4359_; lean_object* v_res_4360_; 
v_t_boxed_4359_ = lean_unbox_uint64(v_t_4358_);
lean_dec_ref(v_t_4358_);
v_res_4360_ = lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284_(v_t_boxed_4359_);
return v_res_4360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(lean_object* v_H_4363_, size_t v_t_4364_){
_start:
{
lean_object* v___x_4365_; lean_object* v___x_4366_; 
v___x_4365_ = lean_usize_to_nat(v_t_4364_);
v___x_4366_ = lean_apply_1(v_H_4363_, v___x_4365_);
return v___x_4366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295____boxed(lean_object* v_H_4367_, lean_object* v_t_4368_){
_start:
{
size_t v_t_12__boxed_4369_; lean_object* v_res_4370_; 
v_t_12__boxed_4369_ = lean_unbox_usize(v_t_4368_);
lean_dec(v_t_4368_);
v_res_4370_ = lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v_H_4367_, v_t_12__boxed_4369_);
return v_res_4370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(lean_object* v_motive_4371_, lean_object* v_H_4372_, size_t v_t_4373_){
_start:
{
lean_object* v___x_4374_; 
v___x_4374_ = lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v_H_4372_, v_t_4373_);
return v___x_4374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295____boxed(lean_object* v_motive_4375_, lean_object* v_H_4376_, lean_object* v_t_4377_){
_start:
{
size_t v_t_23__boxed_4378_; lean_object* v_res_4379_; 
v_t_23__boxed_4378_ = lean_unbox_usize(v_t_4377_);
lean_dec(v_t_4377_);
v_res_4379_ = lp_mathlib_USize_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v_motive_4375_, v_H_4376_, v_t_23__boxed_4378_);
return v_res_4379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(size_t v_t_4380_, lean_object* v_ofBitVec_4381_){
_start:
{
lean_object* v___x_4382_; 
v___x_4382_ = lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v_ofBitVec_4381_, v_t_4380_);
return v___x_4382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297____boxed(lean_object* v_t_4383_, lean_object* v_ofBitVec_4384_){
_start:
{
size_t v_t_boxed_4385_; lean_object* v_res_4386_; 
v_t_boxed_4385_ = lean_unbox_usize(v_t_4383_);
lean_dec(v_t_4383_);
v_res_4386_ = lp_mathlib_USize_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(v_t_boxed_4385_, v_ofBitVec_4384_);
return v_res_4386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(lean_object* v_motive_4387_, size_t v_t_4388_, lean_object* v_ofBitVec_4389_){
_start:
{
lean_object* v___x_4390_; 
v___x_4390_ = lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v_ofBitVec_4389_, v_t_4388_);
return v___x_4390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297____boxed(lean_object* v_motive_4391_, lean_object* v_t_4392_, lean_object* v_ofBitVec_4393_){
_start:
{
size_t v_t_boxed_4394_; lean_object* v_res_4395_; 
v_t_boxed_4394_ = lean_unbox_usize(v_t_4392_);
lean_dec(v_t_4392_);
v_res_4395_ = lp_mathlib_USize_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_297_(v_motive_4391_, v_t_boxed_4394_, v_ofBitVec_4393_);
return v_res_4395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299_(size_t v_t_4396_){
_start:
{
lean_object* v___f_4397_; lean_object* v___x_4398_; 
v___f_4397_ = ((lean_object*)(lp_mathlib_UInt8___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_));
v___x_4398_ = lp_mathlib_USize_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_295_(v___f_4397_, v_t_4396_);
return v___x_4398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299____boxed(lean_object* v_t_4399_){
_start:
{
size_t v_t_boxed_4400_; lean_object* v_res_4401_; 
v_t_boxed_4400_ = lean_unbox_usize(v_t_4399_);
lean_dec(v_t_4399_);
v_res_4401_ = lp_mathlib_USize___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_299_(v_t_boxed_4400_);
return v_res_4401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(lean_object* v_H_4404_, double v_t_4405_){
_start:
{
uint64_t v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; 
v___x_4406_ = lean_float_to_bits(v_t_4405_);
v___x_4407_ = lean_box_uint64(v___x_4406_);
v___x_4408_ = lean_apply_1(v_H_4404_, v___x_4407_);
return v___x_4408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310____boxed(lean_object* v_H_4409_, lean_object* v_t_4410_){
_start:
{
double v_t_12__boxed_4411_; lean_object* v_res_4412_; 
v_t_12__boxed_4411_ = lean_unbox_float(v_t_4410_);
lean_dec_ref(v_t_4410_);
v_res_4412_ = lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v_H_4409_, v_t_12__boxed_4411_);
return v_res_4412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(lean_object* v_motive_4413_, lean_object* v_H_4414_, double v_t_4415_){
_start:
{
lean_object* v___x_4416_; 
v___x_4416_ = lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v_H_4414_, v_t_4415_);
return v___x_4416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310____boxed(lean_object* v_motive_4417_, lean_object* v_H_4418_, lean_object* v_t_4419_){
_start:
{
double v_t_25__boxed_4420_; lean_object* v_res_4421_; 
v_t_25__boxed_4420_ = lean_unbox_float(v_t_4419_);
lean_dec_ref(v_t_4419_);
v_res_4421_ = lp_mathlib_Float_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v_motive_4417_, v_H_4418_, v_t_25__boxed_4420_);
return v_res_4421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(double v_t_4422_, lean_object* v_ofModel_4423_){
_start:
{
lean_object* v___x_4424_; 
v___x_4424_ = lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v_ofModel_4423_, v_t_4422_);
return v___x_4424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312____boxed(lean_object* v_t_4425_, lean_object* v_ofModel_4426_){
_start:
{
double v_t_boxed_4427_; lean_object* v_res_4428_; 
v_t_boxed_4427_ = lean_unbox_float(v_t_4425_);
lean_dec_ref(v_t_4425_);
v_res_4428_ = lp_mathlib_Float_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(v_t_boxed_4427_, v_ofModel_4426_);
return v_res_4428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(lean_object* v_motive_4429_, double v_t_4430_, lean_object* v_ofModel_4431_){
_start:
{
lean_object* v___x_4432_; 
v___x_4432_ = lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v_ofModel_4431_, v_t_4430_);
return v___x_4432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312____boxed(lean_object* v_motive_4433_, lean_object* v_t_4434_, lean_object* v_ofModel_4435_){
_start:
{
double v_t_boxed_4436_; lean_object* v_res_4437_; 
v_t_boxed_4436_ = lean_unbox_float(v_t_4434_);
lean_dec_ref(v_t_4434_);
v_res_4437_ = lp_mathlib_Float_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_312_(v_motive_4433_, v_t_boxed_4436_, v_ofModel_4435_);
return v_res_4437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(lean_object* v_mk_4438_, uint64_t v_t_4439_){
_start:
{
lean_object* v___x_4440_; lean_object* v___x_4441_; 
v___x_4440_ = lean_box_uint64(v_t_4439_);
v___x_4441_ = lean_apply_2(v_mk_4438_, v___x_4440_, lean_box(0));
return v___x_4441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314____boxed(lean_object* v_mk_4442_, lean_object* v_t_4443_){
_start:
{
uint64_t v_t_boxed_4444_; lean_object* v_res_4445_; 
v_t_boxed_4444_ = lean_unbox_uint64(v_t_4443_);
lean_dec_ref(v_t_4443_);
v_res_4445_ = lp_mathlib_Float_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(v_mk_4442_, v_t_boxed_4444_);
return v_res_4445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(lean_object* v_motive_4446_, lean_object* v_mk_4447_, uint64_t v_t_4448_){
_start:
{
lean_object* v___x_4449_; lean_object* v___x_4450_; 
v___x_4449_ = lean_box_uint64(v_t_4448_);
v___x_4450_ = lean_apply_2(v_mk_4447_, v___x_4449_, lean_box(0));
return v___x_4450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314____boxed(lean_object* v_motive_4451_, lean_object* v_mk_4452_, lean_object* v_t_4453_){
_start:
{
uint64_t v_t_boxed_4454_; lean_object* v_res_4455_; 
v_t_boxed_4454_ = lean_unbox_uint64(v_t_4453_);
lean_dec_ref(v_t_4453_);
v_res_4455_ = lp_mathlib_Float_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_314_(v_motive_4451_, v_mk_4452_, v_t_boxed_4454_);
return v_res_4455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(uint64_t v_t_4456_, lean_object* v_mk_4457_){
_start:
{
lean_object* v___x_4458_; lean_object* v___x_4459_; 
v___x_4458_ = lean_box_uint64(v_t_4456_);
v___x_4459_ = lean_apply_2(v_mk_4457_, v___x_4458_, lean_box(0));
return v___x_4459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316____boxed(lean_object* v_t_4460_, lean_object* v_mk_4461_){
_start:
{
uint64_t v_t_boxed_4462_; lean_object* v_res_4463_; 
v_t_boxed_4462_ = lean_unbox_uint64(v_t_4460_);
lean_dec_ref(v_t_4460_);
v_res_4463_ = lp_mathlib_Float_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(v_t_boxed_4462_, v_mk_4461_);
return v_res_4463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(lean_object* v_motive_4464_, uint64_t v_t_4465_, lean_object* v_mk_4466_){
_start:
{
lean_object* v___x_4467_; lean_object* v___x_4468_; 
v___x_4467_ = lean_box_uint64(v_t_4465_);
v___x_4468_ = lean_apply_2(v_mk_4466_, v___x_4467_, lean_box(0));
return v___x_4468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316____boxed(lean_object* v_motive_4469_, lean_object* v_t_4470_, lean_object* v_mk_4471_){
_start:
{
uint64_t v_t_boxed_4472_; lean_object* v_res_4473_; 
v_t_boxed_4472_ = lean_unbox_uint64(v_t_4470_);
lean_dec_ref(v_t_4470_);
v_res_4473_ = lp_mathlib_Float_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_316_(v_motive_4469_, v_t_boxed_4472_, v_mk_4471_);
return v_res_4473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318_(uint64_t v_t_4474_){
_start:
{
lean_object* v___x_4475_; lean_object* v___x_4476_; lean_object* v___x_4477_; 
v___x_4475_ = lean_unsigned_to_nat(1u);
v___x_4476_ = lp_mathlib_UInt64___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_284_(v_t_4474_);
v___x_4477_ = lean_nat_add(v___x_4475_, v___x_4476_);
lean_dec(v___x_4476_);
return v___x_4477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318____boxed(lean_object* v_t_4478_){
_start:
{
uint64_t v_t_boxed_4479_; lean_object* v_res_4480_; 
v_t_boxed_4479_ = lean_unbox_uint64(v_t_4478_);
lean_dec_ref(v_t_4478_);
v_res_4480_ = lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318_(v_t_boxed_4479_);
return v_res_4480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(uint64_t v_toModel_4483_){
_start:
{
lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v___x_4486_; 
v___x_4484_ = lean_unsigned_to_nat(1u);
v___x_4485_ = lp_mathlib_Float_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_318_(v_toModel_4483_);
v___x_4486_ = lean_nat_add(v___x_4484_, v___x_4485_);
lean_dec(v___x_4485_);
return v___x_4486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed(lean_object* v_toModel_4487_){
_start:
{
uint64_t v_toModel_boxed_4488_; lean_object* v_res_4489_; 
v_toModel_boxed_4488_ = lean_unbox_uint64(v_toModel_4487_);
lean_dec_ref(v_toModel_4487_);
v_res_4489_ = lp_mathlib_Float___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(v_toModel_boxed_4488_);
return v_res_4489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(double v_t_4491_){
_start:
{
lean_object* v___f_4492_; lean_object* v___x_4493_; 
v___f_4492_ = ((lean_object*)(lp_mathlib_Float___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_));
v___x_4493_ = lp_mathlib_Float_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_310_(v___f_4492_, v_t_4491_);
return v___x_4493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322____boxed(lean_object* v_t_4494_){
_start:
{
double v_t_boxed_4495_; lean_object* v_res_4496_; 
v_t_boxed_4495_ = lean_unbox_float(v_t_4494_);
lean_dec_ref(v_t_4494_);
v_res_4496_ = lp_mathlib_Float___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_322_(v_t_boxed_4495_);
return v_res_4496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(lean_object* v_H_4499_, float v_t_4500_){
_start:
{
uint32_t v___x_4501_; lean_object* v___x_4502_; lean_object* v___x_4503_; 
v___x_4501_ = lean_float32_to_bits(v_t_4500_);
v___x_4502_ = lean_box_uint32(v___x_4501_);
v___x_4503_ = lean_apply_1(v_H_4499_, v___x_4502_);
return v___x_4503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333____boxed(lean_object* v_H_4504_, lean_object* v_t_4505_){
_start:
{
float v_t_12__boxed_4506_; lean_object* v_res_4507_; 
v_t_12__boxed_4506_ = lean_unbox_float32(v_t_4505_);
lean_dec_ref(v_t_4505_);
v_res_4507_ = lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v_H_4504_, v_t_12__boxed_4506_);
return v_res_4507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(lean_object* v_motive_4508_, lean_object* v_H_4509_, float v_t_4510_){
_start:
{
lean_object* v___x_4511_; 
v___x_4511_ = lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v_H_4509_, v_t_4510_);
return v___x_4511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333____boxed(lean_object* v_motive_4512_, lean_object* v_H_4513_, lean_object* v_t_4514_){
_start:
{
float v_t_25__boxed_4515_; lean_object* v_res_4516_; 
v_t_25__boxed_4515_ = lean_unbox_float32(v_t_4514_);
lean_dec_ref(v_t_4514_);
v_res_4516_ = lp_mathlib_Float32_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v_motive_4512_, v_H_4513_, v_t_25__boxed_4515_);
return v_res_4516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(float v_t_4517_, lean_object* v_ofModel_4518_){
_start:
{
lean_object* v___x_4519_; 
v___x_4519_ = lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v_ofModel_4518_, v_t_4517_);
return v___x_4519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335____boxed(lean_object* v_t_4520_, lean_object* v_ofModel_4521_){
_start:
{
float v_t_boxed_4522_; lean_object* v_res_4523_; 
v_t_boxed_4522_ = lean_unbox_float32(v_t_4520_);
lean_dec_ref(v_t_4520_);
v_res_4523_ = lp_mathlib_Float32_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(v_t_boxed_4522_, v_ofModel_4521_);
return v_res_4523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(lean_object* v_motive_4524_, float v_t_4525_, lean_object* v_ofModel_4526_){
_start:
{
lean_object* v___x_4527_; 
v___x_4527_ = lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v_ofModel_4526_, v_t_4525_);
return v___x_4527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335____boxed(lean_object* v_motive_4528_, lean_object* v_t_4529_, lean_object* v_ofModel_4530_){
_start:
{
float v_t_boxed_4531_; lean_object* v_res_4532_; 
v_t_boxed_4531_ = lean_unbox_float32(v_t_4529_);
lean_dec_ref(v_t_4529_);
v_res_4532_ = lp_mathlib_Float32_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_335_(v_motive_4528_, v_t_boxed_4531_, v_ofModel_4530_);
return v_res_4532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(lean_object* v_mk_4533_, uint32_t v_t_4534_){
_start:
{
lean_object* v___x_4535_; lean_object* v___x_4536_; 
v___x_4535_ = lean_box_uint32(v_t_4534_);
v___x_4536_ = lean_apply_2(v_mk_4533_, v___x_4535_, lean_box(0));
return v___x_4536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337____boxed(lean_object* v_mk_4537_, lean_object* v_t_4538_){
_start:
{
uint32_t v_t_boxed_4539_; lean_object* v_res_4540_; 
v_t_boxed_4539_ = lean_unbox_uint32(v_t_4538_);
lean_dec(v_t_4538_);
v_res_4540_ = lp_mathlib_Float32_Model_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(v_mk_4537_, v_t_boxed_4539_);
return v_res_4540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(lean_object* v_motive_4541_, lean_object* v_mk_4542_, uint32_t v_t_4543_){
_start:
{
lean_object* v___x_4544_; lean_object* v___x_4545_; 
v___x_4544_ = lean_box_uint32(v_t_4543_);
v___x_4545_ = lean_apply_2(v_mk_4542_, v___x_4544_, lean_box(0));
return v___x_4545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337____boxed(lean_object* v_motive_4546_, lean_object* v_mk_4547_, lean_object* v_t_4548_){
_start:
{
uint32_t v_t_boxed_4549_; lean_object* v_res_4550_; 
v_t_boxed_4549_ = lean_unbox_uint32(v_t_4548_);
lean_dec(v_t_4548_);
v_res_4550_ = lp_mathlib_Float32_Model_rec_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_337_(v_motive_4546_, v_mk_4547_, v_t_boxed_4549_);
return v_res_4550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(uint32_t v_t_4551_, lean_object* v_mk_4552_){
_start:
{
lean_object* v___x_4553_; lean_object* v___x_4554_; 
v___x_4553_ = lean_box_uint32(v_t_4551_);
v___x_4554_ = lean_apply_2(v_mk_4552_, v___x_4553_, lean_box(0));
return v___x_4554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339____boxed(lean_object* v_t_4555_, lean_object* v_mk_4556_){
_start:
{
uint32_t v_t_boxed_4557_; lean_object* v_res_4558_; 
v_t_boxed_4557_ = lean_unbox_uint32(v_t_4555_);
lean_dec(v_t_4555_);
v_res_4558_ = lp_mathlib_Float32_Model_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(v_t_boxed_4557_, v_mk_4556_);
return v_res_4558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(lean_object* v_motive_4559_, uint32_t v_t_4560_, lean_object* v_mk_4561_){
_start:
{
lean_object* v___x_4562_; lean_object* v___x_4563_; 
v___x_4562_ = lean_box_uint32(v_t_4560_);
v___x_4563_ = lean_apply_2(v_mk_4561_, v___x_4562_, lean_box(0));
return v___x_4563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339____boxed(lean_object* v_motive_4564_, lean_object* v_t_4565_, lean_object* v_mk_4566_){
_start:
{
uint32_t v_t_boxed_4567_; lean_object* v_res_4568_; 
v_t_boxed_4567_ = lean_unbox_uint32(v_t_4565_);
lean_dec(v_t_4565_);
v_res_4568_ = lp_mathlib_Float32_Model_recOn_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_339_(v_motive_4564_, v_t_boxed_4567_, v_mk_4566_);
return v_res_4568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341_(uint32_t v_t_4569_){
_start:
{
lean_object* v___x_4570_; lean_object* v___x_4571_; lean_object* v___x_4572_; 
v___x_4570_ = lean_unsigned_to_nat(1u);
v___x_4571_ = lp_mathlib_UInt32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_269_(v_t_4569_);
v___x_4572_ = lean_nat_add(v___x_4570_, v___x_4571_);
lean_dec(v___x_4571_);
return v___x_4572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341____boxed(lean_object* v_t_4573_){
_start:
{
uint32_t v_t_boxed_4574_; lean_object* v_res_4575_; 
v_t_boxed_4574_ = lean_unbox_uint32(v_t_4573_);
lean_dec(v_t_4573_);
v_res_4575_ = lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341_(v_t_boxed_4574_);
return v_res_4575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(uint32_t v_toModel_4578_){
_start:
{
lean_object* v___x_4579_; lean_object* v___x_4580_; lean_object* v___x_4581_; 
v___x_4579_ = lean_unsigned_to_nat(1u);
v___x_4580_ = lp_mathlib_Float32_Model___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_341_(v_toModel_4578_);
v___x_4581_ = lean_nat_add(v___x_4579_, v___x_4580_);
lean_dec(v___x_4580_);
return v___x_4581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed(lean_object* v_toModel_4582_){
_start:
{
uint32_t v_toModel_boxed_4583_; lean_object* v_res_4584_; 
v_toModel_boxed_4583_ = lean_unbox_uint32(v_toModel_4582_);
lean_dec(v_toModel_4582_);
v_res_4584_ = lp_mathlib_Float32___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(v_toModel_boxed_4583_);
return v_res_4584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(float v_t_4586_){
_start:
{
lean_object* v___f_4587_; lean_object* v___x_4588_; 
v___f_4587_ = ((lean_object*)(lp_mathlib_Float32___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_));
v___x_4588_ = lp_mathlib_Float32_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_333_(v___f_4587_, v_t_4586_);
return v___x_4588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345____boxed(lean_object* v_t_4589_){
_start:
{
float v_t_boxed_4590_; lean_object* v_res_4591_; 
v_t_boxed_4590_ = lean_unbox_float32(v_t_4589_);
lean_dec_ref(v_t_4589_);
v_res_4591_ = lp_mathlib_Float32___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_345_(v_t_boxed_4590_);
return v_res_4591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(lean_object* v_ofByteArray_4594_, lean_object* v_t_4595_){
_start:
{
lean_object* v___x_4596_; lean_object* v___x_4597_; 
v___x_4596_ = lean_string_to_utf8(v_t_4595_);
v___x_4597_ = lean_apply_2(v_ofByteArray_4594_, v___x_4596_, lean_box(0));
return v___x_4597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(lean_object* v_motive_4598_, lean_object* v_ofByteArray_4599_, lean_object* v_t_4600_){
_start:
{
lean_object* v___x_4601_; 
v___x_4601_ = lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(v_ofByteArray_4599_, v_t_4600_);
return v___x_4601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_5_(lean_object* v_t_4602_, lean_object* v_ofByteArray_4603_){
_start:
{
lean_object* v___x_4604_; 
v___x_4604_ = lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(v_ofByteArray_4603_, v_t_4602_);
return v___x_4604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_5_(lean_object* v_motive_4605_, lean_object* v_t_4606_, lean_object* v_ofByteArray_4607_){
_start:
{
lean_object* v___x_4608_; 
v___x_4608_ = lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(v_ofByteArray_4607_, v_t_4606_);
return v___x_4608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(lean_object* v_mk_4609_, lean_object* v_t_4610_){
_start:
{
lean_object* v___x_4611_; lean_object* v___x_4612_; 
v___x_4611_ = lean_byte_array_data(v_t_4610_);
v___x_4612_ = lean_apply_1(v_mk_4609_, v___x_4611_);
return v___x_4612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(lean_object* v_motive_4613_, lean_object* v_mk_4614_, lean_object* v_t_4615_){
_start:
{
lean_object* v___x_4616_; 
v___x_4616_ = lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(v_mk_4614_, v_t_4615_);
return v___x_4616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_9_(lean_object* v_t_4617_, lean_object* v_mk_4618_){
_start:
{
lean_object* v___x_4619_; 
v___x_4619_ = lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(v_mk_4618_, v_t_4617_);
return v___x_4619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_9_(lean_object* v_motive_4620_, lean_object* v_t_4621_, lean_object* v_mk_4622_){
_start:
{
lean_object* v___x_4623_; 
v___x_4623_ = lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(v_mk_4622_, v_t_4621_);
return v___x_4623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(lean_object* v_mk_4624_, lean_object* v_t_4625_){
_start:
{
lean_object* v___x_4626_; lean_object* v___x_4627_; 
v___x_4626_ = lean_array_to_list(v_t_4625_);
v___x_4627_ = lean_apply_1(v_mk_4624_, v___x_4626_);
return v___x_4627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_rec_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(lean_object* v_00_u03b1_4628_, lean_object* v_motive_4629_, lean_object* v_mk_4630_, lean_object* v_t_4631_){
_start:
{
lean_object* v___x_4632_; 
v___x_4632_ = lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(v_mk_4630_, v_t_4631_);
return v___x_4632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_13_(lean_object* v_t_4633_, lean_object* v_mk_4634_){
_start:
{
lean_object* v___x_4635_; 
v___x_4635_ = lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(v_mk_4634_, v_t_4633_);
return v___x_4635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_recOn_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_13_(lean_object* v_00_u03b1_4636_, lean_object* v_motive_4637_, lean_object* v_t_4638_, lean_object* v_mk_4639_){
_start:
{
lean_object* v___x_4640_; 
v___x_4640_ = lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(v_mk_4639_, v_t_4638_);
return v___x_4640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object* v_inst_4641_, lean_object* v_toList_4642_){
_start:
{
lean_object* v___x_4643_; lean_object* v___x_4644_; lean_object* v___x_4645_; 
v___x_4643_ = lean_unsigned_to_nat(1u);
v___x_4644_ = lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_4641_, v_toList_4642_);
v___x_4645_ = lean_nat_add(v___x_4643_, v___x_4644_);
lean_dec(v___x_4644_);
return v___x_4645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object* v_inst_4646_, lean_object* v_t_4647_){
_start:
{
lean_object* v___f_4648_; lean_object* v___x_4649_; 
v___f_4648_ = lean_alloc_closure((void*)(lp_mathlib_Array___sizeOf__1___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_), 2, 1);
lean_closure_set(v___f_4648_, 0, v_inst_4646_);
v___x_4649_ = lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(v___f_4648_, v_t_4647_);
return v___x_4649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(lean_object* v_00_u03b1_4650_, lean_object* v_inst_4651_, lean_object* v_t_4652_){
_start:
{
lean_object* v___x_4653_; 
v___x_4653_ = lp_mathlib_Array___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(v_inst_4651_, v_t_4652_);
return v___x_4653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object* v_inst_4654_, lean_object* v_m_4655_){
_start:
{
lean_object* v___x_4656_; 
v___x_4656_ = lp_mathlib_Array___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15_(v_inst_4654_, v_m_4655_);
return v___x_4656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object* v_inst_4657_){
_start:
{
lean_object* v___f_4658_; 
v___f_4658_ = lean_alloc_closure((void*)(lp_mathlib_Array___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_), 2, 1);
lean_closure_set(v___f_4658_, 0, v_inst_4657_);
return v___f_4658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__inst_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_(lean_object* v_00_u03b1_4659_, lean_object* v_inst_4660_){
_start:
{
lean_object* v___f_4661_; 
v___f_4661_ = lean_alloc_closure((void*)(lp_mathlib_Array___sizeOf__inst___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_17_), 2, 1);
lean_closure_set(v___f_4661_, 0, v_inst_4660_);
return v___f_4661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0(lean_object* v___x_4662_, uint8_t v_head_4663_, lean_object* v_tail_4664_, lean_object* v_tail__ih_4665_){
_start:
{
lean_object* v___x_4666_; lean_object* v___x_4667_; lean_object* v___x_4668_; 
v___x_4666_ = lp_mathlib_UInt8___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3310122970____hygCtx___hyg_239_(v_head_4663_);
v___x_4667_ = lean_nat_add(v___x_4662_, v___x_4666_);
lean_dec(v___x_4666_);
v___x_4668_ = lean_nat_add(v___x_4667_, v_tail__ih_4665_);
lean_dec(v___x_4667_);
return v___x_4668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0___boxed(lean_object* v___x_4669_, lean_object* v_head_4670_, lean_object* v_tail_4671_, lean_object* v_tail__ih_4672_){
_start:
{
uint8_t v_head_boxed_4673_; lean_object* v_res_4674_; 
v_head_boxed_4673_ = lean_unbox(v_head_4670_);
v_res_4674_ = lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___lam__0(v___x_4669_, v_head_boxed_4673_, v_tail_4671_, v_tail__ih_4672_);
lean_dec(v_tail__ih_4672_);
lean_dec(v_tail_4671_);
lean_dec(v___x_4669_);
return v_res_4674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0(lean_object* v_t_4677_){
_start:
{
lean_object* v___x_4678_; lean_object* v___f_4679_; lean_object* v___x_4680_; 
v___x_4678_ = lean_unsigned_to_nat(1u);
v___f_4679_ = ((lean_object*)(lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0___closed__0));
v___x_4680_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v___x_4678_, v___f_4679_, v_t_4677_);
return v___x_4680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___lam__0(lean_object* v_toList_4681_){
_start:
{
lean_object* v___x_4682_; lean_object* v___x_4683_; lean_object* v___x_4684_; 
v___x_4682_ = lean_unsigned_to_nat(1u);
v___x_4683_ = lp_mathlib_List___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11____at___00Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0_spec__0(v_toList_4681_);
v___x_4684_ = lean_nat_add(v___x_4682_, v___x_4683_);
lean_dec(v___x_4683_);
return v___x_4684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0(lean_object* v_t_4686_){
_start:
{
lean_object* v___f_4687_; lean_object* v___x_4688_; 
v___f_4687_ = ((lean_object*)(lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0___closed__0));
v___x_4688_ = lp_mathlib_Array_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_11_(v___f_4687_, v_t_4686_);
return v___x_4688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_(lean_object* v_data_4689_){
_start:
{
lean_object* v___x_4690_; lean_object* v___x_4691_; lean_object* v___x_4692_; 
v___x_4690_ = lean_unsigned_to_nat(1u);
v___x_4691_ = lp_mathlib_Array___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_15____at___00ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19__spec__0(v_data_4689_);
v___x_4692_ = lean_nat_add(v___x_4690_, v___x_4691_);
lean_dec(v___x_4691_);
return v___x_4692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_(lean_object* v_t_4694_){
_start:
{
lean_object* v___f_4695_; lean_object* v___x_4696_; 
v___f_4695_ = ((lean_object*)(lp_mathlib_ByteArray___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_));
v___x_4696_ = lp_mathlib_ByteArray_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_7_(v___f_4695_, v_t_4694_);
return v___x_4696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String___sizeOf__1___lam__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_(lean_object* v_toByteArray_4699_, lean_object* v_isValidUTF8_4700_){
_start:
{
lean_object* v___x_4701_; lean_object* v___x_4702_; lean_object* v___x_4703_; 
v___x_4701_ = lean_unsigned_to_nat(1u);
v___x_4702_ = lp_mathlib_ByteArray___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_19_(v_toByteArray_4699_);
v___x_4703_ = lean_nat_add(v___x_4701_, v___x_4702_);
lean_dec(v___x_4702_);
return v___x_4703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_(lean_object* v_t_4705_){
_start:
{
lean_object* v___f_4706_; lean_object* v___x_4707_; 
v___f_4706_ = ((lean_object*)(lp_mathlib_String___sizeOf__1___closed__0_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_));
v___x_4707_ = lp_mathlib_String_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_3_(v___f_4706_, v_t_4705_);
return v___x_4707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(lean_object* v_anonymous_4710_, lean_object* v_str_4711_, lean_object* v_num_4712_, lean_object* v_t_4713_){
_start:
{
switch(lean_obj_tag(v_t_4713_))
{
case 0:
{
lean_dec(v_num_4712_);
lean_dec(v_str_4711_);
lean_inc(v_anonymous_4710_);
return v_anonymous_4710_;
}
case 1:
{
lean_object* v_pre_4714_; lean_object* v_str_4715_; lean_object* v___x_4716_; lean_object* v___x_4717_; 
v_pre_4714_ = lean_ctor_get(v_t_4713_, 0);
lean_inc_n(v_pre_4714_, 2);
v_str_4715_ = lean_ctor_get(v_t_4713_, 1);
lean_inc_ref(v_str_4715_);
lean_dec_ref_known(v_t_4713_, 2);
lean_inc(v_str_4711_);
v___x_4716_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4710_, v_str_4711_, v_num_4712_, v_pre_4714_);
v___x_4717_ = lean_apply_3(v_str_4711_, v_pre_4714_, v_str_4715_, v___x_4716_);
return v___x_4717_;
}
default: 
{
lean_object* v_pre_4718_; lean_object* v_i_4719_; lean_object* v___x_4720_; lean_object* v___x_4721_; 
v_pre_4718_ = lean_ctor_get(v_t_4713_, 0);
lean_inc_n(v_pre_4718_, 2);
v_i_4719_ = lean_ctor_get(v_t_4713_, 1);
lean_inc(v_i_4719_);
lean_dec_ref_known(v_t_4713_, 2);
lean_inc(v_num_4712_);
v___x_4720_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4710_, v_str_4711_, v_num_4712_, v_pre_4718_);
v___x_4721_ = lean_apply_3(v_num_4712_, v_pre_4718_, v_i_4719_, v___x_4720_);
return v___x_4721_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3____boxed(lean_object* v_anonymous_4722_, lean_object* v_str_4723_, lean_object* v_num_4724_, lean_object* v_t_4725_){
_start:
{
lean_object* v_res_4726_; 
v_res_4726_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4722_, v_str_4723_, v_num_4724_, v_t_4725_);
lean_dec(v_anonymous_4722_);
return v_res_4726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(lean_object* v_motive_4727_, lean_object* v_anonymous_4728_, lean_object* v_str_4729_, lean_object* v_num_4730_, lean_object* v_t_4731_){
_start:
{
lean_object* v___x_4732_; 
v___x_4732_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4728_, v_str_4729_, v_num_4730_, v_t_4731_);
return v___x_4732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_rec_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3____boxed(lean_object* v_motive_4733_, lean_object* v_anonymous_4734_, lean_object* v_str_4735_, lean_object* v_num_4736_, lean_object* v_t_4737_){
_start:
{
lean_object* v_res_4738_; 
v_res_4738_ = lp_mathlib_Lean_Name_rec_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_motive_4733_, v_anonymous_4734_, v_str_4735_, v_num_4736_, v_t_4737_);
lean_dec(v_anonymous_4734_);
return v_res_4738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(lean_object* v_t_4739_, lean_object* v_anonymous_4740_, lean_object* v_str_4741_, lean_object* v_num_4742_){
_start:
{
lean_object* v___x_4743_; 
v___x_4743_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4740_, v_str_4741_, v_num_4742_, v_t_4739_);
return v___x_4743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5____boxed(lean_object* v_t_4744_, lean_object* v_anonymous_4745_, lean_object* v_str_4746_, lean_object* v_num_4747_){
_start:
{
lean_object* v_res_4748_; 
v_res_4748_ = lp_mathlib_Lean_Name_recOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(v_t_4744_, v_anonymous_4745_, v_str_4746_, v_num_4747_);
lean_dec(v_anonymous_4745_);
return v_res_4748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(lean_object* v_motive_4749_, lean_object* v_t_4750_, lean_object* v_anonymous_4751_, lean_object* v_str_4752_, lean_object* v_num_4753_){
_start:
{
lean_object* v___x_4754_; 
v___x_4754_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v_anonymous_4751_, v_str_4752_, v_num_4753_, v_t_4750_);
return v___x_4754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_recOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5____boxed(lean_object* v_motive_4755_, lean_object* v_t_4756_, lean_object* v_anonymous_4757_, lean_object* v_str_4758_, lean_object* v_num_4759_){
_start:
{
lean_object* v_res_4760_; 
v_res_4760_ = lp_mathlib_Lean_Name_recOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_5_(v_motive_4755_, v_t_4756_, v_anonymous_4757_, v_str_4758_, v_num_4759_);
lean_dec(v_anonymous_4757_);
return v_res_4760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object* v_F__1_4761_, lean_object* v_pre_4762_, lean_object* v_str_4763_, lean_object* v_pre__ih_4764_){
_start:
{
lean_object* v___x_4765_; lean_object* v___x_4766_; lean_object* v___x_4767_; 
v___x_4765_ = l_Lean_Name_str___override(v_pre_4762_, v_str_4763_);
lean_inc_ref(v_pre__ih_4764_);
v___x_4766_ = lean_apply_2(v_F__1_4761_, v___x_4765_, v_pre__ih_4764_);
v___x_4767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4767_, 0, v___x_4766_);
lean_ctor_set(v___x_4767_, 1, v_pre__ih_4764_);
return v___x_4767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg___lam__1_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object* v_F__1_4768_, lean_object* v_pre_4769_, lean_object* v_i_4770_, lean_object* v_pre__ih_4771_){
_start:
{
lean_object* v___x_4772_; lean_object* v___x_4773_; lean_object* v___x_4774_; 
v___x_4772_ = l_Lean_Name_num___override(v_pre_4769_, v_i_4770_);
lean_inc_ref(v_pre__ih_4771_);
v___x_4773_ = lean_apply_2(v_F__1_4768_, v___x_4772_, v_pre__ih_4771_);
v___x_4774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4774_, 0, v___x_4773_);
lean_ctor_set(v___x_4774_, 1, v_pre__ih_4771_);
return v___x_4774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object* v_t_4775_, lean_object* v_F__1_4776_){
_start:
{
lean_object* v___f_4777_; lean_object* v___f_4778_; lean_object* v___x_4779_; lean_object* v___x_4780_; lean_object* v___x_4781_; lean_object* v___x_4782_; lean_object* v___x_4783_; 
lean_inc_n(v_F__1_4776_, 2);
v___f_4777_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Name_brecOn_go___redArg___lam__0_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_), 4, 1);
lean_closure_set(v___f_4777_, 0, v_F__1_4776_);
v___f_4778_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Name_brecOn_go___redArg___lam__1_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_), 4, 1);
lean_closure_set(v___f_4778_, 0, v_F__1_4776_);
v___x_4779_ = lean_box(0);
v___x_4780_ = lean_box(0);
v___x_4781_ = lean_apply_2(v_F__1_4776_, v___x_4779_, v___x_4780_);
v___x_4782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4782_, 0, v___x_4781_);
lean_ctor_set(v___x_4782_, 1, v___x_4780_);
v___x_4783_ = lp_mathlib_Lean_Name_rec___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_3_(v___x_4782_, v___f_4777_, v___f_4778_, v_t_4775_);
lean_dec_ref_known(v___x_4782_, 2);
return v___x_4783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_go_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(lean_object* v_motive_4784_, lean_object* v_t_4785_, lean_object* v_F__1_4786_){
_start:
{
lean_object* v___x_4787_; 
v___x_4787_ = lp_mathlib_Lean_Name_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(v_t_4785_, v_F__1_4786_);
return v___x_4787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(lean_object* v_t_4788_, lean_object* v_F__1_4789_){
_start:
{
lean_object* v___x_4790_; lean_object* v_fst_4791_; 
v___x_4790_ = lp_mathlib_Lean_Name_brecOn_go___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_7_(v_t_4788_, v_F__1_4789_);
v_fst_4791_ = lean_ctor_get(v___x_4790_, 0);
lean_inc(v_fst_4791_);
lean_dec_ref(v___x_4790_);
return v_fst_4791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_brecOn_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(lean_object* v_motive_4792_, lean_object* v_t_4793_, lean_object* v_F__1_4794_){
_start:
{
lean_object* v___x_4795_; 
v___x_4795_ = lp_mathlib_Lean_Name_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(v_t_4793_, v_F__1_4794_);
return v___x_4795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3_(lean_object* v_x_4796_, lean_object* v_f_4797_){
_start:
{
switch(lean_obj_tag(v_x_4796_))
{
case 0:
{
lean_object* v___x_4798_; 
v___x_4798_ = lean_unsigned_to_nat(1u);
return v___x_4798_;
}
case 1:
{
lean_object* v_str_4799_; lean_object* v_fst_4800_; lean_object* v___x_4801_; lean_object* v___x_4802_; lean_object* v___x_4803_; lean_object* v___x_4804_; 
v_str_4799_ = lean_ctor_get(v_x_4796_, 1);
lean_inc_ref(v_str_4799_);
lean_dec_ref_known(v_x_4796_, 2);
v_fst_4800_ = lean_ctor_get(v_f_4797_, 0);
v___x_4801_ = lean_unsigned_to_nat(1u);
v___x_4802_ = lean_nat_add(v___x_4801_, v_fst_4800_);
v___x_4803_ = lp_mathlib_String___sizeOf__1_00___x40_Mathlib_Util_CompileInductive_3964927263____hygCtx___hyg_23_(v_str_4799_);
v___x_4804_ = lean_nat_add(v___x_4802_, v___x_4803_);
lean_dec(v___x_4803_);
lean_dec(v___x_4802_);
return v___x_4804_;
}
default: 
{
lean_object* v_i_4805_; lean_object* v_fst_4806_; lean_object* v___x_4807_; lean_object* v___x_4808_; lean_object* v___x_4809_; 
v_i_4805_ = lean_ctor_get(v_x_4796_, 1);
lean_inc(v_i_4805_);
lean_dec_ref_known(v_x_4796_, 2);
v_fst_4806_ = lean_ctor_get(v_f_4797_, 0);
v___x_4807_ = lean_unsigned_to_nat(1u);
v___x_4808_ = lean_nat_add(v___x_4807_, v_fst_4806_);
v___x_4809_ = lean_nat_add(v___x_4808_, v_i_4805_);
lean_dec(v_i_4805_);
lean_dec(v___x_4808_);
return v___x_4809_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3____boxed(lean_object* v_x_4810_, lean_object* v_f_4811_){
_start:
{
lean_object* v_res_4812_; 
v_res_4812_ = lp_mathlib_Lean_Name_sizeOf___f_00___x40_Mathlib_Util_CompileInductive_1775691164____hygCtx___hyg_3_(v_x_4810_, v_f_4811_);
lean_dec(v_f_4811_);
return v_res_4812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_sizeOf_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3_(lean_object* v_x_4814_){
_start:
{
lean_object* v___x_4815_; lean_object* v___x_4816_; 
v___x_4815_ = ((lean_object*)(lp_mathlib_Lean_Name_sizeOf___closed__0_00___x40_Mathlib_Util_CompileInductive_2201498507____hygCtx___hyg_3_));
v___x_4816_ = lp_mathlib_Lean_Name_brecOn___redArg_00___x40_Mathlib_Util_CompileInductive_350617726____hygCtx___hyg_9_(v_x_4814_, v___x_4815_);
return v___x_4816_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Compiler_CSimpAttr(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_FoldConsts(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_AssocList(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Compiler_CSimpAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_FoldConsts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_AssocList(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Compiler_CSimpAttr(uint8_t builtin);
lean_object* initialize_Lean_Util_FoldConsts(uint8_t builtin);
lean_object* initialize_Lean_Data_AssocList(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Compiler_CSimpAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_FoldConsts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_AssocList(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
}
#ifdef __cplusplus
}
#endif
