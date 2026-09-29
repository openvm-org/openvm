// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions.SectionState
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.Util public import ProofWidgets.Component.FilterDetails public meta import ProofWidgets.Util
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
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_binInsertM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_instMonadControlStateRefT_x27(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlReaderT(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
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
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps;
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_click__suggestions_debug;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_WithRpcRef_mk___redArg(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_InteractiveMessage;
lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_withCurrHeartbeats___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_IO_CancelToken_isSet(lean_object*);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadRefCoreM;
extern lean_object* l_Lean_Core_instAddMessageContextCoreM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_throwInterruptException___redArg(lean_object*);
lean_object* l_instMonadEIO___aux__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_EIO_catchExceptions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_as_task(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
uint64_t lean_string_hash(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableFilterDetailsProps_enc_00___x40_ProofWidgets_Component_FilterDetails_2598060976____hygCtx___hyg_1_(lean_object*, lean_object*);
extern lean_object* l_instMonadBaseIO;
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_FilterDetails;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
extern lean_object* l_Lean_interruptExceptionId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instLTResult(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instLTResult___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Tactic.ClickSuggestions.SectionState"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Mathlib.Tactic.ClickSuggestions.SectionState.insertResult"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "an error occurred when checking for duplicate entries:\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 1}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__3_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__8_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__9_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__9_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "class"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "error"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__15_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " Failures: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__18_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__20_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ul"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "padding-left"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__25_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "30px"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__26_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__27_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__29_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = " (local hypotheses)"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " (current file)"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "li"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " failed:\n        "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "br"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__18_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_2_ = lean_box(0);
v___x_3_ = lp_proofwidgets_ProofWidgets_instInhabitedHtml_default;
v___x_4_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4_, 0, v___x_2_);
lean_ctor_set(v___x_4_, 1, v___x_3_);
lean_ctor_set(v___x_4_, 2, v_inst_1_);
lean_ctor_set(v___x_4_, 3, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult___redArg(lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(v_inst_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult(lean_object* v_a_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0(lean_object* v_inst_13_, lean_object* v_x1_14_, lean_object* v_x2_15_){
_start:
{
lean_object* v_key_16_; lean_object* v_key_17_; lean_object* v___x_18_; uint8_t v___x_19_; 
v_key_16_ = lean_ctor_get(v_x1_14_, 2);
lean_inc(v_key_16_);
lean_dec_ref(v_x1_14_);
v_key_17_ = lean_ctor_get(v_x2_15_, 2);
lean_inc(v_key_17_);
lean_dec_ref(v_x2_15_);
v___x_18_ = lean_apply_2(v_inst_13_, v_key_16_, v_key_17_);
v___x_19_ = lean_unbox(v___x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0___boxed(lean_object* v_inst_20_, lean_object* v_x1_21_, lean_object* v_x2_22_){
_start:
{
uint8_t v_res_23_; lean_object* v_r_24_; 
v_res_23_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0(v_inst_20_, v_x1_21_, v_x2_22_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg(lean_object* v_inst_25_){
_start:
{
lean_object* v___f_26_; 
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_26_, 0, v_inst_25_);
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___f_29_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_29_, 0, v_inst_28_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instLTResult(lean_object* v_00_u03b1_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_box(0);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_instLTResult___boxed(lean_object* v_00_u03b1_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instLTResult(v_00_u03b1_33_, v_inst_34_);
lean_dec_ref(v_inst_34_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg(lean_object* v_result_36_, lean_object* v_isDup_37_, lean_object* v_as_38_, size_t v_sz_39_, size_t v_i_40_, lean_object* v_b_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
uint8_t v___x_47_; 
v___x_47_ = lean_usize_dec_lt(v_i_40_, v_sz_39_);
if (v___x_47_ == 0)
{
lean_object* v___x_48_; 
lean_dec_ref(v_isDup_37_);
lean_dec_ref(v_result_36_);
v___x_48_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_48_, 0, v_b_41_);
return v___x_48_;
}
else
{
lean_object* v_snd_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_99_; 
v_snd_49_ = lean_ctor_get(v_b_41_, 1);
v_isSharedCheck_99_ = !lean_is_exclusive(v_b_41_);
if (v_isSharedCheck_99_ == 0)
{
lean_object* v_unused_100_; 
v_unused_100_ = lean_ctor_get(v_b_41_, 0);
lean_dec(v_unused_100_);
v___x_51_ = v_b_41_;
v_isShared_52_ = v_isSharedCheck_99_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_snd_49_);
lean_dec(v_b_41_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_99_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v_a_53_; lean_object* v_filtered_54_; lean_object* v_key_55_; lean_object* v___x_56_; 
v_a_53_ = lean_array_uget_borrowed(v_as_38_, v_i_40_);
v_filtered_54_ = lean_ctor_get(v_a_53_, 0);
lean_inc(v_filtered_54_);
v_key_55_ = lean_ctor_get(v_a_53_, 2);
v___x_56_ = lean_box(0);
if (lean_obj_tag(v_filtered_54_) == 0)
{
goto v___jp_57_;
}
else
{
lean_object* v___x_67_; uint8_t v_isShared_68_; uint8_t v_isSharedCheck_97_; 
v_isSharedCheck_97_ = !lean_is_exclusive(v_filtered_54_);
if (v_isSharedCheck_97_ == 0)
{
lean_object* v_unused_98_; 
v_unused_98_ = lean_ctor_get(v_filtered_54_, 0);
lean_dec(v_unused_98_);
v___x_67_ = v_filtered_54_;
v_isShared_68_ = v_isSharedCheck_97_;
goto v_resetjp_66_;
}
else
{
lean_dec(v_filtered_54_);
v___x_67_ = lean_box(0);
v_isShared_68_ = v_isSharedCheck_97_;
goto v_resetjp_66_;
}
v_resetjp_66_:
{
lean_object* v_key_69_; lean_object* v___x_70_; 
v_key_69_ = lean_ctor_get(v_result_36_, 2);
lean_inc_ref(v_isDup_37_);
lean_inc(v___y_45_);
lean_inc_ref(v___y_44_);
lean_inc(v___y_43_);
lean_inc_ref(v___y_42_);
lean_inc(v_key_69_);
lean_inc(v_key_55_);
v___x_70_ = lean_apply_7(v_isDup_37_, v_key_55_, v_key_69_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, lean_box(0));
if (lean_obj_tag(v___x_70_) == 0)
{
lean_object* v_a_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_84_; 
v_a_71_ = lean_ctor_get(v___x_70_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v___x_70_);
if (v_isSharedCheck_84_ == 0)
{
v___x_73_ = v___x_70_;
v_isShared_74_ = v_isSharedCheck_84_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_a_71_);
lean_dec(v___x_70_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_84_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
uint8_t v___x_75_; 
v___x_75_ = lean_unbox(v_a_71_);
lean_dec(v_a_71_);
if (v___x_75_ == 0)
{
lean_del_object(v___x_73_);
lean_del_object(v___x_67_);
goto v___jp_57_;
}
else
{
lean_object* v___x_77_; 
lean_del_object(v___x_51_);
lean_dec_ref(v_isDup_37_);
lean_dec_ref(v_result_36_);
lean_inc(v_snd_49_);
if (v_isShared_68_ == 0)
{
lean_ctor_set(v___x_67_, 0, v_snd_49_);
v___x_77_ = v___x_67_;
goto v_reusejp_76_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_snd_49_);
v___x_77_ = v_reuseFailAlloc_83_;
goto v_reusejp_76_;
}
v_reusejp_76_:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_81_; 
v___x_78_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_snd_49_);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 0, v___x_79_);
v___x_81_ = v___x_73_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v___x_79_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
}
else
{
lean_object* v_a_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_96_; 
lean_del_object(v___x_67_);
v_a_85_ = lean_ctor_get(v___x_70_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_70_);
if (v_isSharedCheck_96_ == 0)
{
v___x_87_ = v___x_70_;
v_isShared_88_ = v_isSharedCheck_96_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_a_85_);
lean_dec(v___x_70_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_96_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
uint8_t v___y_90_; uint8_t v___x_94_; 
v___x_94_ = l_Lean_Exception_isInterrupt(v_a_85_);
if (v___x_94_ == 0)
{
uint8_t v___x_95_; 
lean_inc(v_a_85_);
v___x_95_ = l_Lean_Exception_isRuntime(v_a_85_);
v___y_90_ = v___x_95_;
goto v___jp_89_;
}
else
{
v___y_90_ = v___x_94_;
goto v___jp_89_;
}
v___jp_89_:
{
if (v___y_90_ == 0)
{
lean_del_object(v___x_87_);
lean_dec(v_a_85_);
goto v___jp_57_;
}
else
{
lean_object* v___x_92_; 
lean_del_object(v___x_51_);
lean_dec(v_snd_49_);
lean_dec_ref(v_isDup_37_);
lean_dec_ref(v_result_36_);
if (v_isShared_88_ == 0)
{
v___x_92_ = v___x_87_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v_a_85_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
}
}
}
v___jp_57_:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_61_; 
v___x_58_ = lean_unsigned_to_nat(1u);
v___x_59_ = lean_nat_add(v_snd_49_, v___x_58_);
lean_dec(v_snd_49_);
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 1, v___x_59_);
lean_ctor_set(v___x_51_, 0, v___x_56_);
v___x_61_ = v___x_51_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___x_56_);
lean_ctor_set(v_reuseFailAlloc_65_, 1, v___x_59_);
v___x_61_ = v_reuseFailAlloc_65_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
size_t v___x_62_; size_t v___x_63_; 
v___x_62_ = ((size_t)1ULL);
v___x_63_ = lean_usize_add(v_i_40_, v___x_62_);
v_i_40_ = v___x_63_;
v_b_41_ = v___x_61_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg___boxed(lean_object* v_result_101_, lean_object* v_isDup_102_, lean_object* v_as_103_, lean_object* v_sz_104_, lean_object* v_i_105_, lean_object* v_b_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
size_t v_sz_boxed_112_; size_t v_i_boxed_113_; lean_object* v_res_114_; 
v_sz_boxed_112_ = lean_unbox_usize(v_sz_104_);
lean_dec(v_sz_104_);
v_i_boxed_113_ = lean_unbox_usize(v_i_105_);
lean_dec(v_i_105_);
v_res_114_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg(v_result_101_, v_isDup_102_, v_as_103_, v_sz_boxed_112_, v_i_boxed_113_, v_b_106_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec_ref(v_as_103_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg(lean_object* v_isDup_118_, lean_object* v_result_119_, lean_object* v_results_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_filtered_126_; 
v_filtered_126_ = lean_ctor_get(v_result_119_, 0);
if (lean_obj_tag(v_filtered_126_) == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec_ref(v_result_119_);
lean_dec_ref(v_isDup_118_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; size_t v_sz_131_; size_t v___x_132_; lean_object* v___x_133_; 
v___x_129_ = lean_box(0);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___closed__0));
v_sz_131_ = lean_array_size(v_results_120_);
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg(v_result_119_, v_isDup_118_, v_results_120_, v_sz_131_, v___x_132_, v___x_130_, v_a_121_, v_a_122_, v_a_123_, v_a_124_);
if (lean_obj_tag(v___x_133_) == 0)
{
lean_object* v_a_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_146_; 
v_a_134_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_146_ == 0)
{
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_146_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_a_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_146_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v_fst_138_; 
v_fst_138_ = lean_ctor_get(v_a_134_, 0);
lean_inc(v_fst_138_);
lean_dec(v_a_134_);
if (lean_obj_tag(v_fst_138_) == 0)
{
lean_object* v___x_140_; 
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 0, v___x_129_);
v___x_140_ = v___x_136_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_129_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
else
{
lean_object* v_val_142_; lean_object* v___x_144_; 
v_val_142_ = lean_ctor_get(v_fst_138_, 0);
lean_inc(v_val_142_);
lean_dec_ref_known(v_fst_138_, 1);
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 0, v_val_142_);
v___x_144_ = v___x_136_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v_val_142_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
v_a_147_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_133_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_133_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg___boxed(lean_object* v_isDup_155_, lean_object* v_result_156_, lean_object* v_results_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg(v_isDup_155_, v_result_156_, v_results_157_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
lean_dec(v_a_161_);
lean_dec_ref(v_a_160_);
lean_dec(v_a_159_);
lean_dec_ref(v_a_158_);
lean_dec_ref(v_results_157_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate(lean_object* v_00_u03b1_164_, lean_object* v_isDup_165_, lean_object* v_result_166_, lean_object* v_results_167_, lean_object* v_a_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg(v_isDup_165_, v_result_166_, v_results_167_, v_a_168_, v_a_169_, v_a_170_, v_a_171_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___boxed(lean_object* v_00_u03b1_174_, lean_object* v_isDup_175_, lean_object* v_result_176_, lean_object* v_results_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate(v_00_u03b1_174_, v_isDup_175_, v_result_176_, v_results_177_, v_a_178_, v_a_179_, v_a_180_, v_a_181_);
lean_dec(v_a_181_);
lean_dec_ref(v_a_180_);
lean_dec(v_a_179_);
lean_dec_ref(v_a_178_);
lean_dec_ref(v_results_177_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0(lean_object* v_00_u03b1_184_, lean_object* v_result_185_, lean_object* v_isDup_186_, lean_object* v_as_187_, size_t v_sz_188_, size_t v_i_189_, lean_object* v_b_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___redArg(v_result_185_, v_isDup_186_, v_as_187_, v_sz_188_, v_i_189_, v_b_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0___boxed(lean_object* v_00_u03b1_197_, lean_object* v_result_198_, lean_object* v_isDup_199_, lean_object* v_as_200_, lean_object* v_sz_201_, lean_object* v_i_202_, lean_object* v_b_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_){
_start:
{
size_t v_sz_boxed_209_; size_t v_i_boxed_210_; lean_object* v_res_211_; 
v_sz_boxed_209_ = lean_unbox_usize(v_sz_201_);
lean_dec(v_sz_201_);
v_i_boxed_210_ = lean_unbox_usize(v_i_202_);
lean_dec(v_i_202_);
v_res_211_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate_spec__0(v_00_u03b1_197_, v_result_198_, v_isDup_199_, v_as_200_, v_sz_boxed_209_, v_i_boxed_210_, v_b_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
lean_dec(v___y_205_);
lean_dec_ref(v___y_204_);
lean_dec_ref(v_as_200_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0(lean_object* v_res_212_, lean_object* v_x_213_){
_start:
{
lean_inc_ref(v_res_212_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0___boxed(lean_object* v_res_214_, lean_object* v_x_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0(v_res_214_, v_x_215_);
lean_dec_ref(v_x_215_);
lean_dec_ref(v_res_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1(lean_object* v_res_217_, lean_object* v_x_218_){
_start:
{
lean_inc_ref(v_res_217_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1___boxed(lean_object* v_res_219_, lean_object* v_x_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1(v_res_219_, v_x_220_);
lean_dec_ref(v_res_219_);
return v_res_221_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2(lean_object* v_inst_222_, lean_object* v_x1_223_, lean_object* v_x2_224_){
_start:
{
uint8_t v___x_225_; 
v___x_225_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0(v_inst_222_, v_x1_223_, v_x2_224_);
if (v___x_225_ == 0)
{
uint8_t v___x_226_; 
v___x_226_ = 1;
return v___x_226_;
}
else
{
uint8_t v___x_227_; 
v___x_227_ = 0;
return v___x_227_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2___boxed(lean_object* v_inst_228_, lean_object* v_x1_229_, lean_object* v_x2_230_){
_start:
{
uint8_t v_res_231_; lean_object* v_r_232_; 
v_res_231_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2(v_inst_228_, v_x1_229_, v_x2_230_);
v_r_232_ = lean_box(v_res_231_);
return v_r_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4(lean_object* v___x_233_, lean_object* v_x_234_){
_start:
{
lean_inc_ref(v___x_233_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4___boxed(lean_object* v___x_235_, lean_object* v_x_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4(v___x_235_, v_x_236_);
lean_dec_ref(v_x_236_);
lean_dec_ref(v___x_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3(lean_object* v___x_238_, lean_object* v_x_239_){
_start:
{
lean_inc_ref(v___x_238_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3___boxed(lean_object* v___x_240_, lean_object* v_x_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3(v___x_240_, v_x_241_);
lean_dec_ref(v___x_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg(lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_res_264_, lean_object* v_arr_265_, lean_object* v_isDup_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_){
_start:
{
lean_object* v___x_272_; 
lean_inc_ref(v_res_264_);
v___x_272_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_Result_insertInArray_findDuplicate___redArg(v_isDup_266_, v_res_264_, v_arr_265_, v_a_267_, v_a_268_, v_a_269_, v_a_270_);
if (lean_obj_tag(v___x_272_) == 0)
{
lean_object* v_a_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_337_; 
v_a_273_ = lean_ctor_get(v___x_272_, 0);
v_isSharedCheck_337_ = !lean_is_exclusive(v___x_272_);
if (v_isSharedCheck_337_ == 0)
{
v___x_275_ = v___x_272_;
v_isShared_276_ = v_isSharedCheck_337_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_a_273_);
lean_dec(v___x_272_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_337_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
if (lean_obj_tag(v_a_273_) == 1)
{
lean_object* v_val_277_; lean_object* v___x_278_; lean_object* v___x_279_; uint8_t v___x_280_; 
v_val_277_ = lean_ctor_get(v_a_273_, 0);
lean_inc(v_val_277_);
lean_dec_ref_known(v_a_273_, 1);
v___x_278_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instInhabitedResult_default___redArg(v_inst_263_);
v___x_279_ = lean_array_get(v___x_278_, v_arr_265_, v_val_277_);
lean_dec_ref(v___x_278_);
lean_inc_ref(v_res_264_);
lean_inc_ref(v_inst_262_);
v___x_280_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_instOrdResult___redArg___lam__0(v_inst_262_, v_res_264_, v___x_279_);
if (v___x_280_ == 0)
{
lean_object* v___f_281_; lean_object* v___f_282_; lean_object* v___f_283_; lean_object* v___y_285_; lean_object* v___x_291_; uint8_t v___x_292_; 
lean_inc_ref_n(v_res_264_, 2);
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_281_, 0, v_res_264_);
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_282_, 0, v_res_264_);
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_283_, 0, v_inst_262_);
v___x_291_ = lean_array_get_size(v_arr_265_);
v___x_292_ = lean_nat_dec_lt(v_val_277_, v___x_291_);
if (v___x_292_ == 0)
{
lean_dec(v_val_277_);
v___y_285_ = v_arr_265_;
goto v___jp_284_;
}
else
{
lean_object* v_v_293_; lean_object* v_unfiltered_294_; lean_object* v_key_295_; lean_object* v_pattern_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_307_; 
v_v_293_ = lean_array_fget(v_arr_265_, v_val_277_);
v_unfiltered_294_ = lean_ctor_get(v_v_293_, 1);
v_key_295_ = lean_ctor_get(v_v_293_, 2);
v_pattern_296_ = lean_ctor_get(v_v_293_, 3);
v_isSharedCheck_307_ = !lean_is_exclusive(v_v_293_);
if (v_isSharedCheck_307_ == 0)
{
lean_object* v_unused_308_; 
v_unused_308_ = lean_ctor_get(v_v_293_, 0);
lean_dec(v_unused_308_);
v___x_298_ = v_v_293_;
v_isShared_299_ = v_isSharedCheck_307_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_pattern_296_);
lean_inc(v_key_295_);
lean_inc(v_unfiltered_294_);
lean_dec(v_v_293_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_307_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___x_300_; lean_object* v_xs_x27_301_; lean_object* v___x_302_; lean_object* v___x_304_; 
v___x_300_ = lean_box(0);
v_xs_x27_301_ = lean_array_fset(v_arr_265_, v_val_277_, v___x_300_);
v___x_302_ = lean_box(0);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_302_);
v___x_304_ = v___x_298_;
goto v_reusejp_303_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v___x_302_);
lean_ctor_set(v_reuseFailAlloc_306_, 1, v_unfiltered_294_);
lean_ctor_set(v_reuseFailAlloc_306_, 2, v_key_295_);
lean_ctor_set(v_reuseFailAlloc_306_, 3, v_pattern_296_);
v___x_304_ = v_reuseFailAlloc_306_;
goto v_reusejp_303_;
}
v_reusejp_303_:
{
lean_object* v___x_305_; 
v___x_305_ = lean_array_fset(v_xs_x27_301_, v_val_277_, v___x_304_);
lean_dec(v_val_277_);
v___y_285_ = v___x_305_;
goto v___jp_284_;
}
}
}
v___jp_284_:
{
lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_289_; 
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9));
v___x_287_ = l_Array_binInsertM___redArg(v___x_286_, v___f_283_, v___f_281_, v___f_282_, v___y_285_, v_res_264_);
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 0, v___x_287_);
v___x_289_ = v___x_275_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v___x_287_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
else
{
lean_object* v_unfiltered_309_; lean_object* v_key_310_; lean_object* v_pattern_311_; lean_object* v___x_313_; uint8_t v_isShared_314_; uint8_t v_isSharedCheck_327_; 
lean_dec(v_val_277_);
v_unfiltered_309_ = lean_ctor_get(v_res_264_, 1);
v_key_310_ = lean_ctor_get(v_res_264_, 2);
v_pattern_311_ = lean_ctor_get(v_res_264_, 3);
v_isSharedCheck_327_ = !lean_is_exclusive(v_res_264_);
if (v_isSharedCheck_327_ == 0)
{
lean_object* v_unused_328_; 
v_unused_328_ = lean_ctor_get(v_res_264_, 0);
lean_dec(v_unused_328_);
v___x_313_ = v_res_264_;
v_isShared_314_ = v_isSharedCheck_327_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_pattern_311_);
lean_inc(v_key_310_);
lean_inc(v_unfiltered_309_);
lean_dec(v_res_264_);
v___x_313_ = lean_box(0);
v_isShared_314_ = v_isSharedCheck_327_;
goto v_resetjp_312_;
}
v_resetjp_312_:
{
lean_object* v___f_315_; lean_object* v___x_316_; lean_object* v___x_318_; 
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_315_, 0, v_inst_262_);
v___x_316_ = lean_box(0);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 0, v___x_316_);
v___x_318_ = v___x_313_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v___x_316_);
lean_ctor_set(v_reuseFailAlloc_326_, 1, v_unfiltered_309_);
lean_ctor_set(v_reuseFailAlloc_326_, 2, v_key_310_);
lean_ctor_set(v_reuseFailAlloc_326_, 3, v_pattern_311_);
v___x_318_ = v_reuseFailAlloc_326_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_324_; 
lean_inc_ref_n(v___x_318_, 2);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_319_, 0, v___x_318_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_320_, 0, v___x_318_);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9));
v___x_322_ = l_Array_binInsertM___redArg(v___x_321_, v___f_315_, v___f_319_, v___f_320_, v_arr_265_, v___x_318_);
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 0, v___x_322_);
v___x_324_ = v___x_275_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_322_);
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
}
else
{
lean_object* v___f_329_; lean_object* v___f_330_; lean_object* v___f_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_335_; 
lean_dec(v_a_273_);
lean_dec(v_inst_263_);
lean_inc_ref_n(v_res_264_, 2);
v___f_329_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_329_, 0, v_res_264_);
v___f_330_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_330_, 0, v_res_264_);
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_331_, 0, v_inst_262_);
v___x_332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___closed__9));
v___x_333_ = l_Array_binInsertM___redArg(v___x_332_, v___f_331_, v___f_329_, v___f_330_, v_arr_265_, v_res_264_);
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 0, v___x_333_);
v___x_335_ = v___x_275_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v___x_333_);
v___x_335_ = v_reuseFailAlloc_336_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
return v___x_335_;
}
}
}
}
else
{
lean_object* v_a_338_; lean_object* v___x_340_; uint8_t v_isShared_341_; uint8_t v_isSharedCheck_345_; 
lean_dec_ref(v_arr_265_);
lean_dec_ref(v_res_264_);
lean_dec(v_inst_263_);
lean_dec_ref(v_inst_262_);
v_a_338_ = lean_ctor_get(v___x_272_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_272_);
if (v_isSharedCheck_345_ == 0)
{
v___x_340_ = v___x_272_;
v_isShared_341_ = v_isSharedCheck_345_;
goto v_resetjp_339_;
}
else
{
lean_inc(v_a_338_);
lean_dec(v___x_272_);
v___x_340_ = lean_box(0);
v_isShared_341_ = v_isSharedCheck_345_;
goto v_resetjp_339_;
}
v_resetjp_339_:
{
lean_object* v___x_343_; 
if (v_isShared_341_ == 0)
{
v___x_343_ = v___x_340_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v_a_338_);
v___x_343_ = v_reuseFailAlloc_344_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
return v___x_343_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg___boxed(lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_res_348_, lean_object* v_arr_349_, lean_object* v_isDup_350_, lean_object* v_a_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg(v_inst_346_, v_inst_347_, v_res_348_, v_arr_349_, v_isDup_350_, v_a_351_, v_a_352_, v_a_353_, v_a_354_);
lean_dec(v_a_354_);
lean_dec_ref(v_a_353_);
lean_dec(v_a_352_);
lean_dec_ref(v_a_351_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray(lean_object* v_00_u03b1_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_res_360_, lean_object* v_arr_361_, lean_object* v_isDup_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg(v_inst_358_, v_inst_359_, v_res_360_, v_arr_361_, v_isDup_362_, v_a_363_, v_a_364_, v_a_365_, v_a_366_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___boxed(lean_object* v_00_u03b1_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_res_372_, lean_object* v_arr_373_, lean_object* v_isDup_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray(v_00_u03b1_369_, v_inst_370_, v_inst_371_, v_res_372_, v_arr_373_, v_isDup_374_, v_a_375_, v_a_376_, v_a_377_, v_a_378_);
lean_dec(v_a_378_);
lean_dec_ref(v_a_377_);
lean_dec(v_a_376_);
lean_dec_ref(v_a_375_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0(lean_object* v_a_384_, lean_object* v___x_385_, lean_object* v_____r_386_){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_713__overap_397_; lean_object* v___x_398_; 
v___x_388_ = l_Lean_Exception_toMessageData(v_a_384_);
v___x_389_ = l_Lean_MessageData_toString(v___x_388_);
v___x_390_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__0));
v___x_391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__1));
v___x_392_ = lean_unsigned_to_nat(87u);
v___x_393_ = lean_unsigned_to_nat(4u);
v___x_394_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___closed__2));
v___x_395_ = lean_string_append(v___x_394_, v___x_389_);
lean_dec_ref(v___x_389_);
v___x_396_ = l_mkPanicMessageWithDecl(v___x_390_, v___x_391_, v___x_392_, v___x_393_, v___x_395_);
lean_dec_ref(v___x_395_);
v___x_713__overap_397_ = l_panic___redArg(v___x_385_, v___x_396_);
v___x_398_ = lean_apply_1(v___x_713__overap_397_, lean_box(0));
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0___boxed(lean_object* v_a_399_, lean_object* v___x_400_, lean_object* v_____r_401_, lean_object* v___y_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0(v_a_399_, v___x_400_, v_____r_401_);
lean_dec_ref(v___x_400_);
return v_res_403_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0(void){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = l_Array_instInhabited(lean_box(0));
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_405_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__0);
v___x_406_ = l_instMonadBaseIO;
v___x_407_ = l_instInhabitedOfMonad___redArg(v___x_406_, v___x_405_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg(lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_s_412_, lean_object* v_res_413_, lean_object* v_isDup_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_){
_start:
{
lean_object* v_results_420_; lean_object* v_errors_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_445_; 
v_results_420_ = lean_ctor_get(v_s_412_, 0);
v_errors_421_ = lean_ctor_get(v_s_412_, 1);
v_isSharedCheck_445_ = !lean_is_exclusive(v_s_412_);
if (v_isSharedCheck_445_ == 0)
{
v___x_423_ = v_s_412_;
v_isShared_424_ = v_isSharedCheck_445_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_errors_421_);
lean_inc(v_results_420_);
lean_dec(v_s_412_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_445_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v_a_426_; lean_object* v___y_432_; lean_object* v___x_433_; 
v___x_433_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Result_insertInArray___redArg(v_inst_410_, v_inst_411_, v_res_413_, v_results_420_, v_isDup_414_, v_a_415_, v_a_416_, v_a_417_, v_a_418_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v_a_426_ = v_a_434_;
goto v___jp_425_;
}
else
{
lean_object* v_a_435_; lean_object* v___x_436_; 
v_a_435_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_435_);
lean_dec_ref_known(v___x_433_, 1);
v___x_436_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__1);
if (lean_obj_tag(v_a_435_) == 1)
{
lean_object* v_id_437_; lean_object* v___x_438_; uint8_t v___x_439_; 
v_id_437_ = lean_ctor_get(v_a_435_, 0);
v___x_438_ = l_Lean_interruptExceptionId;
v___x_439_ = l_Lean_instBEqInternalExceptionId_beq(v_id_437_, v___x_438_);
if (v___x_439_ == 0)
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = lean_box(0);
v___x_441_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0(v_a_435_, v___x_436_, v___x_440_);
v___y_432_ = v___x_441_;
goto v___jp_431_;
}
else
{
lean_object* v___x_442_; 
lean_dec_ref_known(v_a_435_, 2);
v___x_442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___closed__2));
v_a_426_ = v___x_442_;
goto v___jp_425_;
}
}
else
{
lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_443_ = lean_box(0);
v___x_444_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___lam__0(v_a_435_, v___x_436_, v___x_443_);
v___y_432_ = v___x_444_;
goto v___jp_431_;
}
}
v___jp_425_:
{
lean_object* v___x_428_; 
if (v_isShared_424_ == 0)
{
lean_ctor_set(v___x_423_, 0, v_a_426_);
v___x_428_ = v___x_423_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_426_);
lean_ctor_set(v_reuseFailAlloc_430_, 1, v_errors_421_);
v___x_428_ = v_reuseFailAlloc_430_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
lean_object* v___x_429_; 
v___x_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
return v___x_429_;
}
}
v___jp_431_:
{
v_a_426_ = v___y_432_;
goto v___jp_425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg___boxed(lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_s_448_, lean_object* v_res_449_, lean_object* v_isDup_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg(v_inst_446_, v_inst_447_, v_s_448_, v_res_449_, v_isDup_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
lean_dec(v_a_454_);
lean_dec_ref(v_a_453_);
lean_dec(v_a_452_);
lean_dec_ref(v_a_451_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult(lean_object* v_00_u03b1_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_s_460_, lean_object* v_res_461_, lean_object* v_isDup_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___redArg(v_inst_458_, v_inst_459_, v_s_460_, v_res_461_, v_isDup_462_, v_a_463_, v_a_464_, v_a_465_, v_a_466_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult___boxed(lean_object* v_00_u03b1_469_, lean_object* v_inst_470_, lean_object* v_inst_471_, lean_object* v_s_472_, lean_object* v_res_473_, lean_object* v_isDup_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState_insertResult(v_00_u03b1_469_, v_inst_470_, v_inst_471_, v_s_472_, v_res_473_, v_isDup_474_, v_a_475_, v_a_476_, v_a_477_, v_a_478_);
lean_dec(v_a_478_);
lean_dec_ref(v_a_477_);
lean_dec(v_a_476_);
lean_dec_ref(v_a_475_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorIdx(uint8_t v_x_481_){
_start:
{
switch(v_x_481_)
{
case 0:
{
lean_object* v___x_482_; 
v___x_482_ = lean_unsigned_to_nat(0u);
return v___x_482_;
}
case 1:
{
lean_object* v___x_483_; 
v___x_483_ = lean_unsigned_to_nat(1u);
return v___x_483_;
}
default: 
{
lean_object* v___x_484_; 
v___x_484_ = lean_unsigned_to_nat(2u);
return v___x_484_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorIdx___boxed(lean_object* v_x_485_){
_start:
{
uint8_t v_x_boxed_486_; lean_object* v_res_487_; 
v_x_boxed_486_ = lean_unbox(v_x_485_);
v_res_487_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorIdx(v_x_boxed_486_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___redArg(lean_object* v_k_488_){
_start:
{
lean_inc(v_k_488_);
return v_k_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___redArg___boxed(lean_object* v_k_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___redArg(v_k_489_);
lean_dec(v_k_489_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim(lean_object* v_motive_491_, lean_object* v_ctorIdx_492_, uint8_t v_t_493_, lean_object* v_h_494_, lean_object* v_k_495_){
_start:
{
lean_inc(v_k_495_);
return v_k_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim___boxed(lean_object* v_motive_496_, lean_object* v_ctorIdx_497_, lean_object* v_t_498_, lean_object* v_h_499_, lean_object* v_k_500_){
_start:
{
uint8_t v_t_boxed_501_; lean_object* v_res_502_; 
v_t_boxed_501_ = lean_unbox(v_t_498_);
v_res_502_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_ctorElim(v_motive_496_, v_ctorIdx_497_, v_t_boxed_501_, v_h_499_, v_k_500_);
lean_dec(v_k_500_);
lean_dec(v_ctorIdx_497_);
return v_res_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___redArg(lean_object* v_hyp_503_){
_start:
{
lean_inc(v_hyp_503_);
return v_hyp_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___redArg___boxed(lean_object* v_hyp_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___redArg(v_hyp_504_);
lean_dec(v_hyp_504_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim(lean_object* v_motive_506_, uint8_t v_t_507_, lean_object* v_h_508_, lean_object* v_hyp_509_){
_start:
{
lean_inc(v_hyp_509_);
return v_hyp_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim___boxed(lean_object* v_motive_510_, lean_object* v_t_511_, lean_object* v_h_512_, lean_object* v_hyp_513_){
_start:
{
uint8_t v_t_boxed_514_; lean_object* v_res_515_; 
v_t_boxed_514_ = lean_unbox(v_t_511_);
v_res_515_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_hyp_elim(v_motive_510_, v_t_boxed_514_, v_h_512_, v_hyp_513_);
lean_dec(v_hyp_513_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___redArg(lean_object* v_currFile_516_){
_start:
{
lean_inc(v_currFile_516_);
return v_currFile_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___redArg___boxed(lean_object* v_currFile_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___redArg(v_currFile_517_);
lean_dec(v_currFile_517_);
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim(lean_object* v_motive_519_, uint8_t v_t_520_, lean_object* v_h_521_, lean_object* v_currFile_522_){
_start:
{
lean_inc(v_currFile_522_);
return v_currFile_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim___boxed(lean_object* v_motive_523_, lean_object* v_t_524_, lean_object* v_h_525_, lean_object* v_currFile_526_){
_start:
{
uint8_t v_t_boxed_527_; lean_object* v_res_528_; 
v_t_boxed_527_ = lean_unbox(v_t_524_);
v_res_528_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_currFile_elim(v_motive_523_, v_t_boxed_527_, v_h_525_, v_currFile_526_);
lean_dec(v_currFile_526_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___redArg(lean_object* v_imported_529_){
_start:
{
lean_inc(v_imported_529_);
return v_imported_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___redArg___boxed(lean_object* v_imported_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___redArg(v_imported_530_);
lean_dec(v_imported_530_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim(lean_object* v_motive_532_, uint8_t v_t_533_, lean_object* v_h_534_, lean_object* v_imported_535_){
_start:
{
lean_inc(v_imported_535_);
return v_imported_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim___boxed(lean_object* v_motive_536_, lean_object* v_t_537_, lean_object* v_h_538_, lean_object* v_imported_539_){
_start:
{
uint8_t v_t_boxed_540_; lean_object* v_res_541_; 
v_t_boxed_540_ = lean_unbox(v_t_537_);
v_res_541_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_SectionKind_imported_elim(v_motive_536_, v_t_boxed_540_, v_h_538_, v_imported_539_);
lean_dec(v_imported_539_);
return v_res_541_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30(void){
_start:
{
lean_object* v___x_608_; lean_object* v___x_609_; 
v___x_608_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__29));
v___x_609_ = l_Lean_Json_mkObj(v___x_608_);
return v___x_609_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31(void){
_start:
{
lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_610_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__30);
v___x_611_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__24));
v___x_612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_612_, 0, v___x_611_);
lean_ctor_set(v___x_612_, 1, v___x_610_);
return v___x_612_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32(void){
_start:
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v___x_613_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__31);
v___x_614_ = lean_unsigned_to_nat(1u);
v___x_615_ = lean_mk_empty_array_with_capacity(v___x_614_);
v___x_616_ = lean_array_push(v___x_615_, v___x_613_);
return v___x_616_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33(void){
_start:
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_617_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__22));
v___x_618_ = lean_unsigned_to_nat(2u);
v___x_619_ = lean_mk_empty_array_with_capacity(v___x_618_);
v___x_620_ = lean_array_push(v___x_619_, v___x_617_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors(lean_object* v_errors_621_){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_622_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__0));
v___x_623_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__4));
v___x_624_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__23));
v___x_625_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__32);
v___x_626_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_626_, 0, v___x_624_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
lean_ctor_set(v___x_626_, 2, v_errors_621_);
v___x_627_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33, &lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33_once, _init_lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__33);
v___x_628_ = lean_array_push(v___x_627_, v___x_626_);
v___x_629_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_629_, 0, v___x_622_);
lean_ctor_set(v___x_629_, 1, v___x_623_);
lean_ctor_set(v___x_629_, 2, v___x_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0(lean_object* v_c_630_, lean_object* v_props_631_, lean_object* v_children_632_){
_start:
{
lean_object* v_toModule_633_; lean_object* v_export_634_; lean_object* v_javascript_635_; uint64_t v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; 
v_toModule_633_ = lean_ctor_get(v_c_630_, 0);
v_export_634_ = lean_ctor_get(v_c_630_, 1);
v_javascript_635_ = lean_ctor_get(v_toModule_633_, 0);
v___x_636_ = lean_string_hash(v_javascript_635_);
v___x_637_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_instRpcEncodableFilterDetailsProps_enc_00___x40_ProofWidgets_Component_FilterDetails_2598060976____hygCtx___hyg_1_), 2, 1);
lean_closure_set(v___x_637_, 0, v_props_631_);
lean_inc_ref(v_export_634_);
v___x_638_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_638_, 0, v_export_634_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
lean_ctor_set(v___x_638_, 2, v_children_632_);
lean_ctor_set_uint64(v___x_638_, sizeof(void*)*3, v___x_636_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0___boxed(lean_object* v_c_639_, lean_object* v_props_640_, lean_object* v_children_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0(v_c_639_, v_props_640_, v_children_641_);
lean_dec_ref(v_c_639_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(lean_object* v_as_643_, size_t v_i_644_, size_t v_stop_645_, lean_object* v_b_646_){
_start:
{
lean_object* v___y_648_; uint8_t v___x_652_; 
v___x_652_ = lean_usize_dec_eq(v_i_644_, v_stop_645_);
if (v___x_652_ == 0)
{
lean_object* v___x_653_; lean_object* v_filtered_654_; 
v___x_653_ = lean_array_uget_borrowed(v_as_643_, v_i_644_);
v_filtered_654_ = lean_ctor_get(v___x_653_, 0);
if (lean_obj_tag(v_filtered_654_) == 0)
{
v___y_648_ = v_b_646_;
goto v___jp_647_;
}
else
{
lean_object* v_val_655_; lean_object* v___x_656_; 
v_val_655_ = lean_ctor_get(v_filtered_654_, 0);
lean_inc(v_val_655_);
v___x_656_ = lean_array_push(v_b_646_, v_val_655_);
v___y_648_ = v___x_656_;
goto v___jp_647_;
}
}
else
{
return v_b_646_;
}
v___jp_647_:
{
size_t v___x_649_; size_t v___x_650_; 
v___x_649_ = ((size_t)1ULL);
v___x_650_ = lean_usize_add(v_i_644_, v___x_649_);
v_i_644_ = v___x_650_;
v_b_646_ = v___y_648_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg___boxed(lean_object* v_as_657_, lean_object* v_i_658_, lean_object* v_stop_659_, lean_object* v_b_660_){
_start:
{
size_t v_i_boxed_661_; size_t v_stop_boxed_662_; lean_object* v_res_663_; 
v_i_boxed_661_ = lean_unbox_usize(v_i_658_);
lean_dec(v_i_658_);
v_stop_boxed_662_ = lean_unbox_usize(v_stop_659_);
lean_dec(v_stop_659_);
v_res_663_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(v_as_657_, v_i_boxed_661_, v_stop_boxed_662_, v_b_660_);
lean_dec_ref(v_as_657_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg(lean_object* v_as_666_, lean_object* v_start_667_, lean_object* v_stop_668_){
_start:
{
lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___closed__0));
v___x_670_ = lean_nat_dec_lt(v_start_667_, v_stop_668_);
if (v___x_670_ == 0)
{
return v___x_669_;
}
else
{
lean_object* v___x_671_; uint8_t v___x_672_; 
v___x_671_ = lean_array_get_size(v_as_666_);
v___x_672_ = lean_nat_dec_le(v_stop_668_, v___x_671_);
if (v___x_672_ == 0)
{
uint8_t v___x_673_; 
v___x_673_ = lean_nat_dec_lt(v_start_667_, v___x_671_);
if (v___x_673_ == 0)
{
return v___x_669_;
}
else
{
size_t v___x_674_; size_t v___x_675_; lean_object* v___x_676_; 
v___x_674_ = lean_usize_of_nat(v_start_667_);
v___x_675_ = lean_usize_of_nat(v___x_671_);
v___x_676_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(v_as_666_, v___x_674_, v___x_675_, v___x_669_);
return v___x_676_;
}
}
else
{
size_t v___x_677_; size_t v___x_678_; lean_object* v___x_679_; 
v___x_677_ = lean_usize_of_nat(v_start_667_);
v___x_678_ = lean_usize_of_nat(v_stop_668_);
v___x_679_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(v_as_666_, v___x_677_, v___x_678_, v___x_669_);
return v___x_679_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg___boxed(lean_object* v_as_680_, lean_object* v_start_681_, lean_object* v_stop_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg(v_as_680_, v_start_681_, v_stop_682_);
lean_dec(v_stop_682_);
lean_dec(v_start_681_);
lean_dec_ref(v_as_680_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg(size_t v_sz_684_, size_t v_i_685_, lean_object* v_bs_686_){
_start:
{
uint8_t v___x_687_; 
v___x_687_ = lean_usize_dec_lt(v_i_685_, v_sz_684_);
if (v___x_687_ == 0)
{
return v_bs_686_;
}
else
{
lean_object* v_v_688_; lean_object* v_unfiltered_689_; lean_object* v___x_690_; lean_object* v_bs_x27_691_; size_t v___x_692_; size_t v___x_693_; lean_object* v___x_694_; 
v_v_688_ = lean_array_uget_borrowed(v_bs_686_, v_i_685_);
v_unfiltered_689_ = lean_ctor_get(v_v_688_, 1);
lean_inc_ref(v_unfiltered_689_);
v___x_690_ = lean_unsigned_to_nat(0u);
v_bs_x27_691_ = lean_array_uset(v_bs_686_, v_i_685_, v___x_690_);
v___x_692_ = ((size_t)1ULL);
v___x_693_ = lean_usize_add(v_i_685_, v___x_692_);
v___x_694_ = lean_array_uset(v_bs_x27_691_, v_i_685_, v_unfiltered_689_);
v_i_685_ = v___x_693_;
v_bs_686_ = v___x_694_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg___boxed(lean_object* v_sz_696_, lean_object* v_i_697_, lean_object* v_bs_698_){
_start:
{
size_t v_sz_boxed_699_; size_t v_i_boxed_700_; lean_object* v_res_701_; 
v_sz_boxed_699_ = lean_unbox_usize(v_sz_696_);
lean_dec(v_sz_696_);
v_i_boxed_700_ = lean_unbox_usize(v_i_697_);
lean_dec(v_i_697_);
v_res_701_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg(v_sz_boxed_699_, v_i_boxed_700_, v_bs_698_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg(lean_object* v_tactic_712_, uint8_t v_kind_713_, lean_object* v_s_714_){
_start:
{
lean_object* v___y_716_; lean_object* v___y_717_; lean_object* v___y_718_; lean_object* v___y_719_; uint8_t v___y_720_; lean_object* v___y_721_; lean_object* v___y_722_; lean_object* v___y_756_; lean_object* v___y_757_; uint8_t v___y_758_; lean_object* v___y_759_; lean_object* v_all_760_; lean_object* v_filtered_761_; lean_object* v_results_765_; lean_object* v_errors_766_; lean_object* v___y_768_; uint8_t v___y_769_; lean_object* v___y_770_; uint8_t v___y_792_; lean_object* v___x_801_; lean_object* v___x_802_; uint8_t v___x_803_; 
v_results_765_ = lean_ctor_get(v_s_714_, 0);
lean_inc_ref(v_results_765_);
v_errors_766_ = lean_ctor_get(v_s_714_, 1);
lean_inc_ref(v_errors_766_);
lean_dec_ref(v_s_714_);
v___x_801_ = lean_array_get_size(v_results_765_);
v___x_802_ = lean_unsigned_to_nat(0u);
v___x_803_ = lean_nat_dec_eq(v___x_801_, v___x_802_);
if (v___x_803_ == 0)
{
v___y_792_ = v___x_803_;
goto v___jp_791_;
}
else
{
lean_object* v___x_804_; uint8_t v___x_805_; 
v___x_804_ = lean_array_get_size(v_errors_766_);
v___x_805_ = lean_nat_dec_eq(v___x_804_, v___x_802_);
v___y_792_ = v___x_805_;
goto v___jp_791_;
}
v___jp_715_:
{
lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v_header_735_; 
v___x_723_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__11));
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__0));
v___x_725_ = lean_string_append(v_tactic_712_, v___x_724_);
v___x_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_726_, 0, v___x_725_);
v___x_727_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__2));
lean_inc_ref(v___y_722_);
v___x_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_728_, 0, v___y_722_);
v___x_729_ = lean_unsigned_to_nat(4u);
v___x_730_ = lean_mk_empty_array_with_capacity(v___x_729_);
v___x_731_ = lean_array_push(v___x_730_, v___x_726_);
v___x_732_ = lean_array_push(v___x_731_, v___y_717_);
v___x_733_ = lean_array_push(v___x_732_, v___x_727_);
v___x_734_ = lean_array_push(v___x_733_, v___x_728_);
lean_inc_ref(v___y_721_);
v_header_735_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_header_735_, 0, v___x_723_);
lean_ctor_set(v_header_735_, 1, v___y_721_);
lean_ctor_set(v_header_735_, 2, v___x_734_);
if (v_kind_713_ == 2)
{
lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
lean_dec_ref(v___y_721_);
v___x_736_ = lp_proofwidgets_ProofWidgets_FilterDetails;
v___x_737_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_737_, 0, v_header_735_);
lean_ctor_set(v___x_737_, 1, v___y_719_);
lean_ctor_set(v___x_737_, 2, v___y_716_);
lean_ctor_set_uint8(v___x_737_, sizeof(void*)*3, v___y_720_);
v___x_738_ = lean_mk_empty_array_with_capacity(v___y_718_);
v___x_739_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__0(v___x_736_, v___x_737_, v___x_738_);
return v___x_739_;
}
else
{
lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; 
lean_dec_ref(v___y_719_);
v___x_740_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__0));
v___x_741_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__1));
v___x_742_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_742_, 0, v___y_720_);
v___x_743_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_741_);
lean_ctor_set(v___x_743_, 1, v___x_742_);
v___x_744_ = lean_unsigned_to_nat(1u);
v___x_745_ = lean_mk_empty_array_with_capacity(v___x_744_);
lean_inc_ref(v___x_745_);
v___x_746_ = lean_array_push(v___x_745_, v___x_743_);
v___x_747_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors___closed__5));
v___x_748_ = lean_array_push(v___x_745_, v_header_735_);
v___x_749_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_749_, 0, v___x_747_);
lean_ctor_set(v___x_749_, 1, v___y_721_);
lean_ctor_set(v___x_749_, 2, v___x_748_);
v___x_750_ = lean_unsigned_to_nat(2u);
v___x_751_ = lean_mk_empty_array_with_capacity(v___x_750_);
v___x_752_ = lean_array_push(v___x_751_, v___x_749_);
v___x_753_ = lean_array_push(v___x_752_, v___y_716_);
v___x_754_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_754_, 0, v___x_740_);
lean_ctor_set(v___x_754_, 1, v___x_746_);
lean_ctor_set(v___x_754_, 2, v___x_753_);
return v___x_754_;
}
}
v___jp_755_:
{
switch(v_kind_713_)
{
case 0:
{
lean_object* v___x_762_; 
v___x_762_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__3));
v___y_716_ = v_all_760_;
v___y_717_ = v___y_756_;
v___y_718_ = v___y_757_;
v___y_719_ = v_filtered_761_;
v___y_720_ = v___y_758_;
v___y_721_ = v___y_759_;
v___y_722_ = v___x_762_;
goto v___jp_715_;
}
case 1:
{
lean_object* v___x_763_; 
v___x_763_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__4));
v___y_716_ = v_all_760_;
v___y_717_ = v___y_756_;
v___y_718_ = v___y_757_;
v___y_719_ = v_filtered_761_;
v___y_720_ = v___y_758_;
v___y_721_ = v___y_759_;
v___y_722_ = v___x_763_;
goto v___jp_715_;
}
default: 
{
lean_object* v___x_764_; 
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__5));
v___y_716_ = v_all_760_;
v___y_717_ = v___y_756_;
v___y_718_ = v___y_757_;
v___y_719_ = v_filtered_761_;
v___y_720_ = v___y_758_;
v___y_721_ = v___y_759_;
v___y_722_ = v___x_764_;
goto v___jp_715_;
}
}
}
v___jp_767_:
{
lean_object* v___x_771_; lean_object* v___x_772_; size_t v_sz_773_; size_t v___x_774_; lean_object* v___x_775_; lean_object* v_all_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v_filtered_779_; lean_object* v___x_780_; uint8_t v___x_781_; 
v___x_771_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__6));
v___x_772_ = lean_mk_empty_array_with_capacity(v___y_768_);
v_sz_773_ = lean_array_size(v_results_765_);
v___x_774_ = ((size_t)0ULL);
lean_inc_ref(v_results_765_);
v___x_775_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg(v_sz_773_, v___x_774_, v_results_765_);
lean_inc_ref_n(v___x_772_, 2);
v_all_776_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_all_776_, 0, v___x_771_);
lean_ctor_set(v_all_776_, 1, v___x_772_);
lean_ctor_set(v_all_776_, 2, v___x_775_);
v___x_777_ = lean_array_get_size(v_results_765_);
v___x_778_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg(v_results_765_, v___y_768_, v___x_777_);
lean_dec_ref(v_results_765_);
v_filtered_779_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_filtered_779_, 0, v___x_771_);
lean_ctor_set(v_filtered_779_, 1, v___x_772_);
lean_ctor_set(v_filtered_779_, 2, v___x_778_);
v___x_780_ = lean_array_get_size(v_errors_766_);
v___x_781_ = lean_nat_dec_eq(v___x_780_, v___y_768_);
if (v___x_781_ == 0)
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v_all_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v_filtered_790_; 
v___x_782_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_SectionState_0__Mathlib_Tactic_ClickSuggestions_renderSection_renderErrors(v_errors_766_);
v___x_783_ = lean_unsigned_to_nat(2u);
v___x_784_ = lean_mk_empty_array_with_capacity(v___x_783_);
lean_inc_ref(v___x_784_);
v___x_785_ = lean_array_push(v___x_784_, v_all_776_);
lean_inc_ref(v___x_782_);
v___x_786_ = lean_array_push(v___x_785_, v___x_782_);
lean_inc_ref_n(v___x_772_, 2);
v_all_787_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_all_787_, 0, v___x_771_);
lean_ctor_set(v_all_787_, 1, v___x_772_);
lean_ctor_set(v_all_787_, 2, v___x_786_);
v___x_788_ = lean_array_push(v___x_784_, v_filtered_779_);
v___x_789_ = lean_array_push(v___x_788_, v___x_782_);
v_filtered_790_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_filtered_790_, 0, v___x_771_);
lean_ctor_set(v_filtered_790_, 1, v___x_772_);
lean_ctor_set(v_filtered_790_, 2, v___x_789_);
v___y_756_ = v___y_770_;
v___y_757_ = v___y_768_;
v___y_758_ = v___y_769_;
v___y_759_ = v___x_772_;
v_all_760_ = v_all_787_;
v_filtered_761_ = v_filtered_790_;
goto v___jp_755_;
}
else
{
lean_dec_ref(v_errors_766_);
v___y_756_ = v___y_770_;
v___y_757_ = v___y_768_;
v___y_758_ = v___y_769_;
v___y_759_ = v___x_772_;
v_all_760_ = v_all_776_;
v_filtered_761_ = v_filtered_779_;
goto v___jp_755_;
}
}
v___jp_791_:
{
if (v___y_792_ == 0)
{
uint8_t v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; uint8_t v___x_796_; 
v___x_793_ = 1;
v___x_794_ = lean_unsigned_to_nat(0u);
v___x_795_ = lean_array_get_size(v_results_765_);
v___x_796_ = lean_nat_dec_lt(v___x_794_, v___x_795_);
if (v___x_796_ == 0)
{
lean_object* v___x_797_; 
v___x_797_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__7));
v___y_768_ = v___x_794_;
v___y_769_ = v___x_793_;
v___y_770_ = v___x_797_;
goto v___jp_767_;
}
else
{
lean_object* v___x_798_; lean_object* v_pattern_799_; 
v___x_798_ = lean_array_fget_borrowed(v_results_765_, v___x_794_);
v_pattern_799_ = lean_ctor_get(v___x_798_, 3);
lean_inc_ref(v_pattern_799_);
v___y_768_ = v___x_794_;
v___y_769_ = v___x_793_;
v___y_770_ = v_pattern_799_;
goto v___jp_767_;
}
}
else
{
lean_object* v___x_800_; 
lean_dec_ref(v_errors_766_);
lean_dec_ref(v_results_765_);
lean_dec_ref(v_tactic_712_);
v___x_800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___closed__7));
return v___x_800_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg___boxed(lean_object* v_tactic_806_, lean_object* v_kind_807_, lean_object* v_s_808_){
_start:
{
uint8_t v_kind_boxed_809_; lean_object* v_res_810_; 
v_kind_boxed_809_ = lean_unbox(v_kind_807_);
v_res_810_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg(v_tactic_806_, v_kind_boxed_809_, v_s_808_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection(lean_object* v_00_u03b1_811_, lean_object* v_tactic_812_, uint8_t v_kind_813_, lean_object* v_s_814_){
_start:
{
lean_object* v___x_815_; 
v___x_815_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___redArg(v_tactic_812_, v_kind_813_, v_s_814_);
return v___x_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection___boxed(lean_object* v_00_u03b1_816_, lean_object* v_tactic_817_, lean_object* v_kind_818_, lean_object* v_s_819_){
_start:
{
uint8_t v_kind_boxed_820_; lean_object* v_res_821_; 
v_kind_boxed_820_ = lean_unbox(v_kind_818_);
v_res_821_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_renderSection(v_00_u03b1_816_, v_tactic_817_, v_kind_boxed_820_, v_s_819_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1(lean_object* v_00_u03b1_822_, size_t v_sz_823_, size_t v_i_824_, lean_object* v_bs_825_){
_start:
{
lean_object* v___x_826_; 
v___x_826_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___redArg(v_sz_823_, v_i_824_, v_bs_825_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1___boxed(lean_object* v_00_u03b1_827_, lean_object* v_sz_828_, lean_object* v_i_829_, lean_object* v_bs_830_){
_start:
{
size_t v_sz_boxed_831_; size_t v_i_boxed_832_; lean_object* v_res_833_; 
v_sz_boxed_831_ = lean_unbox_usize(v_sz_828_);
lean_dec(v_sz_828_);
v_i_boxed_832_ = lean_unbox_usize(v_i_829_);
lean_dec(v_i_829_);
v_res_833_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__1(v_00_u03b1_827_, v_sz_boxed_831_, v_i_boxed_832_, v_bs_830_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2(lean_object* v_00_u03b1_834_, lean_object* v_as_835_, lean_object* v_start_836_, lean_object* v_stop_837_){
_start:
{
lean_object* v___x_838_; 
v___x_838_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___redArg(v_as_835_, v_start_836_, v_stop_837_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2___boxed(lean_object* v_00_u03b1_839_, lean_object* v_as_840_, lean_object* v_start_841_, lean_object* v_stop_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2(v_00_u03b1_839_, v_as_840_, v_start_841_, v_stop_842_);
lean_dec(v_stop_842_);
lean_dec(v_start_841_);
lean_dec_ref(v_as_840_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2(lean_object* v_00_u03b1_844_, lean_object* v_as_845_, size_t v_i_846_, size_t v_stop_847_, lean_object* v_b_848_){
_start:
{
lean_object* v___x_849_; 
v___x_849_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___redArg(v_as_845_, v_i_846_, v_stop_847_, v_b_848_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2___boxed(lean_object* v_00_u03b1_850_, lean_object* v_as_851_, lean_object* v_i_852_, lean_object* v_stop_853_, lean_object* v_b_854_){
_start:
{
size_t v_i_boxed_855_; size_t v_stop_boxed_856_; lean_object* v_res_857_; 
v_i_boxed_855_ = lean_unbox_usize(v_i_852_);
lean_dec(v_i_852_);
v_stop_boxed_856_ = lean_unbox_usize(v_stop_853_);
lean_dec(v_stop_853_);
v_res_857_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_ClickSuggestions_renderSection_spec__2_spec__2(v_00_u03b1_850_, v_as_851_, v_i_boxed_855_, v_stop_boxed_856_, v_b_854_);
lean_dec_ref(v_as_851_);
return v_res_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0(lean_object* v_____x_858_){
_start:
{
lean_object* v_fst_860_; lean_object* v___x_861_; 
v_fst_860_ = lean_ctor_get(v_____x_858_, 0);
lean_inc(v_fst_860_);
v___x_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_861_, 0, v_fst_860_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0___boxed(lean_object* v_____x_862_, lean_object* v___y_863_){
_start:
{
lean_object* v_res_864_; 
v_res_864_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__0(v_____x_862_);
lean_dec_ref(v_____x_862_);
return v_res_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1(lean_object* v_e_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_){
_start:
{
if (lean_obj_tag(v_e_865_) == 0)
{
lean_object* v_a_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_880_; 
v_a_873_ = lean_ctor_get(v_e_865_, 0);
v_isSharedCheck_880_ = !lean_is_exclusive(v_e_865_);
if (v_isSharedCheck_880_ == 0)
{
v___x_875_ = v_e_865_;
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_a_873_);
lean_dec(v_e_865_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
lean_object* v___x_878_; 
if (v_isShared_876_ == 0)
{
v___x_878_ = v___x_875_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_879_, 0, v_a_873_);
v___x_878_ = v_reuseFailAlloc_879_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
return v___x_878_;
}
}
}
else
{
lean_object* v_a_881_; lean_object* v___x_883_; uint8_t v_isShared_884_; uint8_t v_isSharedCheck_888_; 
v_a_881_ = lean_ctor_get(v_e_865_, 0);
v_isSharedCheck_888_ = !lean_is_exclusive(v_e_865_);
if (v_isSharedCheck_888_ == 0)
{
v___x_883_ = v_e_865_;
v_isShared_884_ = v_isSharedCheck_888_;
goto v_resetjp_882_;
}
else
{
lean_inc(v_a_881_);
lean_dec(v_e_865_);
v___x_883_ = lean_box(0);
v_isShared_884_ = v_isSharedCheck_888_;
goto v_resetjp_882_;
}
v_resetjp_882_:
{
lean_object* v___x_886_; 
if (v_isShared_884_ == 0)
{
lean_ctor_set_tag(v___x_883_, 0);
v___x_886_ = v___x_883_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v_a_881_);
v___x_886_ = v_reuseFailAlloc_887_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
return v___x_886_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1___boxed(lean_object* v_e_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_){
_start:
{
lean_object* v_res_897_; 
v_res_897_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__1(v_e_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
lean_dec(v___y_895_);
lean_dec_ref(v___y_894_);
lean_dec(v___y_893_);
lean_dec_ref(v___y_892_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
return v_res_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2(lean_object* v_k_902_, lean_object* v___x_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v___x_911_; 
lean_inc(v___y_909_);
lean_inc_ref(v___y_908_);
lean_inc(v___y_907_);
lean_inc_ref(v___y_906_);
lean_inc(v___y_905_);
v___x_911_ = lean_apply_7(v_k_902_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_, lean_box(0));
if (lean_obj_tag(v___x_911_) == 0)
{
lean_object* v_a_912_; lean_object* v___x_914_; uint8_t v_isShared_915_; uint8_t v_isSharedCheck_922_; 
lean_dec_ref(v___x_903_);
v_a_912_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_922_ == 0)
{
v___x_914_ = v___x_911_;
v_isShared_915_ = v_isSharedCheck_922_;
goto v_resetjp_913_;
}
else
{
lean_inc(v_a_912_);
lean_dec(v___x_911_);
v___x_914_ = lean_box(0);
v_isShared_915_ = v_isSharedCheck_922_;
goto v_resetjp_913_;
}
v_resetjp_913_:
{
lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_920_; 
v___x_916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_916_, 0, v_a_912_);
v___x_917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
v___x_918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_918_, 0, v___x_917_);
if (v_isShared_915_ == 0)
{
lean_ctor_set(v___x_914_, 0, v___x_918_);
v___x_920_ = v___x_914_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v___x_918_);
v___x_920_ = v_reuseFailAlloc_921_;
goto v_reusejp_919_;
}
v_reusejp_919_:
{
return v___x_920_;
}
}
}
else
{
lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_940_; 
v_a_923_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_940_ == 0)
{
v___x_925_ = v___x_911_;
v_isShared_926_ = v_isSharedCheck_940_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_911_);
v___x_925_ = lean_box(0);
v_isShared_926_ = v_isSharedCheck_940_;
goto v_resetjp_924_;
}
v_resetjp_924_:
{
lean_object* v___x_928_; 
lean_inc(v_a_923_);
if (v_isShared_926_ == 0)
{
v___x_928_ = v___x_925_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v_a_923_);
v___x_928_ = v_reuseFailAlloc_939_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
uint8_t v___y_930_; uint8_t v___x_937_; 
v___x_937_ = l_Lean_Exception_isInterrupt(v_a_923_);
if (v___x_937_ == 0)
{
uint8_t v___x_938_; 
v___x_938_ = l_Lean_Exception_isRuntime(v_a_923_);
v___y_930_ = v___x_938_;
goto v___jp_929_;
}
else
{
lean_dec(v_a_923_);
v___y_930_ = v___x_937_;
goto v___jp_929_;
}
v___jp_929_:
{
if (v___y_930_ == 0)
{
lean_object* v_options_931_; lean_object* v___x_932_; lean_object* v___x_933_; uint8_t v___x_934_; 
v_options_931_ = lean_ctor_get(v___y_908_, 2);
v___x_932_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_click__suggestions_debug;
v___x_933_ = l_Lean_Option_get___redArg(v___x_903_, v_options_931_, v___x_932_);
v___x_934_ = lean_unbox(v___x_933_);
lean_dec(v___x_933_);
if (v___x_934_ == 0)
{
lean_object* v___x_935_; lean_object* v___x_936_; 
lean_dec_ref(v___x_928_);
v___x_935_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___closed__1));
v___x_936_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_936_, 0, v___x_935_);
return v___x_936_;
}
else
{
return v___x_928_;
}
}
else
{
lean_dec_ref(v___x_903_);
return v___x_928_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___boxed(lean_object* v_k_941_, lean_object* v___x_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_){
_start:
{
lean_object* v_res_950_; 
v_res_950_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2(v_k_941_, v___x_942_, v___y_943_, v___y_944_, v___y_945_, v___y_946_, v___y_947_, v___y_948_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
lean_dec(v___y_946_);
lean_dec_ref(v___y_945_);
lean_dec(v___y_944_);
return v_res_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3(lean_object* v___x_961_, lean_object* v_a_962_, lean_object* v_ex_963_){
_start:
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_965_ = l_Lean_Exception_toMessageData(v_ex_963_);
v___x_966_ = l_Lean_Server_WithRpcRef_mk___redArg(v___x_965_);
v___x_967_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__0));
v___x_968_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__1));
v___x_969_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__3));
v___x_970_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___closed__5));
v___x_971_ = lp_proofwidgets_ProofWidgets_InteractiveMessage;
v___x_972_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(v___x_961_, v___x_971_, v___x_966_, v___x_968_);
v___x_973_ = lean_unsigned_to_nat(4u);
v___x_974_ = lean_mk_empty_array_with_capacity(v___x_973_);
v___x_975_ = lean_array_push(v___x_974_, v_a_962_);
v___x_976_ = lean_array_push(v___x_975_, v___x_969_);
v___x_977_ = lean_array_push(v___x_976_, v___x_970_);
v___x_978_ = lean_array_push(v___x_977_, v___x_972_);
v___x_979_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_979_, 0, v___x_967_);
lean_ctor_set(v___x_979_, 1, v___x_968_);
lean_ctor_set(v___x_979_, 2, v___x_978_);
v___x_980_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_980_, 0, v___x_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___boxed(lean_object* v___x_981_, lean_object* v_a_982_, lean_object* v_ex_983_, lean_object* v___y_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3(v___x_981_, v_a_982_, v_ex_983_);
return v_res_985_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0(void){
_start:
{
lean_object* v___x_986_; 
v___x_986_ = l_instMonadEIO(lean_box(0));
return v___x_986_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1(void){
_start:
{
lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_987_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__0);
v___x_988_ = l_StateRefT_x27_instMonad___redArg(v___x_987_);
return v___x_988_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4(void){
_start:
{
lean_object* v___x_991_; 
v___x_991_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_991_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5(void){
_start:
{
lean_object* v___x_992_; lean_object* v___f_993_; 
v___x_992_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4);
v___f_993_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_993_, 0, v___x_992_);
return v___f_993_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6(void){
_start:
{
lean_object* v___x_994_; lean_object* v___f_995_; 
v___x_994_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__4);
v___f_995_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_995_, 0, v___x_994_);
return v___f_995_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7(void){
_start:
{
lean_object* v___f_996_; lean_object* v___f_997_; lean_object* v___x_998_; 
v___f_996_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__6);
v___f_997_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__5);
v___x_998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_998_, 0, v___f_997_);
lean_ctor_set(v___x_998_, 1, v___f_996_);
return v___x_998_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8(void){
_start:
{
lean_object* v___x_999_; lean_object* v___f_1000_; 
v___x_999_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7);
v___f_1000_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1000_, 0, v___x_999_);
return v___f_1000_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9(void){
_start:
{
lean_object* v___x_1001_; lean_object* v___f_1002_; 
v___x_1001_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__7);
v___f_1002_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_1002_, 0, v___x_1001_);
return v___f_1002_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10(void){
_start:
{
lean_object* v___f_1003_; lean_object* v___f_1004_; lean_object* v___x_1005_; 
v___f_1003_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__9);
v___f_1004_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__8);
v___x_1005_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1005_, 0, v___f_1004_);
lean_ctor_set(v___x_1005_, 1, v___f_1003_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4(lean_object* v_val_1006_, lean_object* v_a_1007_, lean_object* v___x_1008_, lean_object* v___f_1009_, lean_object* v___f_1010_, lean_object* v___x_1011_, lean_object* v___x_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_ref_1016_){
_start:
{
lean_object* v___x_1018_; lean_object* v___x_1042_; lean_object* v_toApplicative_1043_; lean_object* v_toFunctor_1044_; lean_object* v_toSeq_1045_; lean_object* v_toSeqLeft_1046_; lean_object* v_toSeqRight_1047_; lean_object* v___f_1048_; lean_object* v___f_1049_; lean_object* v___f_1050_; lean_object* v___f_1051_; lean_object* v___x_1052_; lean_object* v___f_1053_; lean_object* v___f_1054_; lean_object* v___f_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v_cancelTk_x3f_1058_; 
v___x_1018_ = lean_st_mk_ref(v_val_1006_);
v___x_1042_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1);
v_toApplicative_1043_ = lean_ctor_get(v___x_1042_, 0);
v_toFunctor_1044_ = lean_ctor_get(v_toApplicative_1043_, 0);
v_toSeq_1045_ = lean_ctor_get(v_toApplicative_1043_, 2);
v_toSeqLeft_1046_ = lean_ctor_get(v_toApplicative_1043_, 3);
v_toSeqRight_1047_ = lean_ctor_get(v_toApplicative_1043_, 4);
v___f_1048_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__2));
v___f_1049_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__3));
lean_inc_ref_n(v_toFunctor_1044_, 2);
v___f_1050_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1050_, 0, v_toFunctor_1044_);
v___f_1051_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1051_, 0, v_toFunctor_1044_);
v___x_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1052_, 0, v___f_1050_);
lean_ctor_set(v___x_1052_, 1, v___f_1051_);
lean_inc(v_toSeqRight_1047_);
v___f_1053_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1053_, 0, v_toSeqRight_1047_);
lean_inc(v_toSeqLeft_1046_);
v___f_1054_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1054_, 0, v_toSeqLeft_1046_);
lean_inc(v_toSeq_1045_);
v___f_1055_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1055_, 0, v_toSeq_1045_);
v___x_1056_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1052_);
lean_ctor_set(v___x_1056_, 1, v___f_1048_);
lean_ctor_set(v___x_1056_, 2, v___f_1055_);
lean_ctor_set(v___x_1056_, 3, v___f_1054_);
lean_ctor_set(v___x_1056_, 4, v___f_1053_);
v___x_1057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1056_);
lean_ctor_set(v___x_1057_, 1, v___f_1049_);
v_cancelTk_x3f_1058_ = lean_ctor_get(v_a_1007_, 12);
if (lean_obj_tag(v_cancelTk_x3f_1058_) == 1)
{
lean_object* v_val_1059_; uint8_t v___x_1060_; 
v_val_1059_ = lean_ctor_get(v_cancelTk_x3f_1058_, 0);
v___x_1060_ = l_IO_CancelToken_isSet(v_val_1059_);
if (v___x_1060_ == 0)
{
lean_dec_ref_known(v___x_1057_, 2);
goto v___jp_1019_;
}
else
{
lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_9723__overap_1066_; lean_object* v___x_1067_; 
v___x_1061_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__10);
v___x_1062_ = l_Lean_Core_instMonadRefCoreM;
v___x_1063_ = l_Lean_Core_instAddMessageContextCoreM;
v___x_1064_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_1063_, v___x_1057_);
v___x_1065_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1061_);
lean_ctor_set(v___x_1065_, 1, v___x_1062_);
lean_ctor_set(v___x_1065_, 2, v___x_1064_);
v___x_9723__overap_1066_ = l_Lean_throwInterruptException___redArg(v___x_1065_);
lean_inc(v_ref_1016_);
lean_inc_ref(v_a_1007_);
v___x_1067_ = lean_apply_3(v___x_9723__overap_1066_, v_a_1007_, v_ref_1016_, lean_box(0));
if (lean_obj_tag(v___x_1067_) == 0)
{
lean_dec_ref_known(v___x_1067_, 1);
goto v___jp_1019_;
}
else
{
lean_object* v_a_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1075_; 
lean_dec(v___x_1018_);
lean_dec(v_ref_1016_);
lean_dec_ref(v___x_1012_);
lean_dec_ref(v___x_1011_);
lean_dec_ref(v___f_1010_);
lean_dec_ref(v___f_1009_);
lean_dec_ref(v___x_1008_);
v_a_1068_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1070_ = v___x_1067_;
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_a_1068_);
lean_dec(v___x_1067_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v___x_1073_; 
if (v_isShared_1071_ == 0)
{
v___x_1073_ = v___x_1070_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v_a_1068_);
v___x_1073_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
return v___x_1073_;
}
}
}
}
}
else
{
lean_dec_ref_known(v___x_1057_, 2);
goto v___jp_1019_;
}
v___jp_1019_:
{
lean_object* v___x_1020_; lean_object* v___x_9694__overap_1021_; lean_object* v___x_1022_; 
v___x_1020_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_1020_, 0, lean_box(0));
lean_closure_set(v___x_1020_, 1, lean_box(0));
lean_closure_set(v___x_1020_, 2, v___x_1008_);
lean_closure_set(v___x_1020_, 3, lean_box(0));
lean_closure_set(v___x_1020_, 4, lean_box(0));
lean_closure_set(v___x_1020_, 5, v___f_1009_);
lean_closure_set(v___x_1020_, 6, v___f_1010_);
v___x_9694__overap_1021_ = l_Lean_Core_withCurrHeartbeats___redArg(v___x_1011_, v___x_1012_, v___x_1020_);
lean_inc(v_ref_1016_);
lean_inc_ref(v_a_1007_);
lean_inc(v___x_1018_);
lean_inc_ref(v_a_1015_);
lean_inc(v_a_1014_);
lean_inc_ref(v_a_1013_);
v___x_1022_ = lean_apply_7(v___x_9694__overap_1021_, v_a_1013_, v_a_1014_, v_a_1015_, v___x_1018_, v_a_1007_, v_ref_1016_, lean_box(0));
if (lean_obj_tag(v___x_1022_) == 0)
{
lean_object* v_a_1023_; lean_object* v___x_1025_; uint8_t v_isShared_1026_; uint8_t v_isSharedCheck_1033_; 
v_a_1023_ = lean_ctor_get(v___x_1022_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1025_ = v___x_1022_;
v_isShared_1026_ = v_isSharedCheck_1033_;
goto v_resetjp_1024_;
}
else
{
lean_inc(v_a_1023_);
lean_dec(v___x_1022_);
v___x_1025_ = lean_box(0);
v_isShared_1026_ = v_isSharedCheck_1033_;
goto v_resetjp_1024_;
}
v_resetjp_1024_:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1031_; 
v___x_1027_ = lean_st_ref_get(v___x_1018_);
lean_dec(v___x_1018_);
lean_dec(v___x_1027_);
v___x_1028_ = lean_st_ref_get(v_ref_1016_);
lean_dec(v_ref_1016_);
v___x_1029_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1029_, 0, v_a_1023_);
lean_ctor_set(v___x_1029_, 1, v___x_1028_);
if (v_isShared_1026_ == 0)
{
lean_ctor_set(v___x_1025_, 0, v___x_1029_);
v___x_1031_ = v___x_1025_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1029_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
return v___x_1031_;
}
}
}
else
{
lean_object* v_a_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1041_; 
lean_dec(v___x_1018_);
lean_dec(v_ref_1016_);
v_a_1034_ = lean_ctor_get(v___x_1022_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1036_ = v___x_1022_;
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_a_1034_);
lean_dec(v___x_1022_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v___x_1039_; 
if (v_isShared_1037_ == 0)
{
v___x_1039_ = v___x_1036_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v_a_1034_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
return v___x_1039_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___boxed(lean_object* v_val_1076_, lean_object* v_a_1077_, lean_object* v___x_1078_, lean_object* v___f_1079_, lean_object* v___f_1080_, lean_object* v___x_1081_, lean_object* v___x_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_, lean_object* v_ref_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v_res_1088_; 
v_res_1088_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4(v_val_1076_, v_a_1077_, v___x_1078_, v___f_1079_, v___f_1080_, v___x_1081_, v___x_1082_, v_a_1083_, v_a_1084_, v_a_1085_, v_ref_1086_);
lean_dec_ref(v_a_1085_);
lean_dec(v_a_1084_);
lean_dec_ref(v_a_1083_);
lean_dec_ref(v_a_1077_);
return v_res_1088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5(lean_object* v_val_1089_){
_start:
{
lean_object* v___x_1091_; lean_object* v___x_1092_; 
v___x_1091_ = lean_st_mk_ref(v_val_1089_);
v___x_1092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1092_, 0, v___x_1091_);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5___boxed(lean_object* v_val_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v_res_1095_; 
v_res_1095_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5(v_val_1093_);
return v_res_1095_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2(void){
_start:
{
lean_object* v___x_1098_; 
v___x_1098_ = l_instMonadControlReaderT(lean_box(0), lean_box(0));
return v___x_1098_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3(void){
_start:
{
lean_object* v___x_1099_; 
v___x_1099_ = l_instMonadControlStateRefT_x27(lean_box(0), lean_box(0), lean_box(0));
return v___x_1099_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4(void){
_start:
{
lean_object* v___x_1100_; lean_object* v___x_1101_; 
v___x_1100_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1);
v___x_1101_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_1101_, 0, lean_box(0));
lean_closure_set(v___x_1101_, 1, lean_box(0));
lean_closure_set(v___x_1101_, 2, v___x_1100_);
return v___x_1101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5(void){
_start:
{
lean_object* v___x_1102_; lean_object* v___x_1103_; 
v___x_1102_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__4);
v___x_1103_ = l_instMonadControlTOfPure___redArg(v___x_1102_);
return v___x_1103_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6(void){
_start:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___f_1106_; 
v___x_1104_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5);
v___x_1105_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3);
v___f_1106_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1106_, 0, v___x_1105_);
lean_closure_set(v___f_1106_, 1, v___x_1104_);
return v___f_1106_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7(void){
_start:
{
lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___f_1109_; 
v___x_1107_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__5);
v___x_1108_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__3);
v___f_1109_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_1109_, 0, v___x_1108_);
lean_closure_set(v___f_1109_, 1, v___x_1107_);
return v___f_1109_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8(void){
_start:
{
lean_object* v___f_1110_; lean_object* v___f_1111_; lean_object* v___x_1112_; 
v___f_1110_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__7);
v___f_1111_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__6);
v___x_1112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1112_, 0, v___f_1111_);
lean_ctor_set(v___x_1112_, 1, v___f_1110_);
return v___x_1112_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9(void){
_start:
{
lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___f_1115_; 
v___x_1113_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8);
v___x_1114_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1115_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1115_, 0, v___x_1114_);
lean_closure_set(v___f_1115_, 1, v___x_1113_);
return v___f_1115_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10(void){
_start:
{
lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___f_1118_; 
v___x_1116_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__8);
v___x_1117_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1118_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_1118_, 0, v___x_1117_);
lean_closure_set(v___f_1118_, 1, v___x_1116_);
return v___f_1118_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11(void){
_start:
{
lean_object* v___f_1119_; lean_object* v___f_1120_; lean_object* v___x_1121_; 
v___f_1119_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__10);
v___f_1120_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__9);
v___x_1121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1121_, 0, v___f_1120_);
lean_ctor_set(v___x_1121_, 1, v___f_1119_);
return v___x_1121_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12(void){
_start:
{
lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___f_1124_; 
v___x_1122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11);
v___x_1123_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1124_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1124_, 0, v___x_1123_);
lean_closure_set(v___f_1124_, 1, v___x_1122_);
return v___f_1124_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13(void){
_start:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___f_1127_; 
v___x_1125_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__11);
v___x_1126_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1127_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_1127_, 0, v___x_1126_);
lean_closure_set(v___f_1127_, 1, v___x_1125_);
return v___f_1127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14(void){
_start:
{
lean_object* v___f_1128_; lean_object* v___f_1129_; lean_object* v___x_1130_; 
v___f_1128_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__13);
v___f_1129_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__12);
v___x_1130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1130_, 0, v___f_1129_);
lean_ctor_set(v___x_1130_, 1, v___f_1128_);
return v___x_1130_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15(void){
_start:
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___f_1133_; 
v___x_1131_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14);
v___x_1132_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1133_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1133_, 0, v___x_1132_);
lean_closure_set(v___f_1133_, 1, v___x_1131_);
return v___f_1133_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16(void){
_start:
{
lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___f_1136_; 
v___x_1134_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__14);
v___x_1135_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__2);
v___f_1136_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_1136_, 0, v___x_1135_);
lean_closure_set(v___f_1136_, 1, v___x_1134_);
return v___f_1136_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17(void){
_start:
{
lean_object* v___f_1137_; lean_object* v___f_1138_; lean_object* v___x_1139_; 
v___f_1137_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__16);
v___f_1138_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__15);
v___x_1139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1139_, 0, v___f_1138_);
lean_ctor_set(v___x_1139_, 1, v___f_1137_);
return v___x_1139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg(lean_object* v_premise_1142_, lean_object* v_k_1143_, lean_object* v_a_1144_, lean_object* v_a_1145_, lean_object* v_a_1146_, lean_object* v_a_1147_, lean_object* v_a_1148_, lean_object* v_a_1149_){
_start:
{
lean_object* v___x_1151_; lean_object* v_toApplicative_1152_; lean_object* v_toFunctor_1153_; lean_object* v_toSeq_1154_; lean_object* v_toSeqLeft_1155_; lean_object* v_toSeqRight_1156_; lean_object* v___f_1157_; lean_object* v___f_1158_; lean_object* v___f_1159_; lean_object* v___f_1160_; lean_object* v___x_1161_; lean_object* v___f_1162_; lean_object* v___f_1163_; lean_object* v___f_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v_toApplicative_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1230_; 
v___x_1151_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__1);
v_toApplicative_1152_ = lean_ctor_get(v___x_1151_, 0);
v_toFunctor_1153_ = lean_ctor_get(v_toApplicative_1152_, 0);
v_toSeq_1154_ = lean_ctor_get(v_toApplicative_1152_, 2);
v_toSeqLeft_1155_ = lean_ctor_get(v_toApplicative_1152_, 3);
v_toSeqRight_1156_ = lean_ctor_get(v_toApplicative_1152_, 4);
v___f_1157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__2));
v___f_1158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___closed__3));
lean_inc_ref_n(v_toFunctor_1153_, 2);
v___f_1159_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1159_, 0, v_toFunctor_1153_);
v___f_1160_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1160_, 0, v_toFunctor_1153_);
v___x_1161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___f_1159_);
lean_ctor_set(v___x_1161_, 1, v___f_1160_);
lean_inc(v_toSeqRight_1156_);
v___f_1162_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1162_, 0, v_toSeqRight_1156_);
lean_inc(v_toSeqLeft_1155_);
v___f_1163_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1163_, 0, v_toSeqLeft_1155_);
lean_inc(v_toSeq_1154_);
v___f_1164_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1164_, 0, v_toSeq_1154_);
v___x_1165_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1161_);
lean_ctor_set(v___x_1165_, 1, v___f_1157_);
lean_ctor_set(v___x_1165_, 2, v___f_1164_);
lean_ctor_set(v___x_1165_, 3, v___f_1163_);
lean_ctor_set(v___x_1165_, 4, v___f_1162_);
v___x_1166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1166_, 0, v___x_1165_);
lean_ctor_set(v___x_1166_, 1, v___f_1158_);
v___x_1167_ = l_StateRefT_x27_instMonad___redArg(v___x_1166_);
v_toApplicative_1168_ = lean_ctor_get(v___x_1167_, 0);
v_isSharedCheck_1230_ = !lean_is_exclusive(v___x_1167_);
if (v_isSharedCheck_1230_ == 0)
{
lean_object* v_unused_1231_; 
v_unused_1231_ = lean_ctor_get(v___x_1167_, 1);
lean_dec(v_unused_1231_);
v___x_1170_ = v___x_1167_;
v_isShared_1171_ = v_isSharedCheck_1230_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_toApplicative_1168_);
lean_dec(v___x_1167_);
v___x_1170_ = lean_box(0);
v_isShared_1171_ = v_isSharedCheck_1230_;
goto v_resetjp_1169_;
}
v_resetjp_1169_:
{
lean_object* v_toFunctor_1172_; lean_object* v_toSeq_1173_; lean_object* v_toSeqLeft_1174_; lean_object* v_toSeqRight_1175_; lean_object* v___x_1177_; uint8_t v_isShared_1178_; uint8_t v_isSharedCheck_1228_; 
v_toFunctor_1172_ = lean_ctor_get(v_toApplicative_1168_, 0);
v_toSeq_1173_ = lean_ctor_get(v_toApplicative_1168_, 2);
v_toSeqLeft_1174_ = lean_ctor_get(v_toApplicative_1168_, 3);
v_toSeqRight_1175_ = lean_ctor_get(v_toApplicative_1168_, 4);
v_isSharedCheck_1228_ = !lean_is_exclusive(v_toApplicative_1168_);
if (v_isSharedCheck_1228_ == 0)
{
lean_object* v_unused_1229_; 
v_unused_1229_ = lean_ctor_get(v_toApplicative_1168_, 1);
lean_dec(v_unused_1229_);
v___x_1177_ = v_toApplicative_1168_;
v_isShared_1178_ = v_isSharedCheck_1228_;
goto v_resetjp_1176_;
}
else
{
lean_inc(v_toSeqRight_1175_);
lean_inc(v_toSeqLeft_1174_);
lean_inc(v_toSeq_1173_);
lean_inc(v_toFunctor_1172_);
lean_dec(v_toApplicative_1168_);
v___x_1177_ = lean_box(0);
v_isShared_1178_ = v_isSharedCheck_1228_;
goto v_resetjp_1176_;
}
v_resetjp_1176_:
{
lean_object* v___f_1179_; lean_object* v___f_1180_; lean_object* v___f_1181_; lean_object* v___f_1182_; lean_object* v___x_1183_; lean_object* v___f_1184_; lean_object* v___f_1185_; lean_object* v___f_1186_; lean_object* v___x_1188_; 
v___f_1179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__0));
v___f_1180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__1));
lean_inc_ref(v_toFunctor_1172_);
v___f_1181_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1181_, 0, v_toFunctor_1172_);
v___f_1182_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1182_, 0, v_toFunctor_1172_);
v___x_1183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1183_, 0, v___f_1181_);
lean_ctor_set(v___x_1183_, 1, v___f_1182_);
v___f_1184_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1184_, 0, v_toSeqRight_1175_);
v___f_1185_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1185_, 0, v_toSeqLeft_1174_);
v___f_1186_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1186_, 0, v_toSeq_1173_);
if (v_isShared_1178_ == 0)
{
lean_ctor_set(v___x_1177_, 4, v___f_1184_);
lean_ctor_set(v___x_1177_, 3, v___f_1185_);
lean_ctor_set(v___x_1177_, 2, v___f_1186_);
lean_ctor_set(v___x_1177_, 1, v___f_1179_);
lean_ctor_set(v___x_1177_, 0, v___x_1183_);
v___x_1188_ = v___x_1177_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v___x_1183_);
lean_ctor_set(v_reuseFailAlloc_1227_, 1, v___f_1179_);
lean_ctor_set(v_reuseFailAlloc_1227_, 2, v___f_1186_);
lean_ctor_set(v_reuseFailAlloc_1227_, 3, v___f_1185_);
lean_ctor_set(v_reuseFailAlloc_1227_, 4, v___f_1184_);
v___x_1188_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
lean_object* v___x_1190_; 
if (v_isShared_1171_ == 0)
{
lean_ctor_set(v___x_1170_, 1, v___f_1180_);
lean_ctor_set(v___x_1170_, 0, v___x_1188_);
v___x_1190_ = v___x_1170_;
goto v_reusejp_1189_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v___x_1188_);
lean_ctor_set(v_reuseFailAlloc_1226_, 1, v___f_1180_);
v___x_1190_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1189_;
}
v_reusejp_1189_:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; 
v___x_1191_ = l_ReaderT_instMonad___redArg(v___x_1190_);
lean_inc_ref(v___x_1191_);
v___x_1192_ = l_ReaderT_instMonad___redArg(v___x_1191_);
v___x_1193_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__17);
v___x_1194_ = l_Lean_KVMap_instValueBool;
v___x_1195_ = lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps;
v___x_1196_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_Premise_toHtml(v_premise_1142_, v_a_1146_, v_a_1147_, v_a_1148_, v_a_1149_);
if (lean_obj_tag(v___x_1196_) == 0)
{
lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1217_; 
v_a_1197_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1217_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1217_ == 0)
{
v___x_1199_ = v___x_1196_;
v_isShared_1200_ = v_isSharedCheck_1217_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_dec(v___x_1196_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1217_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___f_1203_; lean_object* v___f_1204_; lean_object* v___f_1205_; lean_object* v___f_1206_; lean_object* v___f_1207_; lean_object* v___f_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1215_; 
v___x_1201_ = lean_st_ref_get(v_a_1147_);
v___x_1202_ = lean_st_ref_get(v_a_1149_);
v___f_1203_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__18));
v___f_1204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___closed__19));
v___f_1205_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__2___boxed), 9, 2);
lean_closure_set(v___f_1205_, 0, v_k_1143_);
lean_closure_set(v___f_1205_, 1, v___x_1194_);
v___f_1206_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__3___boxed), 4, 2);
lean_closure_set(v___f_1206_, 0, v___x_1195_);
lean_closure_set(v___f_1206_, 1, v_a_1197_);
lean_inc_ref(v_a_1146_);
lean_inc(v_a_1145_);
lean_inc_ref(v_a_1144_);
lean_inc_ref(v_a_1148_);
v___f_1207_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__4___boxed), 12, 10);
lean_closure_set(v___f_1207_, 0, v___x_1201_);
lean_closure_set(v___f_1207_, 1, v_a_1148_);
lean_closure_set(v___f_1207_, 2, v___x_1191_);
lean_closure_set(v___f_1207_, 3, v___f_1205_);
lean_closure_set(v___f_1207_, 4, v___f_1204_);
lean_closure_set(v___f_1207_, 5, v___x_1192_);
lean_closure_set(v___f_1207_, 6, v___x_1193_);
lean_closure_set(v___f_1207_, 7, v_a_1144_);
lean_closure_set(v___f_1207_, 8, v_a_1145_);
lean_closure_set(v___f_1207_, 9, v_a_1146_);
v___f_1208_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___lam__5___boxed), 2, 1);
lean_closure_set(v___f_1208_, 0, v___x_1202_);
v___x_1209_ = lean_alloc_closure((void*)(l_instMonadEIO___aux__13___boxed), 6, 5);
lean_closure_set(v___x_1209_, 0, lean_box(0));
lean_closure_set(v___x_1209_, 1, lean_box(0));
lean_closure_set(v___x_1209_, 2, lean_box(0));
lean_closure_set(v___x_1209_, 3, v___f_1208_);
lean_closure_set(v___x_1209_, 4, v___f_1207_);
v___x_1210_ = lean_alloc_closure((void*)(l_instMonadEIO___aux__13___boxed), 6, 5);
lean_closure_set(v___x_1210_, 0, lean_box(0));
lean_closure_set(v___x_1210_, 1, lean_box(0));
lean_closure_set(v___x_1210_, 2, lean_box(0));
lean_closure_set(v___x_1210_, 3, v___x_1209_);
lean_closure_set(v___x_1210_, 4, v___f_1203_);
v___x_1211_ = lean_alloc_closure((void*)(l_EIO_catchExceptions___boxed), 5, 4);
lean_closure_set(v___x_1211_, 0, lean_box(0));
lean_closure_set(v___x_1211_, 1, lean_box(0));
lean_closure_set(v___x_1211_, 2, v___x_1210_);
lean_closure_set(v___x_1211_, 3, v___f_1206_);
v___x_1212_ = lean_unsigned_to_nat(0u);
v___x_1213_ = lean_io_as_task(v___x_1211_, v___x_1212_);
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 0, v___x_1213_);
v___x_1215_ = v___x_1199_;
goto v_reusejp_1214_;
}
else
{
lean_object* v_reuseFailAlloc_1216_; 
v_reuseFailAlloc_1216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1216_, 0, v___x_1213_);
v___x_1215_ = v_reuseFailAlloc_1216_;
goto v_reusejp_1214_;
}
v_reusejp_1214_:
{
return v___x_1215_;
}
}
}
else
{
lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1225_; 
lean_dec_ref(v___x_1192_);
lean_dec_ref(v___x_1191_);
lean_dec_ref(v_k_1143_);
v_a_1218_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1220_ = v___x_1196_;
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1196_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v___x_1223_; 
if (v_isShared_1221_ == 0)
{
v___x_1223_ = v___x_1220_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v_a_1218_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg___boxed(lean_object* v_premise_1232_, lean_object* v_k_1233_, lean_object* v_a_1234_, lean_object* v_a_1235_, lean_object* v_a_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_, lean_object* v_a_1239_, lean_object* v_a_1240_){
_start:
{
lean_object* v_res_1241_; 
v_res_1241_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg(v_premise_1232_, v_k_1233_, v_a_1234_, v_a_1235_, v_a_1236_, v_a_1237_, v_a_1238_, v_a_1239_);
lean_dec(v_a_1239_);
lean_dec_ref(v_a_1238_);
lean_dec(v_a_1237_);
lean_dec_ref(v_a_1236_);
lean_dec(v_a_1235_);
lean_dec_ref(v_a_1234_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask(lean_object* v_00_u03b1_1242_, lean_object* v_premise_1243_, lean_object* v_k_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_, lean_object* v_a_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_, lean_object* v_a_1250_){
_start:
{
lean_object* v___x_1252_; 
v___x_1252_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___redArg(v_premise_1243_, v_k_1244_, v_a_1245_, v_a_1246_, v_a_1247_, v_a_1248_, v_a_1249_, v_a_1250_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask___boxed(lean_object* v_00_u03b1_1253_, lean_object* v_premise_1254_, lean_object* v_k_1255_, lean_object* v_a_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_, lean_object* v_a_1262_){
_start:
{
lean_object* v_res_1263_; 
v_res_1263_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_spawnTask(v_00_u03b1_1253_, v_premise_1254_, v_k_1255_, v_a_1256_, v_a_1257_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
lean_dec(v_a_1261_);
lean_dec_ref(v_a_1260_);
lean_dec(v_a_1259_);
lean_dec_ref(v_a_1258_);
lean_dec(v_a_1257_);
lean_dec_ref(v_a_1256_);
return v_res_1263_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_FilterDetails(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_FilterDetails(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_FilterDetails(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_FilterDetails(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions_SectionState(builtin);
}
#ifdef __cplusplus
}
#endif
