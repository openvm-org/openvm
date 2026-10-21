// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Datatypes
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Linarith.Lemmas public import Mathlib.Tactic.NormNum.Basic public import Mathlib.Util.SynthesizeUsing
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
uint8_t lp_mathlib_Mathlib_Ineq_cmp(uint8_t, uint8_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Int_repr___boxed(lean_object*);
lean_object* l_instToFormatOfToString___redArg(lean_object*);
lean_object* l_instToFormatProd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_format___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Ineq_toString(uint8_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
uint8_t lp_mathlib_Mathlib_Ineq_max(uint8_t, uint8_t);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_io_mono_nanos_now();
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_string_length(lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_instReprIneq_repr(uint8_t, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
uint8_t lp_mathlib_Mathlib_instDecidableEqIneq(uint8_t, uint8_t);
lean_object* lp_mathlib_Lean_Expr_ineq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Lean_Expr_zero_x3f(lean_object*);
lean_object* lp_mathlib_Lean_Expr_ofNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName(uint8_t);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_addRawTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linarith"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__7_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__7_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__7_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__8_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__8_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__8_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__9_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__7_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__8_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(160, 128, 21, 181, 24, 166, 24, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__9_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__9_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__10_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Datatypes"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__10_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__10_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__11_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__9_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__10_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(107, 154, 50, 145, 121, 207, 176, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__11_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__11_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__12_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__11_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(214, 176, 82, 23, 24, 142, 43, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__12_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__12_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__13_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__13_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__13_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__14_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__12_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__13_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(59, 244, 118, 215, 125, 188, 134, 26)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__14_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__14_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__15_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__15_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__15_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__16_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__14_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__15_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(30, 173, 245, 42, 71, 248, 24, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__16_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__16_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__17_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__16_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(215, 248, 107, 253, 223, 30, 21, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__17_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__17_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__18_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__17_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(22, 108, 237, 196, 25, 55, 167, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__18_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__18_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__19_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__18_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__8_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(189, 108, 245, 172, 222, 200, 237, 245)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__19_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__19_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__20_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__19_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__10_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(138, 192, 176, 174, 48, 89, 0, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__20_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__20_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__21_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__20_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)(((size_t)(189509315) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(43, 233, 89, 184, 130, 74, 98, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__21_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__21_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__22_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__22_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__22_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__23_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__21_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__22_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(208, 246, 153, 140, 133, 144, 220, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__23_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__23_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__24_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__24_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__24_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__25_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__23_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__24_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(212, 179, 104, 55, 192, 186, 235, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__25_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__25_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__26_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__25_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(237, 192, 3, 242, 28, 89, 183, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__26_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__26_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "detail"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__0_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 12, 183, 160, 66, 250, 13, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__13_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_contains(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_contains___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_vars_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_vars(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instInhabitedComp_default___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6;
static const lean_ctor_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__7 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__8 = (const lean_object*)&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__2_value;
static const lean_string_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4;
static lean_once_cell_t lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5;
static const lean_ctor_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__6 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "coeffs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_instReprComp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_vars(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_coeffOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_coeffOf___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_scale(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_add(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_cmp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_cmp___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_isContr(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_isContr___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_reprFast, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_repr___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__9_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "declName"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__13_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__14_value),LEAN_SCALAR_PTR_LITERAL(113, 211, 58, 33, 138, 196, 138, 106)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "decl_name%"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Preprocessing: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = " has branched, with branches:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_comp, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessorToGlobalBranchingPreprocessor = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorToGlobalBranchingPreprocessor___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "invalid comparison, rhs not zero: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "gt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 16, 15, 58, 66, 186, 138, 31)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__1_value),LEAN_SCALAR_PTR_LITERAL(239, 75, 137, 103, 59, 22, 209, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "normNum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__3_value),LEAN_SCALAR_PTR_LITERAL(235, 202, 36, 226, 215, 147, 189, 233)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "norm_num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__6_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__6_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "zero_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__9_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__10_value),LEAN_SCALAR_PTR_LITERAL(200, 16, 162, 252, 150, 115, 215, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_62_; uint8_t v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_63_ = 0;
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__26_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_65_ = l_Lean_registerTraceClass(v___x_62_, v___x_63_, v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2____boxed(lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_();
return v_res_67_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_72_ = lean_unsigned_to_nat(3248332875u);
v___x_73_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__20_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_74_ = l_Lean_Name_num___override(v___x_73_, v___x_72_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__22_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_76_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__2_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_);
v___x_77_ = l_Lean_Name_str___override(v___x_76_, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_78_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__24_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_79_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__3_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_);
v___x_80_ = l_Lean_Name_str___override(v___x_79_, v___x_78_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = lean_unsigned_to_nat(2u);
v___x_82_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__4_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_);
v___x_83_ = l_Lean_Name_num___override(v___x_82_, v___x_81_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_85_; uint8_t v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_));
v___x_86_ = 0;
v___x_87_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__5_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_);
v___x_88_ = l_Lean_registerTraceClass(v___x_85_, v___x_86_, v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2____boxed(lean_object* v_a_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_();
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg(lean_object* v_e_91_, lean_object* v___y_92_){
_start:
{
uint8_t v___x_94_; 
v___x_94_ = l_Lean_Expr_hasMVar(v_e_91_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; 
v___x_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_95_, 0, v_e_91_);
return v___x_95_;
}
else
{
lean_object* v___x_96_; lean_object* v_mctx_97_; lean_object* v___x_98_; lean_object* v_fst_99_; lean_object* v_snd_100_; lean_object* v___x_101_; lean_object* v_cache_102_; lean_object* v_zetaDeltaFVarIds_103_; lean_object* v_postponed_104_; lean_object* v_diag_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_114_; 
v___x_96_ = lean_st_ref_get(v___y_92_);
v_mctx_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc_ref(v_mctx_97_);
lean_dec(v___x_96_);
v___x_98_ = l_Lean_instantiateMVarsCore(v_mctx_97_, v_e_91_);
v_fst_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_fst_99_);
v_snd_100_ = lean_ctor_get(v___x_98_, 1);
lean_inc(v_snd_100_);
lean_dec_ref(v___x_98_);
v___x_101_ = lean_st_ref_take(v___y_92_);
v_cache_102_ = lean_ctor_get(v___x_101_, 1);
v_zetaDeltaFVarIds_103_ = lean_ctor_get(v___x_101_, 2);
v_postponed_104_ = lean_ctor_get(v___x_101_, 3);
v_diag_105_ = lean_ctor_get(v___x_101_, 4);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_114_ == 0)
{
lean_object* v_unused_115_; 
v_unused_115_ = lean_ctor_get(v___x_101_, 0);
lean_dec(v_unused_115_);
v___x_107_ = v___x_101_;
v_isShared_108_ = v_isSharedCheck_114_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_diag_105_);
lean_inc(v_postponed_104_);
lean_inc(v_zetaDeltaFVarIds_103_);
lean_inc(v_cache_102_);
lean_dec(v___x_101_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_114_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___x_110_; 
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 0, v_snd_100_);
v___x_110_ = v___x_107_;
goto v_reusejp_109_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v_snd_100_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v_cache_102_);
lean_ctor_set(v_reuseFailAlloc_113_, 2, v_zetaDeltaFVarIds_103_);
lean_ctor_set(v_reuseFailAlloc_113_, 3, v_postponed_104_);
lean_ctor_set(v_reuseFailAlloc_113_, 4, v_diag_105_);
v___x_110_ = v_reuseFailAlloc_113_;
goto v_reusejp_109_;
}
v_reusejp_109_:
{
lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_111_ = lean_st_ref_set(v___y_92_, v___x_110_);
v___x_112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_112_, 0, v_fst_99_);
return v___x_112_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg___boxed(lean_object* v_e_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg(v_e_116_, v___y_117_);
lean_dec(v___y_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0(lean_object* v_e_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg(v_e_120_, v___y_122_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___boxed(lean_object* v_e_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0(v_e_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1(lean_object* v_x_134_, lean_object* v_x_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
if (lean_obj_tag(v_x_134_) == 0)
{
lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_141_ = l_List_reverse___redArg(v_x_135_);
v___x_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
return v___x_142_;
}
else
{
lean_object* v_head_143_; lean_object* v_tail_144_; lean_object* v___x_146_; uint8_t v_isShared_147_; uint8_t v_isSharedCheck_166_; 
v_head_143_ = lean_ctor_get(v_x_134_, 0);
v_tail_144_ = lean_ctor_get(v_x_134_, 1);
v_isSharedCheck_166_ = !lean_is_exclusive(v_x_134_);
if (v_isSharedCheck_166_ == 0)
{
v___x_146_ = v_x_134_;
v_isShared_147_ = v_isSharedCheck_166_;
goto v_resetjp_145_;
}
else
{
lean_inc(v_tail_144_);
lean_inc(v_head_143_);
lean_dec(v_x_134_);
v___x_146_ = lean_box(0);
v_isShared_147_ = v_isSharedCheck_166_;
goto v_resetjp_145_;
}
v_resetjp_145_:
{
lean_object* v___y_149_; lean_object* v___x_163_; 
lean_inc(v___y_139_);
lean_inc_ref(v___y_138_);
lean_inc(v___y_137_);
lean_inc_ref(v___y_136_);
v___x_163_ = lean_infer_type(v_head_143_, v___y_136_, v___y_137_, v___y_138_, v___y_139_);
if (lean_obj_tag(v___x_163_) == 0)
{
lean_object* v_a_164_; lean_object* v___x_165_; 
v_a_164_ = lean_ctor_get(v___x_163_, 0);
lean_inc(v_a_164_);
lean_dec_ref_known(v___x_163_, 1);
v___x_165_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__0___redArg(v_a_164_, v___y_137_);
v___y_149_ = v___x_165_;
goto v___jp_148_;
}
else
{
v___y_149_ = v___x_163_;
goto v___jp_148_;
}
v___jp_148_:
{
if (lean_obj_tag(v___y_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_152_; 
v_a_150_ = lean_ctor_get(v___y_149_, 0);
lean_inc(v_a_150_);
lean_dec_ref_known(v___y_149_, 1);
if (v_isShared_147_ == 0)
{
lean_ctor_set(v___x_146_, 1, v_x_135_);
lean_ctor_set(v___x_146_, 0, v_a_150_);
v___x_152_ = v___x_146_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_150_);
lean_ctor_set(v_reuseFailAlloc_154_, 1, v_x_135_);
v___x_152_ = v_reuseFailAlloc_154_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
v_x_134_ = v_tail_144_;
v_x_135_ = v___x_152_;
goto _start;
}
}
else
{
lean_object* v_a_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
lean_del_object(v___x_146_);
lean_dec(v_tail_144_);
lean_dec(v_x_135_);
v_a_155_ = lean_ctor_get(v___y_149_, 0);
v_isSharedCheck_162_ = !lean_is_exclusive(v___y_149_);
if (v_isSharedCheck_162_ == 0)
{
v___x_157_ = v___y_149_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_a_155_);
lean_dec(v___y_149_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_a_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1___boxed(lean_object* v_x_167_, lean_object* v_x_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1(v_x_167_, v_x_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__2(lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
if (lean_obj_tag(v_a_175_) == 0)
{
lean_object* v___x_177_; 
v___x_177_ = l_List_reverse___redArg(v_a_176_);
return v___x_177_;
}
else
{
lean_object* v_head_178_; lean_object* v_tail_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_188_; 
v_head_178_ = lean_ctor_get(v_a_175_, 0);
v_tail_179_ = lean_ctor_get(v_a_175_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_188_ == 0)
{
v___x_181_ = v_a_175_;
v_isShared_182_ = v_isSharedCheck_188_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_tail_179_);
lean_inc(v_head_178_);
lean_dec(v_a_175_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_188_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_183_; lean_object* v___x_185_; 
v___x_183_ = l_Lean_MessageData_ofExpr(v_head_178_);
if (v_isShared_182_ == 0)
{
lean_ctor_set(v___x_181_, 1, v_a_176_);
lean_ctor_set(v___x_181_, 0, v___x_183_);
v___x_185_ = v___x_181_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_183_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_a_176_);
v___x_185_ = v_reuseFailAlloc_187_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
v_a_175_ = v_tail_179_;
v_a_176_ = v___x_185_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(lean_object* v_l_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_195_ = lean_box(0);
v___x_196_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__1(v_l_189_, v___x_195_, v_a_190_, v_a_191_, v_a_192_, v_a_193_);
if (lean_obj_tag(v___x_196_) == 0)
{
lean_object* v_a_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_206_; 
v_a_197_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_206_ == 0)
{
v___x_199_ = v___x_196_;
v_isShared_200_ = v_isSharedCheck_206_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_a_197_);
lean_dec(v___x_196_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_206_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_204_; 
v___x_201_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linarithGetProofsMessage_spec__2(v_a_197_, v___x_195_);
v___x_202_ = l_Lean_MessageData_ofList(v___x_201_);
if (v_isShared_200_ == 0)
{
lean_ctor_set(v___x_199_, 0, v___x_202_);
v___x_204_ = v___x_199_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v___x_202_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
else
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_214_; 
v_a_207_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_214_ == 0)
{
v___x_209_ = v___x_196_;
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_196_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_212_; 
if (v_isShared_210_ == 0)
{
v___x_212_ = v___x_209_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_a_207_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage___boxed(lean_object* v_l_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_, lean_object* v_a_219_, lean_object* v_a_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(v_l_215_, v_a_216_, v_a_217_, v_a_218_, v_a_219_);
lean_dec(v_a_219_);
lean_dec_ref(v_a_218_);
lean_dec(v_a_217_);
lean_dec_ref(v_a_216_);
return v_res_221_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = l_instMonadEIO(lean_box(0));
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1(void){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_223_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__0);
v___x_224_ = l_StateRefT_x27_instMonad___redArg(v___x_223_);
return v___x_224_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8(void){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_231_ = l_Lean_Core_instMonadTraceCoreM;
v___x_232_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__7));
v___x_233_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_232_, v___x_231_);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9(void){
_start:
{
lean_object* v___x_234_; lean_object* v___f_235_; lean_object* v___x_236_; 
v___x_234_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__8);
v___f_235_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__6));
v___x_236_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_235_, v___x_234_);
return v___x_236_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_240_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_241_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__11));
v___x_242_ = l_Lean_Name_append(v___x_241_, v___x_240_);
return v___x_242_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_245_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__7));
v___x_247_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__14));
v___x_248_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_247_, v___x_246_, v___x_245_);
return v___x_248_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16(void){
_start:
{
lean_object* v___x_249_; lean_object* v___f_250_; lean_object* v___f_251_; lean_object* v___x_252_; 
v___x_249_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__15);
v___f_250_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__6));
v___f_251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__13));
v___x_252_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_251_, v___f_250_, v___x_249_);
return v___x_252_;
}
}
static double _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17(void){
_start:
{
lean_object* v___x_253_; double v___x_254_; 
v___x_253_ = lean_unsigned_to_nat(0u);
v___x_254_ = lean_float_of_nat(v___x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg(lean_object* v_inst_256_, lean_object* v_s_257_, lean_object* v_l_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v___x_267_; lean_object* v_toApplicative_268_; lean_object* v_toFunctor_269_; lean_object* v_toSeq_270_; lean_object* v_toSeqLeft_271_; lean_object* v_toSeqRight_272_; lean_object* v___f_273_; lean_object* v___f_274_; lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___x_277_; lean_object* v___f_278_; lean_object* v___f_279_; lean_object* v___f_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v_toApplicative_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_342_; 
v___x_267_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__1);
v_toApplicative_268_ = lean_ctor_get(v___x_267_, 0);
v_toFunctor_269_ = lean_ctor_get(v_toApplicative_268_, 0);
v_toSeq_270_ = lean_ctor_get(v_toApplicative_268_, 2);
v_toSeqLeft_271_ = lean_ctor_get(v_toApplicative_268_, 3);
v_toSeqRight_272_ = lean_ctor_get(v_toApplicative_268_, 4);
v___f_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__2));
v___f_274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_269_, 2);
v___f_275_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_275_, 0, v_toFunctor_269_);
v___f_276_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_276_, 0, v_toFunctor_269_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___f_275_);
lean_ctor_set(v___x_277_, 1, v___f_276_);
lean_inc(v_toSeqRight_272_);
v___f_278_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_278_, 0, v_toSeqRight_272_);
lean_inc(v_toSeqLeft_271_);
v___f_279_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_279_, 0, v_toSeqLeft_271_);
lean_inc(v_toSeq_270_);
v___f_280_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_280_, 0, v_toSeq_270_);
v___x_281_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_281_, 0, v___x_277_);
lean_ctor_set(v___x_281_, 1, v___f_273_);
lean_ctor_set(v___x_281_, 2, v___f_280_);
lean_ctor_set(v___x_281_, 3, v___f_279_);
lean_ctor_set(v___x_281_, 4, v___f_278_);
v___x_282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_282_, 0, v___x_281_);
lean_ctor_set(v___x_282_, 1, v___f_274_);
v___x_283_ = l_StateRefT_x27_instMonad___redArg(v___x_282_);
v_toApplicative_284_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_342_ == 0)
{
lean_object* v_unused_343_; 
v_unused_343_ = lean_ctor_get(v___x_283_, 1);
lean_dec(v_unused_343_);
v___x_286_ = v___x_283_;
v_isShared_287_ = v_isSharedCheck_342_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_toApplicative_284_);
lean_dec(v___x_283_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_342_;
goto v_resetjp_285_;
}
v___jp_264_:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = lean_box(0);
v___x_266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
return v___x_266_;
}
v_resetjp_285_:
{
lean_object* v_toFunctor_288_; lean_object* v_toSeq_289_; lean_object* v_toSeqLeft_290_; lean_object* v_toSeqRight_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_340_; 
v_toFunctor_288_ = lean_ctor_get(v_toApplicative_284_, 0);
v_toSeq_289_ = lean_ctor_get(v_toApplicative_284_, 2);
v_toSeqLeft_290_ = lean_ctor_get(v_toApplicative_284_, 3);
v_toSeqRight_291_ = lean_ctor_get(v_toApplicative_284_, 4);
v_isSharedCheck_340_ = !lean_is_exclusive(v_toApplicative_284_);
if (v_isSharedCheck_340_ == 0)
{
lean_object* v_unused_341_; 
v_unused_341_ = lean_ctor_get(v_toApplicative_284_, 1);
lean_dec(v_unused_341_);
v___x_293_ = v_toApplicative_284_;
v_isShared_294_ = v_isSharedCheck_340_;
goto v_resetjp_292_;
}
else
{
lean_inc(v_toSeqRight_291_);
lean_inc(v_toSeqLeft_290_);
lean_inc(v_toSeq_289_);
lean_inc(v_toFunctor_288_);
lean_dec(v_toApplicative_284_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_340_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___f_295_; lean_object* v___f_296_; lean_object* v___f_297_; lean_object* v___f_298_; lean_object* v___x_299_; lean_object* v___f_300_; lean_object* v___f_301_; lean_object* v___f_302_; lean_object* v___x_304_; 
v___f_295_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__4));
v___f_296_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__5));
lean_inc_ref(v_toFunctor_288_);
v___f_297_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_297_, 0, v_toFunctor_288_);
v___f_298_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_298_, 0, v_toFunctor_288_);
v___x_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_299_, 0, v___f_297_);
lean_ctor_set(v___x_299_, 1, v___f_298_);
v___f_300_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_300_, 0, v_toSeqRight_291_);
v___f_301_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_301_, 0, v_toSeqLeft_290_);
v___f_302_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_302_, 0, v_toSeq_289_);
if (v_isShared_294_ == 0)
{
lean_ctor_set(v___x_293_, 4, v___f_300_);
lean_ctor_set(v___x_293_, 3, v___f_301_);
lean_ctor_set(v___x_293_, 2, v___f_302_);
lean_ctor_set(v___x_293_, 1, v___f_295_);
lean_ctor_set(v___x_293_, 0, v___x_299_);
v___x_304_ = v___x_293_;
goto v_reusejp_303_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_299_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v___f_295_);
lean_ctor_set(v_reuseFailAlloc_339_, 2, v___f_302_);
lean_ctor_set(v_reuseFailAlloc_339_, 3, v___f_301_);
lean_ctor_set(v_reuseFailAlloc_339_, 4, v___f_300_);
v___x_304_ = v_reuseFailAlloc_339_;
goto v_reusejp_303_;
}
v_reusejp_303_:
{
lean_object* v___x_306_; 
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 1, v___f_296_);
lean_ctor_set(v___x_286_, 0, v___x_304_);
v___x_306_ = v___x_286_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_304_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v___f_296_);
v___x_306_ = v_reuseFailAlloc_338_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
lean_object* v___x_307_; lean_object* v_options_308_; uint8_t v_hasTrace_309_; 
v___x_307_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__9);
v_options_308_ = lean_ctor_get(v_a_261_, 2);
v_hasTrace_309_ = lean_ctor_get_uint8(v_options_308_, sizeof(void*)*1);
if (v_hasTrace_309_ == 0)
{
lean_dec_ref(v___x_306_);
lean_dec(v_l_258_);
lean_dec(v_s_257_);
lean_dec_ref(v_inst_256_);
goto v___jp_264_;
}
else
{
lean_object* v_inheritedTraceOptions_310_; lean_object* v___x_311_; lean_object* v___x_312_; uint8_t v___x_313_; 
v_inheritedTraceOptions_310_ = lean_ctor_get(v_a_261_, 13);
v___x_311_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_312_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12);
v___x_313_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_310_, v_options_308_, v___x_312_);
if (v___x_313_ == 0)
{
lean_dec_ref(v___x_306_);
lean_dec(v_l_258_);
lean_dec(v_s_257_);
lean_dec_ref(v_inst_256_);
goto v___jp_264_;
}
else
{
lean_object* v___x_314_; 
v___x_314_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage(v_l_258_, v_a_259_, v_a_260_, v_a_261_, v_a_262_);
if (lean_obj_tag(v___x_314_) == 0)
{
lean_object* v_a_315_; lean_object* v___x_316_; lean_object* v_toMonadRef_317_; lean_object* v___x_318_; lean_object* v___x_319_; double v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_793__overap_328_; lean_object* v___x_329_; 
v_a_315_ = lean_ctor_get(v___x_314_, 0);
lean_inc(v_a_315_);
lean_dec_ref_known(v___x_314_, 1);
v___x_316_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__16);
v_toMonadRef_317_ = lean_ctor_get(v___x_316_, 0);
v___x_318_ = l_Lean_Meta_instAddMessageContextMetaM;
v___x_319_ = lean_box(0);
v___x_320_ = lean_float_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18));
v___x_322_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_322_, 0, v___x_311_);
lean_ctor_set(v___x_322_, 1, v___x_319_);
lean_ctor_set(v___x_322_, 2, v___x_321_);
lean_ctor_set_float(v___x_322_, sizeof(void*)*3, v___x_320_);
lean_ctor_set_float(v___x_322_, sizeof(void*)*3 + 8, v___x_320_);
lean_ctor_set_uint8(v___x_322_, sizeof(void*)*3 + 16, v_hasTrace_309_);
v___x_323_ = lean_apply_1(v_inst_256_, v_s_257_);
v___x_324_ = lean_unsigned_to_nat(1u);
v___x_325_ = lean_mk_empty_array_with_capacity(v___x_324_);
v___x_326_ = lean_array_push(v___x_325_, v_a_315_);
v___x_327_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_327_, 0, v___x_322_);
lean_ctor_set(v___x_327_, 1, v___x_323_);
lean_ctor_set(v___x_327_, 2, v___x_326_);
lean_inc_ref(v_toMonadRef_317_);
v___x_793__overap_328_ = l_Lean_addRawTrace___redArg(v___x_306_, v___x_307_, v_toMonadRef_317_, v___x_318_, v___x_327_);
lean_inc(v_a_262_);
lean_inc_ref(v_a_261_);
lean_inc(v_a_260_);
lean_inc_ref(v_a_259_);
v___x_329_ = lean_apply_5(v___x_793__overap_328_, v_a_259_, v_a_260_, v_a_261_, v_a_262_, lean_box(0));
return v___x_329_;
}
else
{
lean_object* v_a_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_337_; 
lean_dec_ref(v___x_306_);
lean_dec(v_s_257_);
lean_dec_ref(v_inst_256_);
v_a_330_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_337_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_337_ == 0)
{
v___x_332_ = v___x_314_;
v_isShared_333_ = v_isSharedCheck_337_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_a_330_);
lean_dec(v___x_314_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_337_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_335_; 
if (v_isShared_333_ == 0)
{
v___x_335_ = v___x_332_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v_a_330_);
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
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___boxed(lean_object* v_inst_344_, lean_object* v_s_345_, lean_object* v_l_346_, lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_, lean_object* v_a_350_, lean_object* v_a_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg(v_inst_344_, v_s_345_, v_l_346_, v_a_347_, v_a_348_, v_a_349_, v_a_350_);
lean_dec(v_a_350_);
lean_dec_ref(v_a_349_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs(lean_object* v_00_u03b1_353_, lean_object* v_inst_354_, lean_object* v_s_355_, lean_object* v_l_356_, lean_object* v_a_357_, lean_object* v_a_358_, lean_object* v_a_359_, lean_object* v_a_360_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg(v_inst_354_, v_s_355_, v_l_356_, v_a_357_, v_a_358_, v_a_359_, v_a_360_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___boxed(lean_object* v_00_u03b1_363_, lean_object* v_inst_364_, lean_object* v_s_365_, lean_object* v_l_366_, lean_object* v_a_367_, lean_object* v_a_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs(v_00_u03b1_363_, v_inst_364_, v_s_365_, v_l_366_, v_a_367_, v_a_368_, v_a_369_, v_a_370_);
lean_dec(v_a_370_);
lean_dec_ref(v_a_369_);
lean_dec(v_a_368_);
lean_dec_ref(v_a_367_);
return v_res_372_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = lean_unsigned_to_nat(0u);
v___x_374_ = lean_nat_to_int(v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(lean_object* v_x_375_, lean_object* v_x_376_){
_start:
{
if (lean_obj_tag(v_x_375_) == 0)
{
return v_x_376_;
}
else
{
if (lean_obj_tag(v_x_376_) == 0)
{
return v_x_375_;
}
else
{
lean_object* v_head_377_; lean_object* v_head_378_; lean_object* v_tail_379_; lean_object* v_tail_380_; lean_object* v_fst_381_; lean_object* v_snd_382_; lean_object* v_fst_383_; lean_object* v_snd_384_; uint8_t v___x_385_; 
v_head_377_ = lean_ctor_get(v_x_376_, 0);
v_head_378_ = lean_ctor_get(v_x_375_, 0);
lean_inc(v_head_378_);
v_tail_379_ = lean_ctor_get(v_x_375_, 1);
v_tail_380_ = lean_ctor_get(v_x_376_, 1);
v_fst_381_ = lean_ctor_get(v_head_377_, 0);
v_snd_382_ = lean_ctor_get(v_head_377_, 1);
v_fst_383_ = lean_ctor_get(v_head_378_, 0);
v_snd_384_ = lean_ctor_get(v_head_378_, 1);
v___x_385_ = lean_nat_dec_lt(v_fst_383_, v_fst_381_);
if (v___x_385_ == 0)
{
lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_417_; 
lean_inc(v_tail_379_);
v_isSharedCheck_417_ = !lean_is_exclusive(v_x_375_);
if (v_isSharedCheck_417_ == 0)
{
lean_object* v_unused_418_; lean_object* v_unused_419_; 
v_unused_418_ = lean_ctor_get(v_x_375_, 1);
lean_dec(v_unused_418_);
v_unused_419_ = lean_ctor_get(v_x_375_, 0);
lean_dec(v_unused_419_);
v___x_387_ = v_x_375_;
v_isShared_388_ = v_isSharedCheck_417_;
goto v_resetjp_386_;
}
else
{
lean_dec(v_x_375_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_417_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
uint8_t v___x_389_; 
v___x_389_ = lean_nat_dec_lt(v_fst_381_, v_fst_383_);
if (v___x_389_ == 0)
{
lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_410_; 
lean_inc(v_snd_384_);
lean_inc(v_fst_383_);
lean_inc(v_snd_382_);
lean_inc(v_tail_380_);
lean_del_object(v___x_387_);
v_isSharedCheck_410_ = !lean_is_exclusive(v_x_376_);
if (v_isSharedCheck_410_ == 0)
{
lean_object* v_unused_411_; lean_object* v_unused_412_; 
v_unused_411_ = lean_ctor_get(v_x_376_, 1);
lean_dec(v_unused_411_);
v_unused_412_ = lean_ctor_get(v_x_376_, 0);
lean_dec(v_unused_412_);
v___x_391_ = v_x_376_;
v_isShared_392_ = v_isSharedCheck_410_;
goto v_resetjp_390_;
}
else
{
lean_dec(v_x_376_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_410_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_407_; 
v_isSharedCheck_407_ = !lean_is_exclusive(v_head_378_);
if (v_isSharedCheck_407_ == 0)
{
lean_object* v_unused_408_; lean_object* v_unused_409_; 
v_unused_408_ = lean_ctor_get(v_head_378_, 1);
lean_dec(v_unused_408_);
v_unused_409_ = lean_ctor_get(v_head_378_, 0);
lean_dec(v_unused_409_);
v___x_394_ = v_head_378_;
v_isShared_395_ = v_isSharedCheck_407_;
goto v_resetjp_393_;
}
else
{
lean_dec(v_head_378_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_407_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v_sum_396_; lean_object* v___x_397_; uint8_t v___x_398_; 
v_sum_396_ = lean_int_add(v_snd_384_, v_snd_382_);
lean_dec(v_snd_382_);
lean_dec(v_snd_384_);
v___x_397_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0);
v___x_398_ = lean_int_dec_eq(v_sum_396_, v___x_397_);
if (v___x_398_ == 0)
{
lean_object* v___x_400_; 
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 1, v_sum_396_);
v___x_400_ = v___x_394_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_fst_383_);
lean_ctor_set(v_reuseFailAlloc_405_, 1, v_sum_396_);
v___x_400_ = v_reuseFailAlloc_405_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
lean_object* v___x_401_; lean_object* v___x_403_; 
v___x_401_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(v_tail_379_, v_tail_380_);
if (v_isShared_392_ == 0)
{
lean_ctor_set(v___x_391_, 1, v___x_401_);
lean_ctor_set(v___x_391_, 0, v___x_400_);
v___x_403_ = v___x_391_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v___x_400_);
lean_ctor_set(v_reuseFailAlloc_404_, 1, v___x_401_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
else
{
lean_dec(v_sum_396_);
lean_del_object(v___x_394_);
lean_del_object(v___x_391_);
lean_dec(v_fst_383_);
v_x_375_ = v_tail_379_;
v_x_376_ = v_tail_380_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_413_; lean_object* v___x_415_; 
v___x_413_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(v_tail_379_, v_x_376_);
if (v_isShared_388_ == 0)
{
lean_ctor_set(v___x_387_, 1, v___x_413_);
v___x_415_ = v___x_387_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_head_378_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v___x_413_);
v___x_415_ = v_reuseFailAlloc_416_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
return v___x_415_;
}
}
}
}
else
{
lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_427_; 
lean_inc(v_tail_380_);
lean_inc(v_head_377_);
lean_dec(v_head_378_);
v_isSharedCheck_427_ = !lean_is_exclusive(v_x_376_);
if (v_isSharedCheck_427_ == 0)
{
lean_object* v_unused_428_; lean_object* v_unused_429_; 
v_unused_428_ = lean_ctor_get(v_x_376_, 1);
lean_dec(v_unused_428_);
v_unused_429_ = lean_ctor_get(v_x_376_, 0);
lean_dec(v_unused_429_);
v___x_421_ = v_x_376_;
v_isShared_422_ = v_isSharedCheck_427_;
goto v_resetjp_420_;
}
else
{
lean_dec(v_x_376_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_427_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_423_; lean_object* v___x_425_; 
v___x_423_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(v_x_375_, v_tail_380_);
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 1, v___x_423_);
v___x_425_ = v___x_421_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_head_377_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v___x_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0(lean_object* v_c_430_, lean_object* v_a_431_, lean_object* v_a_432_){
_start:
{
if (lean_obj_tag(v_a_431_) == 0)
{
lean_object* v___x_433_; 
v___x_433_ = l_List_reverse___redArg(v_a_432_);
return v___x_433_;
}
else
{
lean_object* v_head_434_; lean_object* v_tail_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_453_; 
v_head_434_ = lean_ctor_get(v_a_431_, 0);
v_tail_435_ = lean_ctor_get(v_a_431_, 1);
v_isSharedCheck_453_ = !lean_is_exclusive(v_a_431_);
if (v_isSharedCheck_453_ == 0)
{
v___x_437_ = v_a_431_;
v_isShared_438_ = v_isSharedCheck_453_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_tail_435_);
lean_inc(v_head_434_);
lean_dec(v_a_431_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_453_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v_fst_439_; lean_object* v_snd_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_452_; 
v_fst_439_ = lean_ctor_get(v_head_434_, 0);
v_snd_440_ = lean_ctor_get(v_head_434_, 1);
v_isSharedCheck_452_ = !lean_is_exclusive(v_head_434_);
if (v_isSharedCheck_452_ == 0)
{
v___x_442_ = v_head_434_;
v_isShared_443_ = v_isSharedCheck_452_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_snd_440_);
lean_inc(v_fst_439_);
lean_dec(v_head_434_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_452_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_444_; lean_object* v___x_446_; 
v___x_444_ = lean_int_mul(v_snd_440_, v_c_430_);
lean_dec(v_snd_440_);
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 1, v___x_444_);
v___x_446_ = v___x_442_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_fst_439_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v___x_444_);
v___x_446_ = v_reuseFailAlloc_451_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_448_; 
if (v_isShared_438_ == 0)
{
lean_ctor_set(v___x_437_, 1, v_a_432_);
lean_ctor_set(v___x_437_, 0, v___x_446_);
v___x_448_ = v___x_437_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_446_);
lean_ctor_set(v_reuseFailAlloc_450_, 1, v_a_432_);
v___x_448_ = v_reuseFailAlloc_450_;
goto v_reusejp_447_;
}
v_reusejp_447_:
{
v_a_431_ = v_tail_435_;
v_a_432_ = v___x_448_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0___boxed(lean_object* v_c_454_, lean_object* v_a_455_, lean_object* v_a_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0(v_c_454_, v_a_455_, v_a_456_);
lean_dec(v_c_454_);
return v_res_457_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_458_ = lean_unsigned_to_nat(1u);
v___x_459_ = lean_nat_to_int(v___x_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale(lean_object* v_c_460_, lean_object* v_l_461_){
_start:
{
lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_462_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0);
v___x_463_ = lean_int_dec_eq(v_c_460_, v___x_462_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; uint8_t v___x_465_; 
v___x_464_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___closed__0);
v___x_465_ = lean_int_dec_eq(v_c_460_, v___x_464_);
if (v___x_465_ == 0)
{
lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_466_ = lean_box(0);
v___x_467_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_scale_spec__0(v_c_460_, v_l_461_, v___x_466_);
return v___x_467_;
}
else
{
return v_l_461_;
}
}
else
{
lean_object* v___x_468_; 
lean_dec(v_l_461_);
v___x_468_ = lean_box(0);
return v___x_468_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale___boxed(lean_object* v_c_469_, lean_object* v_l_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale(v_c_469_, v_l_470_);
lean_dec(v_c_469_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get(lean_object* v_n_472_, lean_object* v_x_473_){
_start:
{
if (lean_obj_tag(v_x_473_) == 0)
{
lean_object* v___x_474_; 
v___x_474_ = lean_box(0);
return v___x_474_;
}
else
{
lean_object* v_head_475_; lean_object* v_tail_476_; lean_object* v_fst_477_; lean_object* v_snd_478_; uint8_t v___x_479_; 
v_head_475_ = lean_ctor_get(v_x_473_, 0);
v_tail_476_ = lean_ctor_get(v_x_473_, 1);
v_fst_477_ = lean_ctor_get(v_head_475_, 0);
v_snd_478_ = lean_ctor_get(v_head_475_, 1);
v___x_479_ = lean_nat_dec_lt(v_fst_477_, v_n_472_);
if (v___x_479_ == 0)
{
uint8_t v___x_480_; 
v___x_480_ = lean_nat_dec_eq(v_fst_477_, v_n_472_);
if (v___x_480_ == 0)
{
v_x_473_ = v_tail_476_;
goto _start;
}
else
{
lean_object* v___x_482_; 
lean_inc(v_snd_478_);
v___x_482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_482_, 0, v_snd_478_);
return v___x_482_;
}
}
else
{
lean_object* v___x_483_; 
v___x_483_ = lean_box(0);
return v___x_483_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get___boxed(lean_object* v_n_484_, lean_object* v_x_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get(v_n_484_, v_x_485_);
lean_dec(v_x_485_);
lean_dec(v_n_484_);
return v_res_486_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_contains(lean_object* v_n_487_, lean_object* v_a_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get(v_n_487_, v_a_488_);
if (lean_obj_tag(v___x_489_) == 0)
{
uint8_t v___x_490_; 
v___x_490_ = 0;
return v___x_490_;
}
else
{
uint8_t v___x_491_; 
lean_dec_ref_known(v___x_489_, 1);
v___x_491_ = 1;
return v___x_491_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_contains___boxed(lean_object* v_n_492_, lean_object* v_a_493_){
_start:
{
uint8_t v_res_494_; lean_object* v_r_495_; 
v_res_494_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_contains(v_n_492_, v_a_493_);
lean_dec(v_a_493_);
lean_dec(v_n_492_);
v_r_495_ = lean_box(v_res_494_);
return v_r_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind(lean_object* v_n_496_, lean_object* v_l_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_get(v_n_496_, v_l_497_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v___x_499_; 
v___x_499_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0);
return v___x_499_;
}
else
{
lean_object* v_val_500_; 
v_val_500_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_val_500_);
lean_dec_ref_known(v___x_498_, 1);
return v_val_500_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind___boxed(lean_object* v_n_501_, lean_object* v_l_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind(v_n_501_, v_l_502_);
lean_dec(v_l_502_);
lean_dec(v_n_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_vars_spec__0(lean_object* v_a_504_, lean_object* v_a_505_){
_start:
{
if (lean_obj_tag(v_a_504_) == 0)
{
lean_object* v___x_506_; 
v___x_506_ = l_List_reverse___redArg(v_a_505_);
return v___x_506_;
}
else
{
lean_object* v_head_507_; lean_object* v_tail_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_517_; 
v_head_507_ = lean_ctor_get(v_a_504_, 0);
v_tail_508_ = lean_ctor_get(v_a_504_, 1);
v_isSharedCheck_517_ = !lean_is_exclusive(v_a_504_);
if (v_isSharedCheck_517_ == 0)
{
v___x_510_ = v_a_504_;
v_isShared_511_ = v_isSharedCheck_517_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_tail_508_);
lean_inc(v_head_507_);
lean_dec(v_a_504_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_517_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v_fst_512_; lean_object* v___x_514_; 
v_fst_512_ = lean_ctor_get(v_head_507_, 0);
lean_inc(v_fst_512_);
lean_dec(v_head_507_);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 1, v_a_505_);
lean_ctor_set(v___x_510_, 0, v_fst_512_);
v___x_514_ = v___x_510_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v_fst_512_);
lean_ctor_set(v_reuseFailAlloc_516_, 1, v_a_505_);
v___x_514_ = v_reuseFailAlloc_516_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
v_a_504_ = v_tail_508_;
v_a_505_ = v___x_514_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_vars(lean_object* v_l_518_){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = lean_box(0);
v___x_520_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_Linexp_vars_spec__0(v_l_518_, v___x_519_);
return v___x_520_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp(lean_object* v_x_521_, lean_object* v_x_522_){
_start:
{
if (lean_obj_tag(v_x_521_) == 0)
{
if (lean_obj_tag(v_x_522_) == 0)
{
uint8_t v___x_523_; 
v___x_523_ = 1;
return v___x_523_;
}
else
{
uint8_t v___x_524_; 
v___x_524_ = 0;
return v___x_524_;
}
}
else
{
if (lean_obj_tag(v_x_522_) == 0)
{
uint8_t v___x_525_; 
v___x_525_ = 2;
return v___x_525_;
}
else
{
lean_object* v_head_526_; lean_object* v_head_527_; lean_object* v_tail_528_; lean_object* v_tail_529_; lean_object* v_fst_530_; lean_object* v_snd_531_; lean_object* v_fst_532_; lean_object* v_snd_533_; uint8_t v___x_534_; 
v_head_526_ = lean_ctor_get(v_x_522_, 0);
v_head_527_ = lean_ctor_get(v_x_521_, 0);
v_tail_528_ = lean_ctor_get(v_x_521_, 1);
v_tail_529_ = lean_ctor_get(v_x_522_, 1);
v_fst_530_ = lean_ctor_get(v_head_526_, 0);
v_snd_531_ = lean_ctor_get(v_head_526_, 1);
v_fst_532_ = lean_ctor_get(v_head_527_, 0);
v_snd_533_ = lean_ctor_get(v_head_527_, 1);
v___x_534_ = lean_nat_dec_lt(v_fst_532_, v_fst_530_);
if (v___x_534_ == 0)
{
uint8_t v___x_535_; 
v___x_535_ = lean_nat_dec_lt(v_fst_530_, v_fst_532_);
if (v___x_535_ == 0)
{
uint8_t v___x_536_; 
v___x_536_ = lean_int_dec_lt(v_snd_533_, v_snd_531_);
if (v___x_536_ == 0)
{
uint8_t v___x_537_; 
v___x_537_ = lean_int_dec_lt(v_snd_531_, v_snd_533_);
if (v___x_537_ == 0)
{
v_x_521_ = v_tail_528_;
v_x_522_ = v_tail_529_;
goto _start;
}
else
{
uint8_t v___x_539_; 
v___x_539_ = 2;
return v___x_539_;
}
}
else
{
uint8_t v___x_540_; 
v___x_540_ = 0;
return v___x_540_;
}
}
else
{
uint8_t v___x_541_; 
v___x_541_ = 2;
return v___x_541_;
}
}
else
{
uint8_t v___x_542_; 
v___x_542_ = 0;
return v___x_542_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp___boxed(lean_object* v_x_543_, lean_object* v_x_544_){
_start:
{
uint8_t v_res_545_; lean_object* v_r_546_; 
v_res_545_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp(v_x_543_, v_x_544_);
lean_dec(v_x_544_);
lean_dec(v_x_543_);
v_r_546_ = lean_box(v_res_545_);
return v_r_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__1(lean_object* v_a_552_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lean_nat_to_int(v_a_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2_spec__3(lean_object* v_x_554_, lean_object* v_x_555_, lean_object* v_x_556_){
_start:
{
if (lean_obj_tag(v_x_556_) == 0)
{
lean_dec(v_x_554_);
return v_x_555_;
}
else
{
lean_object* v_head_557_; lean_object* v_tail_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_567_; 
v_head_557_ = lean_ctor_get(v_x_556_, 0);
v_tail_558_ = lean_ctor_get(v_x_556_, 1);
v_isSharedCheck_567_ = !lean_is_exclusive(v_x_556_);
if (v_isSharedCheck_567_ == 0)
{
v___x_560_ = v_x_556_;
v_isShared_561_ = v_isSharedCheck_567_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_tail_558_);
lean_inc(v_head_557_);
lean_dec(v_x_556_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_567_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_563_; 
lean_inc(v_x_554_);
if (v_isShared_561_ == 0)
{
lean_ctor_set_tag(v___x_560_, 5);
lean_ctor_set(v___x_560_, 1, v_x_554_);
lean_ctor_set(v___x_560_, 0, v_x_555_);
v___x_563_ = v___x_560_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v_x_555_);
lean_ctor_set(v_reuseFailAlloc_566_, 1, v_x_554_);
v___x_563_ = v_reuseFailAlloc_566_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
lean_object* v___x_564_; 
v___x_564_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_564_, 0, v___x_563_);
lean_ctor_set(v___x_564_, 1, v_head_557_);
v_x_555_ = v___x_564_;
v_x_556_ = v_tail_558_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(lean_object* v_x_568_, lean_object* v_x_569_){
_start:
{
if (lean_obj_tag(v_x_568_) == 0)
{
lean_object* v___x_570_; 
lean_dec(v_x_569_);
v___x_570_ = lean_box(0);
return v___x_570_;
}
else
{
lean_object* v_tail_571_; 
v_tail_571_ = lean_ctor_get(v_x_568_, 1);
if (lean_obj_tag(v_tail_571_) == 0)
{
lean_object* v_head_572_; 
lean_dec(v_x_569_);
v_head_572_ = lean_ctor_get(v_x_568_, 0);
lean_inc(v_head_572_);
lean_dec_ref_known(v_x_568_, 2);
return v_head_572_;
}
else
{
lean_object* v_head_573_; lean_object* v___x_574_; 
lean_inc(v_tail_571_);
v_head_573_ = lean_ctor_get(v_x_568_, 0);
lean_inc(v_head_573_);
lean_dec_ref_known(v_x_568_, 2);
v___x_574_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2_spec__3(v_x_569_, v_head_573_, v_tail_571_);
return v___x_574_;
}
}
}
}
static lean_object* _init_lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__0));
v___x_584_ = lean_string_length(v___x_583_);
return v___x_584_;
}
}
static lean_object* _init_lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_585_ = lean_obj_once(&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5, &lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5_once, _init_lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__5);
v___x_586_ = lean_nat_to_int(v___x_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(lean_object* v_x_591_){
_start:
{
lean_object* v_fst_592_; lean_object* v_snd_593_; lean_object* v___x_595_; uint8_t v_isShared_596_; uint8_t v_isSharedCheck_625_; 
v_fst_592_ = lean_ctor_get(v_x_591_, 0);
v_snd_593_ = lean_ctor_get(v_x_591_, 1);
v_isSharedCheck_625_ = !lean_is_exclusive(v_x_591_);
if (v_isSharedCheck_625_ == 0)
{
v___x_595_ = v_x_591_;
v_isShared_596_ = v_isSharedCheck_625_;
goto v_resetjp_594_;
}
else
{
lean_inc(v_snd_593_);
lean_inc(v_fst_592_);
lean_dec(v_x_591_);
v___x_595_ = lean_box(0);
v_isShared_596_ = v_isSharedCheck_625_;
goto v_resetjp_594_;
}
v_resetjp_594_:
{
lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_601_; 
v___x_597_ = l_Nat_reprFast(v_fst_592_);
v___x_598_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_598_, 0, v___x_597_);
v___x_599_ = lean_box(0);
if (v_isShared_596_ == 0)
{
lean_ctor_set_tag(v___x_595_, 1);
lean_ctor_set(v___x_595_, 1, v___x_599_);
lean_ctor_set(v___x_595_, 0, v___x_598_);
v___x_601_ = v___x_595_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v___x_598_);
lean_ctor_set(v_reuseFailAlloc_624_, 1, v___x_599_);
v___x_601_ = v_reuseFailAlloc_624_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
lean_object* v___y_603_; lean_object* v___x_616_; lean_object* v___x_617_; uint8_t v___x_618_; 
v___x_616_ = lean_unsigned_to_nat(0u);
v___x_617_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add___closed__0);
v___x_618_ = lean_int_dec_lt(v_snd_593_, v___x_617_);
if (v___x_618_ == 0)
{
lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_619_ = l_Int_repr(v_snd_593_);
lean_dec(v_snd_593_);
v___x_620_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
v___y_603_ = v___x_620_;
goto v___jp_602_;
}
else
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_621_ = l_Int_repr(v_snd_593_);
lean_dec(v_snd_593_);
v___x_622_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_622_, 0, v___x_621_);
v___x_623_ = l_Repr_addAppParen(v___x_622_, v___x_616_);
v___y_603_ = v___x_623_;
goto v___jp_602_;
}
v___jp_602_:
{
lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; uint8_t v___x_614_; lean_object* v___x_615_; 
v___x_604_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_604_, 0, v___y_603_);
lean_ctor_set(v___x_604_, 1, v___x_601_);
v___x_605_ = l_List_reverse___redArg(v___x_604_);
v___x_606_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__3));
v___x_607_ = lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(v___x_605_, v___x_606_);
v___x_608_ = lean_obj_once(&lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6, &lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6_once, _init_lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__6);
v___x_609_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__7));
v___x_610_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_610_, 0, v___x_609_);
lean_ctor_set(v___x_610_, 1, v___x_607_);
v___x_611_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__8));
v___x_612_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_612_, 0, v___x_610_);
lean_ctor_set(v___x_612_, 1, v___x_611_);
v___x_613_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_613_, 0, v___x_608_);
lean_ctor_set(v___x_613_, 1, v___x_612_);
v___x_614_ = 0;
v___x_615_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_615_, 0, v___x_613_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*1, v___x_614_);
return v___x_615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4_spec__6(lean_object* v_x_626_, lean_object* v_x_627_, lean_object* v_x_628_){
_start:
{
if (lean_obj_tag(v_x_628_) == 0)
{
lean_dec(v_x_626_);
return v_x_627_;
}
else
{
lean_object* v_head_629_; lean_object* v_tail_630_; lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_640_; 
v_head_629_ = lean_ctor_get(v_x_628_, 0);
v_tail_630_ = lean_ctor_get(v_x_628_, 1);
v_isSharedCheck_640_ = !lean_is_exclusive(v_x_628_);
if (v_isSharedCheck_640_ == 0)
{
v___x_632_ = v_x_628_;
v_isShared_633_ = v_isSharedCheck_640_;
goto v_resetjp_631_;
}
else
{
lean_inc(v_tail_630_);
lean_inc(v_head_629_);
lean_dec(v_x_628_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_640_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
lean_object* v___x_635_; 
lean_inc(v_x_626_);
if (v_isShared_633_ == 0)
{
lean_ctor_set_tag(v___x_632_, 5);
lean_ctor_set(v___x_632_, 1, v_x_626_);
lean_ctor_set(v___x_632_, 0, v_x_627_);
v___x_635_ = v___x_632_;
goto v_reusejp_634_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_x_627_);
lean_ctor_set(v_reuseFailAlloc_639_, 1, v_x_626_);
v___x_635_ = v_reuseFailAlloc_639_;
goto v_reusejp_634_;
}
v_reusejp_634_:
{
lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_636_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(v_head_629_);
v___x_637_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_637_, 0, v___x_635_);
lean_ctor_set(v___x_637_, 1, v___x_636_);
v_x_627_ = v___x_637_;
v_x_628_ = v_tail_630_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4(lean_object* v_x_641_, lean_object* v_x_642_, lean_object* v_x_643_){
_start:
{
if (lean_obj_tag(v_x_643_) == 0)
{
lean_dec(v_x_641_);
return v_x_642_;
}
else
{
lean_object* v_head_644_; lean_object* v_tail_645_; lean_object* v___x_647_; uint8_t v_isShared_648_; uint8_t v_isSharedCheck_655_; 
v_head_644_ = lean_ctor_get(v_x_643_, 0);
v_tail_645_ = lean_ctor_get(v_x_643_, 1);
v_isSharedCheck_655_ = !lean_is_exclusive(v_x_643_);
if (v_isSharedCheck_655_ == 0)
{
v___x_647_ = v_x_643_;
v_isShared_648_ = v_isSharedCheck_655_;
goto v_resetjp_646_;
}
else
{
lean_inc(v_tail_645_);
lean_inc(v_head_644_);
lean_dec(v_x_643_);
v___x_647_ = lean_box(0);
v_isShared_648_ = v_isSharedCheck_655_;
goto v_resetjp_646_;
}
v_resetjp_646_:
{
lean_object* v___x_650_; 
lean_inc(v_x_641_);
if (v_isShared_648_ == 0)
{
lean_ctor_set_tag(v___x_647_, 5);
lean_ctor_set(v___x_647_, 1, v_x_641_);
lean_ctor_set(v___x_647_, 0, v_x_642_);
v___x_650_ = v___x_647_;
goto v_reusejp_649_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_x_642_);
lean_ctor_set(v_reuseFailAlloc_654_, 1, v_x_641_);
v___x_650_ = v_reuseFailAlloc_654_;
goto v_reusejp_649_;
}
v_reusejp_649_:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_651_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(v_head_644_);
v___x_652_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_652_, 0, v___x_650_);
lean_ctor_set(v___x_652_, 1, v___x_651_);
v___x_653_ = lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4_spec__6(v_x_641_, v___x_652_, v_tail_645_);
return v___x_653_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1(lean_object* v_x_656_, lean_object* v_x_657_){
_start:
{
if (lean_obj_tag(v_x_656_) == 0)
{
lean_object* v___x_658_; 
lean_dec(v_x_657_);
v___x_658_ = lean_box(0);
return v___x_658_;
}
else
{
lean_object* v_tail_659_; 
v_tail_659_ = lean_ctor_get(v_x_656_, 1);
if (lean_obj_tag(v_tail_659_) == 0)
{
lean_object* v_head_660_; lean_object* v___x_661_; 
lean_dec(v_x_657_);
v_head_660_ = lean_ctor_get(v_x_656_, 0);
lean_inc(v_head_660_);
lean_dec_ref_known(v_x_656_, 2);
v___x_661_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(v_head_660_);
return v___x_661_;
}
else
{
lean_object* v_head_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
lean_inc(v_tail_659_);
v_head_662_ = lean_ctor_get(v_x_656_, 0);
lean_inc(v_head_662_);
lean_dec_ref_known(v_x_656_, 2);
v___x_663_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(v_head_662_);
v___x_664_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1_spec__4(v_x_657_, v___x_663_, v_tail_659_);
return v___x_664_;
}
}
}
}
static lean_object* _init_lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_670_ = ((lean_object*)(lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__2));
v___x_671_ = lean_string_length(v___x_670_);
return v___x_671_;
}
}
static lean_object* _init_lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_672_ = lean_obj_once(&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4, &lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4_once, _init_lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__4);
v___x_673_ = lean_nat_to_int(v___x_672_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg(lean_object* v_a_678_){
_start:
{
if (lean_obj_tag(v_a_678_) == 0)
{
lean_object* v___x_679_; 
v___x_679_ = ((lean_object*)(lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__1));
return v___x_679_;
}
else
{
lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; uint8_t v___x_688_; lean_object* v___x_689_; 
v___x_680_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__3));
v___x_681_ = lp_mathlib_Std_Format_joinSep___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__1(v_a_678_, v___x_680_);
v___x_682_ = lean_obj_once(&lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5, &lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5_once, _init_lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__5);
v___x_683_ = ((lean_object*)(lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__6));
v___x_684_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_684_, 0, v___x_683_);
lean_ctor_set(v___x_684_, 1, v___x_681_);
v___x_685_ = ((lean_object*)(lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg___closed__7));
v___x_686_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_686_, 0, v___x_684_);
lean_ctor_set(v___x_686_, 1, v___x_685_);
v___x_687_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_687_, 0, v___x_682_);
lean_ctor_set(v___x_687_, 1, v___x_686_);
v___x_688_ = 0;
v___x_689_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_689_, 0, v___x_687_);
lean_ctor_set_uint8(v___x_689_, sizeof(void*)*1, v___x_688_);
return v___x_689_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_703_ = lean_unsigned_to_nat(7u);
v___x_704_ = lean_nat_to_int(v___x_703_);
return v___x_704_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_708_ = lean_unsigned_to_nat(10u);
v___x_709_ = lean_nat_to_int(v___x_708_);
return v___x_709_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__0));
v___x_712_ = lean_string_length(v___x_711_);
return v___x_712_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_713_; lean_object* v___x_714_; 
v___x_713_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__12);
v___x_714_ = lean_nat_to_int(v___x_713_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg(lean_object* v_x_719_){
_start:
{
uint8_t v_str_720_; lean_object* v_coeffs_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_755_; 
v_str_720_ = lean_ctor_get_uint8(v_x_719_, sizeof(void*)*1);
v_coeffs_721_ = lean_ctor_get(v_x_719_, 0);
v_isSharedCheck_755_ = !lean_is_exclusive(v_x_719_);
if (v_isSharedCheck_755_ == 0)
{
v___x_723_ = v_x_719_;
v_isShared_724_ = v_isSharedCheck_755_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_coeffs_721_);
lean_dec(v_x_719_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_755_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; uint8_t v___x_731_; lean_object* v___x_733_; 
v___x_725_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__5));
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__6));
v___x_727_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__7);
v___x_728_ = lean_unsigned_to_nat(0u);
v___x_729_ = lp_mathlib_Mathlib_instReprIneq_repr(v_str_720_, v___x_728_);
v___x_730_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_727_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = 0;
if (v_isShared_724_ == 0)
{
lean_ctor_set_tag(v___x_723_, 6);
lean_ctor_set(v___x_723_, 0, v___x_730_);
v___x_733_ = v___x_723_;
goto v_reusejp_732_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v___x_730_);
v___x_733_ = v_reuseFailAlloc_754_;
goto v_reusejp_732_;
}
v_reusejp_732_:
{
lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
lean_ctor_set_uint8(v___x_733_, sizeof(void*)*1, v___x_731_);
v___x_734_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_726_);
lean_ctor_set(v___x_734_, 1, v___x_733_);
v___x_735_ = ((lean_object*)(lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg___closed__2));
v___x_736_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_734_);
lean_ctor_set(v___x_736_, 1, v___x_735_);
v___x_737_ = lean_box(1);
v___x_738_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_738_, 0, v___x_736_);
lean_ctor_set(v___x_738_, 1, v___x_737_);
v___x_739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__9));
v___x_740_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_738_);
lean_ctor_set(v___x_740_, 1, v___x_739_);
v___x_741_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_741_, 0, v___x_740_);
lean_ctor_set(v___x_741_, 1, v___x_725_);
v___x_742_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__10);
v___x_743_ = lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg(v_coeffs_721_);
v___x_744_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_744_, 0, v___x_742_);
lean_ctor_set(v___x_744_, 1, v___x_743_);
v___x_745_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set_uint8(v___x_745_, sizeof(void*)*1, v___x_731_);
v___x_746_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_746_, 0, v___x_741_);
lean_ctor_set(v___x_746_, 1, v___x_745_);
v___x_747_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__13);
v___x_748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__14));
v___x_749_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_749_, 0, v___x_748_);
lean_ctor_set(v___x_749_, 1, v___x_746_);
v___x_750_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg___closed__15));
v___x_751_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_749_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v___x_752_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_752_, 0, v___x_747_);
lean_ctor_set(v___x_752_, 1, v___x_751_);
v___x_753_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_753_, 0, v___x_752_);
lean_ctor_set_uint8(v___x_753_, sizeof(void*)*1, v___x_731_);
return v___x_753_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr(lean_object* v_x_756_, lean_object* v_prec_757_){
_start:
{
lean_object* v___x_758_; 
v___x_758_ = lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___redArg(v_x_756_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr___boxed(lean_object* v_x_759_, lean_object* v_prec_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_mathlib_Mathlib_Tactic_Linarith_instReprComp_repr(v_x_759_, v_prec_760_);
lean_dec(v_prec_760_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0(lean_object* v_a_762_, lean_object* v_n_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___redArg(v_a_762_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0___boxed(lean_object* v_a_765_, lean_object* v_n_766_){
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_mathlib_List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0(v_a_765_, v_n_766_);
lean_dec(v_n_766_);
return v_res_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0(lean_object* v_x_768_, lean_object* v_x_769_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___redArg(v_x_768_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0___boxed(lean_object* v_x_771_, lean_object* v_x_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib_Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0(v_x_771_, v_x_772_);
lean_dec(v_x_772_);
return v_res_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_vars(lean_object* v_a_776_){
_start:
{
lean_object* v_coeffs_777_; lean_object* v___x_778_; 
v_coeffs_777_ = lean_ctor_get(v_a_776_, 0);
lean_inc(v_coeffs_777_);
lean_dec_ref(v_a_776_);
v___x_778_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_vars(v_coeffs_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_coeffOf(lean_object* v_c_779_, lean_object* v_a_780_){
_start:
{
lean_object* v_coeffs_781_; lean_object* v___x_782_; 
v_coeffs_781_ = lean_ctor_get(v_c_779_, 0);
v___x_782_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_zfind(v_a_780_, v_coeffs_781_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_coeffOf___boxed(lean_object* v_c_783_, lean_object* v_a_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_mathlib_Mathlib_Tactic_Linarith_Comp_coeffOf(v_c_783_, v_a_784_);
lean_dec(v_a_784_);
lean_dec_ref(v_c_783_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_scale(lean_object* v_c_786_, lean_object* v_n_787_){
_start:
{
uint8_t v_str_788_; lean_object* v_coeffs_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_798_; 
v_str_788_ = lean_ctor_get_uint8(v_c_786_, sizeof(void*)*1);
v_coeffs_789_ = lean_ctor_get(v_c_786_, 0);
v_isSharedCheck_798_ = !lean_is_exclusive(v_c_786_);
if (v_isSharedCheck_798_ == 0)
{
v___x_791_ = v_c_786_;
v_isShared_792_ = v_isSharedCheck_798_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_coeffs_789_);
lean_dec(v_c_786_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_798_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_796_; 
v___x_793_ = lean_nat_to_int(v_n_787_);
v___x_794_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_scale(v___x_793_, v_coeffs_789_);
lean_dec(v___x_793_);
if (v_isShared_792_ == 0)
{
lean_ctor_set(v___x_791_, 0, v___x_794_);
v___x_796_ = v___x_791_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v___x_794_);
lean_ctor_set_uint8(v_reuseFailAlloc_797_, sizeof(void*)*1, v_str_788_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
return v___x_796_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_add(lean_object* v_c1_799_, lean_object* v_c2_800_){
_start:
{
uint8_t v_str_801_; lean_object* v_coeffs_802_; uint8_t v_str_803_; lean_object* v_coeffs_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_813_; 
v_str_801_ = lean_ctor_get_uint8(v_c1_799_, sizeof(void*)*1);
v_coeffs_802_ = lean_ctor_get(v_c1_799_, 0);
lean_inc(v_coeffs_802_);
lean_dec_ref(v_c1_799_);
v_str_803_ = lean_ctor_get_uint8(v_c2_800_, sizeof(void*)*1);
v_coeffs_804_ = lean_ctor_get(v_c2_800_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v_c2_800_);
if (v_isSharedCheck_813_ == 0)
{
v___x_806_ = v_c2_800_;
v_isShared_807_ = v_isSharedCheck_813_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_coeffs_804_);
lean_dec(v_c2_800_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_813_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
uint8_t v___x_808_; lean_object* v___x_809_; lean_object* v___x_811_; 
v___x_808_ = lp_mathlib_Mathlib_Ineq_max(v_str_801_, v_str_803_);
v___x_809_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_add(v_coeffs_802_, v_coeffs_804_);
if (v_isShared_807_ == 0)
{
lean_ctor_set(v___x_806_, 0, v___x_809_);
v___x_811_ = v___x_806_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v___x_809_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
lean_ctor_set_uint8(v___x_811_, sizeof(void*)*1, v___x_808_);
return v___x_811_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_cmp(lean_object* v_x_814_, lean_object* v_x_815_){
_start:
{
uint8_t v_str_816_; lean_object* v_coeffs_817_; uint8_t v_str_818_; lean_object* v_coeffs_819_; uint8_t v___x_820_; 
v_str_816_ = lean_ctor_get_uint8(v_x_814_, sizeof(void*)*1);
v_coeffs_817_ = lean_ctor_get(v_x_814_, 0);
v_str_818_ = lean_ctor_get_uint8(v_x_815_, sizeof(void*)*1);
v_coeffs_819_ = lean_ctor_get(v_x_815_, 0);
v___x_820_ = lp_mathlib_Mathlib_Ineq_cmp(v_str_816_, v_str_818_);
if (v___x_820_ == 1)
{
uint8_t v___x_821_; 
v___x_821_ = lp_mathlib_Mathlib_Tactic_Linarith_Linexp_cmp(v_coeffs_817_, v_coeffs_819_);
return v___x_821_;
}
else
{
return v___x_820_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_cmp___boxed(lean_object* v_x_822_, lean_object* v_x_823_){
_start:
{
uint8_t v_res_824_; lean_object* v_r_825_; 
v_res_824_ = lp_mathlib_Mathlib_Tactic_Linarith_Comp_cmp(v_x_822_, v_x_823_);
lean_dec_ref(v_x_823_);
lean_dec_ref(v_x_822_);
v_r_825_ = lean_box(v_res_824_);
return v_r_825_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Comp_isContr(lean_object* v_c_826_){
_start:
{
uint8_t v_str_827_; lean_object* v_coeffs_828_; uint8_t v___x_829_; 
v_str_827_ = lean_ctor_get_uint8(v_c_826_, sizeof(void*)*1);
v_coeffs_828_ = lean_ctor_get(v_c_826_, 0);
v___x_829_ = l_List_isEmpty___redArg(v_coeffs_828_);
if (v___x_829_ == 0)
{
return v___x_829_;
}
else
{
uint8_t v___x_830_; uint8_t v___x_831_; 
v___x_830_ = 2;
v___x_831_ = lp_mathlib_Mathlib_instDecidableEqIneq(v_str_827_, v___x_830_);
return v___x_831_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_isContr___boxed(lean_object* v_c_832_){
_start:
{
uint8_t v_res_833_; lean_object* v_r_834_; 
v_res_833_ = lp_mathlib_Mathlib_Tactic_Linarith_Comp_isContr(v_c_832_);
lean_dec_ref(v_c_832_);
v_r_834_ = lean_box(v_res_833_);
return v_r_834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0(lean_object* v___f_838_, lean_object* v_p_839_){
_start:
{
uint8_t v_str_840_; lean_object* v_coeffs_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v_str_840_ = lean_ctor_get_uint8(v_p_839_, sizeof(void*)*1);
v_coeffs_841_ = lean_ctor_get(v_p_839_, 0);
lean_inc(v_coeffs_841_);
lean_dec_ref(v_p_839_);
v___x_842_ = l_List_format___redArg(v___f_838_, v_coeffs_841_);
v___x_843_ = lp_mathlib_Mathlib_Ineq_toString(v_str_840_);
v___x_844_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_844_, 0, v___x_843_);
v___x_845_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_845_, 0, v___x_842_);
lean_ctor_set(v___x_845_, 1, v___x_844_);
v___x_846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0___closed__1));
v___x_847_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_845_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
return v___x_847_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1(void){
_start:
{
lean_object* v___f_849_; lean_object* v___x_850_; 
v___f_849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__0));
v___x_850_ = l_instToFormatOfToString___redArg(v___f_849_);
return v___x_850_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3(void){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; 
v___x_852_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__2));
v___x_853_ = l_instToFormatOfToString___redArg(v___x_852_);
return v___x_853_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4(void){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___f_856_; 
v___x_854_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3, &lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__3);
v___x_855_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__1);
v___f_856_ = lean_alloc_closure((void*)(l_instToFormatProd___redArg___lam__0), 3, 2);
lean_closure_set(v___f_856_, 0, v___x_855_);
lean_closure_set(v___f_856_, 1, v___x_854_);
return v___f_856_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5(void){
_start:
{
lean_object* v___f_857_; lean_object* v___f_858_; 
v___f_857_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__4);
v___f_858_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___lam__0), 2, 1);
lean_closure_set(v___f_858_, 0, v___f_857_);
return v___f_858_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat(void){
_start:
{
lean_object* v___f_859_; 
v___f_859_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat___closed__5);
return v___f_859_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11(void){
_start:
{
lean_object* v___x_885_; lean_object* v___x_886_; 
v___x_885_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__9));
v___x_886_ = l_Lean_mkAtom(v___x_885_);
return v___x_886_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12(void){
_start:
{
lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; 
v___x_887_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__11);
v___x_888_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4));
v___x_889_ = lean_array_push(v___x_888_, v___x_887_);
return v___x_889_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17(void){
_start:
{
lean_object* v___x_898_; lean_object* v___x_899_; 
v___x_898_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__16));
v___x_899_ = l_Lean_mkAtom(v___x_898_);
return v___x_899_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18(void){
_start:
{
lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_900_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__17);
v___x_901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4));
v___x_902_ = lean_array_push(v___x_901_, v___x_900_);
return v___x_902_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19(void){
_start:
{
lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_903_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__18);
v___x_904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__15));
v___x_905_ = lean_box(2);
v___x_906_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_906_, 0, v___x_905_);
lean_ctor_set(v___x_906_, 1, v___x_904_);
lean_ctor_set(v___x_906_, 2, v___x_903_);
return v___x_906_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20(void){
_start:
{
lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v___x_907_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__19);
v___x_908_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__12);
v___x_909_ = lean_array_push(v___x_908_, v___x_907_);
return v___x_909_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21(void){
_start:
{
lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; 
v___x_910_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__20);
v___x_911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__10));
v___x_912_ = lean_box(2);
v___x_913_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_913_, 0, v___x_912_);
lean_ctor_set(v___x_913_, 1, v___x_911_);
lean_ctor_set(v___x_913_, 2, v___x_910_);
return v___x_913_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22(void){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; 
v___x_914_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__21);
v___x_915_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4));
v___x_916_ = lean_array_push(v___x_915_, v___x_914_);
return v___x_916_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23(void){
_start:
{
lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_917_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__22);
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__8));
v___x_919_ = lean_box(2);
v___x_920_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_920_, 0, v___x_919_);
lean_ctor_set(v___x_920_, 1, v___x_918_);
lean_ctor_set(v___x_920_, 2, v___x_917_);
return v___x_920_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24(void){
_start:
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v___x_921_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__23);
v___x_922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4));
v___x_923_ = lean_array_push(v___x_922_, v___x_921_);
return v___x_923_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25(void){
_start:
{
lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_924_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__24);
v___x_925_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__6));
v___x_926_ = lean_box(2);
v___x_927_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_927_, 0, v___x_926_);
lean_ctor_set(v___x_927_, 1, v___x_925_);
lean_ctor_set(v___x_927_, 2, v___x_924_);
return v___x_927_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26(void){
_start:
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
v___x_928_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__25);
v___x_929_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__4));
v___x_930_ = lean_array_push(v___x_929_, v___x_928_);
return v___x_930_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27(void){
_start:
{
lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_931_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__26);
v___x_932_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__3));
v___x_933_ = lean_box(2);
v___x_934_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
lean_ctor_set(v___x_934_, 1, v___x_932_);
lean_ctor_set(v___x_934_, 2, v___x_931_);
return v___x_934_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam(void){
_start:
{
lean_object* v___x_935_; 
v___x_935_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27, &lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__27);
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0(lean_object* v_pp_936_, lean_object* v_x_937_, lean_object* v_x_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_){
_start:
{
if (lean_obj_tag(v_x_938_) == 0)
{
lean_object* v___x_944_; 
lean_dec_ref(v_pp_936_);
v___x_944_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_944_, 0, v_x_937_);
return v___x_944_;
}
else
{
lean_object* v_head_945_; lean_object* v_tail_946_; lean_object* v_transform_947_; lean_object* v___x_948_; 
v_head_945_ = lean_ctor_get(v_x_938_, 0);
lean_inc(v_head_945_);
v_tail_946_ = lean_ctor_get(v_x_938_, 1);
lean_inc(v_tail_946_);
lean_dec_ref_known(v_x_938_, 2);
v_transform_947_ = lean_ctor_get(v_pp_936_, 1);
lean_inc_ref(v_transform_947_);
lean_inc(v___y_942_);
lean_inc_ref(v___y_941_);
lean_inc(v___y_940_);
lean_inc_ref(v___y_939_);
v___x_948_ = lean_apply_6(v_transform_947_, v_head_945_, v___y_939_, v___y_940_, v___y_941_, v___y_942_, lean_box(0));
if (lean_obj_tag(v___x_948_) == 0)
{
lean_object* v_a_949_; lean_object* v___x_950_; 
v_a_949_ = lean_ctor_get(v___x_948_, 0);
lean_inc(v_a_949_);
lean_dec_ref_known(v___x_948_, 1);
v___x_950_ = l_List_appendTR___redArg(v_a_949_, v_x_937_);
v_x_937_ = v___x_950_;
v_x_938_ = v_tail_946_;
goto _start;
}
else
{
lean_dec(v_tail_946_);
lean_dec(v_x_937_);
lean_dec_ref(v_pp_936_);
return v___x_948_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0___boxed(lean_object* v_pp_952_, lean_object* v_x_953_, lean_object* v_x_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0(v_pp_952_, v_x_953_, v_x_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0(lean_object* v_pp_961_, lean_object* v___x_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_){
_start:
{
lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_969_ = l_List_reverse___redArg(v___y_963_);
v___x_970_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_Preprocessor_globalize_spec__0(v_pp_961_, v___x_962_, v___x_969_, v___y_964_, v___y_965_, v___y_966_, v___y_967_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0___boxed(lean_object* v_pp_971_, lean_object* v___x_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0(v_pp_971_, v___x_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize(lean_object* v_pp_980_){
_start:
{
lean_object* v_toPreprocessorBase_981_; lean_object* v___x_982_; lean_object* v___f_983_; lean_object* v___x_984_; 
v_toPreprocessorBase_981_ = lean_ctor_get(v_pp_980_, 0);
lean_inc_ref(v_toPreprocessorBase_981_);
v___x_982_ = lean_box(0);
v___f_983_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_Preprocessor_globalize___lam__0___boxed), 8, 2);
lean_closure_set(v___f_983_, 0, v_pp_980_);
lean_closure_set(v___f_983_, 1, v___x_982_);
v___x_984_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_984_, 0, v_toPreprocessorBase_981_);
lean_ctor_set(v___x_984_, 1, v___f_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0(lean_object* v_transform_985_, lean_object* v_g_986_, lean_object* v_l_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_){
_start:
{
lean_object* v___x_993_; 
lean_inc(v___y_991_);
lean_inc_ref(v___y_990_);
lean_inc(v___y_989_);
lean_inc_ref(v___y_988_);
v___x_993_ = lean_apply_6(v_transform_985_, v_l_987_, v___y_988_, v___y_989_, v___y_990_, v___y_991_, lean_box(0));
if (lean_obj_tag(v___x_993_) == 0)
{
lean_object* v_a_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1004_; 
v_a_994_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1004_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1004_ == 0)
{
v___x_996_ = v___x_993_;
v_isShared_997_ = v_isSharedCheck_1004_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_a_994_);
lean_dec(v___x_993_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1004_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1002_; 
v___x_998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_998_, 0, v_g_986_);
lean_ctor_set(v___x_998_, 1, v_a_994_);
v___x_999_ = lean_box(0);
v___x_1000_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1000_, 0, v___x_998_);
lean_ctor_set(v___x_1000_, 1, v___x_999_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 0, v___x_1000_);
v___x_1002_ = v___x_996_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v___x_1000_);
v___x_1002_ = v_reuseFailAlloc_1003_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
return v___x_1002_;
}
}
}
else
{
lean_object* v_a_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1012_; 
lean_dec(v_g_986_);
v_a_1005_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_1007_ = v___x_993_;
v_isShared_1008_ = v_isSharedCheck_1012_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_a_1005_);
lean_dec(v___x_993_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0___boxed(lean_object* v_transform_1013_, lean_object* v_g_1014_, lean_object* v_l_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v_res_1021_; 
v_res_1021_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0(v_transform_1013_, v_g_1014_, v_l_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
return v_res_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching(lean_object* v_pp_1022_){
_start:
{
lean_object* v_toPreprocessorBase_1023_; lean_object* v_transform_1024_; lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1032_; 
v_toPreprocessorBase_1023_ = lean_ctor_get(v_pp_1022_, 0);
v_transform_1024_ = lean_ctor_get(v_pp_1022_, 1);
v_isSharedCheck_1032_ = !lean_is_exclusive(v_pp_1022_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1026_ = v_pp_1022_;
v_isShared_1027_ = v_isSharedCheck_1032_;
goto v_resetjp_1025_;
}
else
{
lean_inc(v_transform_1024_);
lean_inc(v_toPreprocessorBase_1023_);
lean_dec(v_pp_1022_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1032_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
lean_object* v___f_1028_; lean_object* v___x_1030_; 
v___f_1028_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalPreprocessor_branching___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1028_, 0, v_transform_1024_);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 1, v___f_1028_);
v___x_1030_ = v___x_1026_;
goto v_reusejp_1029_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_toPreprocessorBase_1023_);
lean_ctor_set(v_reuseFailAlloc_1031_, 1, v___f_1028_);
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
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1033_ = lean_unsigned_to_nat(32u);
v___x_1034_ = lean_mk_empty_array_with_capacity(v___x_1033_);
v___x_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
return v___x_1035_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1036_ = ((size_t)5ULL);
v___x_1037_ = lean_unsigned_to_nat(0u);
v___x_1038_ = lean_unsigned_to_nat(32u);
v___x_1039_ = lean_mk_empty_array_with_capacity(v___x_1038_);
v___x_1040_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__0);
v___x_1041_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
lean_ctor_set(v___x_1041_, 1, v___x_1039_);
lean_ctor_set(v___x_1041_, 2, v___x_1037_);
lean_ctor_set(v___x_1041_, 3, v___x_1037_);
lean_ctor_set_usize(v___x_1041_, 4, v___x_1036_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg(lean_object* v___y_1042_){
_start:
{
lean_object* v___x_1044_; lean_object* v_traceState_1045_; lean_object* v_traces_1046_; lean_object* v___x_1047_; lean_object* v_traceState_1048_; lean_object* v_env_1049_; lean_object* v_nextMacroScope_1050_; lean_object* v_ngen_1051_; lean_object* v_auxDeclNGen_1052_; lean_object* v_cache_1053_; lean_object* v_messages_1054_; lean_object* v_infoState_1055_; lean_object* v_snapshotTasks_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1075_; 
v___x_1044_ = lean_st_ref_get(v___y_1042_);
v_traceState_1045_ = lean_ctor_get(v___x_1044_, 4);
lean_inc_ref(v_traceState_1045_);
lean_dec(v___x_1044_);
v_traces_1046_ = lean_ctor_get(v_traceState_1045_, 0);
lean_inc_ref(v_traces_1046_);
lean_dec_ref(v_traceState_1045_);
v___x_1047_ = lean_st_ref_take(v___y_1042_);
v_traceState_1048_ = lean_ctor_get(v___x_1047_, 4);
v_env_1049_ = lean_ctor_get(v___x_1047_, 0);
v_nextMacroScope_1050_ = lean_ctor_get(v___x_1047_, 1);
v_ngen_1051_ = lean_ctor_get(v___x_1047_, 2);
v_auxDeclNGen_1052_ = lean_ctor_get(v___x_1047_, 3);
v_cache_1053_ = lean_ctor_get(v___x_1047_, 5);
v_messages_1054_ = lean_ctor_get(v___x_1047_, 6);
v_infoState_1055_ = lean_ctor_get(v___x_1047_, 7);
v_snapshotTasks_1056_ = lean_ctor_get(v___x_1047_, 8);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1058_ = v___x_1047_;
v_isShared_1059_ = v_isSharedCheck_1075_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_snapshotTasks_1056_);
lean_inc(v_infoState_1055_);
lean_inc(v_messages_1054_);
lean_inc(v_cache_1053_);
lean_inc(v_traceState_1048_);
lean_inc(v_auxDeclNGen_1052_);
lean_inc(v_ngen_1051_);
lean_inc(v_nextMacroScope_1050_);
lean_inc(v_env_1049_);
lean_dec(v___x_1047_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1075_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
uint64_t v_tid_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1073_; 
v_tid_1060_ = lean_ctor_get_uint64(v_traceState_1048_, sizeof(void*)*1);
v_isSharedCheck_1073_ = !lean_is_exclusive(v_traceState_1048_);
if (v_isSharedCheck_1073_ == 0)
{
lean_object* v_unused_1074_; 
v_unused_1074_ = lean_ctor_get(v_traceState_1048_, 0);
lean_dec(v_unused_1074_);
v___x_1062_ = v_traceState_1048_;
v_isShared_1063_ = v_isSharedCheck_1073_;
goto v_resetjp_1061_;
}
else
{
lean_dec(v_traceState_1048_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1073_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1064_; lean_object* v___x_1066_; 
v___x_1064_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___closed__1);
if (v_isShared_1063_ == 0)
{
lean_ctor_set(v___x_1062_, 0, v___x_1064_);
v___x_1066_ = v___x_1062_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1072_; 
v_reuseFailAlloc_1072_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1072_, 0, v___x_1064_);
lean_ctor_set_uint64(v_reuseFailAlloc_1072_, sizeof(void*)*1, v_tid_1060_);
v___x_1066_ = v_reuseFailAlloc_1072_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
lean_object* v___x_1068_; 
if (v_isShared_1059_ == 0)
{
lean_ctor_set(v___x_1058_, 4, v___x_1066_);
v___x_1068_ = v___x_1058_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v_env_1049_);
lean_ctor_set(v_reuseFailAlloc_1071_, 1, v_nextMacroScope_1050_);
lean_ctor_set(v_reuseFailAlloc_1071_, 2, v_ngen_1051_);
lean_ctor_set(v_reuseFailAlloc_1071_, 3, v_auxDeclNGen_1052_);
lean_ctor_set(v_reuseFailAlloc_1071_, 4, v___x_1066_);
lean_ctor_set(v_reuseFailAlloc_1071_, 5, v_cache_1053_);
lean_ctor_set(v_reuseFailAlloc_1071_, 6, v_messages_1054_);
lean_ctor_set(v_reuseFailAlloc_1071_, 7, v_infoState_1055_);
lean_ctor_set(v_reuseFailAlloc_1071_, 8, v_snapshotTasks_1056_);
v___x_1068_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
lean_object* v___x_1069_; lean_object* v___x_1070_; 
v___x_1069_ = lean_st_ref_set(v___y_1042_, v___x_1068_);
v___x_1070_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1070_, 0, v_traces_1046_);
return v___x_1070_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg___boxed(lean_object* v___y_1076_, lean_object* v___y_1077_){
_start:
{
lean_object* v_res_1078_; 
v_res_1078_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg(v___y_1076_);
lean_dec(v___y_1076_);
return v_res_1078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0(lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg(v___y_1082_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___boxed(lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_){
_start:
{
lean_object* v_res_1090_; 
v_res_1090_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0(v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_);
lean_dec(v___y_1088_);
lean_dec_ref(v___y_1087_);
lean_dec(v___y_1086_);
lean_dec_ref(v___y_1085_);
return v_res_1090_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(lean_object* v_opts_1091_, lean_object* v_opt_1092_){
_start:
{
lean_object* v_name_1093_; lean_object* v_defValue_1094_; lean_object* v_map_1095_; lean_object* v___x_1096_; 
v_name_1093_ = lean_ctor_get(v_opt_1092_, 0);
v_defValue_1094_ = lean_ctor_get(v_opt_1092_, 1);
v_map_1095_ = lean_ctor_get(v_opts_1091_, 0);
v___x_1096_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1095_, v_name_1093_);
if (lean_obj_tag(v___x_1096_) == 0)
{
uint8_t v___x_1097_; 
v___x_1097_ = lean_unbox(v_defValue_1094_);
return v___x_1097_;
}
else
{
lean_object* v_val_1098_; 
v_val_1098_ = lean_ctor_get(v___x_1096_, 0);
lean_inc(v_val_1098_);
lean_dec_ref_known(v___x_1096_, 1);
if (lean_obj_tag(v_val_1098_) == 1)
{
uint8_t v_v_1099_; 
v_v_1099_ = lean_ctor_get_uint8(v_val_1098_, 0);
lean_dec_ref_known(v_val_1098_, 0);
return v_v_1099_;
}
else
{
uint8_t v___x_1100_; 
lean_dec(v_val_1098_);
v___x_1100_ = lean_unbox(v_defValue_1094_);
return v___x_1100_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1___boxed(lean_object* v_opts_1101_, lean_object* v_opt_1102_){
_start:
{
uint8_t v_res_1103_; lean_object* v_r_1104_; 
v_res_1103_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(v_opts_1101_, v_opt_1102_);
lean_dec_ref(v_opt_1102_);
lean_dec_ref(v_opts_1101_);
v_r_1104_ = lean_box(v_res_1103_);
return v_r_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(lean_object* v_mvarId_1105_, lean_object* v_x_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
lean_object* v___x_1112_; 
v___x_1112_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1105_, v_x_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_);
if (lean_obj_tag(v___x_1112_) == 0)
{
lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1120_; 
v_a_1113_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1115_ = v___x_1112_;
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
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
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_a_1113_);
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
lean_object* v_a_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1128_; 
v_a_1121_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1123_ = v___x_1112_;
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_a_1121_);
lean_dec(v___x_1112_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1126_; 
if (v_isShared_1124_ == 0)
{
v___x_1126_ = v___x_1123_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_a_1121_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg___boxed(lean_object* v_mvarId_1129_, lean_object* v_x_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_){
_start:
{
lean_object* v_res_1136_; 
v_res_1136_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(v_mvarId_1129_, v_x_1130_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_);
lean_dec(v___y_1134_);
lean_dec_ref(v___y_1133_);
lean_dec(v___y_1132_);
lean_dec_ref(v___y_1131_);
return v_res_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3(lean_object* v_00_u03b1_1137_, lean_object* v_mvarId_1138_, lean_object* v_x_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_){
_start:
{
lean_object* v___x_1145_; 
v___x_1145_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(v_mvarId_1138_, v_x_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
return v___x_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___boxed(lean_object* v_00_u03b1_1146_, lean_object* v_mvarId_1147_, lean_object* v_x_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_){
_start:
{
lean_object* v_res_1154_; 
v_res_1154_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3(v_00_u03b1_1146_, v_mvarId_1147_, v_x_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_);
lean_dec(v___y_1152_);
lean_dec_ref(v___y_1151_);
lean_dec(v___y_1150_);
lean_dec_ref(v___y_1149_);
return v_res_1154_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1156_; lean_object* v___x_1157_; 
v___x_1156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__0));
v___x_1157_ = l_Lean_stringToMessageData(v___x_1156_);
return v___x_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0(lean_object* v_toPreprocessorBase_1158_, lean_object* v_x_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_){
_start:
{
lean_object* v_name_1165_; lean_object* v_description_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1179_; 
v_name_1165_ = lean_ctor_get(v_toPreprocessorBase_1158_, 0);
v_description_1166_ = lean_ctor_get(v_toPreprocessorBase_1158_, 1);
v_isSharedCheck_1179_ = !lean_is_exclusive(v_toPreprocessorBase_1158_);
if (v_isSharedCheck_1179_ == 0)
{
v___x_1168_ = v_toPreprocessorBase_1158_;
v_isShared_1169_ = v_isSharedCheck_1179_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_description_1166_);
lean_inc(v_name_1165_);
lean_dec(v_toPreprocessorBase_1158_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1179_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
uint8_t v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1174_; 
v___x_1170_ = 0;
v___x_1171_ = l_Lean_MessageData_ofConstName(v_name_1165_, v___x_1170_);
v___x_1172_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___closed__1);
if (v_isShared_1169_ == 0)
{
lean_ctor_set_tag(v___x_1168_, 7);
lean_ctor_set(v___x_1168_, 1, v___x_1172_);
lean_ctor_set(v___x_1168_, 0, v___x_1171_);
v___x_1174_ = v___x_1168_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1178_; 
v_reuseFailAlloc_1178_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1178_, 0, v___x_1171_);
lean_ctor_set(v_reuseFailAlloc_1178_, 1, v___x_1172_);
v___x_1174_ = v_reuseFailAlloc_1178_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; 
v___x_1175_ = l_Lean_stringToMessageData(v_description_1166_);
v___x_1176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1176_, 0, v___x_1174_);
lean_ctor_set(v___x_1176_, 1, v___x_1175_);
v___x_1177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1176_);
return v___x_1177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___boxed(lean_object* v_toPreprocessorBase_1180_, lean_object* v_x_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_){
_start:
{
lean_object* v_res_1187_; 
v_res_1187_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0(v_toPreprocessorBase_1180_, v_x_1181_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
lean_dec(v___y_1185_);
lean_dec_ref(v___y_1184_);
lean_dec(v___y_1183_);
lean_dec_ref(v___y_1182_);
lean_dec_ref(v_x_1181_);
return v_res_1187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(lean_object* v_msgData_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v___x_1194_; lean_object* v_env_1195_; lean_object* v___x_1196_; lean_object* v_mctx_1197_; lean_object* v_lctx_1198_; lean_object* v_options_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1194_ = lean_st_ref_get(v___y_1192_);
v_env_1195_ = lean_ctor_get(v___x_1194_, 0);
lean_inc_ref(v_env_1195_);
lean_dec(v___x_1194_);
v___x_1196_ = lean_st_ref_get(v___y_1190_);
v_mctx_1197_ = lean_ctor_get(v___x_1196_, 0);
lean_inc_ref(v_mctx_1197_);
lean_dec(v___x_1196_);
v_lctx_1198_ = lean_ctor_get(v___y_1189_, 2);
v_options_1199_ = lean_ctor_get(v___y_1191_, 2);
lean_inc_ref(v_options_1199_);
lean_inc_ref(v_lctx_1198_);
v___x_1200_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1200_, 0, v_env_1195_);
lean_ctor_set(v___x_1200_, 1, v_mctx_1197_);
lean_ctor_set(v___x_1200_, 2, v_lctx_1198_);
lean_ctor_set(v___x_1200_, 3, v_options_1199_);
v___x_1201_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
lean_ctor_set(v___x_1201_, 1, v_msgData_1188_);
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
return v___x_1202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8___boxed(lean_object* v_msgData_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_){
_start:
{
lean_object* v_res_1209_; 
v_res_1209_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(v_msgData_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_);
lean_dec(v___y_1207_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
return v_res_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(lean_object* v_cls_1212_, lean_object* v_msg_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_){
_start:
{
lean_object* v_ref_1219_; lean_object* v___x_1220_; lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1265_; 
v_ref_1219_ = lean_ctor_get(v___y_1216_, 5);
v___x_1220_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(v_msg_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_);
v_a_1221_ = lean_ctor_get(v___x_1220_, 0);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1220_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1223_ = v___x_1220_;
v_isShared_1224_ = v_isSharedCheck_1265_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1220_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1265_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v___x_1225_; lean_object* v_traceState_1226_; lean_object* v_env_1227_; lean_object* v_nextMacroScope_1228_; lean_object* v_ngen_1229_; lean_object* v_auxDeclNGen_1230_; lean_object* v_cache_1231_; lean_object* v_messages_1232_; lean_object* v_infoState_1233_; lean_object* v_snapshotTasks_1234_; lean_object* v___x_1236_; uint8_t v_isShared_1237_; uint8_t v_isSharedCheck_1264_; 
v___x_1225_ = lean_st_ref_take(v___y_1217_);
v_traceState_1226_ = lean_ctor_get(v___x_1225_, 4);
v_env_1227_ = lean_ctor_get(v___x_1225_, 0);
v_nextMacroScope_1228_ = lean_ctor_get(v___x_1225_, 1);
v_ngen_1229_ = lean_ctor_get(v___x_1225_, 2);
v_auxDeclNGen_1230_ = lean_ctor_get(v___x_1225_, 3);
v_cache_1231_ = lean_ctor_get(v___x_1225_, 5);
v_messages_1232_ = lean_ctor_get(v___x_1225_, 6);
v_infoState_1233_ = lean_ctor_get(v___x_1225_, 7);
v_snapshotTasks_1234_ = lean_ctor_get(v___x_1225_, 8);
v_isSharedCheck_1264_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1236_ = v___x_1225_;
v_isShared_1237_ = v_isSharedCheck_1264_;
goto v_resetjp_1235_;
}
else
{
lean_inc(v_snapshotTasks_1234_);
lean_inc(v_infoState_1233_);
lean_inc(v_messages_1232_);
lean_inc(v_cache_1231_);
lean_inc(v_traceState_1226_);
lean_inc(v_auxDeclNGen_1230_);
lean_inc(v_ngen_1229_);
lean_inc(v_nextMacroScope_1228_);
lean_inc(v_env_1227_);
lean_dec(v___x_1225_);
v___x_1236_ = lean_box(0);
v_isShared_1237_ = v_isSharedCheck_1264_;
goto v_resetjp_1235_;
}
v_resetjp_1235_:
{
uint64_t v_tid_1238_; lean_object* v_traces_1239_; lean_object* v___x_1241_; uint8_t v_isShared_1242_; uint8_t v_isSharedCheck_1263_; 
v_tid_1238_ = lean_ctor_get_uint64(v_traceState_1226_, sizeof(void*)*1);
v_traces_1239_ = lean_ctor_get(v_traceState_1226_, 0);
v_isSharedCheck_1263_ = !lean_is_exclusive(v_traceState_1226_);
if (v_isSharedCheck_1263_ == 0)
{
v___x_1241_ = v_traceState_1226_;
v_isShared_1242_ = v_isSharedCheck_1263_;
goto v_resetjp_1240_;
}
else
{
lean_inc(v_traces_1239_);
lean_dec(v_traceState_1226_);
v___x_1241_ = lean_box(0);
v_isShared_1242_ = v_isSharedCheck_1263_;
goto v_resetjp_1240_;
}
v_resetjp_1240_:
{
lean_object* v___x_1243_; double v___x_1244_; uint8_t v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1253_; 
v___x_1243_ = lean_box(0);
v___x_1244_ = lean_float_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17);
v___x_1245_ = 0;
v___x_1246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18));
v___x_1247_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1247_, 0, v_cls_1212_);
lean_ctor_set(v___x_1247_, 1, v___x_1243_);
lean_ctor_set(v___x_1247_, 2, v___x_1246_);
lean_ctor_set_float(v___x_1247_, sizeof(void*)*3, v___x_1244_);
lean_ctor_set_float(v___x_1247_, sizeof(void*)*3 + 8, v___x_1244_);
lean_ctor_set_uint8(v___x_1247_, sizeof(void*)*3 + 16, v___x_1245_);
v___x_1248_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___closed__0));
v___x_1249_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1247_);
lean_ctor_set(v___x_1249_, 1, v_a_1221_);
lean_ctor_set(v___x_1249_, 2, v___x_1248_);
lean_inc(v_ref_1219_);
v___x_1250_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1250_, 0, v_ref_1219_);
lean_ctor_set(v___x_1250_, 1, v___x_1249_);
v___x_1251_ = l_Lean_PersistentArray_push___redArg(v_traces_1239_, v___x_1250_);
if (v_isShared_1242_ == 0)
{
lean_ctor_set(v___x_1241_, 0, v___x_1251_);
v___x_1253_ = v___x_1241_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v___x_1251_);
lean_ctor_set_uint64(v_reuseFailAlloc_1262_, sizeof(void*)*1, v_tid_1238_);
v___x_1253_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
lean_object* v___x_1255_; 
if (v_isShared_1237_ == 0)
{
lean_ctor_set(v___x_1236_, 4, v___x_1253_);
v___x_1255_ = v___x_1236_;
goto v_reusejp_1254_;
}
else
{
lean_object* v_reuseFailAlloc_1261_; 
v_reuseFailAlloc_1261_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1261_, 0, v_env_1227_);
lean_ctor_set(v_reuseFailAlloc_1261_, 1, v_nextMacroScope_1228_);
lean_ctor_set(v_reuseFailAlloc_1261_, 2, v_ngen_1229_);
lean_ctor_set(v_reuseFailAlloc_1261_, 3, v_auxDeclNGen_1230_);
lean_ctor_set(v_reuseFailAlloc_1261_, 4, v___x_1253_);
lean_ctor_set(v_reuseFailAlloc_1261_, 5, v_cache_1231_);
lean_ctor_set(v_reuseFailAlloc_1261_, 6, v_messages_1232_);
lean_ctor_set(v_reuseFailAlloc_1261_, 7, v_infoState_1233_);
lean_ctor_set(v_reuseFailAlloc_1261_, 8, v_snapshotTasks_1234_);
v___x_1255_ = v_reuseFailAlloc_1261_;
goto v_reusejp_1254_;
}
v_reusejp_1254_:
{
lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1259_; 
v___x_1256_ = lean_st_ref_set(v___y_1217_, v___x_1255_);
v___x_1257_ = lean_box(0);
if (v_isShared_1224_ == 0)
{
lean_ctor_set(v___x_1223_, 0, v___x_1257_);
v___x_1259_ = v___x_1223_;
goto v_reusejp_1258_;
}
else
{
lean_object* v_reuseFailAlloc_1260_; 
v_reuseFailAlloc_1260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1260_, 0, v___x_1257_);
v___x_1259_ = v_reuseFailAlloc_1260_;
goto v_reusejp_1258_;
}
v_reusejp_1258_:
{
return v___x_1259_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4___boxed(lean_object* v_cls_1266_, lean_object* v_msg_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_){
_start:
{
lean_object* v_res_1273_; 
v_res_1273_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(v_cls_1266_, v_msg_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_);
lean_dec(v___y_1271_);
lean_dec_ref(v___y_1270_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
return v_res_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(lean_object* v_as_x27_1274_, lean_object* v_b_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
if (lean_obj_tag(v_as_x27_1274_) == 0)
{
lean_object* v___x_1281_; 
v___x_1281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1281_, 0, v_b_1275_);
return v___x_1281_;
}
else
{
lean_object* v_head_1282_; lean_object* v_options_1283_; lean_object* v_tail_1284_; lean_object* v_fst_1285_; lean_object* v_snd_1286_; lean_object* v_inheritedTraceOptions_1287_; uint8_t v_hasTrace_1288_; lean_object* v___x_1289_; 
v_head_1282_ = lean_ctor_get(v_as_x27_1274_, 0);
v_options_1283_ = lean_ctor_get(v___y_1278_, 2);
v_tail_1284_ = lean_ctor_get(v_as_x27_1274_, 1);
v_fst_1285_ = lean_ctor_get(v_head_1282_, 0);
v_snd_1286_ = lean_ctor_get(v_head_1282_, 1);
v_inheritedTraceOptions_1287_ = lean_ctor_get(v___y_1278_, 13);
v_hasTrace_1288_ = lean_ctor_get_uint8(v_options_1283_, sizeof(void*)*1);
v___x_1289_ = lean_box(0);
if (v_hasTrace_1288_ == 0)
{
v_as_x27_1274_ = v_tail_1284_;
v_b_1275_ = v___x_1289_;
goto _start;
}
else
{
lean_object* v_cls_1291_; lean_object* v___x_1292_; uint8_t v___x_1293_; 
v_cls_1291_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_1292_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__12);
v___x_1293_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1287_, v_options_1283_, v___x_1292_);
if (v___x_1293_ == 0)
{
v_as_x27_1274_ = v_tail_1284_;
v_b_1275_ = v___x_1289_;
goto _start;
}
else
{
lean_object* v___x_1295_; lean_object* v___x_1296_; 
lean_inc(v_snd_1286_);
v___x_1295_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithGetProofsMessage___boxed), 6, 1);
lean_closure_set(v___x_1295_, 0, v_snd_1286_);
lean_inc(v_fst_1285_);
v___x_1296_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(v_fst_1285_, v___x_1295_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1279_);
if (lean_obj_tag(v___x_1296_) == 0)
{
lean_object* v_a_1297_; lean_object* v___x_1298_; 
v_a_1297_ = lean_ctor_get(v___x_1296_, 0);
lean_inc(v_a_1297_);
lean_dec_ref_known(v___x_1296_, 1);
v___x_1298_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(v_cls_1291_, v_a_1297_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1279_);
if (lean_obj_tag(v___x_1298_) == 0)
{
lean_dec_ref_known(v___x_1298_, 1);
v_as_x27_1274_ = v_tail_1284_;
v_b_1275_ = v___x_1289_;
goto _start;
}
else
{
return v___x_1298_;
}
}
else
{
lean_object* v_a_1300_; lean_object* v___x_1302_; uint8_t v_isShared_1303_; uint8_t v_isSharedCheck_1307_; 
v_a_1300_ = lean_ctor_get(v___x_1296_, 0);
v_isSharedCheck_1307_ = !lean_is_exclusive(v___x_1296_);
if (v_isSharedCheck_1307_ == 0)
{
v___x_1302_ = v___x_1296_;
v_isShared_1303_ = v_isSharedCheck_1307_;
goto v_resetjp_1301_;
}
else
{
lean_inc(v_a_1300_);
lean_dec(v___x_1296_);
v___x_1302_ = lean_box(0);
v_isShared_1303_ = v_isSharedCheck_1307_;
goto v_resetjp_1301_;
}
v_resetjp_1301_:
{
lean_object* v___x_1305_; 
if (v_isShared_1303_ == 0)
{
v___x_1305_ = v___x_1302_;
goto v_reusejp_1304_;
}
else
{
lean_object* v_reuseFailAlloc_1306_; 
v_reuseFailAlloc_1306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1306_, 0, v_a_1300_);
v___x_1305_ = v_reuseFailAlloc_1306_;
goto v_reusejp_1304_;
}
v_reusejp_1304_:
{
return v___x_1305_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg___boxed(lean_object* v_as_x27_1308_, lean_object* v_b_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(v_as_x27_1308_, v_b_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
lean_dec(v___y_1311_);
lean_dec_ref(v___y_1310_);
lean_dec(v_as_x27_1308_);
return v_res_1315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(lean_object* v_a_1316_, lean_object* v_____r_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1323_ = lean_box(0);
v___x_1324_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(v_a_1316_, v___x_1323_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
if (lean_obj_tag(v___x_1324_) == 0)
{
lean_object* v___x_1326_; uint8_t v_isShared_1327_; uint8_t v_isSharedCheck_1331_; 
v_isSharedCheck_1331_ = !lean_is_exclusive(v___x_1324_);
if (v_isSharedCheck_1331_ == 0)
{
lean_object* v_unused_1332_; 
v_unused_1332_ = lean_ctor_get(v___x_1324_, 0);
lean_dec(v_unused_1332_);
v___x_1326_ = v___x_1324_;
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
else
{
lean_dec(v___x_1324_);
v___x_1326_ = lean_box(0);
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
v_resetjp_1325_:
{
lean_object* v___x_1329_; 
if (v_isShared_1327_ == 0)
{
lean_ctor_set(v___x_1326_, 0, v_a_1316_);
v___x_1329_ = v___x_1326_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1330_; 
v_reuseFailAlloc_1330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1330_, 0, v_a_1316_);
v___x_1329_ = v_reuseFailAlloc_1330_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
return v___x_1329_;
}
}
}
else
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1340_; 
lean_dec(v_a_1316_);
v_a_1333_ = lean_ctor_get(v___x_1324_, 0);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1324_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1335_ = v___x_1324_;
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1324_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1338_; 
if (v_isShared_1336_ == 0)
{
v___x_1338_ = v___x_1335_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_a_1333_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1___boxed(lean_object* v_a_1341_, lean_object* v_____r_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_){
_start:
{
lean_object* v_res_1348_; 
v_res_1348_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1341_, v_____r_1342_, v___y_1343_, v___y_1344_, v___y_1345_, v___y_1346_);
lean_dec(v___y_1346_);
lean_dec_ref(v___y_1345_);
lean_dec(v___y_1344_);
lean_dec_ref(v___y_1343_);
return v_res_1348_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4(lean_object* v_e_1349_){
_start:
{
if (lean_obj_tag(v_e_1349_) == 0)
{
uint8_t v___x_1350_; 
v___x_1350_ = 2;
return v___x_1350_;
}
else
{
uint8_t v___x_1351_; 
v___x_1351_ = 0;
return v___x_1351_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4___boxed(lean_object* v_e_1352_){
_start:
{
uint8_t v_res_1353_; lean_object* v_r_1354_; 
v_res_1353_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4(v_e_1352_);
lean_dec_ref(v_e_1352_);
v_r_1354_ = lean_box(v_res_1353_);
return v_r_1354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5(lean_object* v_opts_1355_, lean_object* v_opt_1356_){
_start:
{
lean_object* v_name_1357_; lean_object* v_defValue_1358_; lean_object* v_map_1359_; lean_object* v___x_1360_; 
v_name_1357_ = lean_ctor_get(v_opt_1356_, 0);
v_defValue_1358_ = lean_ctor_get(v_opt_1356_, 1);
v_map_1359_ = lean_ctor_get(v_opts_1355_, 0);
v___x_1360_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1359_, v_name_1357_);
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_inc(v_defValue_1358_);
return v_defValue_1358_;
}
else
{
lean_object* v_val_1361_; 
v_val_1361_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_val_1361_);
lean_dec_ref_known(v___x_1360_, 1);
if (lean_obj_tag(v_val_1361_) == 3)
{
lean_object* v_v_1362_; 
v_v_1362_ = lean_ctor_get(v_val_1361_, 0);
lean_inc(v_v_1362_);
lean_dec_ref_known(v_val_1361_, 1);
return v_v_1362_;
}
else
{
lean_dec(v_val_1361_);
lean_inc(v_defValue_1358_);
return v_defValue_1358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5___boxed(lean_object* v_opts_1363_, lean_object* v_opt_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5(v_opts_1363_, v_opt_1364_);
lean_dec_ref(v_opt_1364_);
lean_dec_ref(v_opts_1363_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4(size_t v_sz_1366_, size_t v_i_1367_, lean_object* v_bs_1368_){
_start:
{
uint8_t v___x_1369_; 
v___x_1369_ = lean_usize_dec_lt(v_i_1367_, v_sz_1366_);
if (v___x_1369_ == 0)
{
return v_bs_1368_;
}
else
{
lean_object* v_v_1370_; lean_object* v_msg_1371_; lean_object* v___x_1372_; lean_object* v_bs_x27_1373_; size_t v___x_1374_; size_t v___x_1375_; lean_object* v___x_1376_; 
v_v_1370_ = lean_array_uget_borrowed(v_bs_1368_, v_i_1367_);
v_msg_1371_ = lean_ctor_get(v_v_1370_, 1);
lean_inc_ref(v_msg_1371_);
v___x_1372_ = lean_unsigned_to_nat(0u);
v_bs_x27_1373_ = lean_array_uset(v_bs_1368_, v_i_1367_, v___x_1372_);
v___x_1374_ = ((size_t)1ULL);
v___x_1375_ = lean_usize_add(v_i_1367_, v___x_1374_);
v___x_1376_ = lean_array_uset(v_bs_x27_1373_, v_i_1367_, v_msg_1371_);
v_i_1367_ = v___x_1375_;
v_bs_1368_ = v___x_1376_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4___boxed(lean_object* v_sz_1378_, lean_object* v_i_1379_, lean_object* v_bs_1380_){
_start:
{
size_t v_sz_boxed_1381_; size_t v_i_boxed_1382_; lean_object* v_res_1383_; 
v_sz_boxed_1381_ = lean_unbox_usize(v_sz_1378_);
lean_dec(v_sz_1378_);
v_i_boxed_1382_ = lean_unbox_usize(v_i_1379_);
lean_dec(v_i_1379_);
v_res_1383_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4(v_sz_boxed_1381_, v_i_boxed_1382_, v_bs_1380_);
return v_res_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2(lean_object* v_oldTraces_1384_, lean_object* v_data_1385_, lean_object* v_ref_1386_, lean_object* v_msg_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_){
_start:
{
lean_object* v_fileName_1393_; lean_object* v_fileMap_1394_; lean_object* v_options_1395_; lean_object* v_currRecDepth_1396_; lean_object* v_maxRecDepth_1397_; lean_object* v_ref_1398_; lean_object* v_currNamespace_1399_; lean_object* v_openDecls_1400_; lean_object* v_initHeartbeats_1401_; lean_object* v_maxHeartbeats_1402_; lean_object* v_quotContext_1403_; lean_object* v_currMacroScope_1404_; uint8_t v_diag_1405_; lean_object* v_cancelTk_x3f_1406_; uint8_t v_suppressElabErrors_1407_; lean_object* v_inheritedTraceOptions_1408_; lean_object* v___x_1409_; lean_object* v_traceState_1410_; lean_object* v_traces_1411_; lean_object* v_ref_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; size_t v_sz_1415_; size_t v___x_1416_; lean_object* v___x_1417_; lean_object* v_msg_1418_; lean_object* v___x_1419_; lean_object* v_a_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1457_; 
v_fileName_1393_ = lean_ctor_get(v___y_1390_, 0);
v_fileMap_1394_ = lean_ctor_get(v___y_1390_, 1);
v_options_1395_ = lean_ctor_get(v___y_1390_, 2);
v_currRecDepth_1396_ = lean_ctor_get(v___y_1390_, 3);
v_maxRecDepth_1397_ = lean_ctor_get(v___y_1390_, 4);
v_ref_1398_ = lean_ctor_get(v___y_1390_, 5);
v_currNamespace_1399_ = lean_ctor_get(v___y_1390_, 6);
v_openDecls_1400_ = lean_ctor_get(v___y_1390_, 7);
v_initHeartbeats_1401_ = lean_ctor_get(v___y_1390_, 8);
v_maxHeartbeats_1402_ = lean_ctor_get(v___y_1390_, 9);
v_quotContext_1403_ = lean_ctor_get(v___y_1390_, 10);
v_currMacroScope_1404_ = lean_ctor_get(v___y_1390_, 11);
v_diag_1405_ = lean_ctor_get_uint8(v___y_1390_, sizeof(void*)*14);
v_cancelTk_x3f_1406_ = lean_ctor_get(v___y_1390_, 12);
v_suppressElabErrors_1407_ = lean_ctor_get_uint8(v___y_1390_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1408_ = lean_ctor_get(v___y_1390_, 13);
v___x_1409_ = lean_st_ref_get(v___y_1391_);
v_traceState_1410_ = lean_ctor_get(v___x_1409_, 4);
lean_inc_ref(v_traceState_1410_);
lean_dec(v___x_1409_);
v_traces_1411_ = lean_ctor_get(v_traceState_1410_, 0);
lean_inc_ref(v_traces_1411_);
lean_dec_ref(v_traceState_1410_);
v_ref_1412_ = l_Lean_replaceRef(v_ref_1386_, v_ref_1398_);
lean_inc_ref(v_inheritedTraceOptions_1408_);
lean_inc(v_cancelTk_x3f_1406_);
lean_inc(v_currMacroScope_1404_);
lean_inc(v_quotContext_1403_);
lean_inc(v_maxHeartbeats_1402_);
lean_inc(v_initHeartbeats_1401_);
lean_inc(v_openDecls_1400_);
lean_inc(v_currNamespace_1399_);
lean_inc(v_maxRecDepth_1397_);
lean_inc(v_currRecDepth_1396_);
lean_inc_ref(v_options_1395_);
lean_inc_ref(v_fileMap_1394_);
lean_inc_ref(v_fileName_1393_);
v___x_1413_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1413_, 0, v_fileName_1393_);
lean_ctor_set(v___x_1413_, 1, v_fileMap_1394_);
lean_ctor_set(v___x_1413_, 2, v_options_1395_);
lean_ctor_set(v___x_1413_, 3, v_currRecDepth_1396_);
lean_ctor_set(v___x_1413_, 4, v_maxRecDepth_1397_);
lean_ctor_set(v___x_1413_, 5, v_ref_1412_);
lean_ctor_set(v___x_1413_, 6, v_currNamespace_1399_);
lean_ctor_set(v___x_1413_, 7, v_openDecls_1400_);
lean_ctor_set(v___x_1413_, 8, v_initHeartbeats_1401_);
lean_ctor_set(v___x_1413_, 9, v_maxHeartbeats_1402_);
lean_ctor_set(v___x_1413_, 10, v_quotContext_1403_);
lean_ctor_set(v___x_1413_, 11, v_currMacroScope_1404_);
lean_ctor_set(v___x_1413_, 12, v_cancelTk_x3f_1406_);
lean_ctor_set(v___x_1413_, 13, v_inheritedTraceOptions_1408_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*14, v_diag_1405_);
lean_ctor_set_uint8(v___x_1413_, sizeof(void*)*14 + 1, v_suppressElabErrors_1407_);
v___x_1414_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1411_);
lean_dec_ref(v_traces_1411_);
v_sz_1415_ = lean_array_size(v___x_1414_);
v___x_1416_ = ((size_t)0ULL);
v___x_1417_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2_spec__4(v_sz_1415_, v___x_1416_, v___x_1414_);
v_msg_1418_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1418_, 0, v_data_1385_);
lean_ctor_set(v_msg_1418_, 1, v_msg_1387_);
lean_ctor_set(v_msg_1418_, 2, v___x_1417_);
v___x_1419_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(v_msg_1418_, v___y_1388_, v___y_1389_, v___x_1413_, v___y_1391_);
lean_dec_ref_known(v___x_1413_, 14);
v_a_1420_ = lean_ctor_get(v___x_1419_, 0);
v_isSharedCheck_1457_ = !lean_is_exclusive(v___x_1419_);
if (v_isSharedCheck_1457_ == 0)
{
v___x_1422_ = v___x_1419_;
v_isShared_1423_ = v_isSharedCheck_1457_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_a_1420_);
lean_dec(v___x_1419_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1457_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1424_; lean_object* v_traceState_1425_; lean_object* v_env_1426_; lean_object* v_nextMacroScope_1427_; lean_object* v_ngen_1428_; lean_object* v_auxDeclNGen_1429_; lean_object* v_cache_1430_; lean_object* v_messages_1431_; lean_object* v_infoState_1432_; lean_object* v_snapshotTasks_1433_; lean_object* v___x_1435_; uint8_t v_isShared_1436_; uint8_t v_isSharedCheck_1456_; 
v___x_1424_ = lean_st_ref_take(v___y_1391_);
v_traceState_1425_ = lean_ctor_get(v___x_1424_, 4);
v_env_1426_ = lean_ctor_get(v___x_1424_, 0);
v_nextMacroScope_1427_ = lean_ctor_get(v___x_1424_, 1);
v_ngen_1428_ = lean_ctor_get(v___x_1424_, 2);
v_auxDeclNGen_1429_ = lean_ctor_get(v___x_1424_, 3);
v_cache_1430_ = lean_ctor_get(v___x_1424_, 5);
v_messages_1431_ = lean_ctor_get(v___x_1424_, 6);
v_infoState_1432_ = lean_ctor_get(v___x_1424_, 7);
v_snapshotTasks_1433_ = lean_ctor_get(v___x_1424_, 8);
v_isSharedCheck_1456_ = !lean_is_exclusive(v___x_1424_);
if (v_isSharedCheck_1456_ == 0)
{
v___x_1435_ = v___x_1424_;
v_isShared_1436_ = v_isSharedCheck_1456_;
goto v_resetjp_1434_;
}
else
{
lean_inc(v_snapshotTasks_1433_);
lean_inc(v_infoState_1432_);
lean_inc(v_messages_1431_);
lean_inc(v_cache_1430_);
lean_inc(v_traceState_1425_);
lean_inc(v_auxDeclNGen_1429_);
lean_inc(v_ngen_1428_);
lean_inc(v_nextMacroScope_1427_);
lean_inc(v_env_1426_);
lean_dec(v___x_1424_);
v___x_1435_ = lean_box(0);
v_isShared_1436_ = v_isSharedCheck_1456_;
goto v_resetjp_1434_;
}
v_resetjp_1434_:
{
uint64_t v_tid_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1454_; 
v_tid_1437_ = lean_ctor_get_uint64(v_traceState_1425_, sizeof(void*)*1);
v_isSharedCheck_1454_ = !lean_is_exclusive(v_traceState_1425_);
if (v_isSharedCheck_1454_ == 0)
{
lean_object* v_unused_1455_; 
v_unused_1455_ = lean_ctor_get(v_traceState_1425_, 0);
lean_dec(v_unused_1455_);
v___x_1439_ = v_traceState_1425_;
v_isShared_1440_ = v_isSharedCheck_1454_;
goto v_resetjp_1438_;
}
else
{
lean_dec(v_traceState_1425_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1454_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1444_; 
v___x_1441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1441_, 0, v_ref_1386_);
lean_ctor_set(v___x_1441_, 1, v_a_1420_);
v___x_1442_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1384_, v___x_1441_);
if (v_isShared_1440_ == 0)
{
lean_ctor_set(v___x_1439_, 0, v___x_1442_);
v___x_1444_ = v___x_1439_;
goto v_reusejp_1443_;
}
else
{
lean_object* v_reuseFailAlloc_1453_; 
v_reuseFailAlloc_1453_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1453_, 0, v___x_1442_);
lean_ctor_set_uint64(v_reuseFailAlloc_1453_, sizeof(void*)*1, v_tid_1437_);
v___x_1444_ = v_reuseFailAlloc_1453_;
goto v_reusejp_1443_;
}
v_reusejp_1443_:
{
lean_object* v___x_1446_; 
if (v_isShared_1436_ == 0)
{
lean_ctor_set(v___x_1435_, 4, v___x_1444_);
v___x_1446_ = v___x_1435_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_env_1426_);
lean_ctor_set(v_reuseFailAlloc_1452_, 1, v_nextMacroScope_1427_);
lean_ctor_set(v_reuseFailAlloc_1452_, 2, v_ngen_1428_);
lean_ctor_set(v_reuseFailAlloc_1452_, 3, v_auxDeclNGen_1429_);
lean_ctor_set(v_reuseFailAlloc_1452_, 4, v___x_1444_);
lean_ctor_set(v_reuseFailAlloc_1452_, 5, v_cache_1430_);
lean_ctor_set(v_reuseFailAlloc_1452_, 6, v_messages_1431_);
lean_ctor_set(v_reuseFailAlloc_1452_, 7, v_infoState_1432_);
lean_ctor_set(v_reuseFailAlloc_1452_, 8, v_snapshotTasks_1433_);
v___x_1446_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1450_; 
v___x_1447_ = lean_st_ref_set(v___y_1391_, v___x_1446_);
v___x_1448_ = lean_box(0);
if (v_isShared_1423_ == 0)
{
lean_ctor_set(v___x_1422_, 0, v___x_1448_);
v___x_1450_ = v___x_1422_;
goto v_reusejp_1449_;
}
else
{
lean_object* v_reuseFailAlloc_1451_; 
v_reuseFailAlloc_1451_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1451_, 0, v___x_1448_);
v___x_1450_ = v_reuseFailAlloc_1451_;
goto v_reusejp_1449_;
}
v_reusejp_1449_:
{
return v___x_1450_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2___boxed(lean_object* v_oldTraces_1458_, lean_object* v_data_1459_, lean_object* v_ref_1460_, lean_object* v_msg_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
lean_object* v_res_1467_; 
v_res_1467_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2(v_oldTraces_1458_, v_data_1459_, v_ref_1460_, v_msg_1461_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_);
lean_dec(v___y_1465_);
lean_dec_ref(v___y_1464_);
lean_dec(v___y_1463_);
lean_dec_ref(v___y_1462_);
return v_res_1467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(lean_object* v_x_1468_){
_start:
{
if (lean_obj_tag(v_x_1468_) == 0)
{
lean_object* v_a_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1477_; 
v_a_1470_ = lean_ctor_get(v_x_1468_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v_x_1468_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1472_ = v_x_1468_;
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_a_1470_);
lean_dec(v_x_1468_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1475_; 
if (v_isShared_1473_ == 0)
{
lean_ctor_set_tag(v___x_1472_, 1);
v___x_1475_ = v___x_1472_;
goto v_reusejp_1474_;
}
else
{
lean_object* v_reuseFailAlloc_1476_; 
v_reuseFailAlloc_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1476_, 0, v_a_1470_);
v___x_1475_ = v_reuseFailAlloc_1476_;
goto v_reusejp_1474_;
}
v_reusejp_1474_:
{
return v___x_1475_;
}
}
}
else
{
lean_object* v_a_1478_; lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1485_; 
v_a_1478_ = lean_ctor_get(v_x_1468_, 0);
v_isSharedCheck_1485_ = !lean_is_exclusive(v_x_1468_);
if (v_isSharedCheck_1485_ == 0)
{
v___x_1480_ = v_x_1468_;
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
else
{
lean_inc(v_a_1478_);
lean_dec(v_x_1468_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
lean_object* v___x_1483_; 
if (v_isShared_1481_ == 0)
{
lean_ctor_set_tag(v___x_1480_, 0);
v___x_1483_ = v___x_1480_;
goto v_reusejp_1482_;
}
else
{
lean_object* v_reuseFailAlloc_1484_; 
v_reuseFailAlloc_1484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1484_, 0, v_a_1478_);
v___x_1483_ = v_reuseFailAlloc_1484_;
goto v_reusejp_1482_;
}
v_reusejp_1482_:
{
return v___x_1483_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg___boxed(lean_object* v_x_1486_, lean_object* v___y_1487_){
_start:
{
lean_object* v_res_1488_; 
v_res_1488_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(v_x_1486_);
return v_res_1488_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1490_; lean_object* v___x_1491_; 
v___x_1490_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__0));
v___x_1491_ = l_Lean_stringToMessageData(v___x_1490_);
return v___x_1491_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2(void){
_start:
{
lean_object* v___x_1492_; double v___x_1493_; 
v___x_1492_ = lean_unsigned_to_nat(1000u);
v___x_1493_ = lean_float_of_nat(v___x_1492_);
return v___x_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2(lean_object* v_cls_1494_, uint8_t v_collapsed_1495_, lean_object* v_tag_1496_, lean_object* v_opts_1497_, uint8_t v_clsEnabled_1498_, lean_object* v_oldTraces_1499_, lean_object* v_msg_1500_, lean_object* v_resStartStop_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_){
_start:
{
lean_object* v_fst_1507_; lean_object* v_snd_1508_; lean_object* v___y_1510_; lean_object* v___y_1511_; lean_object* v_data_1512_; lean_object* v_fst_1523_; lean_object* v_snd_1524_; lean_object* v___x_1525_; uint8_t v___x_1526_; lean_object* v___y_1528_; lean_object* v_a_1529_; uint8_t v___y_1544_; double v___y_1575_; 
v_fst_1507_ = lean_ctor_get(v_resStartStop_1501_, 0);
lean_inc(v_fst_1507_);
v_snd_1508_ = lean_ctor_get(v_resStartStop_1501_, 1);
lean_inc(v_snd_1508_);
lean_dec_ref(v_resStartStop_1501_);
v_fst_1523_ = lean_ctor_get(v_snd_1508_, 0);
lean_inc(v_fst_1523_);
v_snd_1524_ = lean_ctor_get(v_snd_1508_, 1);
lean_inc(v_snd_1524_);
lean_dec(v_snd_1508_);
v___x_1525_ = l_Lean_trace_profiler;
v___x_1526_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(v_opts_1497_, v___x_1525_);
if (v___x_1526_ == 0)
{
v___y_1544_ = v___x_1526_;
goto v___jp_1543_;
}
else
{
lean_object* v___x_1580_; uint8_t v___x_1581_; 
v___x_1580_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1581_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(v_opts_1497_, v___x_1580_);
if (v___x_1581_ == 0)
{
lean_object* v___x_1582_; lean_object* v___x_1583_; double v___x_1584_; double v___x_1585_; double v___x_1586_; 
v___x_1582_ = l_Lean_trace_profiler_threshold;
v___x_1583_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5(v_opts_1497_, v___x_1582_);
v___x_1584_ = lean_float_of_nat(v___x_1583_);
v___x_1585_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__2);
v___x_1586_ = lean_float_div(v___x_1584_, v___x_1585_);
v___y_1575_ = v___x_1586_;
goto v___jp_1574_;
}
else
{
lean_object* v___x_1587_; lean_object* v___x_1588_; double v___x_1589_; 
v___x_1587_ = l_Lean_trace_profiler_threshold;
v___x_1588_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__5(v_opts_1497_, v___x_1587_);
v___x_1589_ = lean_float_of_nat(v___x_1588_);
v___y_1575_ = v___x_1589_;
goto v___jp_1574_;
}
}
v___jp_1509_:
{
lean_object* v___x_1513_; 
lean_inc(v___y_1511_);
v___x_1513_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__2(v_oldTraces_1499_, v_data_1512_, v___y_1511_, v___y_1510_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_);
if (lean_obj_tag(v___x_1513_) == 0)
{
lean_object* v___x_1514_; 
lean_dec_ref_known(v___x_1513_, 1);
v___x_1514_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(v_fst_1507_);
return v___x_1514_;
}
else
{
lean_object* v_a_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1522_; 
lean_dec(v_fst_1507_);
v_a_1515_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1517_ = v___x_1513_;
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_a_1515_);
lean_dec(v___x_1513_);
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
v___jp_1527_:
{
uint8_t v_result_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; double v___x_1533_; lean_object* v_data_1534_; 
v_result_1530_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__4(v_fst_1507_);
v___x_1531_ = lean_box(v_result_1530_);
v___x_1532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1532_, 0, v___x_1531_);
v___x_1533_ = lean_float_once(&lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__17);
lean_inc_ref(v_tag_1496_);
lean_inc_ref(v___x_1532_);
lean_inc(v_cls_1494_);
v_data_1534_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1534_, 0, v_cls_1494_);
lean_ctor_set(v_data_1534_, 1, v___x_1532_);
lean_ctor_set(v_data_1534_, 2, v_tag_1496_);
lean_ctor_set_float(v_data_1534_, sizeof(void*)*3, v___x_1533_);
lean_ctor_set_float(v_data_1534_, sizeof(void*)*3 + 8, v___x_1533_);
lean_ctor_set_uint8(v_data_1534_, sizeof(void*)*3 + 16, v_collapsed_1495_);
if (v___x_1526_ == 0)
{
lean_dec_ref_known(v___x_1532_, 1);
lean_dec(v_snd_1524_);
lean_dec(v_fst_1523_);
lean_dec_ref(v_tag_1496_);
lean_dec(v_cls_1494_);
v___y_1510_ = v_a_1529_;
v___y_1511_ = v___y_1528_;
v_data_1512_ = v_data_1534_;
goto v___jp_1509_;
}
else
{
lean_object* v_data_1535_; double v___x_1536_; double v___x_1537_; 
lean_dec_ref_known(v_data_1534_, 3);
v_data_1535_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1535_, 0, v_cls_1494_);
lean_ctor_set(v_data_1535_, 1, v___x_1532_);
lean_ctor_set(v_data_1535_, 2, v_tag_1496_);
v___x_1536_ = lean_unbox_float(v_fst_1523_);
lean_dec(v_fst_1523_);
lean_ctor_set_float(v_data_1535_, sizeof(void*)*3, v___x_1536_);
v___x_1537_ = lean_unbox_float(v_snd_1524_);
lean_dec(v_snd_1524_);
lean_ctor_set_float(v_data_1535_, sizeof(void*)*3 + 8, v___x_1537_);
lean_ctor_set_uint8(v_data_1535_, sizeof(void*)*3 + 16, v_collapsed_1495_);
v___y_1510_ = v_a_1529_;
v___y_1511_ = v___y_1528_;
v_data_1512_ = v_data_1535_;
goto v___jp_1509_;
}
}
v___jp_1538_:
{
lean_object* v_ref_1539_; lean_object* v___x_1540_; 
v_ref_1539_ = lean_ctor_get(v___y_1504_, 5);
lean_inc(v___y_1505_);
lean_inc_ref(v___y_1504_);
lean_inc(v___y_1503_);
lean_inc_ref(v___y_1502_);
lean_inc(v_fst_1507_);
v___x_1540_ = lean_apply_6(v_msg_1500_, v_fst_1507_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_, lean_box(0));
if (lean_obj_tag(v___x_1540_) == 0)
{
lean_object* v_a_1541_; 
v_a_1541_ = lean_ctor_get(v___x_1540_, 0);
lean_inc(v_a_1541_);
lean_dec_ref_known(v___x_1540_, 1);
v___y_1528_ = v_ref_1539_;
v_a_1529_ = v_a_1541_;
goto v___jp_1527_;
}
else
{
lean_object* v___x_1542_; 
lean_dec_ref_known(v___x_1540_, 1);
v___x_1542_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___closed__1);
v___y_1528_ = v_ref_1539_;
v_a_1529_ = v___x_1542_;
goto v___jp_1527_;
}
}
v___jp_1543_:
{
if (v_clsEnabled_1498_ == 0)
{
if (v___y_1544_ == 0)
{
lean_object* v___x_1545_; lean_object* v_traceState_1546_; lean_object* v_env_1547_; lean_object* v_nextMacroScope_1548_; lean_object* v_ngen_1549_; lean_object* v_auxDeclNGen_1550_; lean_object* v_cache_1551_; lean_object* v_messages_1552_; lean_object* v_infoState_1553_; lean_object* v_snapshotTasks_1554_; lean_object* v___x_1556_; uint8_t v_isShared_1557_; uint8_t v_isSharedCheck_1573_; 
lean_dec(v_snd_1524_);
lean_dec(v_fst_1523_);
lean_dec_ref(v_msg_1500_);
lean_dec_ref(v_tag_1496_);
lean_dec(v_cls_1494_);
v___x_1545_ = lean_st_ref_take(v___y_1505_);
v_traceState_1546_ = lean_ctor_get(v___x_1545_, 4);
v_env_1547_ = lean_ctor_get(v___x_1545_, 0);
v_nextMacroScope_1548_ = lean_ctor_get(v___x_1545_, 1);
v_ngen_1549_ = lean_ctor_get(v___x_1545_, 2);
v_auxDeclNGen_1550_ = lean_ctor_get(v___x_1545_, 3);
v_cache_1551_ = lean_ctor_get(v___x_1545_, 5);
v_messages_1552_ = lean_ctor_get(v___x_1545_, 6);
v_infoState_1553_ = lean_ctor_get(v___x_1545_, 7);
v_snapshotTasks_1554_ = lean_ctor_get(v___x_1545_, 8);
v_isSharedCheck_1573_ = !lean_is_exclusive(v___x_1545_);
if (v_isSharedCheck_1573_ == 0)
{
v___x_1556_ = v___x_1545_;
v_isShared_1557_ = v_isSharedCheck_1573_;
goto v_resetjp_1555_;
}
else
{
lean_inc(v_snapshotTasks_1554_);
lean_inc(v_infoState_1553_);
lean_inc(v_messages_1552_);
lean_inc(v_cache_1551_);
lean_inc(v_traceState_1546_);
lean_inc(v_auxDeclNGen_1550_);
lean_inc(v_ngen_1549_);
lean_inc(v_nextMacroScope_1548_);
lean_inc(v_env_1547_);
lean_dec(v___x_1545_);
v___x_1556_ = lean_box(0);
v_isShared_1557_ = v_isSharedCheck_1573_;
goto v_resetjp_1555_;
}
v_resetjp_1555_:
{
uint64_t v_tid_1558_; lean_object* v_traces_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1572_; 
v_tid_1558_ = lean_ctor_get_uint64(v_traceState_1546_, sizeof(void*)*1);
v_traces_1559_ = lean_ctor_get(v_traceState_1546_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v_traceState_1546_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1561_ = v_traceState_1546_;
v_isShared_1562_ = v_isSharedCheck_1572_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_traces_1559_);
lean_dec(v_traceState_1546_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1572_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v___x_1563_; lean_object* v___x_1565_; 
v___x_1563_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1499_, v_traces_1559_);
lean_dec_ref(v_traces_1559_);
if (v_isShared_1562_ == 0)
{
lean_ctor_set(v___x_1561_, 0, v___x_1563_);
v___x_1565_ = v___x_1561_;
goto v_reusejp_1564_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v___x_1563_);
lean_ctor_set_uint64(v_reuseFailAlloc_1571_, sizeof(void*)*1, v_tid_1558_);
v___x_1565_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1564_;
}
v_reusejp_1564_:
{
lean_object* v___x_1567_; 
if (v_isShared_1557_ == 0)
{
lean_ctor_set(v___x_1556_, 4, v___x_1565_);
v___x_1567_ = v___x_1556_;
goto v_reusejp_1566_;
}
else
{
lean_object* v_reuseFailAlloc_1570_; 
v_reuseFailAlloc_1570_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1570_, 0, v_env_1547_);
lean_ctor_set(v_reuseFailAlloc_1570_, 1, v_nextMacroScope_1548_);
lean_ctor_set(v_reuseFailAlloc_1570_, 2, v_ngen_1549_);
lean_ctor_set(v_reuseFailAlloc_1570_, 3, v_auxDeclNGen_1550_);
lean_ctor_set(v_reuseFailAlloc_1570_, 4, v___x_1565_);
lean_ctor_set(v_reuseFailAlloc_1570_, 5, v_cache_1551_);
lean_ctor_set(v_reuseFailAlloc_1570_, 6, v_messages_1552_);
lean_ctor_set(v_reuseFailAlloc_1570_, 7, v_infoState_1553_);
lean_ctor_set(v_reuseFailAlloc_1570_, 8, v_snapshotTasks_1554_);
v___x_1567_ = v_reuseFailAlloc_1570_;
goto v_reusejp_1566_;
}
v_reusejp_1566_:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; 
v___x_1568_ = lean_st_ref_set(v___y_1505_, v___x_1567_);
v___x_1569_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(v_fst_1507_);
return v___x_1569_;
}
}
}
}
}
else
{
goto v___jp_1538_;
}
}
else
{
goto v___jp_1538_;
}
}
v___jp_1574_:
{
double v___x_1576_; double v___x_1577_; double v___x_1578_; uint8_t v___x_1579_; 
v___x_1576_ = lean_unbox_float(v_snd_1524_);
v___x_1577_ = lean_unbox_float(v_fst_1523_);
v___x_1578_ = lean_float_sub(v___x_1576_, v___x_1577_);
v___x_1579_ = lean_float_decLt(v___y_1575_, v___x_1578_);
v___y_1544_ = v___x_1579_;
goto v___jp_1543_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2___boxed(lean_object* v_cls_1590_, lean_object* v_collapsed_1591_, lean_object* v_tag_1592_, lean_object* v_opts_1593_, lean_object* v_clsEnabled_1594_, lean_object* v_oldTraces_1595_, lean_object* v_msg_1596_, lean_object* v_resStartStop_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_){
_start:
{
uint8_t v_collapsed_boxed_1603_; uint8_t v_clsEnabled_boxed_1604_; lean_object* v_res_1605_; 
v_collapsed_boxed_1603_ = lean_unbox(v_collapsed_1591_);
v_clsEnabled_boxed_1604_ = lean_unbox(v_clsEnabled_1594_);
v_res_1605_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2(v_cls_1590_, v_collapsed_boxed_1603_, v_tag_1592_, v_opts_1593_, v_clsEnabled_boxed_1604_, v_oldTraces_1595_, v_msg_1596_, v_resStartStop_1597_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
lean_dec(v___y_1601_);
lean_dec_ref(v___y_1600_);
lean_dec(v___y_1599_);
lean_dec_ref(v___y_1598_);
lean_dec_ref(v_opts_1593_);
return v_res_1605_;
}
}
static double _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0(void){
_start:
{
lean_object* v___x_1606_; double v___x_1607_; 
v___x_1606_ = lean_unsigned_to_nat(1000000000u);
v___x_1607_ = lean_float_of_nat(v___x_1606_);
return v___x_1607_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2(void){
_start:
{
lean_object* v___x_1609_; lean_object* v___x_1610_; 
v___x_1609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__1));
v___x_1610_ = l_Lean_stringToMessageData(v___x_1609_);
return v___x_1610_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4(void){
_start:
{
lean_object* v___x_1612_; lean_object* v___x_1613_; 
v___x_1612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__3));
v___x_1613_ = l_Lean_stringToMessageData(v___x_1612_);
return v___x_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3(lean_object* v_transform_1614_, lean_object* v_g_1615_, lean_object* v_l_1616_, lean_object* v_cls_1617_, uint8_t v___x_1618_, lean_object* v___x_1619_, lean_object* v___f_1620_, lean_object* v_toPreprocessorBase_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v_options_1627_; uint8_t v_hasTrace_1628_; 
v_options_1627_ = lean_ctor_get(v___y_1624_, 2);
v_hasTrace_1628_ = lean_ctor_get_uint8(v_options_1627_, sizeof(void*)*1);
if (v_hasTrace_1628_ == 0)
{
lean_object* v___x_1629_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
lean_dec_ref(v___f_1620_);
lean_dec_ref(v___x_1619_);
lean_dec(v_cls_1617_);
lean_inc(v___y_1625_);
lean_inc_ref(v___y_1624_);
lean_inc(v___y_1623_);
lean_inc_ref(v___y_1622_);
v___x_1629_ = lean_apply_7(v_transform_1614_, v_g_1615_, v_l_1616_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_, lean_box(0));
if (lean_obj_tag(v___x_1629_) == 0)
{
lean_object* v_a_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; 
v_a_1630_ = lean_ctor_get(v___x_1629_, 0);
lean_inc(v_a_1630_);
lean_dec_ref_known(v___x_1629_, 1);
v___x_1631_ = lean_box(0);
v___x_1632_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(v_a_1630_, v___x_1631_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
if (lean_obj_tag(v___x_1632_) == 0)
{
lean_object* v___x_1634_; uint8_t v_isShared_1635_; uint8_t v_isSharedCheck_1639_; 
v_isSharedCheck_1639_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_1639_ == 0)
{
lean_object* v_unused_1640_; 
v_unused_1640_ = lean_ctor_get(v___x_1632_, 0);
lean_dec(v_unused_1640_);
v___x_1634_ = v___x_1632_;
v_isShared_1635_ = v_isSharedCheck_1639_;
goto v_resetjp_1633_;
}
else
{
lean_dec(v___x_1632_);
v___x_1634_ = lean_box(0);
v_isShared_1635_ = v_isSharedCheck_1639_;
goto v_resetjp_1633_;
}
v_resetjp_1633_:
{
lean_object* v___x_1637_; 
if (v_isShared_1635_ == 0)
{
lean_ctor_set(v___x_1634_, 0, v_a_1630_);
v___x_1637_ = v___x_1634_;
goto v_reusejp_1636_;
}
else
{
lean_object* v_reuseFailAlloc_1638_; 
v_reuseFailAlloc_1638_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1638_, 0, v_a_1630_);
v___x_1637_ = v_reuseFailAlloc_1638_;
goto v_reusejp_1636_;
}
v_reusejp_1636_:
{
return v___x_1637_;
}
}
}
else
{
lean_object* v_a_1641_; lean_object* v___x_1643_; uint8_t v_isShared_1644_; uint8_t v_isSharedCheck_1648_; 
lean_dec(v_a_1630_);
v_a_1641_ = lean_ctor_get(v___x_1632_, 0);
v_isSharedCheck_1648_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1643_ = v___x_1632_;
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
else
{
lean_inc(v_a_1641_);
lean_dec(v___x_1632_);
v___x_1643_ = lean_box(0);
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
v_resetjp_1642_:
{
lean_object* v___x_1646_; 
if (v_isShared_1644_ == 0)
{
v___x_1646_ = v___x_1643_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_a_1641_);
v___x_1646_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
return v___x_1646_;
}
}
}
}
else
{
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
return v___x_1629_;
}
}
else
{
lean_object* v_inheritedTraceOptions_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; uint8_t v___x_1652_; lean_object* v___y_1654_; lean_object* v___y_1655_; lean_object* v_a_1656_; lean_object* v___y_1666_; lean_object* v___y_1667_; lean_object* v_a_1668_; lean_object* v___y_1671_; lean_object* v___y_1672_; lean_object* v___y_1673_; lean_object* v___y_1684_; lean_object* v___y_1685_; lean_object* v_a_1686_; lean_object* v___y_1699_; lean_object* v___y_1700_; lean_object* v_a_1701_; lean_object* v___y_1704_; lean_object* v___y_1705_; lean_object* v___y_1706_; 
v_inheritedTraceOptions_1649_ = lean_ctor_get(v___y_1624_, 13);
v___x_1650_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__11));
lean_inc(v_cls_1617_);
v___x_1651_ = l_Lean_Name_append(v___x_1650_, v_cls_1617_);
v___x_1652_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1649_, v_options_1627_, v___x_1651_);
lean_dec(v___x_1651_);
if (v___x_1652_ == 0)
{
lean_object* v___x_1777_; uint8_t v___x_1778_; 
v___x_1777_ = l_Lean_trace_profiler;
v___x_1778_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(v_options_1627_, v___x_1777_);
if (v___x_1778_ == 0)
{
lean_object* v___x_1779_; 
lean_dec_ref(v___f_1620_);
lean_dec_ref(v___x_1619_);
lean_inc(v___y_1625_);
lean_inc_ref(v___y_1624_);
lean_inc(v___y_1623_);
lean_inc_ref(v___y_1622_);
v___x_1779_ = lean_apply_7(v_transform_1614_, v_g_1615_, v_l_1616_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_, lean_box(0));
if (lean_obj_tag(v___x_1779_) == 0)
{
lean_object* v_a_1780_; lean_object* v___y_1782_; lean_object* v___y_1783_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___x_1804_; lean_object* v___x_1805_; uint8_t v___x_1806_; 
v_a_1780_ = lean_ctor_get(v___x_1779_, 0);
lean_inc(v_a_1780_);
lean_dec_ref_known(v___x_1779_, 1);
v___x_1804_ = lean_unsigned_to_nat(1u);
v___x_1805_ = l_List_lengthTR___redArg(v_a_1780_);
v___x_1806_ = lean_nat_dec_lt(v___x_1804_, v___x_1805_);
lean_dec(v___x_1805_);
if (v___x_1806_ == 0)
{
lean_dec_ref(v_toPreprocessorBase_1621_);
lean_dec(v_cls_1617_);
v___y_1782_ = v___y_1622_;
v___y_1783_ = v___y_1623_;
v___y_1784_ = v___y_1624_;
v___y_1785_ = v___y_1625_;
goto v___jp_1781_;
}
else
{
if (v___x_1652_ == 0)
{
lean_dec_ref(v_toPreprocessorBase_1621_);
lean_dec(v_cls_1617_);
v___y_1782_ = v___y_1622_;
v___y_1783_ = v___y_1623_;
v___y_1784_ = v___y_1624_;
v___y_1785_ = v___y_1625_;
goto v___jp_1781_;
}
else
{
lean_object* v_name_1807_; lean_object* v___x_1809_; uint8_t v_isShared_1810_; uint8_t v_isSharedCheck_1827_; 
v_name_1807_ = lean_ctor_get(v_toPreprocessorBase_1621_, 0);
v_isSharedCheck_1827_ = !lean_is_exclusive(v_toPreprocessorBase_1621_);
if (v_isSharedCheck_1827_ == 0)
{
lean_object* v_unused_1828_; 
v_unused_1828_ = lean_ctor_get(v_toPreprocessorBase_1621_, 1);
lean_dec(v_unused_1828_);
v___x_1809_ = v_toPreprocessorBase_1621_;
v_isShared_1810_ = v_isSharedCheck_1827_;
goto v_resetjp_1808_;
}
else
{
lean_inc(v_name_1807_);
lean_dec(v_toPreprocessorBase_1621_);
v___x_1809_ = lean_box(0);
v_isShared_1810_ = v_isSharedCheck_1827_;
goto v_resetjp_1808_;
}
v_resetjp_1808_:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1814_; 
v___x_1811_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2);
v___x_1812_ = l_Lean_MessageData_ofName(v_name_1807_);
if (v_isShared_1810_ == 0)
{
lean_ctor_set_tag(v___x_1809_, 7);
lean_ctor_set(v___x_1809_, 1, v___x_1812_);
lean_ctor_set(v___x_1809_, 0, v___x_1811_);
v___x_1814_ = v___x_1809_;
goto v_reusejp_1813_;
}
else
{
lean_object* v_reuseFailAlloc_1826_; 
v_reuseFailAlloc_1826_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1826_, 0, v___x_1811_);
lean_ctor_set(v_reuseFailAlloc_1826_, 1, v___x_1812_);
v___x_1814_ = v_reuseFailAlloc_1826_;
goto v_reusejp_1813_;
}
v_reusejp_1813_:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; 
v___x_1815_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4);
v___x_1816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1814_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
v___x_1817_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(v_cls_1617_, v___x_1816_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
if (lean_obj_tag(v___x_1817_) == 0)
{
lean_dec_ref_known(v___x_1817_, 1);
v___y_1782_ = v___y_1622_;
v___y_1783_ = v___y_1623_;
v___y_1784_ = v___y_1624_;
v___y_1785_ = v___y_1625_;
goto v___jp_1781_;
}
else
{
lean_object* v_a_1818_; lean_object* v___x_1820_; uint8_t v_isShared_1821_; uint8_t v_isSharedCheck_1825_; 
lean_dec(v_a_1780_);
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
v_a_1818_ = lean_ctor_get(v___x_1817_, 0);
v_isSharedCheck_1825_ = !lean_is_exclusive(v___x_1817_);
if (v_isSharedCheck_1825_ == 0)
{
v___x_1820_ = v___x_1817_;
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
else
{
lean_inc(v_a_1818_);
lean_dec(v___x_1817_);
v___x_1820_ = lean_box(0);
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
v_resetjp_1819_:
{
lean_object* v___x_1823_; 
if (v_isShared_1821_ == 0)
{
v___x_1823_ = v___x_1820_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v_a_1818_);
v___x_1823_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
return v___x_1823_;
}
}
}
}
}
}
}
v___jp_1781_:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1786_ = lean_box(0);
v___x_1787_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(v_a_1780_, v___x_1786_, v___y_1782_, v___y_1783_, v___y_1784_, v___y_1785_);
lean_dec(v___y_1785_);
lean_dec_ref(v___y_1784_);
lean_dec(v___y_1783_);
lean_dec_ref(v___y_1782_);
if (lean_obj_tag(v___x_1787_) == 0)
{
lean_object* v___x_1789_; uint8_t v_isShared_1790_; uint8_t v_isSharedCheck_1794_; 
v_isSharedCheck_1794_ = !lean_is_exclusive(v___x_1787_);
if (v_isSharedCheck_1794_ == 0)
{
lean_object* v_unused_1795_; 
v_unused_1795_ = lean_ctor_get(v___x_1787_, 0);
lean_dec(v_unused_1795_);
v___x_1789_ = v___x_1787_;
v_isShared_1790_ = v_isSharedCheck_1794_;
goto v_resetjp_1788_;
}
else
{
lean_dec(v___x_1787_);
v___x_1789_ = lean_box(0);
v_isShared_1790_ = v_isSharedCheck_1794_;
goto v_resetjp_1788_;
}
v_resetjp_1788_:
{
lean_object* v___x_1792_; 
if (v_isShared_1790_ == 0)
{
lean_ctor_set(v___x_1789_, 0, v_a_1780_);
v___x_1792_ = v___x_1789_;
goto v_reusejp_1791_;
}
else
{
lean_object* v_reuseFailAlloc_1793_; 
v_reuseFailAlloc_1793_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1793_, 0, v_a_1780_);
v___x_1792_ = v_reuseFailAlloc_1793_;
goto v_reusejp_1791_;
}
v_reusejp_1791_:
{
return v___x_1792_;
}
}
}
else
{
lean_object* v_a_1796_; lean_object* v___x_1798_; uint8_t v_isShared_1799_; uint8_t v_isSharedCheck_1803_; 
lean_dec(v_a_1780_);
v_a_1796_ = lean_ctor_get(v___x_1787_, 0);
v_isSharedCheck_1803_ = !lean_is_exclusive(v___x_1787_);
if (v_isSharedCheck_1803_ == 0)
{
v___x_1798_ = v___x_1787_;
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
else
{
lean_inc(v_a_1796_);
lean_dec(v___x_1787_);
v___x_1798_ = lean_box(0);
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
v_resetjp_1797_:
{
lean_object* v___x_1801_; 
if (v_isShared_1799_ == 0)
{
v___x_1801_ = v___x_1798_;
goto v_reusejp_1800_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v_a_1796_);
v___x_1801_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1800_;
}
v_reusejp_1800_:
{
return v___x_1801_;
}
}
}
}
}
else
{
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
lean_dec_ref(v_toPreprocessorBase_1621_);
lean_dec(v_cls_1617_);
return v___x_1779_;
}
}
else
{
lean_inc_ref(v_options_1627_);
goto v___jp_1716_;
}
}
else
{
lean_inc_ref(v_options_1627_);
goto v___jp_1716_;
}
v___jp_1653_:
{
lean_object* v___x_1657_; double v___x_1658_; double v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
v___x_1657_ = lean_io_get_num_heartbeats();
v___x_1658_ = lean_float_of_nat(v___y_1654_);
v___x_1659_ = lean_float_of_nat(v___x_1657_);
v___x_1660_ = lean_box_float(v___x_1658_);
v___x_1661_ = lean_box_float(v___x_1659_);
v___x_1662_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1662_, 0, v___x_1660_);
lean_ctor_set(v___x_1662_, 1, v___x_1661_);
v___x_1663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1663_, 0, v_a_1656_);
lean_ctor_set(v___x_1663_, 1, v___x_1662_);
v___x_1664_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2(v_cls_1617_, v___x_1618_, v___x_1619_, v_options_1627_, v___x_1652_, v___y_1655_, v___f_1620_, v___x_1663_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
lean_dec_ref(v_options_1627_);
return v___x_1664_;
}
v___jp_1665_:
{
lean_object* v___x_1669_; 
v___x_1669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1669_, 0, v_a_1668_);
v___y_1654_ = v___y_1666_;
v___y_1655_ = v___y_1667_;
v_a_1656_ = v___x_1669_;
goto v___jp_1653_;
}
v___jp_1670_:
{
if (lean_obj_tag(v___y_1673_) == 0)
{
lean_object* v_a_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1681_; 
v_a_1674_ = lean_ctor_get(v___y_1673_, 0);
v_isSharedCheck_1681_ = !lean_is_exclusive(v___y_1673_);
if (v_isSharedCheck_1681_ == 0)
{
v___x_1676_ = v___y_1673_;
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_a_1674_);
lean_dec(v___y_1673_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
lean_object* v___x_1679_; 
if (v_isShared_1677_ == 0)
{
lean_ctor_set_tag(v___x_1676_, 1);
v___x_1679_ = v___x_1676_;
goto v_reusejp_1678_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_a_1674_);
v___x_1679_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1678_;
}
v_reusejp_1678_:
{
v___y_1654_ = v___y_1671_;
v___y_1655_ = v___y_1672_;
v_a_1656_ = v___x_1679_;
goto v___jp_1653_;
}
}
}
else
{
lean_object* v_a_1682_; 
v_a_1682_ = lean_ctor_get(v___y_1673_, 0);
lean_inc(v_a_1682_);
lean_dec_ref_known(v___y_1673_, 1);
v___y_1666_ = v___y_1671_;
v___y_1667_ = v___y_1672_;
v_a_1668_ = v_a_1682_;
goto v___jp_1665_;
}
}
v___jp_1683_:
{
lean_object* v___x_1687_; double v___x_1688_; double v___x_1689_; double v___x_1690_; double v___x_1691_; double v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; 
v___x_1687_ = lean_io_mono_nanos_now();
v___x_1688_ = lean_float_of_nat(v___y_1684_);
v___x_1689_ = lean_float_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__0);
v___x_1690_ = lean_float_div(v___x_1688_, v___x_1689_);
v___x_1691_ = lean_float_of_nat(v___x_1687_);
v___x_1692_ = lean_float_div(v___x_1691_, v___x_1689_);
v___x_1693_ = lean_box_float(v___x_1690_);
v___x_1694_ = lean_box_float(v___x_1692_);
v___x_1695_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1695_, 0, v___x_1693_);
lean_ctor_set(v___x_1695_, 1, v___x_1694_);
v___x_1696_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1696_, 0, v_a_1686_);
lean_ctor_set(v___x_1696_, 1, v___x_1695_);
v___x_1697_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2(v_cls_1617_, v___x_1618_, v___x_1619_, v_options_1627_, v___x_1652_, v___y_1685_, v___f_1620_, v___x_1696_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
lean_dec_ref(v_options_1627_);
return v___x_1697_;
}
v___jp_1698_:
{
lean_object* v___x_1702_; 
v___x_1702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1702_, 0, v_a_1701_);
v___y_1684_ = v___y_1699_;
v___y_1685_ = v___y_1700_;
v_a_1686_ = v___x_1702_;
goto v___jp_1683_;
}
v___jp_1703_:
{
if (lean_obj_tag(v___y_1706_) == 0)
{
lean_object* v_a_1707_; lean_object* v___x_1709_; uint8_t v_isShared_1710_; uint8_t v_isSharedCheck_1714_; 
v_a_1707_ = lean_ctor_get(v___y_1706_, 0);
v_isSharedCheck_1714_ = !lean_is_exclusive(v___y_1706_);
if (v_isSharedCheck_1714_ == 0)
{
v___x_1709_ = v___y_1706_;
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
else
{
lean_inc(v_a_1707_);
lean_dec(v___y_1706_);
v___x_1709_ = lean_box(0);
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
v_resetjp_1708_:
{
lean_object* v___x_1712_; 
if (v_isShared_1710_ == 0)
{
lean_ctor_set_tag(v___x_1709_, 1);
v___x_1712_ = v___x_1709_;
goto v_reusejp_1711_;
}
else
{
lean_object* v_reuseFailAlloc_1713_; 
v_reuseFailAlloc_1713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1713_, 0, v_a_1707_);
v___x_1712_ = v_reuseFailAlloc_1713_;
goto v_reusejp_1711_;
}
v_reusejp_1711_:
{
v___y_1684_ = v___y_1704_;
v___y_1685_ = v___y_1705_;
v_a_1686_ = v___x_1712_;
goto v___jp_1683_;
}
}
}
else
{
lean_object* v_a_1715_; 
v_a_1715_ = lean_ctor_get(v___y_1706_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___y_1706_, 1);
v___y_1699_ = v___y_1704_;
v___y_1700_ = v___y_1705_;
v_a_1701_ = v_a_1715_;
goto v___jp_1698_;
}
}
v___jp_1716_:
{
lean_object* v___x_1717_; lean_object* v_a_1718_; lean_object* v___x_1719_; uint8_t v___x_1720_; 
v___x_1717_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__0___redArg(v___y_1625_);
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
lean_inc(v_a_1718_);
lean_dec_ref(v___x_1717_);
v___x_1719_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1720_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__1(v_options_1627_, v___x_1719_);
if (v___x_1720_ == 0)
{
lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1721_ = lean_io_mono_nanos_now();
lean_inc(v___y_1625_);
lean_inc_ref(v___y_1624_);
lean_inc(v___y_1623_);
lean_inc_ref(v___y_1622_);
v___x_1722_ = lean_apply_7(v_transform_1614_, v_g_1615_, v_l_1616_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_, lean_box(0));
if (lean_obj_tag(v___x_1722_) == 0)
{
lean_object* v_a_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; uint8_t v___x_1726_; 
v_a_1723_ = lean_ctor_get(v___x_1722_, 0);
lean_inc(v_a_1723_);
lean_dec_ref_known(v___x_1722_, 1);
v___x_1724_ = lean_unsigned_to_nat(1u);
v___x_1725_ = l_List_lengthTR___redArg(v_a_1723_);
v___x_1726_ = lean_nat_dec_lt(v___x_1724_, v___x_1725_);
lean_dec(v___x_1725_);
if (v___x_1726_ == 0)
{
lean_object* v___x_1727_; lean_object* v___x_1728_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v___x_1727_ = lean_box(0);
v___x_1728_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1723_, v___x_1727_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1704_ = v___x_1721_;
v___y_1705_ = v_a_1718_;
v___y_1706_ = v___x_1728_;
goto v___jp_1703_;
}
else
{
if (v___x_1652_ == 0)
{
lean_object* v___x_1729_; lean_object* v___x_1730_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v___x_1729_ = lean_box(0);
v___x_1730_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1723_, v___x_1729_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1704_ = v___x_1721_;
v___y_1705_ = v_a_1718_;
v___y_1706_ = v___x_1730_;
goto v___jp_1703_;
}
else
{
lean_object* v_name_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1746_; 
v_name_1731_ = lean_ctor_get(v_toPreprocessorBase_1621_, 0);
v_isSharedCheck_1746_ = !lean_is_exclusive(v_toPreprocessorBase_1621_);
if (v_isSharedCheck_1746_ == 0)
{
lean_object* v_unused_1747_; 
v_unused_1747_ = lean_ctor_get(v_toPreprocessorBase_1621_, 1);
lean_dec(v_unused_1747_);
v___x_1733_ = v_toPreprocessorBase_1621_;
v_isShared_1734_ = v_isSharedCheck_1746_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_name_1731_);
lean_dec(v_toPreprocessorBase_1621_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1746_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1738_; 
v___x_1735_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2);
v___x_1736_ = l_Lean_MessageData_ofName(v_name_1731_);
if (v_isShared_1734_ == 0)
{
lean_ctor_set_tag(v___x_1733_, 7);
lean_ctor_set(v___x_1733_, 1, v___x_1736_);
lean_ctor_set(v___x_1733_, 0, v___x_1735_);
v___x_1738_ = v___x_1733_;
goto v_reusejp_1737_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1735_);
lean_ctor_set(v_reuseFailAlloc_1745_, 1, v___x_1736_);
v___x_1738_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1737_;
}
v_reusejp_1737_:
{
lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; 
v___x_1739_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4);
v___x_1740_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1740_, 0, v___x_1738_);
lean_ctor_set(v___x_1740_, 1, v___x_1739_);
lean_inc(v_cls_1617_);
v___x_1741_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(v_cls_1617_, v___x_1740_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
if (lean_obj_tag(v___x_1741_) == 0)
{
lean_object* v_a_1742_; lean_object* v___x_1743_; 
v_a_1742_ = lean_ctor_get(v___x_1741_, 0);
lean_inc(v_a_1742_);
lean_dec_ref_known(v___x_1741_, 1);
v___x_1743_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1723_, v_a_1742_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1704_ = v___x_1721_;
v___y_1705_ = v_a_1718_;
v___y_1706_ = v___x_1743_;
goto v___jp_1703_;
}
else
{
lean_object* v_a_1744_; 
lean_dec(v_a_1723_);
v_a_1744_ = lean_ctor_get(v___x_1741_, 0);
lean_inc(v_a_1744_);
lean_dec_ref_known(v___x_1741_, 1);
v___y_1699_ = v___x_1721_;
v___y_1700_ = v_a_1718_;
v_a_1701_ = v_a_1744_;
goto v___jp_1698_;
}
}
}
}
}
}
else
{
lean_object* v_a_1748_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v_a_1748_ = lean_ctor_get(v___x_1722_, 0);
lean_inc(v_a_1748_);
lean_dec_ref_known(v___x_1722_, 1);
v___y_1699_ = v___x_1721_;
v___y_1700_ = v_a_1718_;
v_a_1701_ = v_a_1748_;
goto v___jp_1698_;
}
}
else
{
lean_object* v___x_1749_; lean_object* v___x_1750_; 
v___x_1749_ = lean_io_get_num_heartbeats();
lean_inc(v___y_1625_);
lean_inc_ref(v___y_1624_);
lean_inc(v___y_1623_);
lean_inc_ref(v___y_1622_);
v___x_1750_ = lean_apply_7(v_transform_1614_, v_g_1615_, v_l_1616_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_, lean_box(0));
if (lean_obj_tag(v___x_1750_) == 0)
{
lean_object* v_a_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; uint8_t v___x_1754_; 
v_a_1751_ = lean_ctor_get(v___x_1750_, 0);
lean_inc(v_a_1751_);
lean_dec_ref_known(v___x_1750_, 1);
v___x_1752_ = lean_unsigned_to_nat(1u);
v___x_1753_ = l_List_lengthTR___redArg(v_a_1751_);
v___x_1754_ = lean_nat_dec_lt(v___x_1752_, v___x_1753_);
lean_dec(v___x_1753_);
if (v___x_1754_ == 0)
{
lean_object* v___x_1755_; lean_object* v___x_1756_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v___x_1755_ = lean_box(0);
v___x_1756_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1751_, v___x_1755_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1671_ = v___x_1749_;
v___y_1672_ = v_a_1718_;
v___y_1673_ = v___x_1756_;
goto v___jp_1670_;
}
else
{
if (v___x_1652_ == 0)
{
lean_object* v___x_1757_; lean_object* v___x_1758_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v___x_1757_ = lean_box(0);
v___x_1758_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1751_, v___x_1757_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1671_ = v___x_1749_;
v___y_1672_ = v_a_1718_;
v___y_1673_ = v___x_1758_;
goto v___jp_1670_;
}
else
{
lean_object* v_name_1759_; lean_object* v___x_1761_; uint8_t v_isShared_1762_; uint8_t v_isSharedCheck_1774_; 
v_name_1759_ = lean_ctor_get(v_toPreprocessorBase_1621_, 0);
v_isSharedCheck_1774_ = !lean_is_exclusive(v_toPreprocessorBase_1621_);
if (v_isSharedCheck_1774_ == 0)
{
lean_object* v_unused_1775_; 
v_unused_1775_ = lean_ctor_get(v_toPreprocessorBase_1621_, 1);
lean_dec(v_unused_1775_);
v___x_1761_ = v_toPreprocessorBase_1621_;
v_isShared_1762_ = v_isSharedCheck_1774_;
goto v_resetjp_1760_;
}
else
{
lean_inc(v_name_1759_);
lean_dec(v_toPreprocessorBase_1621_);
v___x_1761_ = lean_box(0);
v_isShared_1762_ = v_isSharedCheck_1774_;
goto v_resetjp_1760_;
}
v_resetjp_1760_:
{
lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1766_; 
v___x_1763_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__2);
v___x_1764_ = l_Lean_MessageData_ofName(v_name_1759_);
if (v_isShared_1762_ == 0)
{
lean_ctor_set_tag(v___x_1761_, 7);
lean_ctor_set(v___x_1761_, 1, v___x_1764_);
lean_ctor_set(v___x_1761_, 0, v___x_1763_);
v___x_1766_ = v___x_1761_;
goto v_reusejp_1765_;
}
else
{
lean_object* v_reuseFailAlloc_1773_; 
v_reuseFailAlloc_1773_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1773_, 0, v___x_1763_);
lean_ctor_set(v_reuseFailAlloc_1773_, 1, v___x_1764_);
v___x_1766_ = v_reuseFailAlloc_1773_;
goto v_reusejp_1765_;
}
v_reusejp_1765_:
{
lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; 
v___x_1767_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___closed__4);
v___x_1768_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1768_, 0, v___x_1766_);
lean_ctor_set(v___x_1768_, 1, v___x_1767_);
lean_inc(v_cls_1617_);
v___x_1769_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4(v_cls_1617_, v___x_1768_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
if (lean_obj_tag(v___x_1769_) == 0)
{
lean_object* v_a_1770_; lean_object* v___x_1771_; 
v_a_1770_ = lean_ctor_get(v___x_1769_, 0);
lean_inc(v_a_1770_);
lean_dec_ref_known(v___x_1769_, 1);
v___x_1771_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__1(v_a_1751_, v_a_1770_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
v___y_1671_ = v___x_1749_;
v___y_1672_ = v_a_1718_;
v___y_1673_ = v___x_1771_;
goto v___jp_1670_;
}
else
{
lean_object* v_a_1772_; 
lean_dec(v_a_1751_);
v_a_1772_ = lean_ctor_get(v___x_1769_, 0);
lean_inc(v_a_1772_);
lean_dec_ref_known(v___x_1769_, 1);
v___y_1666_ = v___x_1749_;
v___y_1667_ = v_a_1718_;
v_a_1668_ = v_a_1772_;
goto v___jp_1665_;
}
}
}
}
}
}
else
{
lean_object* v_a_1776_; 
lean_dec_ref(v_toPreprocessorBase_1621_);
v_a_1776_ = lean_ctor_get(v___x_1750_, 0);
lean_inc(v_a_1776_);
lean_dec_ref_known(v___x_1750_, 1);
v___y_1666_ = v___x_1749_;
v___y_1667_ = v_a_1718_;
v_a_1668_ = v_a_1776_;
goto v___jp_1665_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___boxed(lean_object* v_transform_1829_, lean_object* v_g_1830_, lean_object* v_l_1831_, lean_object* v_cls_1832_, lean_object* v___x_1833_, lean_object* v___x_1834_, lean_object* v___f_1835_, lean_object* v_toPreprocessorBase_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_){
_start:
{
uint8_t v___x_17026__boxed_1842_; lean_object* v_res_1843_; 
v___x_17026__boxed_1842_ = lean_unbox(v___x_1833_);
v_res_1843_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3(v_transform_1829_, v_g_1830_, v_l_1831_, v_cls_1832_, v___x_17026__boxed_1842_, v___x_1834_, v___f_1835_, v_toPreprocessorBase_1836_, v___y_1837_, v___y_1838_, v___y_1839_, v___y_1840_);
return v_res_1843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process(lean_object* v_pp_1844_, lean_object* v_g_1845_, lean_object* v_l_1846_, lean_object* v_a_1847_, lean_object* v_a_1848_, lean_object* v_a_1849_, lean_object* v_a_1850_){
_start:
{
lean_object* v_toPreprocessorBase_1852_; lean_object* v_transform_1853_; lean_object* v___f_1854_; lean_object* v_cls_1855_; uint8_t v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___f_1859_; lean_object* v___x_1860_; 
v_toPreprocessorBase_1852_ = lean_ctor_get(v_pp_1844_, 0);
lean_inc_ref_n(v_toPreprocessorBase_1852_, 2);
v_transform_1853_ = lean_ctor_get(v_pp_1844_, 1);
lean_inc_ref(v_transform_1853_);
lean_dec_ref(v_pp_1844_);
v___f_1854_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1854_, 0, v_toPreprocessorBase_1852_);
v_cls_1855_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn___closed__1_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_));
v___x_1856_ = 1;
v___x_1857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linarithTraceProofs___redArg___closed__18));
v___x_1858_ = lean_box(v___x_1856_);
lean_inc(v_g_1845_);
v___f_1859_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___lam__3___boxed), 13, 8);
lean_closure_set(v___f_1859_, 0, v_transform_1853_);
lean_closure_set(v___f_1859_, 1, v_g_1845_);
lean_closure_set(v___f_1859_, 2, v_l_1846_);
lean_closure_set(v___f_1859_, 3, v_cls_1855_);
lean_closure_set(v___f_1859_, 4, v___x_1858_);
lean_closure_set(v___f_1859_, 5, v___x_1857_);
lean_closure_set(v___f_1859_, 6, v___f_1854_);
lean_closure_set(v___f_1859_, 7, v_toPreprocessorBase_1852_);
v___x_1860_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__3___redArg(v_g_1845_, v___f_1859_, v_a_1847_, v_a_1848_, v_a_1849_, v_a_1850_);
return v___x_1860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process___boxed(lean_object* v_pp_1861_, lean_object* v_g_1862_, lean_object* v_l_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_, lean_object* v_a_1868_){
_start:
{
lean_object* v_res_1869_; 
v_res_1869_ = lp_mathlib_Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process(v_pp_1861_, v_g_1862_, v_l_1863_, v_a_1864_, v_a_1865_, v_a_1866_, v_a_1867_);
lean_dec(v_a_1867_);
lean_dec_ref(v_a_1866_);
lean_dec(v_a_1865_);
lean_dec_ref(v_a_1864_);
return v_res_1869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3(lean_object* v_00_u03b1_1870_, lean_object* v_x_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_){
_start:
{
lean_object* v___x_1877_; 
v___x_1877_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___redArg(v_x_1871_);
return v___x_1877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1878_, lean_object* v_x_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_, lean_object* v___y_1882_, lean_object* v___y_1883_, lean_object* v___y_1884_){
_start:
{
lean_object* v_res_1885_; 
v_res_1885_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__2_spec__3(v_00_u03b1_1878_, v_x_1879_, v___y_1880_, v___y_1881_, v___y_1882_, v___y_1883_);
lean_dec(v___y_1883_);
lean_dec_ref(v___y_1882_);
lean_dec(v___y_1881_);
lean_dec_ref(v___y_1880_);
return v_res_1885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5(lean_object* v_as_1886_, lean_object* v_as_x27_1887_, lean_object* v_b_1888_, lean_object* v_a_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_){
_start:
{
lean_object* v___x_1895_; 
v___x_1895_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___redArg(v_as_x27_1887_, v_b_1888_, v___y_1890_, v___y_1891_, v___y_1892_, v___y_1893_);
return v___x_1895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5___boxed(lean_object* v_as_1896_, lean_object* v_as_x27_1897_, lean_object* v_b_1898_, lean_object* v_a_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__5(v_as_1896_, v_as_x27_1897_, v_b_1898_, v_a_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1901_);
lean_dec_ref(v___y_1900_);
lean_dec(v_as_x27_1897_);
lean_dec(v_as_1896_);
return v_res_1905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg(lean_object* v_msg_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v_ref_1919_; lean_object* v___x_1920_; lean_object* v_a_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1929_; 
v_ref_1919_ = lean_ctor_get(v___y_1916_, 5);
v___x_1920_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Linarith_GlobalBranchingPreprocessor_process_spec__4_spec__8(v_msg_1913_, v___y_1914_, v___y_1915_, v___y_1916_, v___y_1917_);
v_a_1921_ = lean_ctor_get(v___x_1920_, 0);
v_isSharedCheck_1929_ = !lean_is_exclusive(v___x_1920_);
if (v_isSharedCheck_1929_ == 0)
{
v___x_1923_ = v___x_1920_;
v_isShared_1924_ = v_isSharedCheck_1929_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_a_1921_);
lean_dec(v___x_1920_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1929_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___x_1925_; lean_object* v___x_1927_; 
lean_inc(v_ref_1919_);
v___x_1925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1925_, 0, v_ref_1919_);
lean_ctor_set(v___x_1925_, 1, v_a_1921_);
if (v_isShared_1924_ == 0)
{
lean_ctor_set_tag(v___x_1923_, 1);
lean_ctor_set(v___x_1923_, 0, v___x_1925_);
v___x_1927_ = v___x_1923_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_1928_; 
v_reuseFailAlloc_1928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1928_, 0, v___x_1925_);
v___x_1927_ = v_reuseFailAlloc_1928_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
return v___x_1927_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg___boxed(lean_object* v_msg_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
lean_object* v_res_1936_; 
v_res_1936_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg(v_msg_1930_, v___y_1931_, v___y_1932_, v___y_1933_, v___y_1934_);
lean_dec(v___y_1934_);
lean_dec_ref(v___y_1933_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
return v_res_1936_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1(void){
_start:
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
v___x_1938_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__0));
v___x_1939_ = l_Lean_stringToMessageData(v___x_1938_);
return v___x_1939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(lean_object* v_e_1940_, lean_object* v_a_1941_, lean_object* v_a_1942_, lean_object* v_a_1943_, lean_object* v_a_1944_){
_start:
{
lean_object* v___x_1946_; 
v___x_1946_ = lp_mathlib_Lean_Expr_ineq_x3f(v_e_1940_, v_a_1941_, v_a_1942_, v_a_1943_, v_a_1944_);
if (lean_obj_tag(v___x_1946_) == 0)
{
lean_object* v_a_1947_; lean_object* v___x_1949_; uint8_t v_isShared_1950_; uint8_t v_isSharedCheck_1978_; 
v_a_1947_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_1978_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_1978_ == 0)
{
v___x_1949_ = v___x_1946_;
v_isShared_1950_ = v_isSharedCheck_1978_;
goto v_resetjp_1948_;
}
else
{
lean_inc(v_a_1947_);
lean_dec(v___x_1946_);
v___x_1949_ = lean_box(0);
v_isShared_1950_ = v_isSharedCheck_1978_;
goto v_resetjp_1948_;
}
v_resetjp_1948_:
{
lean_object* v_snd_1951_; lean_object* v_snd_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_1976_; 
v_snd_1951_ = lean_ctor_get(v_a_1947_, 1);
lean_inc(v_snd_1951_);
v_snd_1952_ = lean_ctor_get(v_snd_1951_, 1);
v_isSharedCheck_1976_ = !lean_is_exclusive(v_snd_1951_);
if (v_isSharedCheck_1976_ == 0)
{
lean_object* v_unused_1977_; 
v_unused_1977_ = lean_ctor_get(v_snd_1951_, 0);
lean_dec(v_unused_1977_);
v___x_1954_ = v_snd_1951_;
v_isShared_1955_ = v_isSharedCheck_1976_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_snd_1952_);
lean_dec(v_snd_1951_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_1976_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v_fst_1956_; lean_object* v_fst_1957_; lean_object* v_snd_1958_; lean_object* v___x_1960_; uint8_t v_isShared_1961_; uint8_t v_isSharedCheck_1975_; 
v_fst_1956_ = lean_ctor_get(v_a_1947_, 0);
lean_inc(v_fst_1956_);
lean_dec(v_a_1947_);
v_fst_1957_ = lean_ctor_get(v_snd_1952_, 0);
v_snd_1958_ = lean_ctor_get(v_snd_1952_, 1);
v_isSharedCheck_1975_ = !lean_is_exclusive(v_snd_1952_);
if (v_isSharedCheck_1975_ == 0)
{
v___x_1960_ = v_snd_1952_;
v_isShared_1961_ = v_isSharedCheck_1975_;
goto v_resetjp_1959_;
}
else
{
lean_inc(v_snd_1958_);
lean_inc(v_fst_1957_);
lean_dec(v_snd_1952_);
v___x_1960_ = lean_box(0);
v_isShared_1961_ = v_isSharedCheck_1975_;
goto v_resetjp_1959_;
}
v_resetjp_1959_:
{
uint8_t v___x_1962_; 
lean_inc(v_snd_1958_);
v___x_1962_ = lp_mathlib_Lean_Expr_zero_x3f(v_snd_1958_);
if (v___x_1962_ == 0)
{
lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1966_; 
lean_del_object(v___x_1960_);
lean_dec(v_fst_1957_);
lean_dec(v_fst_1956_);
lean_del_object(v___x_1949_);
v___x_1963_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___closed__1);
v___x_1964_ = l_Lean_MessageData_ofExpr(v_snd_1958_);
if (v_isShared_1955_ == 0)
{
lean_ctor_set_tag(v___x_1954_, 7);
lean_ctor_set(v___x_1954_, 1, v___x_1964_);
lean_ctor_set(v___x_1954_, 0, v___x_1963_);
v___x_1966_ = v___x_1954_;
goto v_reusejp_1965_;
}
else
{
lean_object* v_reuseFailAlloc_1968_; 
v_reuseFailAlloc_1968_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1968_, 0, v___x_1963_);
lean_ctor_set(v_reuseFailAlloc_1968_, 1, v___x_1964_);
v___x_1966_ = v_reuseFailAlloc_1968_;
goto v_reusejp_1965_;
}
v_reusejp_1965_:
{
lean_object* v___x_1967_; 
v___x_1967_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg(v___x_1966_, v_a_1941_, v_a_1942_, v_a_1943_, v_a_1944_);
return v___x_1967_;
}
}
else
{
lean_object* v___x_1970_; 
lean_dec(v_snd_1958_);
lean_del_object(v___x_1954_);
if (v_isShared_1961_ == 0)
{
lean_ctor_set(v___x_1960_, 1, v_fst_1957_);
lean_ctor_set(v___x_1960_, 0, v_fst_1956_);
v___x_1970_ = v___x_1960_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1974_; 
v_reuseFailAlloc_1974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1974_, 0, v_fst_1956_);
lean_ctor_set(v_reuseFailAlloc_1974_, 1, v_fst_1957_);
v___x_1970_ = v_reuseFailAlloc_1974_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
lean_object* v___x_1972_; 
if (v_isShared_1950_ == 0)
{
lean_ctor_set(v___x_1949_, 0, v___x_1970_);
v___x_1972_ = v___x_1949_;
goto v_reusejp_1971_;
}
else
{
lean_object* v_reuseFailAlloc_1973_; 
v_reuseFailAlloc_1973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1973_, 0, v___x_1970_);
v___x_1972_ = v_reuseFailAlloc_1973_;
goto v_reusejp_1971_;
}
v_reusejp_1971_:
{
return v___x_1972_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_1986_; 
v_a_1979_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_1986_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_1986_ == 0)
{
v___x_1981_ = v___x_1946_;
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_a_1979_);
lean_dec(v___x_1946_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1984_; 
if (v_isShared_1982_ == 0)
{
v___x_1984_ = v___x_1981_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v_a_1979_);
v___x_1984_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
return v___x_1984_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr___boxed(lean_object* v_e_1987_, lean_object* v_a_1988_, lean_object* v_a_1989_, lean_object* v_a_1990_, lean_object* v_a_1991_, lean_object* v_a_1992_){
_start:
{
lean_object* v_res_1993_; 
v_res_1993_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_e_1987_, v_a_1988_, v_a_1989_, v_a_1990_, v_a_1991_);
lean_dec(v_a_1991_);
lean_dec_ref(v_a_1990_);
lean_dec(v_a_1989_);
lean_dec_ref(v_a_1988_);
return v_res_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0(lean_object* v_00_u03b1_1994_, lean_object* v_msg_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_){
_start:
{
lean_object* v___x_2001_; 
v___x_2001_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___redArg(v_msg_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_);
return v___x_2001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0___boxed(lean_object* v_00_u03b1_2002_, lean_object* v_msg_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_){
_start:
{
lean_object* v_res_2009_; 
v_res_2009_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_parseCompAndExpr_spec__0(v_00_u03b1_2002_, v_msg_2003_, v___y_2004_, v___y_2005_, v___y_2006_, v___y_2007_);
lean_dec(v___y_2007_);
lean_dec_ref(v___y_2006_);
lean_dec(v___y_2005_);
lean_dec_ref(v___y_2004_);
return v_res_2009_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8(void){
_start:
{
lean_object* v___x_2027_; 
v___x_2027_ = l_Array_mkArray0(lean_box(0));
return v___x_2027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(lean_object* v_c_2033_, lean_object* v_h_2034_, lean_object* v_a_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_){
_start:
{
lean_object* v___x_2040_; 
lean_inc(v_a_2038_);
lean_inc_ref(v_a_2037_);
lean_inc(v_a_2036_);
lean_inc_ref(v_a_2035_);
lean_inc_ref(v_h_2034_);
v___x_2040_ = lean_infer_type(v_h_2034_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2040_) == 0)
{
lean_object* v_a_2041_; lean_object* v___x_2042_; 
v_a_2041_ = lean_ctor_get(v___x_2040_, 0);
lean_inc_n(v_a_2041_, 2);
lean_dec_ref_known(v___x_2040_, 1);
v___x_2042_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_a_2041_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2042_) == 0)
{
lean_object* v_a_2043_; lean_object* v___x_2045_; uint8_t v_isShared_2046_; uint8_t v_isSharedCheck_2193_; 
v_a_2043_ = lean_ctor_get(v___x_2042_, 0);
v_isSharedCheck_2193_ = !lean_is_exclusive(v___x_2042_);
if (v_isSharedCheck_2193_ == 0)
{
v___x_2045_ = v___x_2042_;
v_isShared_2046_ = v_isSharedCheck_2193_;
goto v_resetjp_2044_;
}
else
{
lean_inc(v_a_2043_);
lean_dec(v___x_2042_);
v___x_2045_ = lean_box(0);
v_isShared_2046_ = v_isSharedCheck_2193_;
goto v_resetjp_2044_;
}
v_resetjp_2044_:
{
lean_object* v_fst_2047_; lean_object* v_snd_2048_; lean_object* v___x_2050_; uint8_t v_isShared_2051_; uint8_t v_isSharedCheck_2192_; 
v_fst_2047_ = lean_ctor_get(v_a_2043_, 0);
v_snd_2048_ = lean_ctor_get(v_a_2043_, 1);
v_isSharedCheck_2192_ = !lean_is_exclusive(v_a_2043_);
if (v_isSharedCheck_2192_ == 0)
{
v___x_2050_ = v_a_2043_;
v_isShared_2051_ = v_isSharedCheck_2192_;
goto v_resetjp_2049_;
}
else
{
lean_inc(v_snd_2048_);
lean_inc(v_fst_2047_);
lean_dec(v_a_2043_);
v___x_2050_ = lean_box(0);
v_isShared_2051_ = v_isSharedCheck_2192_;
goto v_resetjp_2049_;
}
v_resetjp_2049_:
{
lean_object* v___x_2052_; uint8_t v___x_2053_; 
v___x_2052_ = lean_unsigned_to_nat(0u);
v___x_2053_ = lean_nat_dec_eq(v_c_2033_, v___x_2052_);
if (v___x_2053_ == 0)
{
lean_object* v___x_2054_; uint8_t v___x_2055_; 
lean_dec(v_snd_2048_);
v___x_2054_ = lean_unsigned_to_nat(1u);
v___x_2055_ = lean_nat_dec_eq(v_c_2033_, v___x_2054_);
if (v___x_2055_ == 0)
{
lean_object* v___x_2056_; 
lean_del_object(v___x_2050_);
lean_del_object(v___x_2045_);
v___x_2056_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_2041_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2056_) == 0)
{
lean_object* v_a_2057_; lean_object* v_snd_2058_; lean_object* v___x_2060_; uint8_t v_isShared_2061_; uint8_t v_isSharedCheck_2150_; 
v_a_2057_ = lean_ctor_get(v___x_2056_, 0);
lean_inc(v_a_2057_);
lean_dec_ref_known(v___x_2056_, 1);
v_snd_2058_ = lean_ctor_get(v_a_2057_, 1);
v_isSharedCheck_2150_ = !lean_is_exclusive(v_a_2057_);
if (v_isSharedCheck_2150_ == 0)
{
lean_object* v_unused_2151_; 
v_unused_2151_ = lean_ctor_get(v_a_2057_, 0);
lean_dec(v_unused_2151_);
v___x_2060_ = v_a_2057_;
v_isShared_2061_ = v_isSharedCheck_2150_;
goto v_resetjp_2059_;
}
else
{
lean_inc(v_snd_2058_);
lean_dec(v_a_2057_);
v___x_2060_ = lean_box(0);
v_isShared_2061_ = v_isSharedCheck_2150_;
goto v_resetjp_2059_;
}
v_resetjp_2059_:
{
lean_object* v_fst_2062_; lean_object* v___x_2064_; uint8_t v_isShared_2065_; uint8_t v_isSharedCheck_2148_; 
v_fst_2062_ = lean_ctor_get(v_snd_2058_, 0);
v_isSharedCheck_2148_ = !lean_is_exclusive(v_snd_2058_);
if (v_isSharedCheck_2148_ == 0)
{
lean_object* v_unused_2149_; 
v_unused_2149_ = lean_ctor_get(v_snd_2058_, 1);
lean_dec(v_unused_2149_);
v___x_2064_ = v_snd_2058_;
v_isShared_2065_ = v_isSharedCheck_2148_;
goto v_resetjp_2063_;
}
else
{
lean_inc(v_fst_2062_);
lean_dec(v_snd_2058_);
v___x_2064_ = lean_box(0);
v_isShared_2065_ = v_isSharedCheck_2148_;
goto v_resetjp_2063_;
}
v_resetjp_2063_:
{
lean_object* v___x_2066_; 
lean_inc(v_fst_2062_);
v___x_2066_ = lp_mathlib_Lean_Expr_ofNat(v_fst_2062_, v_c_2033_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2066_) == 0)
{
lean_object* v_a_2067_; lean_object* v___x_2068_; 
v_a_2067_ = lean_ctor_get(v___x_2066_, 0);
lean_inc(v_a_2067_);
lean_dec_ref_known(v___x_2066_, 1);
v___x_2068_ = lp_mathlib_Lean_Expr_ofNat(v_fst_2062_, v___x_2052_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2068_) == 0)
{
lean_object* v_a_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; lean_object* v___x_2075_; 
v_a_2069_ = lean_ctor_get(v___x_2068_, 0);
lean_inc(v_a_2069_);
lean_dec_ref_known(v___x_2068_, 1);
v___x_2070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__2));
v___x_2071_ = lean_unsigned_to_nat(2u);
v___x_2072_ = lean_mk_empty_array_with_capacity(v___x_2071_);
lean_inc_ref(v___x_2072_);
v___x_2073_ = lean_array_push(v___x_2072_, v_a_2067_);
v___x_2074_ = lean_array_push(v___x_2073_, v_a_2069_);
v___x_2075_ = l_Lean_Meta_mkAppM(v___x_2070_, v___x_2074_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2075_) == 0)
{
lean_object* v_a_2076_; lean_object* v_ref_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2082_; 
v_a_2076_ = lean_ctor_get(v___x_2075_, 0);
lean_inc(v_a_2076_);
lean_dec_ref_known(v___x_2075_, 1);
v_ref_2077_ = lean_ctor_get(v_a_2037_, 5);
v___x_2078_ = l_Lean_SourceInfo_fromRef(v_ref_2077_, v___x_2055_);
v___x_2079_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__4));
v___x_2080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__5));
lean_inc(v___x_2078_);
if (v_isShared_2061_ == 0)
{
lean_ctor_set_tag(v___x_2060_, 2);
lean_ctor_set(v___x_2060_, 1, v___x_2080_);
lean_ctor_set(v___x_2060_, 0, v___x_2078_);
v___x_2082_ = v___x_2060_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2123_; 
v_reuseFailAlloc_2123_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2123_, 0, v___x_2078_);
lean_ctor_set(v_reuseFailAlloc_2123_, 1, v___x_2080_);
v___x_2082_ = v_reuseFailAlloc_2123_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; 
v___x_2083_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__7));
v___x_2084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam___closed__8));
v___x_2085_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8, &lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__8);
lean_inc_n(v___x_2078_, 2);
v___x_2086_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2086_, 0, v___x_2078_);
lean_ctor_set(v___x_2086_, 1, v___x_2084_);
lean_ctor_set(v___x_2086_, 2, v___x_2085_);
lean_inc_ref_n(v___x_2086_, 3);
v___x_2087_ = l_Lean_Syntax_node1(v___x_2078_, v___x_2083_, v___x_2086_);
v___x_2088_ = l_Lean_Syntax_node5(v___x_2078_, v___x_2079_, v___x_2082_, v___x_2087_, v___x_2086_, v___x_2086_, v___x_2086_);
v___x_2089_ = lp_mathlib_synthesizeUsingTactic_x27___redArg(v_a_2076_, v___x_2088_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_object* v_a_2090_; uint8_t v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc(v_a_2090_);
lean_dec_ref_known(v___x_2089_, 1);
v___x_2091_ = lean_unbox(v_fst_2047_);
v___x_2092_ = lp_mathlib_Mathlib_Ineq_toConstMulName(v___x_2091_);
v___x_2093_ = lean_array_push(v___x_2072_, v_h_2034_);
v___x_2094_ = lean_array_push(v___x_2093_, v_a_2090_);
v___x_2095_ = l_Lean_Meta_mkAppM(v___x_2092_, v___x_2094_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2095_) == 0)
{
lean_object* v_a_2096_; lean_object* v___x_2098_; uint8_t v_isShared_2099_; uint8_t v_isSharedCheck_2106_; 
v_a_2096_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2106_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2106_ == 0)
{
v___x_2098_ = v___x_2095_;
v_isShared_2099_ = v_isSharedCheck_2106_;
goto v_resetjp_2097_;
}
else
{
lean_inc(v_a_2096_);
lean_dec(v___x_2095_);
v___x_2098_ = lean_box(0);
v_isShared_2099_ = v_isSharedCheck_2106_;
goto v_resetjp_2097_;
}
v_resetjp_2097_:
{
lean_object* v___x_2101_; 
if (v_isShared_2065_ == 0)
{
lean_ctor_set(v___x_2064_, 1, v_a_2096_);
lean_ctor_set(v___x_2064_, 0, v_fst_2047_);
v___x_2101_ = v___x_2064_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2105_; 
v_reuseFailAlloc_2105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2105_, 0, v_fst_2047_);
lean_ctor_set(v_reuseFailAlloc_2105_, 1, v_a_2096_);
v___x_2101_ = v_reuseFailAlloc_2105_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
lean_object* v___x_2103_; 
if (v_isShared_2099_ == 0)
{
lean_ctor_set(v___x_2098_, 0, v___x_2101_);
v___x_2103_ = v___x_2098_;
goto v_reusejp_2102_;
}
else
{
lean_object* v_reuseFailAlloc_2104_; 
v_reuseFailAlloc_2104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2104_, 0, v___x_2101_);
v___x_2103_ = v_reuseFailAlloc_2104_;
goto v_reusejp_2102_;
}
v_reusejp_2102_:
{
return v___x_2103_;
}
}
}
}
else
{
lean_object* v_a_2107_; lean_object* v___x_2109_; uint8_t v_isShared_2110_; uint8_t v_isSharedCheck_2114_; 
lean_del_object(v___x_2064_);
lean_dec(v_fst_2047_);
v_a_2107_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2114_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2114_ == 0)
{
v___x_2109_ = v___x_2095_;
v_isShared_2110_ = v_isSharedCheck_2114_;
goto v_resetjp_2108_;
}
else
{
lean_inc(v_a_2107_);
lean_dec(v___x_2095_);
v___x_2109_ = lean_box(0);
v_isShared_2110_ = v_isSharedCheck_2114_;
goto v_resetjp_2108_;
}
v_resetjp_2108_:
{
lean_object* v___x_2112_; 
if (v_isShared_2110_ == 0)
{
v___x_2112_ = v___x_2109_;
goto v_reusejp_2111_;
}
else
{
lean_object* v_reuseFailAlloc_2113_; 
v_reuseFailAlloc_2113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2113_, 0, v_a_2107_);
v___x_2112_ = v_reuseFailAlloc_2113_;
goto v_reusejp_2111_;
}
v_reusejp_2111_:
{
return v___x_2112_;
}
}
}
}
else
{
lean_object* v_a_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2122_; 
lean_dec_ref(v___x_2072_);
lean_del_object(v___x_2064_);
lean_dec(v_fst_2047_);
lean_dec_ref(v_h_2034_);
v_a_2115_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2122_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2122_ == 0)
{
v___x_2117_ = v___x_2089_;
v_isShared_2118_ = v_isSharedCheck_2122_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_a_2115_);
lean_dec(v___x_2089_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2122_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v___x_2120_; 
if (v_isShared_2118_ == 0)
{
v___x_2120_ = v___x_2117_;
goto v_reusejp_2119_;
}
else
{
lean_object* v_reuseFailAlloc_2121_; 
v_reuseFailAlloc_2121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2121_, 0, v_a_2115_);
v___x_2120_ = v_reuseFailAlloc_2121_;
goto v_reusejp_2119_;
}
v_reusejp_2119_:
{
return v___x_2120_;
}
}
}
}
}
else
{
lean_object* v_a_2124_; lean_object* v___x_2126_; uint8_t v_isShared_2127_; uint8_t v_isSharedCheck_2131_; 
lean_dec_ref(v___x_2072_);
lean_del_object(v___x_2064_);
lean_del_object(v___x_2060_);
lean_dec(v_fst_2047_);
lean_dec_ref(v_h_2034_);
v_a_2124_ = lean_ctor_get(v___x_2075_, 0);
v_isSharedCheck_2131_ = !lean_is_exclusive(v___x_2075_);
if (v_isSharedCheck_2131_ == 0)
{
v___x_2126_ = v___x_2075_;
v_isShared_2127_ = v_isSharedCheck_2131_;
goto v_resetjp_2125_;
}
else
{
lean_inc(v_a_2124_);
lean_dec(v___x_2075_);
v___x_2126_ = lean_box(0);
v_isShared_2127_ = v_isSharedCheck_2131_;
goto v_resetjp_2125_;
}
v_resetjp_2125_:
{
lean_object* v___x_2129_; 
if (v_isShared_2127_ == 0)
{
v___x_2129_ = v___x_2126_;
goto v_reusejp_2128_;
}
else
{
lean_object* v_reuseFailAlloc_2130_; 
v_reuseFailAlloc_2130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2130_, 0, v_a_2124_);
v___x_2129_ = v_reuseFailAlloc_2130_;
goto v_reusejp_2128_;
}
v_reusejp_2128_:
{
return v___x_2129_;
}
}
}
}
else
{
lean_object* v_a_2132_; lean_object* v___x_2134_; uint8_t v_isShared_2135_; uint8_t v_isSharedCheck_2139_; 
lean_dec(v_a_2067_);
lean_del_object(v___x_2064_);
lean_del_object(v___x_2060_);
lean_dec(v_fst_2047_);
lean_dec_ref(v_h_2034_);
v_a_2132_ = lean_ctor_get(v___x_2068_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_2068_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_2134_ = v___x_2068_;
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
else
{
lean_inc(v_a_2132_);
lean_dec(v___x_2068_);
v___x_2134_ = lean_box(0);
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
v_resetjp_2133_:
{
lean_object* v___x_2137_; 
if (v_isShared_2135_ == 0)
{
v___x_2137_ = v___x_2134_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v_a_2132_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
return v___x_2137_;
}
}
}
}
else
{
lean_object* v_a_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2147_; 
lean_del_object(v___x_2064_);
lean_dec(v_fst_2062_);
lean_del_object(v___x_2060_);
lean_dec(v_fst_2047_);
lean_dec_ref(v_h_2034_);
v_a_2140_ = lean_ctor_get(v___x_2066_, 0);
v_isSharedCheck_2147_ = !lean_is_exclusive(v___x_2066_);
if (v_isSharedCheck_2147_ == 0)
{
v___x_2142_ = v___x_2066_;
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_a_2140_);
lean_dec(v___x_2066_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v___x_2145_; 
if (v_isShared_2143_ == 0)
{
v___x_2145_ = v___x_2142_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2146_; 
v_reuseFailAlloc_2146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2146_, 0, v_a_2140_);
v___x_2145_ = v_reuseFailAlloc_2146_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
return v___x_2145_;
}
}
}
}
}
}
else
{
lean_object* v_a_2152_; lean_object* v___x_2154_; uint8_t v_isShared_2155_; uint8_t v_isSharedCheck_2159_; 
lean_dec(v_fst_2047_);
lean_dec_ref(v_h_2034_);
lean_dec(v_c_2033_);
v_a_2152_ = lean_ctor_get(v___x_2056_, 0);
v_isSharedCheck_2159_ = !lean_is_exclusive(v___x_2056_);
if (v_isSharedCheck_2159_ == 0)
{
v___x_2154_ = v___x_2056_;
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
else
{
lean_inc(v_a_2152_);
lean_dec(v___x_2056_);
v___x_2154_ = lean_box(0);
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
v_resetjp_2153_:
{
lean_object* v___x_2157_; 
if (v_isShared_2155_ == 0)
{
v___x_2157_ = v___x_2154_;
goto v_reusejp_2156_;
}
else
{
lean_object* v_reuseFailAlloc_2158_; 
v_reuseFailAlloc_2158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2158_, 0, v_a_2152_);
v___x_2157_ = v_reuseFailAlloc_2158_;
goto v_reusejp_2156_;
}
v_reusejp_2156_:
{
return v___x_2157_;
}
}
}
}
else
{
lean_object* v___x_2161_; 
lean_dec(v_a_2041_);
lean_dec(v_c_2033_);
if (v_isShared_2051_ == 0)
{
lean_ctor_set(v___x_2050_, 1, v_h_2034_);
v___x_2161_ = v___x_2050_;
goto v_reusejp_2160_;
}
else
{
lean_object* v_reuseFailAlloc_2165_; 
v_reuseFailAlloc_2165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2165_, 0, v_fst_2047_);
lean_ctor_set(v_reuseFailAlloc_2165_, 1, v_h_2034_);
v___x_2161_ = v_reuseFailAlloc_2165_;
goto v_reusejp_2160_;
}
v_reusejp_2160_:
{
lean_object* v___x_2163_; 
if (v_isShared_2046_ == 0)
{
lean_ctor_set(v___x_2045_, 0, v___x_2161_);
v___x_2163_ = v___x_2045_;
goto v_reusejp_2162_;
}
else
{
lean_object* v_reuseFailAlloc_2164_; 
v_reuseFailAlloc_2164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2164_, 0, v___x_2161_);
v___x_2163_ = v_reuseFailAlloc_2164_;
goto v_reusejp_2162_;
}
v_reusejp_2162_:
{
return v___x_2163_;
}
}
}
}
else
{
lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; 
lean_dec(v_fst_2047_);
lean_del_object(v___x_2045_);
lean_dec(v_a_2041_);
lean_dec_ref(v_h_2034_);
lean_dec(v_c_2033_);
v___x_2166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___closed__11));
v___x_2167_ = lean_unsigned_to_nat(1u);
v___x_2168_ = lean_mk_empty_array_with_capacity(v___x_2167_);
v___x_2169_ = lean_array_push(v___x_2168_, v_snd_2048_);
v___x_2170_ = l_Lean_Meta_mkAppM(v___x_2166_, v___x_2169_, v_a_2035_, v_a_2036_, v_a_2037_, v_a_2038_);
if (lean_obj_tag(v___x_2170_) == 0)
{
lean_object* v_a_2171_; lean_object* v___x_2173_; uint8_t v_isShared_2174_; uint8_t v_isSharedCheck_2183_; 
v_a_2171_ = lean_ctor_get(v___x_2170_, 0);
v_isSharedCheck_2183_ = !lean_is_exclusive(v___x_2170_);
if (v_isSharedCheck_2183_ == 0)
{
v___x_2173_ = v___x_2170_;
v_isShared_2174_ = v_isSharedCheck_2183_;
goto v_resetjp_2172_;
}
else
{
lean_inc(v_a_2171_);
lean_dec(v___x_2170_);
v___x_2173_ = lean_box(0);
v_isShared_2174_ = v_isSharedCheck_2183_;
goto v_resetjp_2172_;
}
v_resetjp_2172_:
{
uint8_t v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2178_; 
v___x_2175_ = 0;
v___x_2176_ = lean_box(v___x_2175_);
if (v_isShared_2051_ == 0)
{
lean_ctor_set(v___x_2050_, 1, v_a_2171_);
lean_ctor_set(v___x_2050_, 0, v___x_2176_);
v___x_2178_ = v___x_2050_;
goto v_reusejp_2177_;
}
else
{
lean_object* v_reuseFailAlloc_2182_; 
v_reuseFailAlloc_2182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2182_, 0, v___x_2176_);
lean_ctor_set(v_reuseFailAlloc_2182_, 1, v_a_2171_);
v___x_2178_ = v_reuseFailAlloc_2182_;
goto v_reusejp_2177_;
}
v_reusejp_2177_:
{
lean_object* v___x_2180_; 
if (v_isShared_2174_ == 0)
{
lean_ctor_set(v___x_2173_, 0, v___x_2178_);
v___x_2180_ = v___x_2173_;
goto v_reusejp_2179_;
}
else
{
lean_object* v_reuseFailAlloc_2181_; 
v_reuseFailAlloc_2181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2181_, 0, v___x_2178_);
v___x_2180_ = v_reuseFailAlloc_2181_;
goto v_reusejp_2179_;
}
v_reusejp_2179_:
{
return v___x_2180_;
}
}
}
}
else
{
lean_object* v_a_2184_; lean_object* v___x_2186_; uint8_t v_isShared_2187_; uint8_t v_isSharedCheck_2191_; 
lean_del_object(v___x_2050_);
v_a_2184_ = lean_ctor_get(v___x_2170_, 0);
v_isSharedCheck_2191_ = !lean_is_exclusive(v___x_2170_);
if (v_isSharedCheck_2191_ == 0)
{
v___x_2186_ = v___x_2170_;
v_isShared_2187_ = v_isSharedCheck_2191_;
goto v_resetjp_2185_;
}
else
{
lean_inc(v_a_2184_);
lean_dec(v___x_2170_);
v___x_2186_ = lean_box(0);
v_isShared_2187_ = v_isSharedCheck_2191_;
goto v_resetjp_2185_;
}
v_resetjp_2185_:
{
lean_object* v___x_2189_; 
if (v_isShared_2187_ == 0)
{
v___x_2189_ = v___x_2186_;
goto v_reusejp_2188_;
}
else
{
lean_object* v_reuseFailAlloc_2190_; 
v_reuseFailAlloc_2190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2190_, 0, v_a_2184_);
v___x_2189_ = v_reuseFailAlloc_2190_;
goto v_reusejp_2188_;
}
v_reusejp_2188_:
{
return v___x_2189_;
}
}
}
}
}
}
}
else
{
lean_dec(v_a_2041_);
lean_dec_ref(v_h_2034_);
lean_dec(v_c_2033_);
return v___x_2042_;
}
}
else
{
lean_object* v_a_2194_; lean_object* v___x_2196_; uint8_t v_isShared_2197_; uint8_t v_isSharedCheck_2201_; 
lean_dec_ref(v_h_2034_);
lean_dec(v_c_2033_);
v_a_2194_ = lean_ctor_get(v___x_2040_, 0);
v_isSharedCheck_2201_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2201_ == 0)
{
v___x_2196_ = v___x_2040_;
v_isShared_2197_ = v_isSharedCheck_2201_;
goto v_resetjp_2195_;
}
else
{
lean_inc(v_a_2194_);
lean_dec(v___x_2040_);
v___x_2196_ = lean_box(0);
v_isShared_2197_ = v_isSharedCheck_2201_;
goto v_resetjp_2195_;
}
v_resetjp_2195_:
{
lean_object* v___x_2199_; 
if (v_isShared_2197_ == 0)
{
v___x_2199_ = v___x_2196_;
goto v_reusejp_2198_;
}
else
{
lean_object* v_reuseFailAlloc_2200_; 
v_reuseFailAlloc_2200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2200_, 0, v_a_2194_);
v___x_2199_ = v_reuseFailAlloc_2200_;
goto v_reusejp_2198_;
}
v_reusejp_2198_:
{
return v___x_2199_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf___boxed(lean_object* v_c_2202_, lean_object* v_h_2203_, lean_object* v_a_2204_, lean_object* v_a_2205_, lean_object* v_a_2206_, lean_object* v_a_2207_, lean_object* v_a_2208_){
_start:
{
lean_object* v_res_2209_; 
v_res_2209_ = lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(v_c_2202_, v_h_2203_, v_a_2204_, v_a_2205_, v_a_2206_, v_a_2207_);
lean_dec(v_a_2207_);
lean_dec_ref(v_a_2206_);
lean_dec(v_a_2205_);
lean_dec_ref(v_a_2204_);
return v_res_2209_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_189509315____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linarith_Datatypes_0__initFn_00___x40_Mathlib_Tactic_Linarith_Datatypes_3248332875____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat = _init_lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_Comp_ToFormat);
lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam = _init_lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_PreprocessorBase_name___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
}
#ifdef __cplusplus
}
#endif
