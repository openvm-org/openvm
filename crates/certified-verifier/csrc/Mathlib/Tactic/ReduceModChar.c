// Lean compiler output
// Module: Mathlib.Tactic.ReduceModChar
// Imports: public import Init public meta import Init public meta import Mathlib.Util.AtLocation public import Mathlib.Data.ZMod.Basic public import Mathlib.RingTheory.Polynomial.Basic public import Mathlib.Tactic.NormNum.PowMod public import Mathlib.Tactic.ReduceModChar.Ext import Mathlib.Tactic.NormNum.DivMod
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* l_Lean_Meta_SimpExtension_getTheorems___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_preDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_mkEqTransResultStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_postDefault___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* lean_io_mono_nanos_now();
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ReduceModChar"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "CharP"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "isInt_of_mod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__3_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value),LEAN_SCALAR_PTR_LITERAL(57, 141, 37, 198, 187, 165, 86, 67)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__3_value),LEAN_SCALAR_PTR_LITERAL(248, 97, 247, 131, 84, 157, 37, 196)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__5_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__6 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__6_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__5_value),LEAN_SCALAR_PTR_LITERAL(93, 4, 3, 35, 188, 254, 191, 190)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__6_value),LEAN_SCALAR_PTR_LITERAL(120, 199, 142, 238, 9, 44, 94, 134)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__8 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__8_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__8_value),LEAN_SCALAR_PTR_LITERAL(242, 7, 29, 140, 31, 32, 204, 87)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__9 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__9_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instMod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__10 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__10_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__11 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__11_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__14 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__14_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsInt"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__19 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__19_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instRing"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__21 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__21_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__21_value),LEAN_SCALAR_PTR_LITERAL(42, 135, 58, 37, 72, 75, 21, 158)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "raw_refl"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__24 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__24_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__19_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value_aux_3),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__24_value),LEAN_SCALAR_PTR_LITERAL(66, 38, 185, 50, 74, 151, 245, 37)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__27 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__27_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__28 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__28_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__27_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__28_value),LEAN_SCALAR_PTR_LITERAL(19, 237, 167, 212, 100, 179, 19, 112)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__32 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__32_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__33 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__33_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__32_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__33_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "instAddMonoidWithOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__37 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__37_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__37_value),LEAN_SCALAR_PTR_LITERAL(129, 65, 157, 144, 0, 78, 170, 16)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsNat"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__44 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__44_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isNat_natCast"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__45 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__45_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__45_value),LEAN_SCALAR_PTR_LITERAL(162, 104, 58, 184, 15, 134, 105, 111)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__16_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__17_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__44_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value_aux_3),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__24_value),LEAN_SCALAR_PTR_LITERAL(163, 125, 133, 66, 231, 251, 113, 144)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__27_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__0_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normPow___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__1;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__3_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__2_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__2_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__3_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__4_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__5_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__5_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__6 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__6_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__8 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__8_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__7 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__7_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__7_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__8_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__9 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__9_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__11 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__11_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__10 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__10_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__10_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__11_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__12 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__12_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__14 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__14_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__13 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__13_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__13_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__14_value),LEAN_SCALAR_PTR_LITERAL(86, 172, 133, 187, 121, 84, 206, 170)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__15 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__15_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__17 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__17_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__16 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__17_value),LEAN_SCALAR_PTR_LITERAL(236, 38, 194, 105, 137, 30, 136, 223)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__18 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__18_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isNat_pow"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__19 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__19_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value),LEAN_SCALAR_PTR_LITERAL(57, 141, 37, 198, 187, 165, 86, 67)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__19_value),LEAN_SCALAR_PTR_LITERAL(224, 207, 57, 62, 133, 30, 81, 81)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__20 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__20_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__22 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__22_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__21 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__21_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__21_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normPow___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__22_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___closed__23 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__0_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__1_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NegZeroClass"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__3_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toNeg"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__4_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(156, 44, 233, 53, 1, 106, 24, 217)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(124, 136, 108, 160, 134, 153, 101, 8)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__6 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__6_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toNegZeroClass"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__7 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__7_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(135, 233, 160, 34, 207, 245, 132, 138)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(107, 179, 145, 12, 37, 42, 18, 108)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__9 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__9_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toSubNegZeroMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__10 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__10_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(203, 24, 17, 79, 61, 156, 198, 150)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(94, 234, 159, 237, 9, 124, 201, 94)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "SubtractionCommMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__12 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__12_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toSubtractionMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__13 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__13_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(100, 8, 183, 201, 110, 57, 85, 213)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(203, 26, 135, 240, 118, 74, 112, 111)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__15 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__15_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toDivisionAddCommMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__16 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__16_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__16_value),LEAN_SCALAR_PTR_LITERAL(65, 138, 55, 164, 85, 246, 87, 209)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommGroup"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__18 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__18_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(121, 151, 225, 139, 113, 68, 25, 156)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__20 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__20_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__21 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__21_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(121, 130, 45, 212, 110, 237, 236, 233)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__21_value),LEAN_SCALAR_PTR_LITERAL(231, 253, 204, 163, 168, 77, 27, 58)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHSub"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__23 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__23_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__23_value),LEAN_SCALAR_PTR_LITERAL(32, 225, 92, 14, 170, 61, 170, 140)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__24 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__24_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "SubNegMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__25 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__25_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toSub"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__26 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__26_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__25_value),LEAN_SCALAR_PTR_LITERAL(161, 3, 69, 109, 235, 35, 121, 64)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(17, 223, 222, 114, 35, 206, 250, 124)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "AddGroup"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__28 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__28_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toSubNegMonoid"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__29 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__29_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__28_value),LEAN_SCALAR_PTR_LITERAL(211, 76, 74, 39, 69, 162, 229, 135)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(237, 249, 208, 19, 139, 128, 45, 144)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "AddGroupWithOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__31 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__31_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toAddGroup"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__32 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__32_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__31_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__32_value),LEAN_SCALAR_PTR_LITERAL(14, 89, 255, 1, 224, 64, 98, 35)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toAddGroupWithOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__34 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__34_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__34_value),LEAN_SCALAR_PTR_LITERAL(99, 161, 243, 168, 232, 89, 236, 229)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__36 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__36_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__31_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__36_value),LEAN_SCALAR_PTR_LITERAL(226, 82, 90, 134, 221, 253, 108, 55)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__38 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__38_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__39 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__39_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__38_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__39_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__41 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__41_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__43 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__43_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__44 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__44_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__43_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__44_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__46 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__46_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__32_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__46_value),LEAN_SCALAR_PTR_LITERAL(52, 219, 71, 246, 148, 114, 208, 126)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__48 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__48_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__49 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__49_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__48_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__49_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__51 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__51_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__51_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__52 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__52_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__53 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__53_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__54 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__54_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__53_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__54_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__56 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__56_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__56_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__57 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__57_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "neg_eq_sub_one_mul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__58 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__58_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value),LEAN_SCALAR_PTR_LITERAL(57, 141, 37, 198, 187, 165, 86, 67)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__58_value),LEAN_SCALAR_PTR_LITERAL(68, 74, 252, 107, 222, 255, 212, 50)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "normNeg: nothing useful to do in negative characteristic"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__60 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__60_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "normNeg: evaluating `"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__62 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__62_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNeg___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = " - 1` should give an integer result"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__64 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__64_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "neg_mul_eq_sub_one_mul"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__0_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value),LEAN_SCALAR_PTR_LITERAL(57, 141, 37, 198, 187, 165, 86, 67)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__0_value),LEAN_SCALAR_PTR_LITERAL(101, 130, 95, 3, 74, 13, 80, 174)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "normNegCoeffMul: nothing useful to do in negative characteristic"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__2_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "normNegCoeffMul: evaluating `"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__4_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_instInhabitedTypeToCharPResult(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_instInhabitedTypeToCharPResult___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__2_value),LEAN_SCALAR_PTR_LITERAL(208, 54, 114, 3, 62, 227, 58, 188)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normPow___closed__16_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__0_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ZMod"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Polynomial"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ring"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__3_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__3_value),LEAN_SCALAR_PTR_LITERAL(191, 110, 189, 234, 0, 87, 224, 95)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instCharP"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__5_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__5_value),LEAN_SCALAR_PTR_LITERAL(77, 107, 138, 254, 124, 25, 127, 63)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 146, 213, 214, 85, 148, 206, 112)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__7 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__7_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CommRing"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__9 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__9_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__10 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__10_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__9_value),LEAN_SCALAR_PTR_LITERAL(78, 130, 181, 61, 179, 129, 164, 15)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__10_value),LEAN_SCALAR_PTR_LITERAL(184, 164, 171, 191, 166, 224, 196, 206)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "commRing"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__13 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__13_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 146, 213, 214, 85, 148, 206, 112)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__13_value),LEAN_SCALAR_PTR_LITERAL(255, 87, 49, 173, 194, 87, 37, 26)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "charP"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__16 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__16_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 146, 213, 214, 85, 148, 206, 112)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__16_value),LEAN_SCALAR_PTR_LITERAL(199, 216, 131, 144, 210, 177, 100, 145)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "inferred type `"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__0_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "` does not have a known characteristic"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__2_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "expected "};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__4_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = " to be a `Type _`, not `Sort "};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__6 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__6_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__8 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__8_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_derive___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0_value;
static const lean_closure_object lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Tactic_ReduceModChar_derive___lam__3___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__1_value;
static const lean_closure_object lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_Simp_postDefault___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0_value)} };
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__7(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0;
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "reduce_mod_char"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_derive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(133, 253, 114, 162, 14, 29, 109, 209)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__1_value;
static const lean_closure_object lp_mathlib_Tactic_ReduceModChar_derive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Tactic_ReduceModChar_derive___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__2_value;
static const lean_closure_object lp_mathlib_Tactic_ReduceModChar_derive___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__3_value;
static const lean_closure_object lp_mathlib_Tactic_ReduceModChar_derive___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Tactic_ReduceModChar_derive___lam__3___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__4_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_derive___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "internal error: reduce_mod_char not registered as simp extension"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__5_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__6;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_derive___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_derive___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 234, 122, 196, 98, 251, 215, 88)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__7 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__7_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_derive___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__8 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__8_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_derive___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__9 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__9_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_derive___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__9_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__10 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__10_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_derive___closed__11;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_derive___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Tactic_ReduceModChar_derive___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 147, 49, 140, 199, 252, 39, 165)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__1_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__2_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_derive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__3 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__3_value;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__4 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__4_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__5 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__5_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char;
static const lean_string_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "reduce_mod_char!"};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__0 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 21, 154, 129, 199, 181, 151, 214)}};
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 75, 219, 180, 33, 92, 25, 71)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__2 = (const lean_object*)&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0;
static lean_once_cell_t lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__11));
v___x_67_ = l_Lean_stringToMessageData(v___x_66_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lean_box(0);
v___x_72_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__14));
v___x_73_ = l_Lean_Expr_const___override(v___x_72_, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_85_ = lean_box(0);
v___x_86_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__22));
v___x_87_ = l_Lean_Expr_const___override(v___x_86_, v___x_85_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_box(0);
v___x_96_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__25));
v___x_97_ = l_Lean_Expr_const___override(v___x_96_, v___x_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_104_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29));
v___x_105_ = l_Lean_Expr_const___override(v___x_104_, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_106_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15);
v___x_107_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__30);
v___x_108_ = l_Lean_Expr_app___override(v___x_107_, v___x_106_);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_115_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34));
v___x_116_ = l_Lean_Expr_const___override(v___x_115_, v___x_114_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15);
v___x_118_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__35);
v___x_119_ = l_Lean_Expr_app___override(v___x_118_, v___x_117_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_126_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_127_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__38));
v___x_128_ = l_Lean_Expr_const___override(v___x_127_, v___x_126_);
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40(void){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_129_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15);
v___x_130_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__39);
v___x_131_ = l_Lean_Expr_app___override(v___x_130_, v___x_129_);
return v___x_131_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_132_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23);
v___x_133_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__40);
v___x_134_ = l_Lean_Expr_app___override(v___x_133_, v___x_132_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_135_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41);
v___x_136_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__36);
v___x_137_ = l_Lean_Expr_app___override(v___x_136_, v___x_135_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__42);
v___x_139_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__31);
v___x_140_ = l_Lean_Expr_app___override(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_148_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_149_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__46));
v___x_150_ = l_Lean_Expr_const___override(v___x_149_, v___x_148_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_151_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15);
v___x_152_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__47);
v___x_153_ = l_Lean_Expr_app___override(v___x_152_, v___x_151_);
return v___x_153_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41);
v___x_155_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__48);
v___x_156_ = l_Lean_Expr_app___override(v___x_155_, v___x_154_);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_163_ = lean_box(0);
v___x_164_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__50));
v___x_165_ = l_Lean_Expr_const___override(v___x_164_, v___x_163_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral(lean_object* v_u_166_, lean_object* v_00_u03b1_167_, lean_object* v_n_168_, lean_object* v_n_x27_169_, lean_object* v_pn_170_, lean_object* v_e_171_, lean_object* v_x_172_, lean_object* v_instCharP_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
uint8_t v___x_179_; lean_object* v___x_180_; 
v___x_179_ = 0;
lean_inc_ref(v_e_171_);
lean_inc_ref(v_00_u03b1_167_);
lean_inc(v_u_166_);
v___x_180_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_166_, v_00_u03b1_167_, v_e_171_, v___x_179_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
if (lean_obj_tag(v___x_180_) == 0)
{
lean_object* v_a_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_290_; 
v_a_181_ = lean_ctor_get(v___x_180_, 0);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_180_);
if (v_isSharedCheck_290_ == 0)
{
v___x_183_ = v___x_180_;
v_isShared_184_ = v_isSharedCheck_290_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_a_181_);
lean_dec(v___x_180_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_290_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___y_188_; lean_object* v___y_189_; lean_object* v_a_190_; lean_object* v___y_213_; lean_object* v___y_214_; lean_object* v___y_215_; lean_object* v___y_216_; lean_object* v___y_217_; lean_object* v___y_218_; lean_object* v___y_219_; lean_object* v___y_220_; lean_object* v_a_221_; lean_object* v_a_252_; lean_object* v___x_278_; 
v___x_185_ = lean_box(0);
lean_inc_n(v_u_166_, 2);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v_u_166_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
lean_inc_ref(v_x_172_);
lean_inc_ref(v_e_171_);
lean_inc_ref(v_00_u03b1_167_);
v___x_278_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v_u_166_, v_00_u03b1_167_, v_e_171_, v_x_172_, v_a_181_);
if (lean_obj_tag(v___x_278_) == 0)
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v_a_281_; lean_object* v___x_283_; uint8_t v_isShared_284_; uint8_t v_isSharedCheck_288_; 
lean_dec_ref_known(v___x_186_, 2);
lean_del_object(v___x_183_);
lean_dec_ref(v_instCharP_173_);
lean_dec_ref(v_x_172_);
lean_dec_ref(v_e_171_);
lean_dec_ref(v_pn_170_);
lean_dec_ref(v_n_x27_169_);
lean_dec_ref(v_n_168_);
lean_dec_ref(v_00_u03b1_167_);
lean_dec(v_u_166_);
v___x_279_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_280_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_279_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
v_a_281_ = lean_ctor_get(v___x_280_, 0);
v_isSharedCheck_288_ = !lean_is_exclusive(v___x_280_);
if (v_isSharedCheck_288_ == 0)
{
v___x_283_ = v___x_280_;
v_isShared_284_ = v_isSharedCheck_288_;
goto v_resetjp_282_;
}
else
{
lean_inc(v_a_281_);
lean_dec(v___x_280_);
v___x_283_ = lean_box(0);
v_isShared_284_ = v_isSharedCheck_288_;
goto v_resetjp_282_;
}
v_resetjp_282_:
{
lean_object* v___x_286_; 
if (v_isShared_284_ == 0)
{
v___x_286_ = v___x_283_;
goto v_reusejp_285_;
}
else
{
lean_object* v_reuseFailAlloc_287_; 
v_reuseFailAlloc_287_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_287_, 0, v_a_281_);
v___x_286_ = v_reuseFailAlloc_287_;
goto v_reusejp_285_;
}
v_reusejp_285_:
{
return v___x_286_;
}
}
}
else
{
lean_object* v_val_289_; 
v_val_289_ = lean_ctor_get(v___x_278_, 0);
lean_inc(v_val_289_);
lean_dec_ref_known(v___x_278_, 1);
v_a_252_ = v_val_289_;
goto v___jp_251_;
}
v___jp_187_:
{
lean_object* v_snd_191_; lean_object* v_fst_192_; lean_object* v_fst_193_; lean_object* v_snd_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_210_; 
v_snd_191_ = lean_ctor_get(v_a_190_, 1);
lean_inc(v_snd_191_);
v_fst_192_ = lean_ctor_get(v_a_190_, 0);
lean_inc(v_fst_192_);
lean_dec_ref(v_a_190_);
v_fst_193_ = lean_ctor_get(v_snd_191_, 0);
lean_inc_n(v_fst_193_, 2);
v_snd_194_ = lean_ctor_get(v_snd_191_, 1);
lean_inc(v_snd_194_);
lean_dec(v_snd_191_);
v___x_195_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__4));
v___x_196_ = l_Lean_Expr_const___override(v___x_195_, v___x_186_);
v___x_197_ = l_Lean_Expr_app___override(v___x_196_, v___y_189_);
v___x_198_ = l_Lean_Expr_app___override(v___x_197_, v_fst_193_);
lean_inc_ref(v_00_u03b1_167_);
v___x_199_ = l_Lean_Expr_app___override(v___x_198_, v_00_u03b1_167_);
lean_inc_ref(v_x_172_);
v___x_200_ = l_Lean_Expr_app___override(v___x_199_, v_x_172_);
v___x_201_ = l_Lean_Expr_app___override(v___x_200_, v_n_168_);
v___x_202_ = l_Lean_Expr_app___override(v___x_201_, v_n_x27_169_);
v___x_203_ = l_Lean_Expr_app___override(v___x_202_, v_instCharP_173_);
lean_inc_ref(v_e_171_);
v___x_204_ = l_Lean_Expr_app___override(v___x_203_, v_e_171_);
v___x_205_ = l_Lean_Expr_app___override(v___x_204_, v___y_188_);
v___x_206_ = l_Lean_Expr_app___override(v___x_205_, v_pn_170_);
v___x_207_ = l_Lean_Expr_app___override(v___x_206_, v_snd_194_);
v___x_208_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_166_, v_00_u03b1_167_, v_e_171_, v_x_172_, v_fst_193_, v_fst_192_, v___x_207_);
lean_dec(v_fst_192_);
lean_dec(v_fst_193_);
if (v_isShared_184_ == 0)
{
lean_ctor_set(v___x_183_, 0, v___x_208_);
v___x_210_ = v___x_183_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v___x_208_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
v___jp_212_:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_222_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__7));
lean_inc_n(v___y_216_, 2);
lean_inc_n(v___y_215_, 3);
v___x_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_223_, 0, v___y_215_);
lean_ctor_set(v___x_223_, 1, v___y_216_);
v___x_224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_224_, 0, v___y_215_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
v___x_225_ = l_Lean_Expr_const___override(v___x_222_, v___x_224_);
lean_inc_ref_n(v___y_213_, 5);
v___x_226_ = l_Lean_Expr_app___override(v___x_225_, v___y_213_);
v___x_227_ = l_Lean_Expr_app___override(v___x_226_, v___y_213_);
v___x_228_ = l_Lean_Expr_app___override(v___x_227_, v___y_213_);
v___x_229_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__9));
v___x_230_ = l_Lean_Expr_const___override(v___x_229_, v___y_216_);
v___x_231_ = l_Lean_Expr_app___override(v___x_230_, v___y_213_);
v___x_232_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__10));
lean_inc_ref(v___y_217_);
v___x_233_ = l_Lean_Name_mkStr2(v___y_217_, v___x_232_);
v___x_234_ = l_Lean_Expr_const___override(v___x_233_, v___x_185_);
v___x_235_ = l_Lean_Expr_app___override(v___x_231_, v___x_234_);
v___x_236_ = l_Lean_Expr_app___override(v___x_228_, v___x_235_);
lean_inc_ref(v___y_219_);
v___x_237_ = l_Lean_Expr_app___override(v___x_236_, v___y_219_);
v___x_238_ = l_Lean_Expr_app___override(v___x_237_, v___y_220_);
lean_inc_ref(v___y_218_);
v___x_239_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(v___y_215_, v___y_213_, v___x_238_, v___y_218_, v_a_221_);
if (lean_obj_tag(v___x_239_) == 0)
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_249_; 
lean_dec_ref(v___y_219_);
lean_dec_ref(v___y_214_);
lean_dec_ref_known(v___x_186_, 2);
lean_del_object(v___x_183_);
lean_dec_ref(v_instCharP_173_);
lean_dec_ref(v_x_172_);
lean_dec_ref(v_e_171_);
lean_dec_ref(v_pn_170_);
lean_dec_ref(v_n_x27_169_);
lean_dec_ref(v_n_168_);
lean_dec_ref(v_00_u03b1_167_);
lean_dec(v_u_166_);
v___x_240_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_241_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_240_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
v_a_242_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_249_ == 0)
{
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_247_; 
if (v_isShared_245_ == 0)
{
v___x_247_ = v___x_244_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v_a_242_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
else
{
lean_object* v_val_250_; 
v_val_250_ = lean_ctor_get(v___x_239_, 0);
lean_inc(v_val_250_);
lean_dec_ref_known(v___x_239_, 1);
v___y_188_ = v___y_214_;
v___y_189_ = v___y_219_;
v_a_190_ = v_val_250_;
goto v___jp_187_;
}
}
v___jp_251_:
{
lean_object* v_snd_253_; lean_object* v_fst_254_; lean_object* v_fst_255_; lean_object* v_snd_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v_snd_253_ = lean_ctor_get(v_a_252_, 1);
lean_inc(v_snd_253_);
v_fst_254_ = lean_ctor_get(v_a_252_, 0);
lean_inc(v_fst_254_);
lean_dec_ref(v_a_252_);
v_fst_255_ = lean_ctor_get(v_snd_253_, 0);
lean_inc_n(v_fst_255_, 4);
v_snd_256_ = lean_ctor_get(v_snd_253_, 1);
lean_inc(v_snd_256_);
lean_dec(v_snd_253_);
v___x_257_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__13));
v___x_258_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__15);
v___x_259_ = lean_box(0);
v___x_260_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_261_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__23);
v___x_262_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__26);
v___x_263_ = l_Lean_Expr_app___override(v___x_262_, v_fst_255_);
v___x_264_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__41);
v___x_265_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__43);
lean_inc_ref_n(v_n_x27_169_, 5);
v___x_266_ = l_Lean_Expr_app___override(v___x_265_, v_n_x27_169_);
v___x_267_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__49);
v___x_268_ = l_Lean_Expr_app___override(v___x_267_, v_n_x27_169_);
v___x_269_ = l_Lean_Expr_app___override(v___x_268_, v_n_x27_169_);
v___x_270_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__51);
v___x_271_ = l_Lean_Expr_app___override(v___x_270_, v_n_x27_169_);
v___x_272_ = l_Lean_Expr_app___override(v___x_269_, v___x_271_);
v___x_273_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_273_, 0, v___x_264_);
lean_ctor_set(v___x_273_, 1, v_n_x27_169_);
lean_ctor_set(v___x_273_, 2, v___x_272_);
lean_inc_ref(v___x_266_);
v___x_274_ = lp_mathlib_Mathlib_Meta_NormNum_evalIntMod_go(v_fst_255_, v_fst_255_, v_fst_254_, v___x_263_, v___x_266_, v___x_273_);
lean_dec(v_fst_254_);
if (lean_obj_tag(v___x_274_) == 0)
{
lean_object* v___x_275_; lean_object* v___x_276_; 
lean_dec_ref(v___x_266_);
lean_dec(v_snd_256_);
lean_dec(v_fst_255_);
lean_dec_ref_known(v___x_186_, 2);
lean_del_object(v___x_183_);
lean_dec_ref(v_instCharP_173_);
lean_dec_ref(v_x_172_);
lean_dec_ref(v_e_171_);
lean_dec_ref(v_pn_170_);
lean_dec_ref(v_n_x27_169_);
lean_dec_ref(v_n_168_);
lean_dec_ref(v_00_u03b1_167_);
lean_dec(v_u_166_);
v___x_275_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_276_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_275_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
return v___x_276_;
}
else
{
lean_object* v_val_277_; 
v_val_277_ = lean_ctor_get(v___x_274_, 0);
lean_inc(v_val_277_);
lean_dec_ref_known(v___x_274_, 1);
v___y_213_ = v___x_258_;
v___y_214_ = v_snd_256_;
v___y_215_ = v___x_259_;
v___y_216_ = v___x_260_;
v___y_217_ = v___x_257_;
v___y_218_ = v___x_261_;
v___y_219_ = v_fst_255_;
v___y_220_ = v___x_266_;
v_a_221_ = v_val_277_;
goto v___jp_212_;
}
}
}
}
else
{
lean_dec_ref(v_instCharP_173_);
lean_dec_ref(v_x_172_);
lean_dec_ref(v_e_171_);
lean_dec_ref(v_pn_170_);
lean_dec_ref(v_n_x27_169_);
lean_dec_ref(v_n_168_);
lean_dec_ref(v_00_u03b1_167_);
lean_dec(v_u_166_);
return v___x_180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normBareNumeral___boxed(lean_object* v_u_291_, lean_object* v_00_u03b1_292_, lean_object* v_n_293_, lean_object* v_n_x27_294_, lean_object* v_pn_295_, lean_object* v_e_296_, lean_object* v_x_297_, lean_object* v_instCharP_298_, lean_object* v_a_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Tactic_ReduceModChar_normBareNumeral(v_u_291_, v_00_u03b1_292_, v_n_293_, v_n_x27_294_, v_pn_295_, v_e_296_, v_x_297_, v_instCharP_298_, v_a_299_, v_a_300_, v_a_301_, v_a_302_);
lean_dec(v_a_302_);
lean_dec_ref(v_a_301_);
lean_dec(v_a_300_);
lean_dec_ref(v_a_299_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0(lean_object* v_00_u03b1_305_, lean_object* v_msg_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v_msg_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___boxed(lean_object* v_00_u03b1_313_, lean_object* v_msg_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0(v_00_u03b1_313_, v_msg_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec(v___y_316_);
lean_dec_ref(v___y_315_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(lean_object* v_k_321_, uint8_t v_allowLevelAssignments_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_322_, v_k_321_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
if (lean_obj_tag(v___x_328_) == 0)
{
lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_336_; 
v_a_329_ = lean_ctor_get(v___x_328_, 0);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_328_);
if (v_isSharedCheck_336_ == 0)
{
v___x_331_ = v___x_328_;
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_dec(v___x_328_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_a_329_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
else
{
lean_object* v_a_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_344_; 
v_a_337_ = lean_ctor_get(v___x_328_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_328_);
if (v_isSharedCheck_344_ == 0)
{
v___x_339_ = v___x_328_;
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_a_337_);
lean_dec(v___x_328_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_342_; 
if (v_isShared_340_ == 0)
{
v___x_342_ = v___x_339_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_a_337_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg___boxed(lean_object* v_k_345_, lean_object* v_allowLevelAssignments_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_352_; lean_object* v_res_353_; 
v_allowLevelAssignments_boxed_352_ = lean_unbox(v_allowLevelAssignments_346_);
v_res_353_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v_k_345_, v_allowLevelAssignments_boxed_352_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___lam__0(lean_object* v_fn_354_, lean_object* v___x_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = l_Lean_Meta_isExprDefEq(v_fn_354_, v___x_355_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___lam__0___boxed(lean_object* v_fn_362_, lean_object* v___x_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Tactic_ReduceModChar_normPow___lam__0(v_fn_362_, v___x_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
lean_dec(v___y_367_);
lean_dec_ref(v___y_366_);
lean_dec(v___y_365_);
lean_dec_ref(v___y_364_);
return v_res_369_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normPow___closed__1(void){
_start:
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_372_ = lean_box(0);
v___x_373_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__0));
v___x_374_ = l_Lean_Expr_const___override(v___x_373_, v___x_372_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow(lean_object* v_u_414_, lean_object* v_00_u03b1_415_, lean_object* v_n_416_, lean_object* v_n_x27_417_, lean_object* v_pn_418_, lean_object* v_e_419_, lean_object* v_x_420_, lean_object* v_instCharP_421_, lean_object* v_a_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = l_Lean_Meta_whnfR(v_e_419_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
if (lean_obj_tag(v___x_427_) == 0)
{
lean_object* v_a_428_; lean_object* v___y_430_; lean_object* v___y_431_; lean_object* v___y_432_; lean_object* v___y_433_; 
v_a_428_ = lean_ctor_get(v___x_427_, 0);
lean_inc(v_a_428_);
lean_dec_ref_known(v___x_427_, 1);
if (lean_obj_tag(v_a_428_) == 5)
{
lean_object* v_fn_436_; 
v_fn_436_ = lean_ctor_get(v_a_428_, 0);
lean_inc_ref(v_fn_436_);
if (lean_obj_tag(v_fn_436_) == 5)
{
lean_object* v_arg_437_; lean_object* v_fn_438_; lean_object* v_arg_439_; lean_object* v___x_440_; 
v_arg_437_ = lean_ctor_get(v_a_428_, 1);
lean_inc_ref(v_arg_437_);
lean_dec_ref_known(v_a_428_, 2);
v_fn_438_ = lean_ctor_get(v_fn_436_, 0);
lean_inc_ref(v_fn_438_);
v_arg_439_ = lean_ctor_get(v_fn_436_, 1);
lean_inc_ref_n(v_arg_439_, 2);
lean_dec_ref_known(v_fn_436_, 2);
lean_inc_ref(v_instCharP_421_);
lean_inc_ref(v_x_420_);
lean_inc_ref(v_pn_418_);
lean_inc_ref(v_n_x27_417_);
lean_inc_ref(v_n_416_);
lean_inc_ref(v_00_u03b1_415_);
lean_inc(v_u_414_);
v___x_440_ = lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27(v_u_414_, v_00_u03b1_415_, v_n_416_, v_n_x27_417_, v_pn_418_, v_arg_439_, v_x_420_, v_instCharP_421_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
if (lean_obj_tag(v___x_440_) == 0)
{
lean_object* v_a_441_; 
v_a_441_ = lean_ctor_get(v___x_440_, 0);
lean_inc(v_a_441_);
lean_dec_ref_known(v___x_440_, 1);
if (lean_obj_tag(v_a_441_) == 1)
{
lean_object* v_inst_442_; lean_object* v_lit_443_; lean_object* v_proof_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_572_; 
v_inst_442_ = lean_ctor_get(v_a_441_, 0);
v_lit_443_ = lean_ctor_get(v_a_441_, 1);
v_proof_444_ = lean_ctor_get(v_a_441_, 2);
v_isSharedCheck_572_ = !lean_is_exclusive(v_a_441_);
if (v_isSharedCheck_572_ == 0)
{
v___x_446_ = v_a_441_;
v_isShared_447_ = v_isSharedCheck_572_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_proof_444_);
lean_inc(v_lit_443_);
lean_inc(v_inst_442_);
lean_dec(v_a_441_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_572_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_448_ = lean_box(0);
v___x_449_ = lean_box(0);
v___x_450_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normPow___closed__1, &lp_mathlib_Tactic_ReduceModChar_normPow___closed__1_once, _init_lp_mathlib_Tactic_ReduceModChar_normPow___closed__1);
v___x_451_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
lean_inc_ref(v_arg_437_);
v___x_452_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_448_, v___x_450_, v_arg_437_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
if (lean_obj_tag(v___x_452_) == 0)
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_563_; 
v_a_453_ = lean_ctor_get(v___x_452_, 0);
v_isSharedCheck_563_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_563_ == 0)
{
v___x_455_ = v___x_452_;
v_isShared_456_ = v_isSharedCheck_563_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_452_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_563_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v_fst_457_; lean_object* v_snd_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_562_; 
v_fst_457_ = lean_ctor_get(v_a_453_, 0);
v_snd_458_ = lean_ctor_get(v_a_453_, 1);
v_isSharedCheck_562_ = !lean_is_exclusive(v_a_453_);
if (v_isSharedCheck_562_ == 0)
{
v___x_460_ = v_a_453_;
v_isShared_461_ = v_isSharedCheck_562_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_snd_458_);
lean_inc(v_fst_457_);
lean_dec(v_a_453_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_562_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
lean_object* v___x_462_; lean_object* v___x_463_; uint8_t v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
lean_inc_n(v_u_414_, 2);
v___x_462_ = l_Lean_Level_succ___override(v_u_414_);
v___x_463_ = lean_box(0);
v___x_464_ = 0;
lean_inc_ref_n(v_00_u03b1_415_, 2);
v___x_465_ = l_Lean_Expr_forallE___override(v___x_463_, v___x_450_, v_00_u03b1_415_, v___x_464_);
v___x_466_ = l_Lean_Expr_forallE___override(v___x_463_, v_00_u03b1_415_, v___x_465_, v___x_464_);
v___x_467_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__4));
if (v_isShared_461_ == 0)
{
lean_ctor_set_tag(v___x_460_, 1);
lean_ctor_set(v___x_460_, 1, v___x_449_);
lean_ctor_set(v___x_460_, 0, v_u_414_);
v___x_469_ = v___x_460_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_u_414_);
lean_ctor_set(v_reuseFailAlloc_561_, 1, v___x_449_);
v___x_469_ = v_reuseFailAlloc_561_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___f_538_; uint8_t v___x_539_; lean_object* v___x_540_; 
lean_inc_ref_n(v___x_469_, 5);
v___x_470_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_448_);
lean_ctor_set(v___x_470_, 1, v___x_469_);
lean_inc(v_u_414_);
v___x_471_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_471_, 0, v_u_414_);
lean_ctor_set(v___x_471_, 1, v___x_470_);
v___x_472_ = l_Lean_Expr_const___override(v___x_467_, v___x_471_);
lean_inc_ref_n(v_00_u03b1_415_, 7);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v_00_u03b1_415_);
v___x_474_ = l_Lean_Expr_app___override(v___x_473_, v___x_450_);
v___x_475_ = l_Lean_Expr_app___override(v___x_474_, v_00_u03b1_415_);
v___x_476_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__6));
v___x_477_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_477_, 0, v_u_414_);
lean_ctor_set(v___x_477_, 1, v___x_451_);
v___x_478_ = l_Lean_Expr_const___override(v___x_476_, v___x_477_);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_00_u03b1_415_);
v___x_480_ = l_Lean_Expr_app___override(v___x_479_, v___x_450_);
v___x_481_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__9));
v___x_482_ = l_Lean_Expr_const___override(v___x_481_, v___x_469_);
v___x_483_ = l_Lean_Expr_app___override(v___x_482_, v_00_u03b1_415_);
v___x_484_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__12));
v___x_485_ = l_Lean_Expr_const___override(v___x_484_, v___x_469_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_00_u03b1_415_);
v___x_487_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__15));
v___x_488_ = l_Lean_Expr_const___override(v___x_487_, v___x_469_);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v_00_u03b1_415_);
v___x_490_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__18));
v___x_491_ = l_Lean_Expr_const___override(v___x_490_, v___x_469_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_00_u03b1_415_);
v___x_493_ = l_Lean_Expr_app___override(v___x_492_, v_x_420_);
lean_inc_ref(v___x_493_);
v___x_533_ = l_Lean_Expr_app___override(v___x_489_, v___x_493_);
v___x_534_ = l_Lean_Expr_app___override(v___x_486_, v___x_533_);
v___x_535_ = l_Lean_Expr_app___override(v___x_483_, v___x_534_);
v___x_536_ = l_Lean_Expr_app___override(v___x_480_, v___x_535_);
v___x_537_ = l_Lean_Expr_app___override(v___x_475_, v___x_536_);
lean_inc_ref(v_fn_438_);
v___f_538_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_normPow___lam__0___boxed), 7, 2);
lean_closure_set(v___f_538_, 0, v_fn_438_);
lean_closure_set(v___f_538_, 1, v___x_537_);
v___x_539_ = 0;
v___x_540_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v___f_538_, v___x_539_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
if (lean_obj_tag(v___x_540_) == 0)
{
lean_object* v_a_541_; uint8_t v___x_542_; 
v_a_541_ = lean_ctor_get(v___x_540_, 0);
lean_inc(v_a_541_);
lean_dec_ref_known(v___x_540_, 1);
v___x_542_ = lean_unbox(v_a_541_);
lean_dec(v_a_541_);
if (v___x_542_ == 0)
{
lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_543_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_544_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_543_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
if (lean_obj_tag(v___x_544_) == 0)
{
lean_dec_ref_known(v___x_544_, 1);
goto v___jp_494_;
}
else
{
lean_object* v_a_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_552_; 
lean_dec_ref(v___x_493_);
lean_dec_ref(v___x_469_);
lean_dec_ref(v___x_466_);
lean_dec(v___x_462_);
lean_dec(v_snd_458_);
lean_dec(v_fst_457_);
lean_del_object(v___x_455_);
lean_del_object(v___x_446_);
lean_dec_ref(v_proof_444_);
lean_dec_ref(v_lit_443_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_arg_439_);
lean_dec_ref(v_fn_438_);
lean_dec_ref(v_arg_437_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
v_a_545_ = lean_ctor_get(v___x_544_, 0);
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_552_ == 0)
{
v___x_547_ = v___x_544_;
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_a_545_);
lean_dec(v___x_544_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_550_; 
if (v_isShared_548_ == 0)
{
v___x_550_ = v___x_547_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v_a_545_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
else
{
goto v___jp_494_;
}
}
else
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_560_; 
lean_dec_ref(v___x_493_);
lean_dec_ref(v___x_469_);
lean_dec_ref(v___x_466_);
lean_dec(v___x_462_);
lean_dec(v_snd_458_);
lean_dec(v_fst_457_);
lean_del_object(v___x_455_);
lean_del_object(v___x_446_);
lean_dec_ref(v_proof_444_);
lean_dec_ref(v_lit_443_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_arg_439_);
lean_dec_ref(v_fn_438_);
lean_dec_ref(v_arg_437_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
v_a_553_ = lean_ctor_get(v___x_540_, 0);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_540_);
if (v_isSharedCheck_560_ == 0)
{
v___x_555_ = v___x_540_;
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_540_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_558_; 
if (v_isShared_556_ == 0)
{
v___x_558_ = v___x_555_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v_a_553_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
v___jp_494_:
{
lean_object* v___x_495_; lean_object* v_fst_496_; lean_object* v_snd_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_532_; 
lean_inc_ref(v_n_x27_417_);
lean_inc(v_fst_457_);
lean_inc_ref(v_lit_443_);
v___x_495_ = lp_mathlib_Mathlib_Meta_NormNum_evalNatPowMod(v_lit_443_, v_fst_457_, v_n_x27_417_);
v_fst_496_ = lean_ctor_get(v___x_495_, 0);
v_snd_497_ = lean_ctor_get(v___x_495_, 1);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_532_ == 0)
{
v___x_499_ = v___x_495_;
v_isShared_500_ = v_isSharedCheck_532_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_snd_497_);
lean_inc(v_fst_496_);
lean_dec(v___x_495_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_532_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_501_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__20));
v___x_502_ = l_Lean_Expr_const___override(v___x_501_, v___x_469_);
v___x_503_ = l_Lean_Expr_app___override(v___x_502_, v_00_u03b1_415_);
v___x_504_ = l_Lean_Expr_app___override(v___x_503_, v___x_493_);
lean_inc_ref(v_fn_438_);
v___x_505_ = l_Lean_Expr_app___override(v___x_504_, v_fn_438_);
v___x_506_ = l_Lean_Expr_app___override(v___x_505_, v_arg_439_);
v___x_507_ = l_Lean_Expr_app___override(v___x_506_, v_lit_443_);
v___x_508_ = l_Lean_Expr_app___override(v___x_507_, v_arg_437_);
v___x_509_ = l_Lean_Expr_app___override(v___x_508_, v_fst_457_);
lean_inc(v_fst_496_);
v___x_510_ = l_Lean_Expr_app___override(v___x_509_, v_fst_496_);
v___x_511_ = l_Lean_Expr_app___override(v___x_510_, v_n_416_);
v___x_512_ = l_Lean_Expr_app___override(v___x_511_, v_n_x27_417_);
v___x_513_ = l_Lean_Expr_app___override(v___x_512_, v_instCharP_421_);
v___x_514_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__23));
if (v_isShared_500_ == 0)
{
lean_ctor_set_tag(v___x_499_, 1);
lean_ctor_set(v___x_499_, 1, v___x_449_);
lean_ctor_set(v___x_499_, 0, v___x_462_);
v___x_516_ = v___x_499_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v___x_462_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v___x_449_);
v___x_516_ = v_reuseFailAlloc_531_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_526_; 
v___x_517_ = l_Lean_Expr_const___override(v___x_514_, v___x_516_);
v___x_518_ = l_Lean_Expr_app___override(v___x_517_, v___x_466_);
v___x_519_ = l_Lean_Expr_app___override(v___x_518_, v_fn_438_);
v___x_520_ = l_Lean_Expr_app___override(v___x_513_, v___x_519_);
v___x_521_ = l_Lean_Expr_app___override(v___x_520_, v_proof_444_);
v___x_522_ = l_Lean_Expr_app___override(v___x_521_, v_snd_458_);
v___x_523_ = l_Lean_Expr_app___override(v___x_522_, v_pn_418_);
v___x_524_ = l_Lean_Expr_app___override(v___x_523_, v_snd_497_);
if (v_isShared_447_ == 0)
{
lean_ctor_set(v___x_446_, 2, v___x_524_);
lean_ctor_set(v___x_446_, 1, v_fst_496_);
v___x_526_ = v___x_446_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_inst_442_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v_fst_496_);
lean_ctor_set(v_reuseFailAlloc_530_, 2, v___x_524_);
v___x_526_ = v_reuseFailAlloc_530_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
lean_object* v___x_528_; 
if (v_isShared_456_ == 0)
{
lean_ctor_set(v___x_455_, 0, v___x_526_);
v___x_528_ = v___x_455_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_526_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
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
lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_571_; 
lean_del_object(v___x_446_);
lean_dec_ref(v_proof_444_);
lean_dec_ref(v_lit_443_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_arg_439_);
lean_dec_ref(v_fn_438_);
lean_dec_ref(v_arg_437_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
v_a_564_ = lean_ctor_get(v___x_452_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_571_ == 0)
{
v___x_566_ = v___x_452_;
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_452_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_569_; 
if (v_isShared_567_ == 0)
{
v___x_569_ = v___x_566_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_a_564_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
}
else
{
lean_object* v___x_573_; lean_object* v___x_574_; 
lean_dec(v_a_441_);
lean_dec_ref(v_arg_439_);
lean_dec_ref(v_fn_438_);
lean_dec_ref(v_arg_437_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
v___x_573_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_574_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_573_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
return v___x_574_;
}
}
else
{
lean_dec_ref(v_arg_439_);
lean_dec_ref(v_fn_438_);
lean_dec_ref(v_arg_437_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
return v___x_440_;
}
}
else
{
lean_dec_ref_known(v_a_428_, 2);
lean_dec_ref(v_fn_436_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
v___y_430_ = v_a_422_;
v___y_431_ = v_a_423_;
v___y_432_ = v_a_424_;
v___y_433_ = v_a_425_;
goto v___jp_429_;
}
}
else
{
lean_dec(v_a_428_);
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
v___y_430_ = v_a_422_;
v___y_431_ = v_a_423_;
v___y_432_ = v_a_424_;
v___y_433_ = v_a_425_;
goto v___jp_429_;
}
v___jp_429_:
{
lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_434_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_435_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_434_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
return v___x_435_;
}
}
else
{
lean_object* v_a_575_; lean_object* v___x_577_; uint8_t v_isShared_578_; uint8_t v_isSharedCheck_582_; 
lean_dec_ref(v_instCharP_421_);
lean_dec_ref(v_x_420_);
lean_dec_ref(v_pn_418_);
lean_dec_ref(v_n_x27_417_);
lean_dec_ref(v_n_416_);
lean_dec_ref(v_00_u03b1_415_);
lean_dec(v_u_414_);
v_a_575_ = lean_ctor_get(v___x_427_, 0);
v_isSharedCheck_582_ = !lean_is_exclusive(v___x_427_);
if (v_isSharedCheck_582_ == 0)
{
v___x_577_ = v___x_427_;
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
else
{
lean_inc(v_a_575_);
lean_dec(v___x_427_);
v___x_577_ = lean_box(0);
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
v_resetjp_576_:
{
lean_object* v___x_580_; 
if (v_isShared_578_ == 0)
{
v___x_580_ = v___x_577_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v_a_575_);
v___x_580_ = v_reuseFailAlloc_581_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
return v___x_580_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27(lean_object* v_u_583_, lean_object* v_00_u03b1_584_, lean_object* v_n_585_, lean_object* v_n_x27_586_, lean_object* v_pn_587_, lean_object* v_e_588_, lean_object* v_x_589_, lean_object* v_instCharP_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_, lean_object* v_a_594_){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = l_Lean_Meta_saveState___redArg(v_a_592_, v_a_594_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v___x_598_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_a_597_);
lean_dec_ref_known(v___x_596_, 1);
lean_inc_ref(v_instCharP_590_);
lean_inc_ref(v_x_589_);
lean_inc_ref(v_e_588_);
lean_inc_ref(v_pn_587_);
lean_inc_ref(v_n_x27_586_);
lean_inc_ref(v_n_585_);
lean_inc_ref(v_00_u03b1_584_);
lean_inc(v_u_583_);
v___x_598_ = lp_mathlib_Tactic_ReduceModChar_normPow(v_u_583_, v_00_u03b1_584_, v_n_585_, v_n_x27_586_, v_pn_587_, v_e_588_, v_x_589_, v_instCharP_590_, v_a_591_, v_a_592_, v_a_593_, v_a_594_);
if (lean_obj_tag(v___x_598_) == 0)
{
lean_dec(v_a_597_);
lean_dec_ref(v_instCharP_590_);
lean_dec_ref(v_x_589_);
lean_dec_ref(v_e_588_);
lean_dec_ref(v_pn_587_);
lean_dec_ref(v_n_x27_586_);
lean_dec_ref(v_n_585_);
lean_dec_ref(v_00_u03b1_584_);
lean_dec(v_u_583_);
return v___x_598_;
}
else
{
lean_object* v_a_599_; uint8_t v___y_601_; uint8_t v___x_612_; 
v_a_599_ = lean_ctor_get(v___x_598_, 0);
lean_inc(v_a_599_);
v___x_612_ = l_Lean_Exception_isInterrupt(v_a_599_);
if (v___x_612_ == 0)
{
uint8_t v___x_613_; 
v___x_613_ = l_Lean_Exception_isRuntime(v_a_599_);
v___y_601_ = v___x_613_;
goto v___jp_600_;
}
else
{
lean_dec(v_a_599_);
v___y_601_ = v___x_612_;
goto v___jp_600_;
}
v___jp_600_:
{
if (v___y_601_ == 0)
{
lean_object* v___x_602_; 
lean_dec_ref_known(v___x_598_, 1);
v___x_602_ = l_Lean_Meta_SavedState_restore___redArg(v_a_597_, v_a_592_, v_a_594_);
lean_dec(v_a_597_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_object* v___x_603_; 
lean_dec_ref_known(v___x_602_, 1);
v___x_603_ = lp_mathlib_Tactic_ReduceModChar_normBareNumeral(v_u_583_, v_00_u03b1_584_, v_n_585_, v_n_x27_586_, v_pn_587_, v_e_588_, v_x_589_, v_instCharP_590_, v_a_591_, v_a_592_, v_a_593_, v_a_594_);
return v___x_603_;
}
else
{
lean_object* v_a_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_611_; 
lean_dec_ref(v_instCharP_590_);
lean_dec_ref(v_x_589_);
lean_dec_ref(v_e_588_);
lean_dec_ref(v_pn_587_);
lean_dec_ref(v_n_x27_586_);
lean_dec_ref(v_n_585_);
lean_dec_ref(v_00_u03b1_584_);
lean_dec(v_u_583_);
v_a_604_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_611_ == 0)
{
v___x_606_ = v___x_602_;
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_a_604_);
lean_dec(v___x_602_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
lean_object* v___x_609_; 
if (v_isShared_607_ == 0)
{
v___x_609_ = v___x_606_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v_a_604_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
}
}
}
}
else
{
lean_dec(v_a_597_);
lean_dec_ref(v_instCharP_590_);
lean_dec_ref(v_x_589_);
lean_dec_ref(v_e_588_);
lean_dec_ref(v_pn_587_);
lean_dec_ref(v_n_x27_586_);
lean_dec_ref(v_n_585_);
lean_dec_ref(v_00_u03b1_584_);
lean_dec(v_u_583_);
return v___x_598_;
}
}
}
}
else
{
lean_object* v_a_614_; lean_object* v___x_616_; uint8_t v_isShared_617_; uint8_t v_isSharedCheck_621_; 
lean_dec_ref(v_instCharP_590_);
lean_dec_ref(v_x_589_);
lean_dec_ref(v_e_588_);
lean_dec_ref(v_pn_587_);
lean_dec_ref(v_n_x27_586_);
lean_dec_ref(v_n_585_);
lean_dec_ref(v_00_u03b1_584_);
lean_dec(v_u_583_);
v_a_614_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_621_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_621_ == 0)
{
v___x_616_ = v___x_596_;
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
else
{
lean_inc(v_a_614_);
lean_dec(v___x_596_);
v___x_616_ = lean_box(0);
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
v_resetjp_615_:
{
lean_object* v___x_619_; 
if (v_isShared_617_ == 0)
{
v___x_619_ = v___x_616_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v_a_614_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27___boxed(lean_object* v_u_622_, lean_object* v_00_u03b1_623_, lean_object* v_n_624_, lean_object* v_n_x27_625_, lean_object* v_pn_626_, lean_object* v_e_627_, lean_object* v_x_628_, lean_object* v_instCharP_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27(v_u_622_, v_00_u03b1_623_, v_n_624_, v_n_x27_625_, v_pn_626_, v_e_627_, v_x_628_, v_instCharP_629_, v_a_630_, v_a_631_, v_a_632_, v_a_633_);
lean_dec(v_a_633_);
lean_dec_ref(v_a_632_);
lean_dec(v_a_631_);
lean_dec_ref(v_a_630_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normPow___boxed(lean_object* v_u_636_, lean_object* v_00_u03b1_637_, lean_object* v_n_638_, lean_object* v_n_x27_639_, lean_object* v_pn_640_, lean_object* v_e_641_, lean_object* v_x_642_, lean_object* v_instCharP_643_, lean_object* v_a_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_mathlib_Tactic_ReduceModChar_normPow(v_u_636_, v_00_u03b1_637_, v_n_638_, v_n_x27_639_, v_pn_640_, v_e_641_, v_x_642_, v_instCharP_643_, v_a_644_, v_a_645_, v_a_646_, v_a_647_);
lean_dec(v_a_647_);
lean_dec_ref(v_a_646_);
lean_dec(v_a_645_);
lean_dec_ref(v_a_644_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0(lean_object* v_00_u03b1_650_, lean_object* v_k_651_, uint8_t v_allowLevelAssignments_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v_k_651_, v_allowLevelAssignments_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___boxed(lean_object* v_00_u03b1_659_, lean_object* v_k_660_, lean_object* v_allowLevelAssignments_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_667_; lean_object* v_res_668_; 
v_allowLevelAssignments_boxed_667_ = lean_unbox(v_allowLevelAssignments_661_);
v_res_668_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0(v_00_u03b1_659_, v_k_660_, v_allowLevelAssignments_boxed_667_, v___y_662_, v___y_663_, v___y_664_, v___y_665_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral(lean_object* v_u_669_, lean_object* v_00_u03b1_670_, lean_object* v_n_671_, lean_object* v_e_672_, lean_object* v_x_673_, lean_object* v_instCharP_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_, lean_object* v_a_678_){
_start:
{
lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_680_ = lean_box(0);
v___x_681_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normPow___closed__1, &lp_mathlib_Tactic_ReduceModChar_normPow___closed__1_once, _init_lp_mathlib_Tactic_ReduceModChar_normPow___closed__1);
lean_inc_ref(v_n_671_);
v___x_682_ = lp_mathlib_Mathlib_Meta_NormNum_deriveNat___redArg(v___x_680_, v___x_681_, v_n_671_, v_a_675_, v_a_676_, v_a_677_, v_a_678_);
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v_a_683_; lean_object* v_fst_684_; lean_object* v_snd_685_; lean_object* v___x_686_; 
v_a_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc(v_a_683_);
lean_dec_ref_known(v___x_682_, 1);
v_fst_684_ = lean_ctor_get(v_a_683_, 0);
lean_inc(v_fst_684_);
v_snd_685_ = lean_ctor_get(v_a_683_, 1);
lean_inc(v_snd_685_);
lean_dec(v_a_683_);
v___x_686_ = lp_mathlib_Tactic_ReduceModChar_normIntNumeral_x27(v_u_669_, v_00_u03b1_670_, v_n_671_, v_fst_684_, v_snd_685_, v_e_672_, v_x_673_, v_instCharP_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_);
return v___x_686_;
}
else
{
lean_object* v_a_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_694_; 
lean_dec_ref(v_instCharP_674_);
lean_dec_ref(v_x_673_);
lean_dec_ref(v_e_672_);
lean_dec_ref(v_n_671_);
lean_dec_ref(v_00_u03b1_670_);
lean_dec(v_u_669_);
v_a_687_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_694_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_694_ == 0)
{
v___x_689_ = v___x_682_;
v_isShared_690_ = v_isSharedCheck_694_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_a_687_);
lean_dec(v___x_682_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_694_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v___x_692_; 
if (v_isShared_690_ == 0)
{
v___x_692_ = v___x_689_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v_a_687_);
v___x_692_ = v_reuseFailAlloc_693_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
return v___x_692_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normIntNumeral___boxed(lean_object* v_u_695_, lean_object* v_00_u03b1_696_, lean_object* v_n_697_, lean_object* v_e_698_, lean_object* v_x_699_, lean_object* v_instCharP_700_, lean_object* v_a_701_, lean_object* v_a_702_, lean_object* v_a_703_, lean_object* v_a_704_, lean_object* v_a_705_){
_start:
{
lean_object* v_res_706_; 
v_res_706_ = lp_mathlib_Tactic_ReduceModChar_normIntNumeral(v_u_695_, v_00_u03b1_696_, v_n_697_, v_e_698_, v_x_699_, v_instCharP_700_, v_a_701_, v_a_702_, v_a_703_, v_a_704_);
lean_dec(v_a_704_);
lean_dec_ref(v_a_703_);
lean_dec(v_a_702_);
lean_dec_ref(v_a_701_);
return v_res_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0(lean_object* v_fn_707_, lean_object* v___x_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_object* v___x_714_; 
v___x_714_ = l_Lean_Meta_isExprDefEq(v_fn_707_, v___x_708_, v___y_709_, v___y_710_, v___y_711_, v___y_712_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0___boxed(lean_object* v_fn_715_, lean_object* v___x_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0(v_fn_715_, v___x_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_);
lean_dec(v___y_720_);
lean_dec_ref(v___y_719_);
lean_dec(v___y_718_);
lean_dec_ref(v___y_717_);
return v_res_722_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42(void){
_start:
{
lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_795_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__41));
v___x_796_ = l_Lean_Expr_lit___override(v___x_795_);
return v___x_796_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61(void){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_829_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__60));
v___x_830_ = l_Lean_stringToMessageData(v___x_829_);
return v___x_830_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63(void){
_start:
{
lean_object* v___x_832_; lean_object* v___x_833_; 
v___x_832_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__62));
v___x_833_ = l_Lean_stringToMessageData(v___x_832_);
return v___x_833_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65(void){
_start:
{
lean_object* v___x_835_; lean_object* v___x_836_; 
v___x_835_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__64));
v___x_836_ = l_Lean_stringToMessageData(v___x_835_);
return v___x_836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg(lean_object* v_u_837_, lean_object* v_00_u03b1_838_, lean_object* v_n_839_, lean_object* v_e_840_, lean_object* v___instRing_841_, lean_object* v_instCharP_842_, lean_object* v_a_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = l_Lean_Meta_whnfR(v_e_840_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v_a_849_; 
v_a_849_ = lean_ctor_get(v___x_848_, 0);
lean_inc(v_a_849_);
lean_dec_ref_known(v___x_848_, 1);
if (lean_obj_tag(v_a_849_) == 5)
{
lean_object* v_fn_850_; lean_object* v_arg_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___f_882_; uint8_t v___x_883_; lean_object* v___x_884_; 
v_fn_850_ = lean_ctor_get(v_a_849_, 0);
lean_inc_ref(v_fn_850_);
v_arg_851_ = lean_ctor_get(v_a_849_, 1);
lean_inc_ref(v_arg_851_);
lean_dec_ref_known(v_a_849_, 2);
v___x_852_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2));
v___x_853_ = lean_box(0);
lean_inc(v_u_837_);
v___x_854_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_854_, 0, v_u_837_);
lean_ctor_set(v___x_854_, 1, v___x_853_);
lean_inc_ref_n(v___x_854_, 7);
v___x_855_ = l_Lean_Expr_const___override(v___x_852_, v___x_854_);
lean_inc_ref_n(v_00_u03b1_838_, 7);
v___x_856_ = l_Lean_Expr_app___override(v___x_855_, v_00_u03b1_838_);
v___x_857_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5));
v___x_858_ = l_Lean_Expr_const___override(v___x_857_, v___x_854_);
v___x_859_ = l_Lean_Expr_app___override(v___x_858_, v_00_u03b1_838_);
v___x_860_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8));
v___x_861_ = l_Lean_Expr_const___override(v___x_860_, v___x_854_);
v___x_862_ = l_Lean_Expr_app___override(v___x_861_, v_00_u03b1_838_);
v___x_863_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11));
v___x_864_ = l_Lean_Expr_const___override(v___x_863_, v___x_854_);
v___x_865_ = l_Lean_Expr_app___override(v___x_864_, v_00_u03b1_838_);
v___x_866_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14));
v___x_867_ = l_Lean_Expr_const___override(v___x_866_, v___x_854_);
v___x_868_ = l_Lean_Expr_app___override(v___x_867_, v_00_u03b1_838_);
v___x_869_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17));
v___x_870_ = l_Lean_Expr_const___override(v___x_869_, v___x_854_);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_00_u03b1_838_);
v___x_872_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19));
v___x_873_ = l_Lean_Expr_const___override(v___x_872_, v___x_854_);
v___x_874_ = l_Lean_Expr_app___override(v___x_873_, v_00_u03b1_838_);
lean_inc_ref(v___instRing_841_);
v___x_875_ = l_Lean_Expr_app___override(v___x_874_, v___instRing_841_);
v___x_876_ = l_Lean_Expr_app___override(v___x_871_, v___x_875_);
v___x_877_ = l_Lean_Expr_app___override(v___x_868_, v___x_876_);
v___x_878_ = l_Lean_Expr_app___override(v___x_865_, v___x_877_);
v___x_879_ = l_Lean_Expr_app___override(v___x_862_, v___x_878_);
v___x_880_ = l_Lean_Expr_app___override(v___x_859_, v___x_879_);
v___x_881_ = l_Lean_Expr_app___override(v___x_856_, v___x_880_);
v___f_882_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_882_, 0, v_fn_850_);
lean_closure_set(v___f_882_, 1, v___x_881_);
v___x_883_ = 0;
v___x_884_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v___f_882_, v___x_883_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
if (lean_obj_tag(v___x_884_) == 0)
{
lean_object* v_a_885_; uint8_t v___x_886_; uint8_t v___x_1022_; 
v_a_885_ = lean_ctor_get(v___x_884_, 0);
lean_inc(v_a_885_);
lean_dec_ref_known(v___x_884_, 1);
v___x_886_ = 1;
v___x_1022_ = lean_unbox(v_a_885_);
lean_dec(v_a_885_);
if (v___x_1022_ == 0)
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v_a_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1032_; 
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v___x_1023_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_1024_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1023_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
v_a_1025_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1032_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1027_ = v___x_1024_;
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_a_1025_);
lean_dec(v___x_1024_);
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
else
{
goto v___jp_887_;
}
v___jp_887_:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_888_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22));
lean_inc_ref_n(v___x_854_, 12);
lean_inc_n(v_u_837_, 3);
v___x_889_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_889_, 0, v_u_837_);
lean_ctor_set(v___x_889_, 1, v___x_854_);
v___x_890_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_890_, 0, v_u_837_);
lean_ctor_set(v___x_890_, 1, v___x_889_);
lean_inc_ref(v___x_890_);
v___x_891_ = l_Lean_Expr_const___override(v___x_888_, v___x_890_);
lean_inc_ref_n(v_00_u03b1_838_, 15);
v___x_892_ = l_Lean_Expr_app___override(v___x_891_, v_00_u03b1_838_);
v___x_893_ = l_Lean_Expr_app___override(v___x_892_, v_00_u03b1_838_);
v___x_894_ = l_Lean_Expr_app___override(v___x_893_, v_00_u03b1_838_);
v___x_895_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__24));
v___x_896_ = l_Lean_Expr_const___override(v___x_895_, v___x_854_);
v___x_897_ = l_Lean_Expr_app___override(v___x_896_, v_00_u03b1_838_);
v___x_898_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27));
v___x_899_ = l_Lean_Expr_const___override(v___x_898_, v___x_854_);
v___x_900_ = l_Lean_Expr_app___override(v___x_899_, v_00_u03b1_838_);
v___x_901_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30));
v___x_902_ = l_Lean_Expr_const___override(v___x_901_, v___x_854_);
v___x_903_ = l_Lean_Expr_app___override(v___x_902_, v_00_u03b1_838_);
v___x_904_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33));
v___x_905_ = l_Lean_Expr_const___override(v___x_904_, v___x_854_);
v___x_906_ = l_Lean_Expr_app___override(v___x_905_, v_00_u03b1_838_);
v___x_907_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35));
v___x_908_ = l_Lean_Expr_const___override(v___x_907_, v___x_854_);
v___x_909_ = l_Lean_Expr_app___override(v___x_908_, v_00_u03b1_838_);
lean_inc_ref(v___instRing_841_);
v___x_910_ = l_Lean_Expr_app___override(v___x_909_, v___instRing_841_);
lean_inc_ref(v___x_910_);
v___x_911_ = l_Lean_Expr_app___override(v___x_906_, v___x_910_);
v___x_912_ = l_Lean_Expr_app___override(v___x_903_, v___x_911_);
v___x_913_ = l_Lean_Expr_app___override(v___x_900_, v___x_912_);
v___x_914_ = l_Lean_Expr_app___override(v___x_897_, v___x_913_);
v___x_915_ = l_Lean_Expr_app___override(v___x_894_, v___x_914_);
v___x_916_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29));
v___x_917_ = l_Lean_Expr_const___override(v___x_916_, v___x_854_);
v___x_918_ = l_Lean_Expr_app___override(v___x_917_, v_00_u03b1_838_);
v___x_919_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34));
v___x_920_ = l_Lean_Expr_const___override(v___x_919_, v___x_854_);
v___x_921_ = l_Lean_Expr_app___override(v___x_920_, v_00_u03b1_838_);
v___x_922_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37));
v___x_923_ = l_Lean_Expr_const___override(v___x_922_, v___x_854_);
v___x_924_ = l_Lean_Expr_app___override(v___x_923_, v_00_u03b1_838_);
v___x_925_ = l_Lean_Expr_app___override(v___x_924_, v___x_910_);
lean_inc_ref(v___x_925_);
v___x_926_ = l_Lean_Expr_app___override(v___x_921_, v___x_925_);
v___x_927_ = l_Lean_Expr_app___override(v___x_918_, v___x_926_);
lean_inc_ref(v_n_839_);
v___x_928_ = l_Lean_Expr_app___override(v___x_927_, v_n_839_);
v___x_929_ = l_Lean_Expr_app___override(v___x_915_, v___x_928_);
v___x_930_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40));
v___x_931_ = l_Lean_Expr_const___override(v___x_930_, v___x_854_);
v___x_932_ = l_Lean_Expr_app___override(v___x_931_, v_00_u03b1_838_);
v___x_933_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42);
v___x_934_ = l_Lean_Expr_app___override(v___x_932_, v___x_933_);
v___x_935_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45));
v___x_936_ = l_Lean_Expr_const___override(v___x_935_, v___x_854_);
v___x_937_ = l_Lean_Expr_app___override(v___x_936_, v_00_u03b1_838_);
v___x_938_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47));
v___x_939_ = l_Lean_Expr_const___override(v___x_938_, v___x_854_);
v___x_940_ = l_Lean_Expr_app___override(v___x_939_, v_00_u03b1_838_);
v___x_941_ = l_Lean_Expr_app___override(v___x_940_, v___x_925_);
v___x_942_ = l_Lean_Expr_app___override(v___x_937_, v___x_941_);
v___x_943_ = l_Lean_Expr_app___override(v___x_934_, v___x_942_);
v___x_944_ = l_Lean_Expr_app___override(v___x_929_, v___x_943_);
v___x_945_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_837_, v_00_u03b1_838_, v___x_944_, v___x_883_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
if (lean_obj_tag(v___x_945_) == 0)
{
lean_object* v_a_946_; 
v_a_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc(v_a_946_);
lean_dec_ref_known(v___x_945_, 1);
switch(lean_obj_tag(v_a_946_))
{
case 1:
{
lean_object* v_inst_947_; lean_object* v_lit_948_; lean_object* v_proof_949_; lean_object* v___x_950_; 
v_inst_947_ = lean_ctor_get(v_a_946_, 0);
lean_inc_ref(v_inst_947_);
v_lit_948_ = lean_ctor_get(v_a_946_, 1);
lean_inc_ref_n(v_lit_948_, 2);
v_proof_949_ = lean_ctor_get(v_a_946_, 2);
lean_inc_ref(v_proof_949_);
lean_dec_ref_known(v_a_946_, 3);
lean_inc_ref(v_00_u03b1_838_);
v___x_950_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_837_, v_00_u03b1_838_, v_inst_947_, v_lit_948_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
if (lean_obj_tag(v___x_950_) == 0)
{
lean_object* v_a_951_; lean_object* v___x_953_; uint8_t v_isShared_954_; uint8_t v_isSharedCheck_997_; 
v_a_951_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_997_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_997_ == 0)
{
v___x_953_ = v___x_950_;
v_isShared_954_ = v_isSharedCheck_997_;
goto v_resetjp_952_;
}
else
{
lean_inc(v_a_951_);
lean_dec(v___x_950_);
v___x_953_ = lean_box(0);
v_isShared_954_ = v_isSharedCheck_997_;
goto v_resetjp_952_;
}
v_resetjp_952_:
{
lean_object* v_fst_955_; lean_object* v_snd_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_995_; 
v_fst_955_ = lean_ctor_get(v_a_951_, 0);
lean_inc_n(v_fst_955_, 2);
v_snd_956_ = lean_ctor_get(v_a_951_, 1);
lean_inc(v_snd_956_);
lean_dec(v_a_951_);
v___x_957_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50));
v___x_958_ = l_Lean_Expr_const___override(v___x_957_, v___x_890_);
lean_inc_ref_n(v_00_u03b1_838_, 7);
v___x_959_ = l_Lean_Expr_app___override(v___x_958_, v_00_u03b1_838_);
v___x_960_ = l_Lean_Expr_app___override(v___x_959_, v_00_u03b1_838_);
v___x_961_ = l_Lean_Expr_app___override(v___x_960_, v_00_u03b1_838_);
v___x_962_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__52));
lean_inc_ref_n(v___x_854_, 4);
v___x_963_ = l_Lean_Expr_const___override(v___x_962_, v___x_854_);
v___x_964_ = l_Lean_Expr_app___override(v___x_963_, v_00_u03b1_838_);
v___x_965_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55));
v___x_966_ = l_Lean_Expr_const___override(v___x_965_, v___x_854_);
v___x_967_ = l_Lean_Expr_app___override(v___x_966_, v_00_u03b1_838_);
v___x_968_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__57));
v___x_969_ = l_Lean_Expr_const___override(v___x_968_, v___x_854_);
v___x_970_ = l_Lean_Expr_app___override(v___x_969_, v_00_u03b1_838_);
v___x_971_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__18));
v___x_972_ = l_Lean_Expr_const___override(v___x_971_, v___x_854_);
v___x_973_ = l_Lean_Expr_app___override(v___x_972_, v_00_u03b1_838_);
lean_inc_ref(v___instRing_841_);
v___x_974_ = l_Lean_Expr_app___override(v___x_973_, v___instRing_841_);
v___x_975_ = l_Lean_Expr_app___override(v___x_970_, v___x_974_);
v___x_976_ = l_Lean_Expr_app___override(v___x_967_, v___x_975_);
v___x_977_ = l_Lean_Expr_app___override(v___x_964_, v___x_976_);
v___x_978_ = l_Lean_Expr_app___override(v___x_961_, v___x_977_);
v___x_979_ = l_Lean_Expr_app___override(v___x_978_, v_fst_955_);
lean_inc_ref(v_arg_851_);
v___x_980_ = l_Lean_Expr_app___override(v___x_979_, v_arg_851_);
v___x_981_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__59));
v___x_982_ = l_Lean_Expr_const___override(v___x_981_, v___x_854_);
v___x_983_ = l_Lean_Expr_app___override(v___x_982_, v_00_u03b1_838_);
v___x_984_ = l_Lean_Expr_app___override(v___x_983_, v___instRing_841_);
v___x_985_ = l_Lean_Expr_app___override(v___x_984_, v_n_839_);
v___x_986_ = l_Lean_Expr_app___override(v___x_985_, v_instCharP_842_);
v___x_987_ = l_Lean_Expr_app___override(v___x_986_, v_arg_851_);
v___x_988_ = l_Lean_Expr_app___override(v___x_987_, v_lit_948_);
v___x_989_ = l_Lean_Expr_app___override(v___x_988_, v_fst_955_);
v___x_990_ = l_Lean_Expr_app___override(v___x_989_, v_proof_949_);
v___x_991_ = l_Lean_Expr_app___override(v___x_990_, v_snd_956_);
v___x_992_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_992_, 0, v___x_991_);
v___x_993_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_993_, 0, v___x_980_);
lean_ctor_set(v___x_993_, 1, v___x_992_);
lean_ctor_set_uint8(v___x_993_, sizeof(void*)*2, v___x_886_);
if (v_isShared_954_ == 0)
{
lean_ctor_set(v___x_953_, 0, v___x_993_);
v___x_995_ = v___x_953_;
goto v_reusejp_994_;
}
else
{
lean_object* v_reuseFailAlloc_996_; 
v_reuseFailAlloc_996_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_996_, 0, v___x_993_);
v___x_995_ = v_reuseFailAlloc_996_;
goto v_reusejp_994_;
}
v_reusejp_994_:
{
return v___x_995_;
}
}
}
else
{
lean_object* v_a_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1005_; 
lean_dec_ref(v_proof_949_);
lean_dec_ref(v_lit_948_);
lean_dec_ref_known(v___x_890_, 2);
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
v_a_998_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_1005_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_1005_ == 0)
{
v___x_1000_ = v___x_950_;
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_a_998_);
lean_dec(v___x_950_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1003_; 
if (v_isShared_1001_ == 0)
{
v___x_1003_ = v___x_1000_;
goto v_reusejp_1002_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v_a_998_);
v___x_1003_ = v_reuseFailAlloc_1004_;
goto v_reusejp_1002_;
}
v_reusejp_1002_:
{
return v___x_1003_;
}
}
}
}
case 2:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; 
lean_dec_ref_known(v_a_946_, 3);
lean_dec_ref_known(v___x_890_, 2);
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v___x_1006_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__61);
v___x_1007_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1006_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
return v___x_1007_;
}
default: 
{
lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; 
lean_dec(v_a_946_);
lean_dec_ref_known(v___x_890_, 2);
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v___x_1008_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__63);
v___x_1009_ = l_Lean_MessageData_ofExpr(v_n_839_);
v___x_1010_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1008_);
lean_ctor_set(v___x_1010_, 1, v___x_1009_);
v___x_1011_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65);
v___x_1012_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1012_, 0, v___x_1010_);
lean_ctor_set(v___x_1012_, 1, v___x_1011_);
v___x_1013_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1012_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
return v___x_1013_;
}
}
}
else
{
lean_object* v_a_1014_; lean_object* v___x_1016_; uint8_t v_isShared_1017_; uint8_t v_isSharedCheck_1021_; 
lean_dec_ref_known(v___x_890_, 2);
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v_a_1014_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_1021_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_1021_ == 0)
{
v___x_1016_ = v___x_945_;
v_isShared_1017_ = v_isSharedCheck_1021_;
goto v_resetjp_1015_;
}
else
{
lean_inc(v_a_1014_);
lean_dec(v___x_945_);
v___x_1016_ = lean_box(0);
v_isShared_1017_ = v_isSharedCheck_1021_;
goto v_resetjp_1015_;
}
v_resetjp_1015_:
{
lean_object* v___x_1019_; 
if (v_isShared_1017_ == 0)
{
v___x_1019_ = v___x_1016_;
goto v_reusejp_1018_;
}
else
{
lean_object* v_reuseFailAlloc_1020_; 
v_reuseFailAlloc_1020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1020_, 0, v_a_1014_);
v___x_1019_ = v_reuseFailAlloc_1020_;
goto v_reusejp_1018_;
}
v_reusejp_1018_:
{
return v___x_1019_;
}
}
}
}
}
else
{
lean_object* v_a_1033_; lean_object* v___x_1035_; uint8_t v_isShared_1036_; uint8_t v_isSharedCheck_1040_; 
lean_dec_ref_known(v___x_854_, 2);
lean_dec_ref(v_arg_851_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v_a_1033_ = lean_ctor_get(v___x_884_, 0);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_884_);
if (v_isSharedCheck_1040_ == 0)
{
v___x_1035_ = v___x_884_;
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
else
{
lean_inc(v_a_1033_);
lean_dec(v___x_884_);
v___x_1035_ = lean_box(0);
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
v_resetjp_1034_:
{
lean_object* v___x_1038_; 
if (v_isShared_1036_ == 0)
{
v___x_1038_ = v___x_1035_;
goto v_reusejp_1037_;
}
else
{
lean_object* v_reuseFailAlloc_1039_; 
v_reuseFailAlloc_1039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1039_, 0, v_a_1033_);
v___x_1038_ = v_reuseFailAlloc_1039_;
goto v_reusejp_1037_;
}
v_reusejp_1037_:
{
return v___x_1038_;
}
}
}
}
else
{
lean_object* v___x_1041_; lean_object* v___x_1042_; 
lean_dec(v_a_849_);
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v___x_1041_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_1042_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1041_, v_a_843_, v_a_844_, v_a_845_, v_a_846_);
return v___x_1042_;
}
}
else
{
lean_object* v_a_1043_; lean_object* v___x_1045_; uint8_t v_isShared_1046_; uint8_t v_isSharedCheck_1050_; 
lean_dec_ref(v_instCharP_842_);
lean_dec_ref(v___instRing_841_);
lean_dec_ref(v_n_839_);
lean_dec_ref(v_00_u03b1_838_);
lean_dec(v_u_837_);
v_a_1043_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_1045_ = v___x_848_;
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
else
{
lean_inc(v_a_1043_);
lean_dec(v___x_848_);
v___x_1045_ = lean_box(0);
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
v_resetjp_1044_:
{
lean_object* v___x_1048_; 
if (v_isShared_1046_ == 0)
{
v___x_1048_ = v___x_1045_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v_a_1043_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNeg___boxed(lean_object* v_u_1051_, lean_object* v_00_u03b1_1052_, lean_object* v_n_1053_, lean_object* v_e_1054_, lean_object* v___instRing_1055_, lean_object* v_instCharP_1056_, lean_object* v_a_1057_, lean_object* v_a_1058_, lean_object* v_a_1059_, lean_object* v_a_1060_, lean_object* v_a_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_mathlib_Tactic_ReduceModChar_normNeg(v_u_1051_, v_00_u03b1_1052_, v_n_1053_, v_e_1054_, v___instRing_1055_, v_instCharP_1056_, v_a_1057_, v_a_1058_, v_a_1059_, v_a_1060_);
lean_dec(v_a_1060_);
lean_dec_ref(v_a_1059_);
lean_dec(v_a_1058_);
lean_dec_ref(v_a_1057_);
return v_res_1062_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3(void){
_start:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___x_1070_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__2));
v___x_1071_ = l_Lean_stringToMessageData(v___x_1070_);
return v___x_1071_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5(void){
_start:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___x_1073_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__4));
v___x_1074_ = l_Lean_stringToMessageData(v___x_1073_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul(lean_object* v_u_1075_, lean_object* v_00_u03b1_1076_, lean_object* v_n_1077_, lean_object* v_e_1078_, lean_object* v___instRing_1079_, lean_object* v_instCharP_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_){
_start:
{
lean_object* v___x_1086_; 
v___x_1086_ = l_Lean_Meta_whnfR(v_e_1078_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
if (lean_obj_tag(v___x_1086_) == 0)
{
lean_object* v_a_1087_; lean_object* v___y_1089_; lean_object* v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; 
v_a_1087_ = lean_ctor_get(v___x_1086_, 0);
lean_inc(v_a_1087_);
lean_dec_ref_known(v___x_1086_, 1);
if (lean_obj_tag(v_a_1087_) == 5)
{
lean_object* v_arg_1095_; 
v_arg_1095_ = lean_ctor_get(v_a_1087_, 1);
lean_inc_ref(v_arg_1095_);
if (lean_obj_tag(v_arg_1095_) == 5)
{
lean_object* v_fn_1096_; 
v_fn_1096_ = lean_ctor_get(v_arg_1095_, 0);
lean_inc_ref(v_fn_1096_);
if (lean_obj_tag(v_fn_1096_) == 5)
{
lean_object* v_fn_1097_; lean_object* v_arg_1098_; lean_object* v_fn_1099_; lean_object* v_arg_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___f_1131_; uint8_t v___x_1132_; lean_object* v___x_1133_; 
v_fn_1097_ = lean_ctor_get(v_a_1087_, 0);
lean_inc_ref(v_fn_1097_);
lean_dec_ref_known(v_a_1087_, 2);
v_arg_1098_ = lean_ctor_get(v_arg_1095_, 1);
lean_inc_ref(v_arg_1098_);
lean_dec_ref_known(v_arg_1095_, 2);
v_fn_1099_ = lean_ctor_get(v_fn_1096_, 0);
lean_inc_ref(v_fn_1099_);
v_arg_1100_ = lean_ctor_get(v_fn_1096_, 1);
lean_inc_ref(v_arg_1100_);
lean_dec_ref_known(v_fn_1096_, 2);
v___x_1101_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__2));
v___x_1102_ = lean_box(0);
lean_inc(v_u_1075_);
v___x_1103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1103_, 0, v_u_1075_);
lean_ctor_set(v___x_1103_, 1, v___x_1102_);
lean_inc_ref_n(v___x_1103_, 7);
v___x_1104_ = l_Lean_Expr_const___override(v___x_1101_, v___x_1103_);
lean_inc_ref_n(v_00_u03b1_1076_, 7);
v___x_1105_ = l_Lean_Expr_app___override(v___x_1104_, v_00_u03b1_1076_);
v___x_1106_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__5));
v___x_1107_ = l_Lean_Expr_const___override(v___x_1106_, v___x_1103_);
v___x_1108_ = l_Lean_Expr_app___override(v___x_1107_, v_00_u03b1_1076_);
v___x_1109_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__8));
v___x_1110_ = l_Lean_Expr_const___override(v___x_1109_, v___x_1103_);
v___x_1111_ = l_Lean_Expr_app___override(v___x_1110_, v_00_u03b1_1076_);
v___x_1112_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__11));
v___x_1113_ = l_Lean_Expr_const___override(v___x_1112_, v___x_1103_);
v___x_1114_ = l_Lean_Expr_app___override(v___x_1113_, v_00_u03b1_1076_);
v___x_1115_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__14));
v___x_1116_ = l_Lean_Expr_const___override(v___x_1115_, v___x_1103_);
v___x_1117_ = l_Lean_Expr_app___override(v___x_1116_, v_00_u03b1_1076_);
v___x_1118_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__17));
v___x_1119_ = l_Lean_Expr_const___override(v___x_1118_, v___x_1103_);
v___x_1120_ = l_Lean_Expr_app___override(v___x_1119_, v_00_u03b1_1076_);
v___x_1121_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__19));
v___x_1122_ = l_Lean_Expr_const___override(v___x_1121_, v___x_1103_);
v___x_1123_ = l_Lean_Expr_app___override(v___x_1122_, v_00_u03b1_1076_);
lean_inc_ref(v___instRing_1079_);
v___x_1124_ = l_Lean_Expr_app___override(v___x_1123_, v___instRing_1079_);
v___x_1125_ = l_Lean_Expr_app___override(v___x_1120_, v___x_1124_);
v___x_1126_ = l_Lean_Expr_app___override(v___x_1117_, v___x_1125_);
v___x_1127_ = l_Lean_Expr_app___override(v___x_1114_, v___x_1126_);
v___x_1128_ = l_Lean_Expr_app___override(v___x_1111_, v___x_1127_);
v___x_1129_ = l_Lean_Expr_app___override(v___x_1108_, v___x_1128_);
v___x_1130_ = l_Lean_Expr_app___override(v___x_1105_, v___x_1129_);
v___f_1131_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1131_, 0, v_fn_1097_);
lean_closure_set(v___f_1131_, 1, v___x_1130_);
v___x_1132_ = 0;
v___x_1133_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v___f_1131_, v___x_1132_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
if (lean_obj_tag(v___x_1133_) == 0)
{
lean_object* v_a_1134_; uint8_t v___x_1135_; lean_object* v___y_1137_; lean_object* v___y_1138_; uint8_t v___x_1299_; 
v_a_1134_ = lean_ctor_get(v___x_1133_, 0);
lean_inc(v_a_1134_);
lean_dec_ref_known(v___x_1133_, 1);
v___x_1135_ = 1;
v___x_1299_ = lean_unbox(v_a_1134_);
lean_dec(v_a_1134_);
if (v___x_1299_ == 0)
{
lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v_a_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1309_; 
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_fn_1099_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___x_1300_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_1301_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1300_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
v_a_1302_ = lean_ctor_get(v___x_1301_, 0);
v_isSharedCheck_1309_ = !lean_is_exclusive(v___x_1301_);
if (v_isSharedCheck_1309_ == 0)
{
v___x_1304_ = v___x_1301_;
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_a_1302_);
lean_dec(v___x_1301_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1307_; 
if (v_isShared_1305_ == 0)
{
v___x_1307_ = v___x_1304_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1308_; 
v_reuseFailAlloc_1308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1308_, 0, v_a_1302_);
v___x_1307_ = v_reuseFailAlloc_1308_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
return v___x_1307_;
}
}
}
else
{
goto v___jp_1252_;
}
v___jp_1136_:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; 
v___x_1139_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__22));
v___x_1140_ = l_Lean_Expr_const___override(v___x_1139_, v___y_1137_);
lean_inc_ref_n(v_00_u03b1_1076_, 15);
v___x_1141_ = l_Lean_Expr_app___override(v___x_1140_, v_00_u03b1_1076_);
v___x_1142_ = l_Lean_Expr_app___override(v___x_1141_, v_00_u03b1_1076_);
v___x_1143_ = l_Lean_Expr_app___override(v___x_1142_, v_00_u03b1_1076_);
v___x_1144_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__24));
lean_inc_ref_n(v___x_1103_, 11);
v___x_1145_ = l_Lean_Expr_const___override(v___x_1144_, v___x_1103_);
v___x_1146_ = l_Lean_Expr_app___override(v___x_1145_, v_00_u03b1_1076_);
v___x_1147_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__27));
v___x_1148_ = l_Lean_Expr_const___override(v___x_1147_, v___x_1103_);
v___x_1149_ = l_Lean_Expr_app___override(v___x_1148_, v_00_u03b1_1076_);
v___x_1150_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__30));
v___x_1151_ = l_Lean_Expr_const___override(v___x_1150_, v___x_1103_);
v___x_1152_ = l_Lean_Expr_app___override(v___x_1151_, v_00_u03b1_1076_);
v___x_1153_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__33));
v___x_1154_ = l_Lean_Expr_const___override(v___x_1153_, v___x_1103_);
v___x_1155_ = l_Lean_Expr_app___override(v___x_1154_, v_00_u03b1_1076_);
v___x_1156_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__35));
v___x_1157_ = l_Lean_Expr_const___override(v___x_1156_, v___x_1103_);
v___x_1158_ = l_Lean_Expr_app___override(v___x_1157_, v_00_u03b1_1076_);
lean_inc_ref(v___instRing_1079_);
v___x_1159_ = l_Lean_Expr_app___override(v___x_1158_, v___instRing_1079_);
lean_inc_ref(v___x_1159_);
v___x_1160_ = l_Lean_Expr_app___override(v___x_1155_, v___x_1159_);
v___x_1161_ = l_Lean_Expr_app___override(v___x_1152_, v___x_1160_);
v___x_1162_ = l_Lean_Expr_app___override(v___x_1149_, v___x_1161_);
v___x_1163_ = l_Lean_Expr_app___override(v___x_1146_, v___x_1162_);
v___x_1164_ = l_Lean_Expr_app___override(v___x_1143_, v___x_1163_);
v___x_1165_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__29));
v___x_1166_ = l_Lean_Expr_const___override(v___x_1165_, v___x_1103_);
v___x_1167_ = l_Lean_Expr_app___override(v___x_1166_, v_00_u03b1_1076_);
v___x_1168_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__34));
v___x_1169_ = l_Lean_Expr_const___override(v___x_1168_, v___x_1103_);
v___x_1170_ = l_Lean_Expr_app___override(v___x_1169_, v_00_u03b1_1076_);
v___x_1171_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37));
v___x_1172_ = l_Lean_Expr_const___override(v___x_1171_, v___x_1103_);
v___x_1173_ = l_Lean_Expr_app___override(v___x_1172_, v_00_u03b1_1076_);
v___x_1174_ = l_Lean_Expr_app___override(v___x_1173_, v___x_1159_);
lean_inc_ref(v___x_1174_);
v___x_1175_ = l_Lean_Expr_app___override(v___x_1170_, v___x_1174_);
v___x_1176_ = l_Lean_Expr_app___override(v___x_1167_, v___x_1175_);
lean_inc_ref(v_n_1077_);
v___x_1177_ = l_Lean_Expr_app___override(v___x_1176_, v_n_1077_);
v___x_1178_ = l_Lean_Expr_app___override(v___x_1164_, v___x_1177_);
v___x_1179_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__40));
v___x_1180_ = l_Lean_Expr_const___override(v___x_1179_, v___x_1103_);
v___x_1181_ = l_Lean_Expr_app___override(v___x_1180_, v_00_u03b1_1076_);
v___x_1182_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__42);
v___x_1183_ = l_Lean_Expr_app___override(v___x_1181_, v___x_1182_);
v___x_1184_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__45));
v___x_1185_ = l_Lean_Expr_const___override(v___x_1184_, v___x_1103_);
v___x_1186_ = l_Lean_Expr_app___override(v___x_1185_, v_00_u03b1_1076_);
v___x_1187_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__47));
v___x_1188_ = l_Lean_Expr_const___override(v___x_1187_, v___x_1103_);
v___x_1189_ = l_Lean_Expr_app___override(v___x_1188_, v_00_u03b1_1076_);
v___x_1190_ = l_Lean_Expr_app___override(v___x_1189_, v___x_1174_);
v___x_1191_ = l_Lean_Expr_app___override(v___x_1186_, v___x_1190_);
v___x_1192_ = l_Lean_Expr_app___override(v___x_1183_, v___x_1191_);
v___x_1193_ = l_Lean_Expr_app___override(v___x_1178_, v___x_1192_);
lean_inc_ref(v___y_1138_);
v___x_1194_ = l_Lean_Expr_app___override(v___y_1138_, v___x_1193_);
lean_inc_ref(v_arg_1100_);
v___x_1195_ = l_Lean_Expr_app___override(v___x_1194_, v_arg_1100_);
lean_inc(v_u_1075_);
v___x_1196_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v_u_1075_, v_00_u03b1_1076_, v___x_1195_, v___x_1132_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
if (lean_obj_tag(v___x_1196_) == 0)
{
lean_object* v_a_1197_; 
v_a_1197_ = lean_ctor_get(v___x_1196_, 0);
lean_inc(v_a_1197_);
lean_dec_ref_known(v___x_1196_, 1);
switch(lean_obj_tag(v_a_1197_))
{
case 1:
{
lean_object* v_inst_1198_; lean_object* v_lit_1199_; lean_object* v_proof_1200_; lean_object* v___x_1201_; 
v_inst_1198_ = lean_ctor_get(v_a_1197_, 0);
lean_inc_ref(v_inst_1198_);
v_lit_1199_ = lean_ctor_get(v_a_1197_, 1);
lean_inc_ref_n(v_lit_1199_, 2);
v_proof_1200_ = lean_ctor_get(v_a_1197_, 2);
lean_inc_ref(v_proof_1200_);
lean_dec_ref_known(v_a_1197_, 3);
lean_inc_ref(v_00_u03b1_1076_);
v___x_1201_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1075_, v_00_u03b1_1076_, v_inst_1198_, v_lit_1199_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
if (lean_obj_tag(v___x_1201_) == 0)
{
lean_object* v_a_1202_; lean_object* v___x_1204_; uint8_t v_isShared_1205_; uint8_t v_isSharedCheck_1227_; 
v_a_1202_ = lean_ctor_get(v___x_1201_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___x_1201_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1204_ = v___x_1201_;
v_isShared_1205_ = v_isSharedCheck_1227_;
goto v_resetjp_1203_;
}
else
{
lean_inc(v_a_1202_);
lean_dec(v___x_1201_);
v___x_1204_ = lean_box(0);
v_isShared_1205_ = v_isSharedCheck_1227_;
goto v_resetjp_1203_;
}
v_resetjp_1203_:
{
lean_object* v_fst_1206_; lean_object* v_snd_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1225_; 
v_fst_1206_ = lean_ctor_get(v_a_1202_, 0);
lean_inc_n(v_fst_1206_, 2);
v_snd_1207_ = lean_ctor_get(v_a_1202_, 1);
lean_inc(v_snd_1207_);
lean_dec(v_a_1202_);
v___x_1208_ = l_Lean_Expr_app___override(v___y_1138_, v_fst_1206_);
lean_inc_ref(v_arg_1098_);
v___x_1209_ = l_Lean_Expr_app___override(v___x_1208_, v_arg_1098_);
v___x_1210_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__1));
v___x_1211_ = l_Lean_Expr_const___override(v___x_1210_, v___x_1103_);
v___x_1212_ = l_Lean_Expr_app___override(v___x_1211_, v_00_u03b1_1076_);
v___x_1213_ = l_Lean_Expr_app___override(v___x_1212_, v___instRing_1079_);
v___x_1214_ = l_Lean_Expr_app___override(v___x_1213_, v_n_1077_);
v___x_1215_ = l_Lean_Expr_app___override(v___x_1214_, v_instCharP_1080_);
v___x_1216_ = l_Lean_Expr_app___override(v___x_1215_, v_arg_1100_);
v___x_1217_ = l_Lean_Expr_app___override(v___x_1216_, v_arg_1098_);
v___x_1218_ = l_Lean_Expr_app___override(v___x_1217_, v_lit_1199_);
v___x_1219_ = l_Lean_Expr_app___override(v___x_1218_, v_fst_1206_);
v___x_1220_ = l_Lean_Expr_app___override(v___x_1219_, v_proof_1200_);
v___x_1221_ = l_Lean_Expr_app___override(v___x_1220_, v_snd_1207_);
v___x_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1221_);
v___x_1223_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1223_, 0, v___x_1209_);
lean_ctor_set(v___x_1223_, 1, v___x_1222_);
lean_ctor_set_uint8(v___x_1223_, sizeof(void*)*2, v___x_1135_);
if (v_isShared_1205_ == 0)
{
lean_ctor_set(v___x_1204_, 0, v___x_1223_);
v___x_1225_ = v___x_1204_;
goto v_reusejp_1224_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v___x_1223_);
v___x_1225_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1224_;
}
v_reusejp_1224_:
{
return v___x_1225_;
}
}
}
else
{
lean_object* v_a_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1235_; 
lean_dec_ref(v_proof_1200_);
lean_dec_ref(v_lit_1199_);
lean_dec_ref(v___y_1138_);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
v_a_1228_ = lean_ctor_get(v___x_1201_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1201_);
if (v_isSharedCheck_1235_ == 0)
{
v___x_1230_ = v___x_1201_;
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_a_1228_);
lean_dec(v___x_1201_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
lean_object* v___x_1233_; 
if (v_isShared_1231_ == 0)
{
v___x_1233_ = v___x_1230_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v_a_1228_);
v___x_1233_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1232_;
}
v_reusejp_1232_:
{
return v___x_1233_;
}
}
}
}
case 2:
{
lean_object* v___x_1236_; lean_object* v___x_1237_; 
lean_dec_ref_known(v_a_1197_, 3);
lean_dec_ref(v___y_1138_);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___x_1236_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3, &lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3_once, _init_lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__3);
v___x_1237_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1236_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
return v___x_1237_;
}
default: 
{
lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; 
lean_dec(v_a_1197_);
lean_dec_ref(v___y_1138_);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___x_1238_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5, &lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5_once, _init_lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___closed__5);
v___x_1239_ = l_Lean_MessageData_ofExpr(v_n_1077_);
v___x_1240_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1240_, 0, v___x_1238_);
lean_ctor_set(v___x_1240_, 1, v___x_1239_);
v___x_1241_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65, &lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65_once, _init_lp_mathlib_Tactic_ReduceModChar_normNeg___closed__65);
v___x_1242_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1242_, 0, v___x_1240_);
lean_ctor_set(v___x_1242_, 1, v___x_1241_);
v___x_1243_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1242_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
return v___x_1243_;
}
}
}
else
{
lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1251_; 
lean_dec_ref(v___y_1138_);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v_a_1244_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1251_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1251_ == 0)
{
v___x_1246_ = v___x_1196_;
v_isShared_1247_ = v_isSharedCheck_1251_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1196_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1251_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v___x_1249_; 
if (v_isShared_1247_ == 0)
{
v___x_1249_ = v___x_1246_;
goto v_reusejp_1248_;
}
else
{
lean_object* v_reuseFailAlloc_1250_; 
v_reuseFailAlloc_1250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1250_, 0, v_a_1244_);
v___x_1249_ = v_reuseFailAlloc_1250_;
goto v_reusejp_1248_;
}
v_reusejp_1248_:
{
return v___x_1249_;
}
}
}
}
v___jp_1252_:
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___f_1277_; lean_object* v___x_1278_; 
v___x_1253_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__50));
lean_inc_ref_n(v___x_1103_, 5);
lean_inc_n(v_u_1075_, 2);
v___x_1254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1254_, 0, v_u_1075_);
lean_ctor_set(v___x_1254_, 1, v___x_1103_);
v___x_1255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1255_, 0, v_u_1075_);
lean_ctor_set(v___x_1255_, 1, v___x_1254_);
lean_inc_ref(v___x_1255_);
v___x_1256_ = l_Lean_Expr_const___override(v___x_1253_, v___x_1255_);
lean_inc_ref_n(v_00_u03b1_1076_, 7);
v___x_1257_ = l_Lean_Expr_app___override(v___x_1256_, v_00_u03b1_1076_);
v___x_1258_ = l_Lean_Expr_app___override(v___x_1257_, v_00_u03b1_1076_);
v___x_1259_ = l_Lean_Expr_app___override(v___x_1258_, v_00_u03b1_1076_);
v___x_1260_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__52));
v___x_1261_ = l_Lean_Expr_const___override(v___x_1260_, v___x_1103_);
v___x_1262_ = l_Lean_Expr_app___override(v___x_1261_, v_00_u03b1_1076_);
v___x_1263_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__55));
v___x_1264_ = l_Lean_Expr_const___override(v___x_1263_, v___x_1103_);
v___x_1265_ = l_Lean_Expr_app___override(v___x_1264_, v_00_u03b1_1076_);
v___x_1266_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__57));
v___x_1267_ = l_Lean_Expr_const___override(v___x_1266_, v___x_1103_);
v___x_1268_ = l_Lean_Expr_app___override(v___x_1267_, v_00_u03b1_1076_);
v___x_1269_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__18));
v___x_1270_ = l_Lean_Expr_const___override(v___x_1269_, v___x_1103_);
v___x_1271_ = l_Lean_Expr_app___override(v___x_1270_, v_00_u03b1_1076_);
lean_inc_ref(v___instRing_1079_);
v___x_1272_ = l_Lean_Expr_app___override(v___x_1271_, v___instRing_1079_);
v___x_1273_ = l_Lean_Expr_app___override(v___x_1268_, v___x_1272_);
v___x_1274_ = l_Lean_Expr_app___override(v___x_1265_, v___x_1273_);
v___x_1275_ = l_Lean_Expr_app___override(v___x_1262_, v___x_1274_);
v___x_1276_ = l_Lean_Expr_app___override(v___x_1259_, v___x_1275_);
lean_inc_ref(v___x_1276_);
v___f_1277_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_normNeg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1277_, 0, v_fn_1099_);
lean_closure_set(v___f_1277_, 1, v___x_1276_);
v___x_1278_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v___f_1277_, v___x_1132_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
if (lean_obj_tag(v___x_1278_) == 0)
{
lean_object* v_a_1279_; uint8_t v___x_1280_; 
v_a_1279_ = lean_ctor_get(v___x_1278_, 0);
lean_inc(v_a_1279_);
lean_dec_ref_known(v___x_1278_, 1);
v___x_1280_ = lean_unbox(v_a_1279_);
lean_dec(v_a_1279_);
if (v___x_1280_ == 0)
{
lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v_a_1283_; lean_object* v___x_1285_; uint8_t v_isShared_1286_; uint8_t v_isSharedCheck_1290_; 
lean_dec_ref(v___x_1276_);
lean_dec_ref_known(v___x_1255_, 2);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___x_1281_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_1282_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1281_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
v_isSharedCheck_1290_ = !lean_is_exclusive(v___x_1282_);
if (v_isSharedCheck_1290_ == 0)
{
v___x_1285_ = v___x_1282_;
v_isShared_1286_ = v_isSharedCheck_1290_;
goto v_resetjp_1284_;
}
else
{
lean_inc(v_a_1283_);
lean_dec(v___x_1282_);
v___x_1285_ = lean_box(0);
v_isShared_1286_ = v_isSharedCheck_1290_;
goto v_resetjp_1284_;
}
v_resetjp_1284_:
{
lean_object* v___x_1288_; 
if (v_isShared_1286_ == 0)
{
v___x_1288_ = v___x_1285_;
goto v_reusejp_1287_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v_a_1283_);
v___x_1288_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1287_;
}
v_reusejp_1287_:
{
return v___x_1288_;
}
}
}
else
{
v___y_1137_ = v___x_1255_;
v___y_1138_ = v___x_1276_;
goto v___jp_1136_;
}
}
else
{
lean_object* v_a_1291_; lean_object* v___x_1293_; uint8_t v_isShared_1294_; uint8_t v_isSharedCheck_1298_; 
lean_dec_ref(v___x_1276_);
lean_dec_ref_known(v___x_1255_, 2);
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v_a_1291_ = lean_ctor_get(v___x_1278_, 0);
v_isSharedCheck_1298_ = !lean_is_exclusive(v___x_1278_);
if (v_isSharedCheck_1298_ == 0)
{
v___x_1293_ = v___x_1278_;
v_isShared_1294_ = v_isSharedCheck_1298_;
goto v_resetjp_1292_;
}
else
{
lean_inc(v_a_1291_);
lean_dec(v___x_1278_);
v___x_1293_ = lean_box(0);
v_isShared_1294_ = v_isSharedCheck_1298_;
goto v_resetjp_1292_;
}
v_resetjp_1292_:
{
lean_object* v___x_1296_; 
if (v_isShared_1294_ == 0)
{
v___x_1296_ = v___x_1293_;
goto v_reusejp_1295_;
}
else
{
lean_object* v_reuseFailAlloc_1297_; 
v_reuseFailAlloc_1297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1297_, 0, v_a_1291_);
v___x_1296_ = v_reuseFailAlloc_1297_;
goto v_reusejp_1295_;
}
v_reusejp_1295_:
{
return v___x_1296_;
}
}
}
}
}
else
{
lean_object* v_a_1310_; lean_object* v___x_1312_; uint8_t v_isShared_1313_; uint8_t v_isSharedCheck_1317_; 
lean_dec_ref_known(v___x_1103_, 2);
lean_dec_ref(v_arg_1100_);
lean_dec_ref(v_fn_1099_);
lean_dec_ref(v_arg_1098_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v_a_1310_ = lean_ctor_get(v___x_1133_, 0);
v_isSharedCheck_1317_ = !lean_is_exclusive(v___x_1133_);
if (v_isSharedCheck_1317_ == 0)
{
v___x_1312_ = v___x_1133_;
v_isShared_1313_ = v_isSharedCheck_1317_;
goto v_resetjp_1311_;
}
else
{
lean_inc(v_a_1310_);
lean_dec(v___x_1133_);
v___x_1312_ = lean_box(0);
v_isShared_1313_ = v_isSharedCheck_1317_;
goto v_resetjp_1311_;
}
v_resetjp_1311_:
{
lean_object* v___x_1315_; 
if (v_isShared_1313_ == 0)
{
v___x_1315_ = v___x_1312_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v_a_1310_);
v___x_1315_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
return v___x_1315_;
}
}
}
}
else
{
lean_dec_ref_known(v_arg_1095_, 2);
lean_dec_ref(v_fn_1096_);
lean_dec_ref_known(v_a_1087_, 2);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___y_1089_ = v_a_1081_;
v___y_1090_ = v_a_1082_;
v___y_1091_ = v_a_1083_;
v___y_1092_ = v_a_1084_;
goto v___jp_1088_;
}
}
else
{
lean_dec_ref(v_arg_1095_);
lean_dec_ref_known(v_a_1087_, 2);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___y_1089_ = v_a_1081_;
v___y_1090_ = v_a_1082_;
v___y_1091_ = v_a_1083_;
v___y_1092_ = v_a_1084_;
goto v___jp_1088_;
}
}
else
{
lean_dec(v_a_1087_);
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v___y_1089_ = v_a_1081_;
v___y_1090_ = v_a_1082_;
v___y_1091_ = v_a_1083_;
v___y_1092_ = v_a_1084_;
goto v___jp_1088_;
}
v___jp_1088_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1093_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12, &lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__12);
v___x_1094_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1093_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
return v___x_1094_;
}
}
else
{
lean_object* v_a_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1325_; 
lean_dec_ref(v_instCharP_1080_);
lean_dec_ref(v___instRing_1079_);
lean_dec_ref(v_n_1077_);
lean_dec_ref(v_00_u03b1_1076_);
lean_dec(v_u_1075_);
v_a_1318_ = lean_ctor_get(v___x_1086_, 0);
v_isSharedCheck_1325_ = !lean_is_exclusive(v___x_1086_);
if (v_isSharedCheck_1325_ == 0)
{
v___x_1320_ = v___x_1086_;
v_isShared_1321_ = v_isSharedCheck_1325_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_a_1318_);
lean_dec(v___x_1086_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1325_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v___x_1323_; 
if (v_isShared_1321_ == 0)
{
v___x_1323_ = v___x_1320_;
goto v_reusejp_1322_;
}
else
{
lean_object* v_reuseFailAlloc_1324_; 
v_reuseFailAlloc_1324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1324_, 0, v_a_1318_);
v___x_1323_ = v_reuseFailAlloc_1324_;
goto v_reusejp_1322_;
}
v_reusejp_1322_:
{
return v___x_1323_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul___boxed(lean_object* v_u_1326_, lean_object* v_00_u03b1_1327_, lean_object* v_n_1328_, lean_object* v_e_1329_, lean_object* v___instRing_1330_, lean_object* v_instCharP_1331_, lean_object* v_a_1332_, lean_object* v_a_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul(v_u_1326_, v_00_u03b1_1327_, v_n_1328_, v_e_1329_, v___instRing_1330_, v_instCharP_1331_, v_a_1332_, v_a_1333_, v_a_1334_, v_a_1335_);
lean_dec(v_a_1335_);
lean_dec_ref(v_a_1334_);
lean_dec(v_a_1333_);
lean_dec_ref(v_a_1332_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg(lean_object* v_x_1338_){
_start:
{
if (lean_obj_tag(v_x_1338_) == 0)
{
lean_object* v___x_1339_; 
v___x_1339_ = lean_unsigned_to_nat(0u);
return v___x_1339_;
}
else
{
lean_object* v___x_1340_; 
v___x_1340_ = lean_unsigned_to_nat(1u);
return v___x_1340_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg___boxed(lean_object* v_x_1341_){
_start:
{
lean_object* v_res_1342_; 
v_res_1342_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg(v_x_1341_);
lean_dec(v_x_1341_);
return v_res_1342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx(lean_object* v_u_1343_, lean_object* v_00_u03b1_1344_, lean_object* v_x_1345_){
_start:
{
lean_object* v___x_1346_; 
v___x_1346_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___redArg(v_x_1345_);
return v___x_1346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx___boxed(lean_object* v_u_1347_, lean_object* v_00_u03b1_1348_, lean_object* v_x_1349_){
_start:
{
lean_object* v_res_1350_; 
v_res_1350_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorIdx(v_u_1347_, v_00_u03b1_1348_, v_x_1349_);
lean_dec(v_x_1349_);
lean_dec_ref(v_00_u03b1_1348_);
lean_dec(v_u_1347_);
return v_res_1350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(lean_object* v_t_1351_, lean_object* v_k_1352_){
_start:
{
if (lean_obj_tag(v_t_1351_) == 0)
{
lean_object* v_n_1353_; lean_object* v_instRing_1354_; lean_object* v_instCharP_1355_; lean_object* v___x_1356_; 
v_n_1353_ = lean_ctor_get(v_t_1351_, 0);
lean_inc_ref(v_n_1353_);
v_instRing_1354_ = lean_ctor_get(v_t_1351_, 1);
lean_inc_ref(v_instRing_1354_);
v_instCharP_1355_ = lean_ctor_get(v_t_1351_, 2);
lean_inc_ref(v_instCharP_1355_);
lean_dec_ref_known(v_t_1351_, 3);
v___x_1356_ = lean_apply_3(v_k_1352_, v_n_1353_, v_instRing_1354_, v_instCharP_1355_);
return v___x_1356_;
}
else
{
return v_k_1352_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim(lean_object* v_u_1357_, lean_object* v_00_u03b1_1358_, lean_object* v_motive_1359_, lean_object* v_ctorIdx_1360_, lean_object* v_t_1361_, lean_object* v_h_1362_, lean_object* v_k_1363_){
_start:
{
lean_object* v___x_1364_; 
v___x_1364_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(v_t_1361_, v_k_1363_);
return v___x_1364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___boxed(lean_object* v_u_1365_, lean_object* v_00_u03b1_1366_, lean_object* v_motive_1367_, lean_object* v_ctorIdx_1368_, lean_object* v_t_1369_, lean_object* v_h_1370_, lean_object* v_k_1371_){
_start:
{
lean_object* v_res_1372_; 
v_res_1372_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim(v_u_1365_, v_00_u03b1_1366_, v_motive_1367_, v_ctorIdx_1368_, v_t_1369_, v_h_1370_, v_k_1371_);
lean_dec(v_ctorIdx_1368_);
lean_dec_ref(v_00_u03b1_1366_);
lean_dec(v_u_1365_);
return v_res_1372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim___redArg(lean_object* v_t_1373_, lean_object* v_intLike_1374_){
_start:
{
lean_object* v___x_1375_; 
v___x_1375_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(v_t_1373_, v_intLike_1374_);
return v___x_1375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim(lean_object* v_u_1376_, lean_object* v_00_u03b1_1377_, lean_object* v_motive_1378_, lean_object* v_t_1379_, lean_object* v_h_1380_, lean_object* v_intLike_1381_){
_start:
{
lean_object* v___x_1382_; 
v___x_1382_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(v_t_1379_, v_intLike_1381_);
return v___x_1382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim___boxed(lean_object* v_u_1383_, lean_object* v_00_u03b1_1384_, lean_object* v_motive_1385_, lean_object* v_t_1386_, lean_object* v_h_1387_, lean_object* v_intLike_1388_){
_start:
{
lean_object* v_res_1389_; 
v_res_1389_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_intLike_elim(v_u_1383_, v_00_u03b1_1384_, v_motive_1385_, v_t_1386_, v_h_1387_, v_intLike_1388_);
lean_dec_ref(v_00_u03b1_1384_);
lean_dec(v_u_1383_);
return v_res_1389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim___redArg(lean_object* v_t_1390_, lean_object* v_failure_1391_){
_start:
{
lean_object* v___x_1392_; 
v___x_1392_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(v_t_1390_, v_failure_1391_);
return v___x_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim(lean_object* v_u_1393_, lean_object* v_00_u03b1_1394_, lean_object* v_motive_1395_, lean_object* v_t_1396_, lean_object* v_h_1397_, lean_object* v_failure_1398_){
_start:
{
lean_object* v___x_1399_; 
v___x_1399_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_ctorElim___redArg(v_t_1396_, v_failure_1398_);
return v___x_1399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim___boxed(lean_object* v_u_1400_, lean_object* v_00_u03b1_1401_, lean_object* v_motive_1402_, lean_object* v_t_1403_, lean_object* v_h_1404_, lean_object* v_failure_1405_){
_start:
{
lean_object* v_res_1406_; 
v_res_1406_ = lp_mathlib_Tactic_ReduceModChar_TypeToCharPResult_failure_elim(v_u_1400_, v_00_u03b1_1401_, v_motive_1402_, v_t_1403_, v_h_1404_, v_failure_1405_);
lean_dec_ref(v_00_u03b1_1401_);
lean_dec(v_u_1400_);
return v_res_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_instInhabitedTypeToCharPResult(lean_object* v_u_1407_, lean_object* v_00_u03b1_1408_){
_start:
{
lean_object* v___x_1409_; 
v___x_1409_ = lean_box(1);
return v___x_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_instInhabitedTypeToCharPResult___boxed(lean_object* v_u_1410_, lean_object* v_00_u03b1_1411_){
_start:
{
lean_object* v_res_1412_; 
v_res_1412_ = lp_mathlib_Tactic_ReduceModChar_instInhabitedTypeToCharPResult(v_u_1410_, v_00_u03b1_1411_);
lean_dec_ref(v_00_u03b1_1411_);
lean_dec(v_u_1410_);
return v_res_1412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0(lean_object* v___x_1415_, lean_object* v___x_1416_, lean_object* v___x_1417_, lean_object* v_t_1418_, lean_object* v___x_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_){
_start:
{
lean_object* v___x_1425_; 
v___x_1425_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v___x_1415_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
if (lean_obj_tag(v___x_1425_) == 0)
{
lean_object* v_a_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1500_; 
v_a_1426_ = lean_ctor_get(v___x_1425_, 0);
v_isSharedCheck_1500_ = !lean_is_exclusive(v___x_1425_);
if (v_isSharedCheck_1500_ == 0)
{
v___x_1428_ = v___x_1425_;
v_isShared_1429_ = v_isSharedCheck_1500_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_a_1426_);
lean_dec(v___x_1425_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1500_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
if (lean_obj_tag(v_a_1426_) == 1)
{
lean_object* v_a_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; uint8_t v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; 
lean_del_object(v___x_1428_);
v_a_1430_ = lean_ctor_get(v_a_1426_, 0);
lean_inc(v_a_1430_);
lean_dec_ref_known(v_a_1426_, 1);
v___x_1431_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__0));
v___x_1432_ = l_Lean_Expr_const___override(v___x_1431_, v___x_1416_);
v___x_1433_ = 0;
v___x_1434_ = lean_box(0);
v___x_1435_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v___x_1432_, v___x_1433_, v___x_1434_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
if (lean_obj_tag(v___x_1435_) == 0)
{
lean_object* v_a_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; 
v_a_1436_ = lean_ctor_get(v___x_1435_, 0);
lean_inc_n(v_a_1436_, 2);
lean_dec_ref_known(v___x_1435_, 1);
v___x_1437_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___closed__0));
lean_inc_n(v___x_1417_, 2);
v___x_1438_ = l_Lean_Expr_const___override(v___x_1437_, v___x_1417_);
lean_inc_ref_n(v_t_1418_, 2);
v___x_1439_ = l_Lean_Expr_app___override(v___x_1438_, v_t_1418_);
v___x_1440_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__37));
v___x_1441_ = l_Lean_Expr_const___override(v___x_1440_, v___x_1417_);
v___x_1442_ = l_Lean_Expr_app___override(v___x_1441_, v_t_1418_);
v___x_1443_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normNeg___closed__34));
v___x_1444_ = l_Lean_Name_mkStr2(v___x_1419_, v___x_1443_);
v___x_1445_ = l_Lean_Expr_const___override(v___x_1444_, v___x_1417_);
v___x_1446_ = l_Lean_Expr_app___override(v___x_1445_, v_t_1418_);
lean_inc(v_a_1430_);
v___x_1447_ = l_Lean_Expr_app___override(v___x_1446_, v_a_1430_);
v___x_1448_ = l_Lean_Expr_app___override(v___x_1442_, v___x_1447_);
v___x_1449_ = l_Lean_Expr_app___override(v___x_1439_, v___x_1448_);
v___x_1450_ = l_Lean_Expr_app___override(v___x_1449_, v_a_1436_);
v___x_1451_ = lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(v___x_1450_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
if (lean_obj_tag(v___x_1451_) == 0)
{
lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1479_; 
v_a_1452_ = lean_ctor_get(v___x_1451_, 0);
v_isSharedCheck_1479_ = !lean_is_exclusive(v___x_1451_);
if (v_isSharedCheck_1479_ == 0)
{
v___x_1454_ = v___x_1451_;
v_isShared_1455_ = v_isSharedCheck_1479_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1451_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1479_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
if (lean_obj_tag(v_a_1452_) == 1)
{
lean_object* v_val_1456_; lean_object* v___x_1457_; 
lean_del_object(v___x_1454_);
v_val_1456_ = lean_ctor_get(v_a_1452_, 0);
lean_inc(v_val_1456_);
lean_dec_ref_known(v_a_1452_, 1);
v___x_1457_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(v_a_1436_, v___y_1421_);
if (lean_obj_tag(v___x_1457_) == 0)
{
lean_object* v_a_1458_; lean_object* v___x_1460_; uint8_t v_isShared_1461_; uint8_t v_isSharedCheck_1466_; 
v_a_1458_ = lean_ctor_get(v___x_1457_, 0);
v_isSharedCheck_1466_ = !lean_is_exclusive(v___x_1457_);
if (v_isSharedCheck_1466_ == 0)
{
v___x_1460_ = v___x_1457_;
v_isShared_1461_ = v_isSharedCheck_1466_;
goto v_resetjp_1459_;
}
else
{
lean_inc(v_a_1458_);
lean_dec(v___x_1457_);
v___x_1460_ = lean_box(0);
v_isShared_1461_ = v_isSharedCheck_1466_;
goto v_resetjp_1459_;
}
v_resetjp_1459_:
{
lean_object* v___x_1462_; lean_object* v___x_1464_; 
v___x_1462_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1462_, 0, v_a_1458_);
lean_ctor_set(v___x_1462_, 1, v_a_1430_);
lean_ctor_set(v___x_1462_, 2, v_val_1456_);
if (v_isShared_1461_ == 0)
{
lean_ctor_set(v___x_1460_, 0, v___x_1462_);
v___x_1464_ = v___x_1460_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v___x_1462_);
v___x_1464_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
return v___x_1464_;
}
}
}
else
{
lean_object* v_a_1467_; lean_object* v___x_1469_; uint8_t v_isShared_1470_; uint8_t v_isSharedCheck_1474_; 
lean_dec(v_val_1456_);
lean_dec(v_a_1430_);
v_a_1467_ = lean_ctor_get(v___x_1457_, 0);
v_isSharedCheck_1474_ = !lean_is_exclusive(v___x_1457_);
if (v_isSharedCheck_1474_ == 0)
{
v___x_1469_ = v___x_1457_;
v_isShared_1470_ = v_isSharedCheck_1474_;
goto v_resetjp_1468_;
}
else
{
lean_inc(v_a_1467_);
lean_dec(v___x_1457_);
v___x_1469_ = lean_box(0);
v_isShared_1470_ = v_isSharedCheck_1474_;
goto v_resetjp_1468_;
}
v_resetjp_1468_:
{
lean_object* v___x_1472_; 
if (v_isShared_1470_ == 0)
{
v___x_1472_ = v___x_1469_;
goto v_reusejp_1471_;
}
else
{
lean_object* v_reuseFailAlloc_1473_; 
v_reuseFailAlloc_1473_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1473_, 0, v_a_1467_);
v___x_1472_ = v_reuseFailAlloc_1473_;
goto v_reusejp_1471_;
}
v_reusejp_1471_:
{
return v___x_1472_;
}
}
}
}
else
{
lean_object* v___x_1475_; lean_object* v___x_1477_; 
lean_dec(v_a_1452_);
lean_dec(v_a_1436_);
lean_dec(v_a_1430_);
v___x_1475_ = lean_box(1);
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 0, v___x_1475_);
v___x_1477_ = v___x_1454_;
goto v_reusejp_1476_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v___x_1475_);
v___x_1477_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1476_;
}
v_reusejp_1476_:
{
return v___x_1477_;
}
}
}
}
else
{
lean_object* v_a_1480_; lean_object* v___x_1482_; uint8_t v_isShared_1483_; uint8_t v_isSharedCheck_1487_; 
lean_dec(v_a_1436_);
lean_dec(v_a_1430_);
v_a_1480_ = lean_ctor_get(v___x_1451_, 0);
v_isSharedCheck_1487_ = !lean_is_exclusive(v___x_1451_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1482_ = v___x_1451_;
v_isShared_1483_ = v_isSharedCheck_1487_;
goto v_resetjp_1481_;
}
else
{
lean_inc(v_a_1480_);
lean_dec(v___x_1451_);
v___x_1482_ = lean_box(0);
v_isShared_1483_ = v_isSharedCheck_1487_;
goto v_resetjp_1481_;
}
v_resetjp_1481_:
{
lean_object* v___x_1485_; 
if (v_isShared_1483_ == 0)
{
v___x_1485_ = v___x_1482_;
goto v_reusejp_1484_;
}
else
{
lean_object* v_reuseFailAlloc_1486_; 
v_reuseFailAlloc_1486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1486_, 0, v_a_1480_);
v___x_1485_ = v_reuseFailAlloc_1486_;
goto v_reusejp_1484_;
}
v_reusejp_1484_:
{
return v___x_1485_;
}
}
}
}
else
{
lean_object* v_a_1488_; lean_object* v___x_1490_; uint8_t v_isShared_1491_; uint8_t v_isSharedCheck_1495_; 
lean_dec(v_a_1430_);
lean_dec_ref(v___x_1419_);
lean_dec_ref(v_t_1418_);
lean_dec(v___x_1417_);
v_a_1488_ = lean_ctor_get(v___x_1435_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1435_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1490_ = v___x_1435_;
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_a_1488_);
lean_dec(v___x_1435_);
v___x_1490_ = lean_box(0);
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
v_resetjp_1489_:
{
lean_object* v___x_1493_; 
if (v_isShared_1491_ == 0)
{
v___x_1493_ = v___x_1490_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v_a_1488_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
else
{
lean_object* v___x_1496_; lean_object* v___x_1498_; 
lean_dec(v_a_1426_);
lean_dec_ref(v___x_1419_);
lean_dec_ref(v_t_1418_);
lean_dec(v___x_1417_);
lean_dec(v___x_1416_);
v___x_1496_ = lean_box(1);
if (v_isShared_1429_ == 0)
{
lean_ctor_set(v___x_1428_, 0, v___x_1496_);
v___x_1498_ = v___x_1428_;
goto v_reusejp_1497_;
}
else
{
lean_object* v_reuseFailAlloc_1499_; 
v_reuseFailAlloc_1499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1499_, 0, v___x_1496_);
v___x_1498_ = v_reuseFailAlloc_1499_;
goto v_reusejp_1497_;
}
v_reusejp_1497_:
{
return v___x_1498_;
}
}
}
}
else
{
lean_object* v_a_1501_; lean_object* v___x_1503_; uint8_t v_isShared_1504_; uint8_t v_isSharedCheck_1508_; 
lean_dec_ref(v___x_1419_);
lean_dec_ref(v_t_1418_);
lean_dec(v___x_1417_);
lean_dec(v___x_1416_);
v_a_1501_ = lean_ctor_get(v___x_1425_, 0);
v_isSharedCheck_1508_ = !lean_is_exclusive(v___x_1425_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1503_ = v___x_1425_;
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
else
{
lean_inc(v_a_1501_);
lean_dec(v___x_1425_);
v___x_1503_ = lean_box(0);
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
v_resetjp_1502_:
{
lean_object* v___x_1506_; 
if (v_isShared_1504_ == 0)
{
v___x_1506_ = v___x_1503_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v_a_1501_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
return v___x_1506_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___boxed(lean_object* v___x_1509_, lean_object* v___x_1510_, lean_object* v___x_1511_, lean_object* v_t_1512_, lean_object* v___x_1513_, lean_object* v___y_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_){
_start:
{
lean_object* v_res_1519_; 
v_res_1519_ = lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0(v___x_1509_, v___x_1510_, v___x_1511_, v_t_1512_, v___x_1513_, v___y_1514_, v___y_1515_, v___y_1516_, v___y_1517_);
lean_dec(v___y_1517_);
lean_dec_ref(v___y_1516_);
lean_dec(v___y_1515_);
lean_dec_ref(v___y_1514_);
return v_res_1519_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8(void){
_start:
{
lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; 
v___x_1534_ = lean_box(0);
v___x_1535_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__7));
v___x_1536_ = l_Lean_Expr_const___override(v___x_1535_, v___x_1534_);
return v___x_1536_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12(void){
_start:
{
lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___x_1542_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normBareNumeral___closed__20));
v___x_1543_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__11));
v___x_1544_ = l_Lean_Expr_const___override(v___x_1543_, v___x_1542_);
return v___x_1544_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15(void){
_start:
{
lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; 
v___x_1549_ = lean_box(0);
v___x_1550_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__14));
v___x_1551_ = l_Lean_Expr_const___override(v___x_1550_, v___x_1549_);
return v___x_1551_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18(void){
_start:
{
lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; 
v___x_1556_ = lean_box(0);
v___x_1557_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__17));
v___x_1558_ = l_Lean_Expr_const___override(v___x_1557_, v___x_1556_);
return v___x_1558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP(lean_object* v_u_1559_, uint8_t v_expensive_1560_, lean_object* v_t_1561_, lean_object* v_a_1562_, lean_object* v_a_1563_, lean_object* v_a_1564_, lean_object* v_a_1565_){
_start:
{
lean_object* v___y_1568_; lean_object* v___y_1569_; lean_object* v___y_1570_; lean_object* v___y_1571_; lean_object* v___x_1583_; lean_object* v_fst_1584_; 
lean_inc_ref(v_t_1561_);
v___x_1583_ = l_Lean_Expr_getAppFnArgs(v_t_1561_);
v_fst_1584_ = lean_ctor_get(v___x_1583_, 0);
lean_inc(v_fst_1584_);
if (lean_obj_tag(v_fst_1584_) == 1)
{
lean_object* v_pre_1585_; 
v_pre_1585_ = lean_ctor_get(v_fst_1584_, 0);
if (lean_obj_tag(v_pre_1585_) == 0)
{
lean_object* v_snd_1586_; lean_object* v___x_1588_; uint8_t v_isShared_1589_; uint8_t v_isSharedCheck_1657_; 
v_snd_1586_ = lean_ctor_get(v___x_1583_, 1);
v_isSharedCheck_1657_ = !lean_is_exclusive(v___x_1583_);
if (v_isSharedCheck_1657_ == 0)
{
lean_object* v_unused_1658_; 
v_unused_1658_ = lean_ctor_get(v___x_1583_, 0);
lean_dec(v_unused_1658_);
v___x_1588_ = v___x_1583_;
v_isShared_1589_ = v_isSharedCheck_1657_;
goto v_resetjp_1587_;
}
else
{
lean_inc(v_snd_1586_);
lean_dec(v___x_1583_);
v___x_1588_ = lean_box(0);
v_isShared_1589_ = v_isSharedCheck_1657_;
goto v_resetjp_1587_;
}
v_resetjp_1587_:
{
lean_object* v_str_1590_; lean_object* v___x_1591_; uint8_t v___x_1592_; 
v_str_1590_ = lean_ctor_get(v_fst_1584_, 1);
lean_inc_ref(v_str_1590_);
lean_dec_ref_known(v_fst_1584_, 2);
v___x_1591_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__1));
v___x_1592_ = lean_string_dec_eq(v_str_1590_, v___x_1591_);
if (v___x_1592_ == 0)
{
lean_object* v___x_1593_; uint8_t v___x_1594_; 
v___x_1593_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__2));
v___x_1594_ = lean_string_dec_eq(v_str_1590_, v___x_1593_);
lean_dec_ref(v_str_1590_);
if (v___x_1594_ == 0)
{
lean_del_object(v___x_1588_);
lean_dec(v_snd_1586_);
v___y_1568_ = v_a_1562_;
v___y_1569_ = v_a_1563_;
v___y_1570_ = v_a_1564_;
v___y_1571_ = v_a_1565_;
goto v___jp_1567_;
}
else
{
lean_object* v___x_1595_; lean_object* v___x_1596_; uint8_t v___x_1597_; 
v___x_1595_ = lean_array_get_size(v_snd_1586_);
v___x_1596_ = lean_unsigned_to_nat(2u);
v___x_1597_ = lean_nat_dec_eq(v___x_1595_, v___x_1596_);
if (v___x_1597_ == 0)
{
lean_del_object(v___x_1588_);
lean_dec(v_snd_1586_);
v___y_1568_ = v_a_1562_;
v___y_1569_ = v_a_1563_;
v___y_1570_ = v_a_1564_;
v___y_1571_ = v_a_1565_;
goto v___jp_1567_;
}
else
{
lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; 
lean_dec_ref(v_t_1561_);
v___x_1598_ = lean_unsigned_to_nat(0u);
v___x_1599_ = lean_array_fget(v_snd_1586_, v___x_1598_);
lean_dec(v_snd_1586_);
lean_inc(v___x_1599_);
lean_inc(v_u_1559_);
v___x_1600_ = lp_mathlib_Tactic_ReduceModChar_typeToCharP(v_u_1559_, v_expensive_1560_, v___x_1599_, v_a_1562_, v_a_1563_, v_a_1564_, v_a_1565_);
if (lean_obj_tag(v___x_1600_) == 0)
{
lean_object* v_a_1601_; lean_object* v___x_1603_; uint8_t v_isShared_1604_; uint8_t v_isSharedCheck_1640_; 
v_a_1601_ = lean_ctor_get(v___x_1600_, 0);
v_isSharedCheck_1640_ = !lean_is_exclusive(v___x_1600_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1603_ = v___x_1600_;
v_isShared_1604_ = v_isSharedCheck_1640_;
goto v_resetjp_1602_;
}
else
{
lean_inc(v_a_1601_);
lean_dec(v___x_1600_);
v___x_1603_ = lean_box(0);
v_isShared_1604_ = v_isSharedCheck_1640_;
goto v_resetjp_1602_;
}
v_resetjp_1602_:
{
if (lean_obj_tag(v_a_1601_) == 0)
{
lean_object* v_n_1605_; lean_object* v_instRing_1606_; lean_object* v_instCharP_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1635_; 
v_n_1605_ = lean_ctor_get(v_a_1601_, 0);
v_instRing_1606_ = lean_ctor_get(v_a_1601_, 1);
v_instCharP_1607_ = lean_ctor_get(v_a_1601_, 2);
v_isSharedCheck_1635_ = !lean_is_exclusive(v_a_1601_);
if (v_isSharedCheck_1635_ == 0)
{
v___x_1609_ = v_a_1601_;
v_isShared_1610_ = v_isSharedCheck_1635_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_instCharP_1607_);
lean_inc(v_instRing_1606_);
lean_inc(v_n_1605_);
lean_dec(v_a_1601_);
v___x_1609_ = lean_box(0);
v_isShared_1610_ = v_isSharedCheck_1635_;
goto v_resetjp_1608_;
}
v_resetjp_1608_:
{
lean_object* v___x_1611_; lean_object* v___x_1613_; 
v___x_1611_ = lean_box(0);
if (v_isShared_1589_ == 0)
{
lean_ctor_set_tag(v___x_1588_, 1);
lean_ctor_set(v___x_1588_, 1, v___x_1611_);
lean_ctor_set(v___x_1588_, 0, v_u_1559_);
v___x_1613_ = v___x_1588_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1634_; 
v_reuseFailAlloc_1634_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1634_, 0, v_u_1559_);
lean_ctor_set(v_reuseFailAlloc_1634_, 1, v___x_1611_);
v___x_1613_ = v_reuseFailAlloc_1634_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1629_; 
v___x_1614_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__18));
lean_inc_ref_n(v___x_1613_, 2);
v___x_1615_ = l_Lean_Expr_const___override(v___x_1614_, v___x_1613_);
lean_inc_n(v___x_1599_, 2);
v___x_1616_ = l_Lean_Expr_app___override(v___x_1615_, v___x_1599_);
lean_inc_ref(v_instRing_1606_);
v___x_1617_ = l_Lean_Expr_app___override(v___x_1616_, v_instRing_1606_);
v___x_1618_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__4));
v___x_1619_ = l_Lean_Expr_const___override(v___x_1618_, v___x_1613_);
v___x_1620_ = l_Lean_Expr_app___override(v___x_1619_, v___x_1599_);
v___x_1621_ = l_Lean_Expr_app___override(v___x_1620_, v_instRing_1606_);
v___x_1622_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__6));
v___x_1623_ = l_Lean_Expr_const___override(v___x_1622_, v___x_1613_);
v___x_1624_ = l_Lean_Expr_app___override(v___x_1623_, v___x_1599_);
v___x_1625_ = l_Lean_Expr_app___override(v___x_1624_, v___x_1617_);
lean_inc_ref(v_n_1605_);
v___x_1626_ = l_Lean_Expr_app___override(v___x_1625_, v_n_1605_);
v___x_1627_ = l_Lean_Expr_app___override(v___x_1626_, v_instCharP_1607_);
if (v_isShared_1610_ == 0)
{
lean_ctor_set(v___x_1609_, 2, v___x_1627_);
lean_ctor_set(v___x_1609_, 1, v___x_1621_);
v___x_1629_ = v___x_1609_;
goto v_reusejp_1628_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v_n_1605_);
lean_ctor_set(v_reuseFailAlloc_1633_, 1, v___x_1621_);
lean_ctor_set(v_reuseFailAlloc_1633_, 2, v___x_1627_);
v___x_1629_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1628_;
}
v_reusejp_1628_:
{
lean_object* v___x_1631_; 
if (v_isShared_1604_ == 0)
{
lean_ctor_set(v___x_1603_, 0, v___x_1629_);
v___x_1631_ = v___x_1603_;
goto v_reusejp_1630_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v___x_1629_);
v___x_1631_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1630_;
}
v_reusejp_1630_:
{
return v___x_1631_;
}
}
}
}
}
else
{
lean_object* v___x_1636_; lean_object* v___x_1638_; 
lean_dec(v___x_1599_);
lean_del_object(v___x_1588_);
lean_dec(v_u_1559_);
v___x_1636_ = lean_box(1);
if (v_isShared_1604_ == 0)
{
lean_ctor_set(v___x_1603_, 0, v___x_1636_);
v___x_1638_ = v___x_1603_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v___x_1636_);
v___x_1638_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
return v___x_1638_;
}
}
}
}
else
{
lean_dec(v___x_1599_);
lean_del_object(v___x_1588_);
lean_dec(v_u_1559_);
return v___x_1600_;
}
}
}
}
else
{
lean_object* v___x_1641_; lean_object* v___x_1642_; uint8_t v___x_1643_; 
lean_dec_ref(v_str_1590_);
lean_del_object(v___x_1588_);
v___x_1641_ = lean_array_get_size(v_snd_1586_);
v___x_1642_ = lean_unsigned_to_nat(1u);
v___x_1643_ = lean_nat_dec_eq(v___x_1641_, v___x_1642_);
if (v___x_1643_ == 0)
{
lean_dec(v_snd_1586_);
v___y_1568_ = v_a_1562_;
v___y_1569_ = v_a_1563_;
v___y_1570_ = v_a_1564_;
v___y_1571_ = v_a_1565_;
goto v___jp_1567_;
}
else
{
lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; 
lean_dec_ref(v_t_1561_);
lean_dec(v_u_1559_);
v___x_1644_ = lean_unsigned_to_nat(0u);
v___x_1645_ = lean_array_fget(v_snd_1586_, v___x_1644_);
lean_dec(v_snd_1586_);
v___x_1646_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8, &lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8_once, _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__8);
lean_inc_n(v___x_1645_, 3);
v___x_1647_ = l_Lean_Expr_app___override(v___x_1646_, v___x_1645_);
v___x_1648_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12, &lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__12);
v___x_1649_ = l_Lean_Expr_app___override(v___x_1648_, v___x_1647_);
v___x_1650_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15, &lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15_once, _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__15);
v___x_1651_ = l_Lean_Expr_app___override(v___x_1650_, v___x_1645_);
v___x_1652_ = l_Lean_Expr_app___override(v___x_1649_, v___x_1651_);
v___x_1653_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18, &lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18_once, _init_lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__18);
v___x_1654_ = l_Lean_Expr_app___override(v___x_1653_, v___x_1645_);
v___x_1655_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1655_, 0, v___x_1645_);
lean_ctor_set(v___x_1655_, 1, v___x_1652_);
lean_ctor_set(v___x_1655_, 2, v___x_1654_);
v___x_1656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1656_, 0, v___x_1655_);
return v___x_1656_;
}
}
}
}
else
{
lean_dec_ref_known(v_fst_1584_, 2);
lean_dec_ref(v___x_1583_);
v___y_1568_ = v_a_1562_;
v___y_1569_ = v_a_1563_;
v___y_1570_ = v_a_1564_;
v___y_1571_ = v_a_1565_;
goto v___jp_1567_;
}
}
else
{
lean_dec(v_fst_1584_);
lean_dec_ref(v___x_1583_);
v___y_1568_ = v_a_1562_;
v___y_1569_ = v_a_1563_;
v___y_1570_ = v_a_1564_;
v___y_1571_ = v_a_1565_;
goto v___jp_1567_;
}
v___jp_1567_:
{
if (v_expensive_1560_ == 0)
{
lean_object* v___x_1572_; lean_object* v___x_1573_; 
lean_dec_ref(v_t_1561_);
lean_dec(v_u_1559_);
v___x_1572_ = lean_box(1);
v___x_1573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1572_);
return v___x_1573_;
}
else
{
uint8_t v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___f_1581_; lean_object* v___x_1582_; 
v___x_1574_ = 0;
v___x_1575_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_normPow___closed__16));
v___x_1576_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___closed__0));
v___x_1577_ = lean_box(0);
v___x_1578_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1578_, 0, v_u_1559_);
lean_ctor_set(v___x_1578_, 1, v___x_1577_);
lean_inc_ref(v___x_1578_);
v___x_1579_ = l_Lean_Expr_const___override(v___x_1576_, v___x_1578_);
lean_inc_ref(v_t_1561_);
v___x_1580_ = l_Lean_Expr_app___override(v___x_1579_, v_t_1561_);
v___f_1581_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_typeToCharP___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1581_, 0, v___x_1580_);
lean_closure_set(v___f_1581_, 1, v___x_1577_);
lean_closure_set(v___f_1581_, 2, v___x_1578_);
lean_closure_set(v___f_1581_, 3, v_t_1561_);
lean_closure_set(v___f_1581_, 4, v___x_1575_);
v___x_1582_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Tactic_ReduceModChar_normPow_spec__0___redArg(v___f_1581_, v___x_1574_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
return v___x_1582_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_typeToCharP___boxed(lean_object* v_u_1659_, lean_object* v_expensive_1660_, lean_object* v_t_1661_, lean_object* v_a_1662_, lean_object* v_a_1663_, lean_object* v_a_1664_, lean_object* v_a_1665_, lean_object* v_a_1666_){
_start:
{
uint8_t v_expensive_boxed_1667_; lean_object* v_res_1668_; 
v_expensive_boxed_1667_ = lean_unbox(v_expensive_1660_);
v_res_1668_ = lp_mathlib_Tactic_ReduceModChar_typeToCharP(v_u_1659_, v_expensive_boxed_1667_, v_t_1661_, v_a_1662_, v_a_1663_, v_a_1664_, v_a_1665_);
lean_dec(v_a_1665_);
lean_dec_ref(v_a_1664_);
lean_dec(v_a_1663_);
lean_dec_ref(v_a_1662_);
return v_res_1668_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1(void){
_start:
{
lean_object* v___x_1670_; lean_object* v___x_1671_; 
v___x_1670_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__0));
v___x_1671_ = l_Lean_stringToMessageData(v___x_1670_);
return v___x_1671_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3(void){
_start:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1673_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__2));
v___x_1674_ = l_Lean_stringToMessageData(v___x_1673_);
return v___x_1674_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5(void){
_start:
{
lean_object* v___x_1676_; lean_object* v___x_1677_; 
v___x_1676_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__4));
v___x_1677_ = l_Lean_stringToMessageData(v___x_1676_);
return v___x_1677_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7(void){
_start:
{
lean_object* v___x_1679_; lean_object* v___x_1680_; 
v___x_1679_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__6));
v___x_1680_ = l_Lean_stringToMessageData(v___x_1679_);
return v___x_1680_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9(void){
_start:
{
lean_object* v___x_1682_; lean_object* v___x_1683_; 
v___x_1682_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__8));
v___x_1683_ = l_Lean_stringToMessageData(v___x_1682_);
return v___x_1683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm(uint8_t v_expensive_1684_, lean_object* v_e_1685_, lean_object* v_a_1686_, lean_object* v_a_1687_, lean_object* v_a_1688_, lean_object* v_a_1689_){
_start:
{
lean_object* v___x_1691_; 
lean_inc(v_a_1689_);
lean_inc_ref(v_a_1688_);
lean_inc(v_a_1687_);
lean_inc_ref(v_a_1686_);
lean_inc_ref(v_e_1685_);
v___x_1691_ = lean_infer_type(v_e_1685_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1691_) == 0)
{
lean_object* v_a_1692_; lean_object* v___x_1693_; 
v_a_1692_ = lean_ctor_get(v___x_1691_, 0);
lean_inc_n(v_a_1692_, 2);
lean_dec_ref_known(v___x_1691_, 1);
v___x_1693_ = l_Lean_Meta_getLevel(v_a_1692_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1693_) == 0)
{
lean_object* v_a_1694_; 
v_a_1694_ = lean_ctor_get(v___x_1693_, 0);
lean_inc(v_a_1694_);
lean_dec_ref_known(v___x_1693_, 1);
if (lean_obj_tag(v_a_1694_) == 1)
{
lean_object* v_a_1695_; lean_object* v___x_1696_; 
v_a_1695_ = lean_ctor_get(v_a_1694_, 0);
lean_inc_n(v_a_1695_, 2);
lean_dec_ref_known(v_a_1694_, 1);
lean_inc(v_a_1692_);
v___x_1696_ = lp_mathlib_Tactic_ReduceModChar_typeToCharP(v_a_1695_, v_expensive_1684_, v_a_1692_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1696_) == 0)
{
lean_object* v_a_1697_; 
v_a_1697_ = lean_ctor_get(v___x_1696_, 0);
lean_inc(v_a_1697_);
lean_dec_ref_known(v___x_1696_, 1);
if (lean_obj_tag(v_a_1697_) == 0)
{
lean_object* v_n_1698_; lean_object* v_instRing_1699_; lean_object* v_instCharP_1700_; lean_object* v___y_1702_; lean_object* v___y_1703_; uint8_t v___y_1704_; lean_object* v___x_1715_; 
v_n_1698_ = lean_ctor_get(v_a_1697_, 0);
lean_inc_ref(v_n_1698_);
v_instRing_1699_ = lean_ctor_get(v_a_1697_, 1);
lean_inc_ref(v_instRing_1699_);
v_instCharP_1700_ = lean_ctor_get(v_a_1697_, 2);
lean_inc_ref(v_instCharP_1700_);
lean_dec_ref_known(v_a_1697_, 3);
v___x_1715_ = l_Lean_Meta_saveState___redArg(v_a_1687_, v_a_1689_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___y_1718_; uint8_t v___y_1719_; lean_object* v___y_1744_; lean_object* v_a_1745_; lean_object* v___x_1748_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
lean_inc(v_a_1716_);
lean_dec_ref_known(v___x_1715_, 1);
lean_inc_ref(v_instCharP_1700_);
lean_inc_ref(v_instRing_1699_);
lean_inc_ref(v_e_1685_);
lean_inc_ref(v_n_1698_);
lean_inc(v_a_1692_);
lean_inc(v_a_1695_);
v___x_1748_ = lp_mathlib_Tactic_ReduceModChar_normIntNumeral(v_a_1695_, v_a_1692_, v_n_1698_, v_e_1685_, v_instRing_1699_, v_instCharP_1700_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1748_) == 0)
{
lean_object* v_a_1749_; lean_object* v___x_1750_; 
v_a_1749_ = lean_ctor_get(v___x_1748_, 0);
lean_inc(v_a_1749_);
lean_dec_ref_known(v___x_1748_, 1);
lean_inc_ref(v_e_1685_);
lean_inc(v_a_1692_);
lean_inc(v_a_1695_);
v___x_1750_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(v_a_1695_, v_a_1692_, v_e_1685_, v_a_1749_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1750_) == 0)
{
lean_dec(v_a_1716_);
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
return v___x_1750_;
}
else
{
lean_object* v_a_1751_; 
v_a_1751_ = lean_ctor_get(v___x_1750_, 0);
lean_inc(v_a_1751_);
v___y_1744_ = v___x_1750_;
v_a_1745_ = v_a_1751_;
goto v___jp_1743_;
}
}
else
{
lean_object* v_a_1752_; lean_object* v___x_1754_; uint8_t v_isShared_1755_; uint8_t v_isSharedCheck_1759_; 
v_a_1752_ = lean_ctor_get(v___x_1748_, 0);
v_isSharedCheck_1759_ = !lean_is_exclusive(v___x_1748_);
if (v_isSharedCheck_1759_ == 0)
{
v___x_1754_ = v___x_1748_;
v_isShared_1755_ = v_isSharedCheck_1759_;
goto v_resetjp_1753_;
}
else
{
lean_inc(v_a_1752_);
lean_dec(v___x_1748_);
v___x_1754_ = lean_box(0);
v_isShared_1755_ = v_isSharedCheck_1759_;
goto v_resetjp_1753_;
}
v_resetjp_1753_:
{
lean_object* v___x_1757_; 
lean_inc(v_a_1752_);
if (v_isShared_1755_ == 0)
{
v___x_1757_ = v___x_1754_;
goto v_reusejp_1756_;
}
else
{
lean_object* v_reuseFailAlloc_1758_; 
v_reuseFailAlloc_1758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1758_, 0, v_a_1752_);
v___x_1757_ = v_reuseFailAlloc_1758_;
goto v_reusejp_1756_;
}
v_reusejp_1756_:
{
v___y_1744_ = v___x_1757_;
v_a_1745_ = v_a_1752_;
goto v___jp_1743_;
}
}
}
v___jp_1717_:
{
if (v___y_1719_ == 0)
{
lean_object* v___x_1720_; 
lean_dec_ref(v___y_1718_);
v___x_1720_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1716_, v_a_1687_, v_a_1689_);
lean_dec(v_a_1716_);
if (lean_obj_tag(v___x_1720_) == 0)
{
lean_object* v___x_1721_; 
lean_dec_ref_known(v___x_1720_, 1);
v___x_1721_ = l_Lean_Meta_saveState___redArg(v_a_1687_, v_a_1689_);
if (lean_obj_tag(v___x_1721_) == 0)
{
lean_object* v_a_1722_; lean_object* v___x_1723_; 
v_a_1722_ = lean_ctor_get(v___x_1721_, 0);
lean_inc(v_a_1722_);
lean_dec_ref_known(v___x_1721_, 1);
lean_inc_ref(v_instCharP_1700_);
lean_inc_ref(v_instRing_1699_);
lean_inc_ref(v_e_1685_);
lean_inc_ref(v_n_1698_);
lean_inc(v_a_1692_);
lean_inc(v_a_1695_);
v___x_1723_ = lp_mathlib_Tactic_ReduceModChar_normNegCoeffMul(v_a_1695_, v_a_1692_, v_n_1698_, v_e_1685_, v_instRing_1699_, v_instCharP_1700_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
if (lean_obj_tag(v___x_1723_) == 0)
{
lean_dec(v_a_1722_);
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
return v___x_1723_;
}
else
{
lean_object* v_a_1724_; uint8_t v___x_1725_; 
v_a_1724_ = lean_ctor_get(v___x_1723_, 0);
lean_inc(v_a_1724_);
v___x_1725_ = l_Lean_Exception_isInterrupt(v_a_1724_);
if (v___x_1725_ == 0)
{
uint8_t v___x_1726_; 
v___x_1726_ = l_Lean_Exception_isRuntime(v_a_1724_);
v___y_1702_ = v___x_1723_;
v___y_1703_ = v_a_1722_;
v___y_1704_ = v___x_1726_;
goto v___jp_1701_;
}
else
{
lean_dec(v_a_1724_);
v___y_1702_ = v___x_1723_;
v___y_1703_ = v_a_1722_;
v___y_1704_ = v___x_1725_;
goto v___jp_1701_;
}
}
}
else
{
lean_object* v_a_1727_; lean_object* v___x_1729_; uint8_t v_isShared_1730_; uint8_t v_isSharedCheck_1734_; 
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1727_ = lean_ctor_get(v___x_1721_, 0);
v_isSharedCheck_1734_ = !lean_is_exclusive(v___x_1721_);
if (v_isSharedCheck_1734_ == 0)
{
v___x_1729_ = v___x_1721_;
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
else
{
lean_inc(v_a_1727_);
lean_dec(v___x_1721_);
v___x_1729_ = lean_box(0);
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
v_resetjp_1728_:
{
lean_object* v___x_1732_; 
if (v_isShared_1730_ == 0)
{
v___x_1732_ = v___x_1729_;
goto v_reusejp_1731_;
}
else
{
lean_object* v_reuseFailAlloc_1733_; 
v_reuseFailAlloc_1733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1733_, 0, v_a_1727_);
v___x_1732_ = v_reuseFailAlloc_1733_;
goto v_reusejp_1731_;
}
v_reusejp_1731_:
{
return v___x_1732_;
}
}
}
}
else
{
lean_object* v_a_1735_; lean_object* v___x_1737_; uint8_t v_isShared_1738_; uint8_t v_isSharedCheck_1742_; 
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1735_ = lean_ctor_get(v___x_1720_, 0);
v_isSharedCheck_1742_ = !lean_is_exclusive(v___x_1720_);
if (v_isSharedCheck_1742_ == 0)
{
v___x_1737_ = v___x_1720_;
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
else
{
lean_inc(v_a_1735_);
lean_dec(v___x_1720_);
v___x_1737_ = lean_box(0);
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
v_resetjp_1736_:
{
lean_object* v___x_1740_; 
if (v_isShared_1738_ == 0)
{
v___x_1740_ = v___x_1737_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v_a_1735_);
v___x_1740_ = v_reuseFailAlloc_1741_;
goto v_reusejp_1739_;
}
v_reusejp_1739_:
{
return v___x_1740_;
}
}
}
}
else
{
lean_dec(v_a_1716_);
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
return v___y_1718_;
}
}
v___jp_1743_:
{
uint8_t v___x_1746_; 
v___x_1746_ = l_Lean_Exception_isInterrupt(v_a_1745_);
if (v___x_1746_ == 0)
{
uint8_t v___x_1747_; 
v___x_1747_ = l_Lean_Exception_isRuntime(v_a_1745_);
v___y_1718_ = v___y_1744_;
v___y_1719_ = v___x_1747_;
goto v___jp_1717_;
}
else
{
lean_dec_ref(v_a_1745_);
v___y_1718_ = v___y_1744_;
v___y_1719_ = v___x_1746_;
goto v___jp_1717_;
}
}
}
else
{
lean_object* v_a_1760_; lean_object* v___x_1762_; uint8_t v_isShared_1763_; uint8_t v_isSharedCheck_1767_; 
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1760_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_1767_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1767_ == 0)
{
v___x_1762_ = v___x_1715_;
v_isShared_1763_ = v_isSharedCheck_1767_;
goto v_resetjp_1761_;
}
else
{
lean_inc(v_a_1760_);
lean_dec(v___x_1715_);
v___x_1762_ = lean_box(0);
v_isShared_1763_ = v_isSharedCheck_1767_;
goto v_resetjp_1761_;
}
v_resetjp_1761_:
{
lean_object* v___x_1765_; 
if (v_isShared_1763_ == 0)
{
v___x_1765_ = v___x_1762_;
goto v_reusejp_1764_;
}
else
{
lean_object* v_reuseFailAlloc_1766_; 
v_reuseFailAlloc_1766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1766_, 0, v_a_1760_);
v___x_1765_ = v_reuseFailAlloc_1766_;
goto v_reusejp_1764_;
}
v_reusejp_1764_:
{
return v___x_1765_;
}
}
}
v___jp_1701_:
{
if (v___y_1704_ == 0)
{
lean_object* v___x_1705_; 
lean_dec_ref(v___y_1702_);
v___x_1705_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1703_, v_a_1687_, v_a_1689_);
lean_dec_ref(v___y_1703_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v___x_1706_; 
lean_dec_ref_known(v___x_1705_, 1);
v___x_1706_ = lp_mathlib_Tactic_ReduceModChar_normNeg(v_a_1695_, v_a_1692_, v_n_1698_, v_e_1685_, v_instRing_1699_, v_instCharP_1700_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
return v___x_1706_;
}
else
{
lean_object* v_a_1707_; lean_object* v___x_1709_; uint8_t v_isShared_1710_; uint8_t v_isSharedCheck_1714_; 
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1707_ = lean_ctor_get(v___x_1705_, 0);
v_isSharedCheck_1714_ = !lean_is_exclusive(v___x_1705_);
if (v_isSharedCheck_1714_ == 0)
{
v___x_1709_ = v___x_1705_;
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
else
{
lean_inc(v_a_1707_);
lean_dec(v___x_1705_);
v___x_1709_ = lean_box(0);
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
v_resetjp_1708_:
{
lean_object* v___x_1712_; 
if (v_isShared_1710_ == 0)
{
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
return v___x_1712_;
}
}
}
}
else
{
lean_dec_ref(v___y_1703_);
lean_dec_ref(v_instCharP_1700_);
lean_dec_ref(v_instRing_1699_);
lean_dec_ref(v_n_1698_);
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
return v___y_1702_;
}
}
}
else
{
lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; 
lean_dec(v_a_1695_);
lean_dec_ref(v_e_1685_);
v___x_1768_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1, &lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1_once, _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__1);
v___x_1769_ = l_Lean_MessageData_ofExpr(v_a_1692_);
v___x_1770_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1770_, 0, v___x_1768_);
lean_ctor_set(v___x_1770_, 1, v___x_1769_);
v___x_1771_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3, &lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3_once, _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__3);
v___x_1772_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1770_);
lean_ctor_set(v___x_1772_, 1, v___x_1771_);
v___x_1773_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1772_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
return v___x_1773_;
}
}
else
{
lean_object* v_a_1774_; lean_object* v___x_1776_; uint8_t v_isShared_1777_; uint8_t v_isSharedCheck_1781_; 
lean_dec(v_a_1695_);
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1774_ = lean_ctor_get(v___x_1696_, 0);
v_isSharedCheck_1781_ = !lean_is_exclusive(v___x_1696_);
if (v_isSharedCheck_1781_ == 0)
{
v___x_1776_ = v___x_1696_;
v_isShared_1777_ = v_isSharedCheck_1781_;
goto v_resetjp_1775_;
}
else
{
lean_inc(v_a_1774_);
lean_dec(v___x_1696_);
v___x_1776_ = lean_box(0);
v_isShared_1777_ = v_isSharedCheck_1781_;
goto v_resetjp_1775_;
}
v_resetjp_1775_:
{
lean_object* v___x_1779_; 
if (v_isShared_1777_ == 0)
{
v___x_1779_ = v___x_1776_;
goto v_reusejp_1778_;
}
else
{
lean_object* v_reuseFailAlloc_1780_; 
v_reuseFailAlloc_1780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1780_, 0, v_a_1774_);
v___x_1779_ = v_reuseFailAlloc_1780_;
goto v_reusejp_1778_;
}
v_reusejp_1778_:
{
return v___x_1779_;
}
}
}
}
else
{
lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; 
lean_dec_ref(v_e_1685_);
v___x_1782_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5, &lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5_once, _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__5);
v___x_1783_ = l_Lean_MessageData_ofExpr(v_a_1692_);
v___x_1784_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1782_);
lean_ctor_set(v___x_1784_, 1, v___x_1783_);
v___x_1785_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7, &lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7_once, _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__7);
v___x_1786_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1784_);
lean_ctor_set(v___x_1786_, 1, v___x_1785_);
v___x_1787_ = l_Lean_MessageData_ofLevel(v_a_1694_);
v___x_1788_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1788_, 0, v___x_1786_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
v___x_1789_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9, &lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9_once, _init_lp_mathlib_Tactic_ReduceModChar_matchAndNorm___closed__9);
v___x_1790_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1790_, 0, v___x_1788_);
lean_ctor_set(v___x_1790_, 1, v___x_1789_);
v___x_1791_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_1790_, v_a_1686_, v_a_1687_, v_a_1688_, v_a_1689_);
return v___x_1791_;
}
}
else
{
lean_object* v_a_1792_; lean_object* v___x_1794_; uint8_t v_isShared_1795_; uint8_t v_isSharedCheck_1799_; 
lean_dec(v_a_1692_);
lean_dec_ref(v_e_1685_);
v_a_1792_ = lean_ctor_get(v___x_1693_, 0);
v_isSharedCheck_1799_ = !lean_is_exclusive(v___x_1693_);
if (v_isSharedCheck_1799_ == 0)
{
v___x_1794_ = v___x_1693_;
v_isShared_1795_ = v_isSharedCheck_1799_;
goto v_resetjp_1793_;
}
else
{
lean_inc(v_a_1792_);
lean_dec(v___x_1693_);
v___x_1794_ = lean_box(0);
v_isShared_1795_ = v_isSharedCheck_1799_;
goto v_resetjp_1793_;
}
v_resetjp_1793_:
{
lean_object* v___x_1797_; 
if (v_isShared_1795_ == 0)
{
v___x_1797_ = v___x_1794_;
goto v_reusejp_1796_;
}
else
{
lean_object* v_reuseFailAlloc_1798_; 
v_reuseFailAlloc_1798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1798_, 0, v_a_1792_);
v___x_1797_ = v_reuseFailAlloc_1798_;
goto v_reusejp_1796_;
}
v_reusejp_1796_:
{
return v___x_1797_;
}
}
}
}
else
{
lean_object* v_a_1800_; lean_object* v___x_1802_; uint8_t v_isShared_1803_; uint8_t v_isSharedCheck_1807_; 
lean_dec_ref(v_e_1685_);
v_a_1800_ = lean_ctor_get(v___x_1691_, 0);
v_isSharedCheck_1807_ = !lean_is_exclusive(v___x_1691_);
if (v_isSharedCheck_1807_ == 0)
{
v___x_1802_ = v___x_1691_;
v_isShared_1803_ = v_isSharedCheck_1807_;
goto v_resetjp_1801_;
}
else
{
lean_inc(v_a_1800_);
lean_dec(v___x_1691_);
v___x_1802_ = lean_box(0);
v_isShared_1803_ = v_isSharedCheck_1807_;
goto v_resetjp_1801_;
}
v_resetjp_1801_:
{
lean_object* v___x_1805_; 
if (v_isShared_1803_ == 0)
{
v___x_1805_ = v___x_1802_;
goto v_reusejp_1804_;
}
else
{
lean_object* v_reuseFailAlloc_1806_; 
v_reuseFailAlloc_1806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1806_, 0, v_a_1800_);
v___x_1805_ = v_reuseFailAlloc_1806_;
goto v_reusejp_1804_;
}
v_reusejp_1804_:
{
return v___x_1805_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_matchAndNorm___boxed(lean_object* v_expensive_1808_, lean_object* v_e_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_, lean_object* v_a_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_){
_start:
{
uint8_t v_expensive_boxed_1815_; lean_object* v_res_1816_; 
v_expensive_boxed_1815_ = lean_unbox(v_expensive_1808_);
v_res_1816_ = lp_mathlib_Tactic_ReduceModChar_matchAndNorm(v_expensive_boxed_1815_, v_e_1809_, v_a_1810_, v_a_1811_, v_a_1812_, v_a_1813_);
lean_dec(v_a_1813_);
lean_dec_ref(v_a_1812_);
lean_dec(v_a_1811_);
lean_dec_ref(v_a_1810_);
return v_res_1816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(lean_object* v_e_1817_, lean_object* v___y_1818_){
_start:
{
uint8_t v___x_1820_; 
v___x_1820_ = l_Lean_Expr_hasMVar(v_e_1817_);
if (v___x_1820_ == 0)
{
lean_object* v___x_1821_; 
v___x_1821_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1821_, 0, v_e_1817_);
return v___x_1821_;
}
else
{
lean_object* v___x_1822_; lean_object* v_mctx_1823_; lean_object* v___x_1824_; lean_object* v_fst_1825_; lean_object* v_snd_1826_; lean_object* v___x_1827_; lean_object* v_cache_1828_; lean_object* v_zetaDeltaFVarIds_1829_; lean_object* v_postponed_1830_; lean_object* v_diag_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1840_; 
v___x_1822_ = lean_st_ref_get(v___y_1818_);
v_mctx_1823_ = lean_ctor_get(v___x_1822_, 0);
lean_inc_ref(v_mctx_1823_);
lean_dec(v___x_1822_);
v___x_1824_ = l_Lean_instantiateMVarsCore(v_mctx_1823_, v_e_1817_);
v_fst_1825_ = lean_ctor_get(v___x_1824_, 0);
lean_inc(v_fst_1825_);
v_snd_1826_ = lean_ctor_get(v___x_1824_, 1);
lean_inc(v_snd_1826_);
lean_dec_ref(v___x_1824_);
v___x_1827_ = lean_st_ref_take(v___y_1818_);
v_cache_1828_ = lean_ctor_get(v___x_1827_, 1);
v_zetaDeltaFVarIds_1829_ = lean_ctor_get(v___x_1827_, 2);
v_postponed_1830_ = lean_ctor_get(v___x_1827_, 3);
v_diag_1831_ = lean_ctor_get(v___x_1827_, 4);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1840_ == 0)
{
lean_object* v_unused_1841_; 
v_unused_1841_ = lean_ctor_get(v___x_1827_, 0);
lean_dec(v_unused_1841_);
v___x_1833_ = v___x_1827_;
v_isShared_1834_ = v_isSharedCheck_1840_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_diag_1831_);
lean_inc(v_postponed_1830_);
lean_inc(v_zetaDeltaFVarIds_1829_);
lean_inc(v_cache_1828_);
lean_dec(v___x_1827_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1840_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
lean_object* v___x_1836_; 
if (v_isShared_1834_ == 0)
{
lean_ctor_set(v___x_1833_, 0, v_snd_1826_);
v___x_1836_ = v___x_1833_;
goto v_reusejp_1835_;
}
else
{
lean_object* v_reuseFailAlloc_1839_; 
v_reuseFailAlloc_1839_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1839_, 0, v_snd_1826_);
lean_ctor_set(v_reuseFailAlloc_1839_, 1, v_cache_1828_);
lean_ctor_set(v_reuseFailAlloc_1839_, 2, v_zetaDeltaFVarIds_1829_);
lean_ctor_set(v_reuseFailAlloc_1839_, 3, v_postponed_1830_);
lean_ctor_set(v_reuseFailAlloc_1839_, 4, v_diag_1831_);
v___x_1836_ = v_reuseFailAlloc_1839_;
goto v_reusejp_1835_;
}
v_reusejp_1835_:
{
lean_object* v___x_1837_; lean_object* v___x_1838_; 
v___x_1837_ = lean_st_ref_set(v___y_1818_, v___x_1836_);
v___x_1838_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1838_, 0, v_fst_1825_);
return v___x_1838_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg___boxed(lean_object* v_e_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_){
_start:
{
lean_object* v_res_1845_; 
v_res_1845_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_1842_, v___y_1843_);
lean_dec(v___y_1843_);
return v_res_1845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0(lean_object* v_e_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_){
_start:
{
lean_object* v___x_1852_; 
v___x_1852_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_1846_, v___y_1848_);
return v___x_1852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___boxed(lean_object* v_e_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
lean_object* v_res_1859_; 
v_res_1859_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0(v_e_1853_, v___y_1854_, v___y_1855_, v___y_1856_, v___y_1857_);
lean_dec(v___y_1857_);
lean_dec_ref(v___y_1856_);
lean_dec(v___y_1855_);
lean_dec_ref(v___y_1854_);
return v_res_1859_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; 
v___x_1860_ = lean_unsigned_to_nat(32u);
v___x_1861_ = lean_mk_empty_array_with_capacity(v___x_1860_);
v___x_1862_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1862_, 0, v___x_1861_);
return v___x_1862_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1(void){
_start:
{
size_t v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; 
v___x_1863_ = ((size_t)5ULL);
v___x_1864_ = lean_unsigned_to_nat(0u);
v___x_1865_ = lean_unsigned_to_nat(32u);
v___x_1866_ = lean_mk_empty_array_with_capacity(v___x_1865_);
v___x_1867_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__0);
v___x_1868_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1868_, 0, v___x_1867_);
lean_ctor_set(v___x_1868_, 1, v___x_1866_);
lean_ctor_set(v___x_1868_, 2, v___x_1864_);
lean_ctor_set(v___x_1868_, 3, v___x_1864_);
lean_ctor_set_usize(v___x_1868_, 4, v___x_1863_);
return v___x_1868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg(lean_object* v___y_1869_){
_start:
{
lean_object* v___x_1871_; lean_object* v_traceState_1872_; lean_object* v_traces_1873_; lean_object* v___x_1874_; lean_object* v_traceState_1875_; lean_object* v_env_1876_; lean_object* v_nextMacroScope_1877_; lean_object* v_ngen_1878_; lean_object* v_auxDeclNGen_1879_; lean_object* v_cache_1880_; lean_object* v_messages_1881_; lean_object* v_infoState_1882_; lean_object* v_snapshotTasks_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1902_; 
v___x_1871_ = lean_st_ref_get(v___y_1869_);
v_traceState_1872_ = lean_ctor_get(v___x_1871_, 4);
lean_inc_ref(v_traceState_1872_);
lean_dec(v___x_1871_);
v_traces_1873_ = lean_ctor_get(v_traceState_1872_, 0);
lean_inc_ref(v_traces_1873_);
lean_dec_ref(v_traceState_1872_);
v___x_1874_ = lean_st_ref_take(v___y_1869_);
v_traceState_1875_ = lean_ctor_get(v___x_1874_, 4);
v_env_1876_ = lean_ctor_get(v___x_1874_, 0);
v_nextMacroScope_1877_ = lean_ctor_get(v___x_1874_, 1);
v_ngen_1878_ = lean_ctor_get(v___x_1874_, 2);
v_auxDeclNGen_1879_ = lean_ctor_get(v___x_1874_, 3);
v_cache_1880_ = lean_ctor_get(v___x_1874_, 5);
v_messages_1881_ = lean_ctor_get(v___x_1874_, 6);
v_infoState_1882_ = lean_ctor_get(v___x_1874_, 7);
v_snapshotTasks_1883_ = lean_ctor_get(v___x_1874_, 8);
v_isSharedCheck_1902_ = !lean_is_exclusive(v___x_1874_);
if (v_isSharedCheck_1902_ == 0)
{
v___x_1885_ = v___x_1874_;
v_isShared_1886_ = v_isSharedCheck_1902_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_snapshotTasks_1883_);
lean_inc(v_infoState_1882_);
lean_inc(v_messages_1881_);
lean_inc(v_cache_1880_);
lean_inc(v_traceState_1875_);
lean_inc(v_auxDeclNGen_1879_);
lean_inc(v_ngen_1878_);
lean_inc(v_nextMacroScope_1877_);
lean_inc(v_env_1876_);
lean_dec(v___x_1874_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1902_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
uint64_t v_tid_1887_; lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1900_; 
v_tid_1887_ = lean_ctor_get_uint64(v_traceState_1875_, sizeof(void*)*1);
v_isSharedCheck_1900_ = !lean_is_exclusive(v_traceState_1875_);
if (v_isSharedCheck_1900_ == 0)
{
lean_object* v_unused_1901_; 
v_unused_1901_ = lean_ctor_get(v_traceState_1875_, 0);
lean_dec(v_unused_1901_);
v___x_1889_ = v_traceState_1875_;
v_isShared_1890_ = v_isSharedCheck_1900_;
goto v_resetjp_1888_;
}
else
{
lean_dec(v_traceState_1875_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1900_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1891_; lean_object* v___x_1893_; 
v___x_1891_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___closed__1);
if (v_isShared_1890_ == 0)
{
lean_ctor_set(v___x_1889_, 0, v___x_1891_);
v___x_1893_ = v___x_1889_;
goto v_reusejp_1892_;
}
else
{
lean_object* v_reuseFailAlloc_1899_; 
v_reuseFailAlloc_1899_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1899_, 0, v___x_1891_);
lean_ctor_set_uint64(v_reuseFailAlloc_1899_, sizeof(void*)*1, v_tid_1887_);
v___x_1893_ = v_reuseFailAlloc_1899_;
goto v_reusejp_1892_;
}
v_reusejp_1892_:
{
lean_object* v___x_1895_; 
if (v_isShared_1886_ == 0)
{
lean_ctor_set(v___x_1885_, 4, v___x_1893_);
v___x_1895_ = v___x_1885_;
goto v_reusejp_1894_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v_env_1876_);
lean_ctor_set(v_reuseFailAlloc_1898_, 1, v_nextMacroScope_1877_);
lean_ctor_set(v_reuseFailAlloc_1898_, 2, v_ngen_1878_);
lean_ctor_set(v_reuseFailAlloc_1898_, 3, v_auxDeclNGen_1879_);
lean_ctor_set(v_reuseFailAlloc_1898_, 4, v___x_1893_);
lean_ctor_set(v_reuseFailAlloc_1898_, 5, v_cache_1880_);
lean_ctor_set(v_reuseFailAlloc_1898_, 6, v_messages_1881_);
lean_ctor_set(v_reuseFailAlloc_1898_, 7, v_infoState_1882_);
lean_ctor_set(v_reuseFailAlloc_1898_, 8, v_snapshotTasks_1883_);
v___x_1895_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1894_;
}
v_reusejp_1894_:
{
lean_object* v___x_1896_; lean_object* v___x_1897_; 
v___x_1896_ = lean_st_ref_set(v___y_1869_, v___x_1895_);
v___x_1897_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1897_, 0, v_traces_1873_);
return v___x_1897_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg___boxed(lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg(v___y_1903_);
lean_dec(v___y_1903_);
return v_res_1905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1(lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
lean_object* v___x_1911_; 
v___x_1911_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg(v___y_1909_);
return v___x_1911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___boxed(lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
lean_object* v_res_1917_; 
v_res_1917_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1(v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
lean_dec(v___y_1913_);
lean_dec_ref(v___y_1912_);
return v_res_1917_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(lean_object* v_opts_1918_, lean_object* v_opt_1919_){
_start:
{
lean_object* v_name_1920_; lean_object* v_defValue_1921_; lean_object* v_map_1922_; lean_object* v___x_1923_; 
v_name_1920_ = lean_ctor_get(v_opt_1919_, 0);
v_defValue_1921_ = lean_ctor_get(v_opt_1919_, 1);
v_map_1922_ = lean_ctor_get(v_opts_1918_, 0);
v___x_1923_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1922_, v_name_1920_);
if (lean_obj_tag(v___x_1923_) == 0)
{
uint8_t v___x_1924_; 
v___x_1924_ = lean_unbox(v_defValue_1921_);
return v___x_1924_;
}
else
{
lean_object* v_val_1925_; 
v_val_1925_ = lean_ctor_get(v___x_1923_, 0);
lean_inc(v_val_1925_);
lean_dec_ref_known(v___x_1923_, 1);
if (lean_obj_tag(v_val_1925_) == 1)
{
uint8_t v_v_1926_; 
v_v_1926_ = lean_ctor_get_uint8(v_val_1925_, 0);
lean_dec_ref_known(v_val_1925_, 0);
return v_v_1926_;
}
else
{
uint8_t v___x_1927_; 
lean_dec(v_val_1925_);
v___x_1927_ = lean_unbox(v_defValue_1921_);
return v___x_1927_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2___boxed(lean_object* v_opts_1928_, lean_object* v_opt_1929_){
_start:
{
uint8_t v_res_1930_; lean_object* v_r_1931_; 
v_res_1930_ = lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(v_opts_1928_, v_opt_1929_);
lean_dec_ref(v_opt_1929_);
lean_dec_ref(v_opts_1928_);
v_r_1931_ = lean_box(v_res_1930_);
return v_r_1931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__0(uint8_t v_expensive_1934_, lean_object* v_x_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_){
_start:
{
lean_object* v___x_1945_; 
v___x_1945_ = lp_mathlib_Tactic_ReduceModChar_matchAndNorm(v_expensive_1934_, v___y_1936_, v___y_1940_, v___y_1941_, v___y_1942_, v___y_1943_);
if (lean_obj_tag(v___x_1945_) == 0)
{
lean_object* v_a_1946_; lean_object* v___x_1948_; uint8_t v_isShared_1949_; uint8_t v_isSharedCheck_1954_; 
v_a_1946_ = lean_ctor_get(v___x_1945_, 0);
v_isSharedCheck_1954_ = !lean_is_exclusive(v___x_1945_);
if (v_isSharedCheck_1954_ == 0)
{
v___x_1948_ = v___x_1945_;
v_isShared_1949_ = v_isSharedCheck_1954_;
goto v_resetjp_1947_;
}
else
{
lean_inc(v_a_1946_);
lean_dec(v___x_1945_);
v___x_1948_ = lean_box(0);
v_isShared_1949_ = v_isSharedCheck_1954_;
goto v_resetjp_1947_;
}
v_resetjp_1947_:
{
lean_object* v___x_1950_; lean_object* v___x_1952_; 
v___x_1950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1950_, 0, v_a_1946_);
if (v_isShared_1949_ == 0)
{
lean_ctor_set(v___x_1948_, 0, v___x_1950_);
v___x_1952_ = v___x_1948_;
goto v_reusejp_1951_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v___x_1950_);
v___x_1952_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1951_;
}
v_reusejp_1951_:
{
return v___x_1952_;
}
}
}
else
{
lean_object* v_a_1955_; lean_object* v___x_1957_; uint8_t v_isShared_1958_; uint8_t v_isSharedCheck_1970_; 
v_a_1955_ = lean_ctor_get(v___x_1945_, 0);
v_isSharedCheck_1970_ = !lean_is_exclusive(v___x_1945_);
if (v_isSharedCheck_1970_ == 0)
{
v___x_1957_ = v___x_1945_;
v_isShared_1958_ = v_isSharedCheck_1970_;
goto v_resetjp_1956_;
}
else
{
lean_inc(v_a_1955_);
lean_dec(v___x_1945_);
v___x_1957_ = lean_box(0);
v_isShared_1958_ = v_isSharedCheck_1970_;
goto v_resetjp_1956_;
}
v_resetjp_1956_:
{
uint8_t v___y_1960_; uint8_t v___x_1968_; 
v___x_1968_ = l_Lean_Exception_isInterrupt(v_a_1955_);
if (v___x_1968_ == 0)
{
uint8_t v___x_1969_; 
lean_inc(v_a_1955_);
v___x_1969_ = l_Lean_Exception_isRuntime(v_a_1955_);
v___y_1960_ = v___x_1969_;
goto v___jp_1959_;
}
else
{
v___y_1960_ = v___x_1968_;
goto v___jp_1959_;
}
v___jp_1959_:
{
if (v___y_1960_ == 0)
{
lean_object* v___x_1961_; lean_object* v___x_1963_; 
lean_dec(v_a_1955_);
v___x_1961_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__0___closed__0));
if (v_isShared_1958_ == 0)
{
lean_ctor_set_tag(v___x_1957_, 0);
lean_ctor_set(v___x_1957_, 0, v___x_1961_);
v___x_1963_ = v___x_1957_;
goto v_reusejp_1962_;
}
else
{
lean_object* v_reuseFailAlloc_1964_; 
v_reuseFailAlloc_1964_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1964_, 0, v___x_1961_);
v___x_1963_ = v_reuseFailAlloc_1964_;
goto v_reusejp_1962_;
}
v_reusejp_1962_:
{
return v___x_1963_;
}
}
else
{
lean_object* v___x_1966_; 
if (v_isShared_1958_ == 0)
{
v___x_1966_ = v___x_1957_;
goto v_reusejp_1965_;
}
else
{
lean_object* v_reuseFailAlloc_1967_; 
v_reuseFailAlloc_1967_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1967_, 0, v_a_1955_);
v___x_1966_ = v_reuseFailAlloc_1967_;
goto v_reusejp_1965_;
}
v_reusejp_1965_:
{
return v___x_1966_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__0___boxed(lean_object* v_expensive_1971_, lean_object* v_x_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
uint8_t v_expensive_boxed_1982_; lean_object* v_res_1983_; 
v_expensive_boxed_1982_ = lean_unbox(v_expensive_1971_);
v_res_1983_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__0(v_expensive_boxed_1982_, v_x_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec(v___y_1978_);
lean_dec_ref(v___y_1977_);
lean_dec(v___y_1976_);
lean_dec_ref(v___y_1975_);
lean_dec(v___y_1974_);
return v_res_1983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__1(lean_object* v_e_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_){
_start:
{
lean_object* v___x_1993_; lean_object* v___x_1994_; 
v___x_1993_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1993_, 0, v_e_1984_);
v___x_1994_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1994_, 0, v___x_1993_);
return v___x_1994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__1___boxed(lean_object* v_e_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_){
_start:
{
lean_object* v_res_2004_; 
v_res_2004_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__1(v_e_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_);
lean_dec(v___y_2002_);
lean_dec_ref(v___y_2001_);
lean_dec(v___y_2000_);
lean_dec_ref(v___y_1999_);
lean_dec(v___y_1998_);
lean_dec_ref(v___y_1997_);
lean_dec(v___y_1996_);
return v_res_2004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__2(lean_object* v___x_2005_, lean_object* v___f_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_){
_start:
{
lean_object* v___x_2016_; 
lean_inc_ref(v___y_2007_);
v___x_2016_ = l_Lean_Meta_Simp_preDefault(v___x_2005_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_, v___y_2013_, v___y_2014_);
if (lean_obj_tag(v___x_2016_) == 0)
{
lean_object* v_a_2017_; 
v_a_2017_ = lean_ctor_get(v___x_2016_, 0);
lean_inc(v_a_2017_);
if (lean_obj_tag(v_a_2017_) == 2)
{
lean_object* v_e_x3f_2018_; lean_object* v___x_2019_; 
lean_dec_ref_known(v___x_2016_, 1);
v_e_x3f_2018_ = lean_ctor_get(v_a_2017_, 0);
lean_inc(v_e_x3f_2018_);
lean_dec_ref_known(v_a_2017_, 1);
v___x_2019_ = lean_box(0);
if (lean_obj_tag(v_e_x3f_2018_) == 0)
{
lean_object* v___x_2020_; 
v___x_2020_ = lean_apply_10(v___f_2006_, v___x_2019_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_, v___y_2013_, v___y_2014_, lean_box(0));
return v___x_2020_;
}
else
{
lean_object* v_val_2021_; lean_object* v_expr_2022_; lean_object* v___x_2023_; 
lean_dec_ref(v___y_2007_);
v_val_2021_ = lean_ctor_get(v_e_x3f_2018_, 0);
lean_inc(v_val_2021_);
lean_dec_ref_known(v_e_x3f_2018_, 1);
v_expr_2022_ = lean_ctor_get(v_val_2021_, 0);
lean_inc(v___y_2014_);
lean_inc_ref(v___y_2013_);
lean_inc(v___y_2012_);
lean_inc_ref(v___y_2011_);
lean_inc_ref(v_expr_2022_);
v___x_2023_ = lean_apply_10(v___f_2006_, v___x_2019_, v_expr_2022_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_, v___y_2013_, v___y_2014_, lean_box(0));
if (lean_obj_tag(v___x_2023_) == 0)
{
lean_object* v_a_2024_; lean_object* v___x_2025_; 
v_a_2024_ = lean_ctor_get(v___x_2023_, 0);
lean_inc(v_a_2024_);
lean_dec_ref_known(v___x_2023_, 1);
v___x_2025_ = l_Lean_Meta_Simp_mkEqTransResultStep(v_val_2021_, v_a_2024_, v___y_2011_, v___y_2012_, v___y_2013_, v___y_2014_);
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec(v___y_2012_);
lean_dec_ref(v___y_2011_);
return v___x_2025_;
}
else
{
lean_dec(v_val_2021_);
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec(v___y_2012_);
lean_dec_ref(v___y_2011_);
return v___x_2023_;
}
}
}
else
{
lean_dec(v_a_2017_);
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec(v___y_2012_);
lean_dec_ref(v___y_2011_);
lean_dec(v___y_2010_);
lean_dec_ref(v___y_2009_);
lean_dec(v___y_2008_);
lean_dec_ref(v___y_2007_);
lean_dec_ref(v___f_2006_);
return v___x_2016_;
}
}
else
{
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec(v___y_2012_);
lean_dec_ref(v___y_2011_);
lean_dec(v___y_2010_);
lean_dec_ref(v___y_2009_);
lean_dec(v___y_2008_);
lean_dec_ref(v___y_2007_);
lean_dec_ref(v___f_2006_);
return v___x_2016_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed(lean_object* v___x_2026_, lean_object* v___f_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_){
_start:
{
lean_object* v_res_2037_; 
v_res_2037_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__2(v___x_2026_, v___f_2027_, v___y_2028_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_, v___y_2034_, v___y_2035_);
return v_res_2037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__3(lean_object* v___x_2038_, lean_object* v_x_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_){
_start:
{
lean_object* v___x_2048_; lean_object* v___x_2049_; 
v___x_2048_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_2048_, 0, v___x_2038_);
v___x_2049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2049_, 0, v___x_2048_);
return v___x_2049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__3___boxed(lean_object* v___x_2050_, lean_object* v_x_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_){
_start:
{
lean_object* v_res_2060_; 
v_res_2060_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__3(v___x_2050_, v_x_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_);
lean_dec(v___y_2058_);
lean_dec_ref(v___y_2057_);
lean_dec(v___y_2056_);
lean_dec_ref(v___y_2055_);
lean_dec(v___y_2054_);
lean_dec_ref(v___y_2053_);
lean_dec(v___y_2052_);
lean_dec_ref(v_x_2051_);
return v_res_2060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__8(lean_object* v_e_2061_, lean_object* v_x_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
lean_object* v___x_2068_; lean_object* v___x_2069_; 
v___x_2068_ = l_Lean_MessageData_ofExpr(v_e_2061_);
v___x_2069_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2069_, 0, v___x_2068_);
return v___x_2069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__8___boxed(lean_object* v_e_2070_, lean_object* v_x_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v_res_2077_; 
v_res_2077_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__8(v_e_2070_, v_x_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
lean_dec(v___y_2075_);
lean_dec_ref(v___y_2074_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
lean_dec_ref(v_x_2071_);
return v_res_2077_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3(void){
_start:
{
lean_object* v___x_2084_; 
v___x_2084_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2084_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4(void){
_start:
{
lean_object* v___x_2085_; lean_object* v___x_2086_; 
v___x_2085_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__3);
v___x_2086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2086_, 0, v___x_2085_);
return v___x_2086_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5(void){
_start:
{
lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; 
v___x_2087_ = lean_unsigned_to_nat(0u);
v___x_2088_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4);
v___x_2089_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2089_, 0, v___x_2088_);
lean_ctor_set(v___x_2089_, 1, v___x_2087_);
return v___x_2089_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6(void){
_start:
{
lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; 
v___x_2090_ = lean_unsigned_to_nat(32u);
v___x_2091_ = lean_mk_empty_array_with_capacity(v___x_2090_);
v___x_2092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2092_, 0, v___x_2091_);
return v___x_2092_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7(void){
_start:
{
size_t v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; 
v___x_2093_ = ((size_t)5ULL);
v___x_2094_ = lean_unsigned_to_nat(0u);
v___x_2095_ = lean_unsigned_to_nat(32u);
v___x_2096_ = lean_mk_empty_array_with_capacity(v___x_2095_);
v___x_2097_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__6);
v___x_2098_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2098_, 0, v___x_2097_);
lean_ctor_set(v___x_2098_, 1, v___x_2096_);
lean_ctor_set(v___x_2098_, 2, v___x_2094_);
lean_ctor_set(v___x_2098_, 3, v___x_2094_);
lean_ctor_set_usize(v___x_2098_, 4, v___x_2093_);
return v___x_2098_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8(void){
_start:
{
lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; 
v___x_2099_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__7);
v___x_2100_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__4);
v___x_2101_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2101_, 0, v___x_2100_);
lean_ctor_set(v___x_2101_, 1, v___x_2100_);
lean_ctor_set(v___x_2101_, 2, v___x_2100_);
lean_ctor_set(v___x_2101_, 3, v___x_2099_);
return v___x_2101_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9(void){
_start:
{
lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; 
v___x_2102_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__8);
v___x_2103_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__5);
v___x_2104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2104_, 0, v___x_2103_);
lean_ctor_set(v___x_2104_, 1, v___x_2102_);
return v___x_2104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6(lean_object* v___x_2105_, lean_object* v_a_2106_, lean_object* v___f_2107_, uint8_t v_hasTrace_2108_, lean_object* v_a_2109_, lean_object* v___f_2110_, lean_object* v_ext_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_){
_start:
{
lean_object* v___x_2117_; 
v___x_2117_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_ext_2111_, v___y_2115_);
if (lean_obj_tag(v___x_2117_) == 0)
{
lean_object* v_a_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; 
v_a_2118_ = lean_ctor_get(v___x_2117_, 0);
lean_inc(v_a_2118_);
lean_dec_ref_known(v___x_2117_, 1);
v___x_2119_ = lean_unsigned_to_nat(1u);
v___x_2120_ = lean_mk_empty_array_with_capacity(v___x_2119_);
v___x_2121_ = lean_array_push(v___x_2120_, v_a_2118_);
v___x_2122_ = l_Lean_Options_empty;
v___x_2123_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_2105_, v___x_2121_, v_a_2106_, v___x_2122_, v___y_2112_, v___y_2114_, v___y_2115_);
if (lean_obj_tag(v___x_2123_) == 0)
{
lean_object* v_a_2124_; lean_object* v___x_2125_; lean_object* v___f_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___f_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; 
v_a_2124_ = lean_ctor_get(v___x_2123_, 0);
lean_inc(v_a_2124_);
lean_dec_ref_known(v___x_2123_, 1);
v___x_2125_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0));
v___f_2126_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed), 11, 2);
lean_closure_set(v___f_2126_, 0, v___x_2125_);
lean_closure_set(v___f_2126_, 1, v___f_2107_);
v___x_2127_ = lean_box(v_hasTrace_2108_);
v___x_2128_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed), 11, 2);
lean_closure_set(v___x_2128_, 0, v___x_2125_);
lean_closure_set(v___x_2128_, 1, v___x_2127_);
v___x_2129_ = lean_box(0);
v___f_2130_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__1));
lean_inc_ref(v_a_2109_);
v___x_2131_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2131_, 0, v_a_2109_);
lean_ctor_set(v___x_2131_, 1, v___x_2129_);
lean_ctor_set_uint8(v___x_2131_, sizeof(void*)*2, v_hasTrace_2108_);
v___x_2132_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2));
v___x_2133_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9);
v___x_2134_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_2134_, 0, v___f_2126_);
lean_ctor_set(v___x_2134_, 1, v___x_2132_);
lean_ctor_set(v___x_2134_, 2, v___f_2130_);
lean_ctor_set(v___x_2134_, 3, v___f_2110_);
lean_ctor_set(v___x_2134_, 4, v___x_2128_);
lean_ctor_set_uint8(v___x_2134_, sizeof(void*)*5, v_hasTrace_2108_);
v___x_2135_ = l_Lean_Meta_Simp_main(v_a_2109_, v_a_2124_, v___x_2133_, v___x_2134_, v___y_2112_, v___y_2113_, v___y_2114_, v___y_2115_);
if (lean_obj_tag(v___x_2135_) == 0)
{
lean_object* v_a_2136_; lean_object* v_fst_2137_; lean_object* v___x_2138_; 
v_a_2136_ = lean_ctor_get(v___x_2135_, 0);
lean_inc(v_a_2136_);
lean_dec_ref_known(v___x_2135_, 1);
v_fst_2137_ = lean_ctor_get(v_a_2136_, 0);
lean_inc(v_fst_2137_);
lean_dec(v_a_2136_);
v___x_2138_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_2131_, v_fst_2137_, v___y_2112_, v___y_2113_, v___y_2114_, v___y_2115_);
return v___x_2138_;
}
else
{
lean_object* v_a_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2146_; 
lean_dec_ref_known(v___x_2131_, 2);
v_a_2139_ = lean_ctor_get(v___x_2135_, 0);
v_isSharedCheck_2146_ = !lean_is_exclusive(v___x_2135_);
if (v_isSharedCheck_2146_ == 0)
{
v___x_2141_ = v___x_2135_;
v_isShared_2142_ = v_isSharedCheck_2146_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_a_2139_);
lean_dec(v___x_2135_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2146_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v___x_2144_; 
if (v_isShared_2142_ == 0)
{
v___x_2144_ = v___x_2141_;
goto v_reusejp_2143_;
}
else
{
lean_object* v_reuseFailAlloc_2145_; 
v_reuseFailAlloc_2145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2145_, 0, v_a_2139_);
v___x_2144_ = v_reuseFailAlloc_2145_;
goto v_reusejp_2143_;
}
v_reusejp_2143_:
{
return v___x_2144_;
}
}
}
}
else
{
lean_object* v_a_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2154_; 
lean_dec_ref(v___f_2110_);
lean_dec_ref(v_a_2109_);
lean_dec_ref(v___f_2107_);
v_a_2147_ = lean_ctor_get(v___x_2123_, 0);
v_isSharedCheck_2154_ = !lean_is_exclusive(v___x_2123_);
if (v_isSharedCheck_2154_ == 0)
{
v___x_2149_ = v___x_2123_;
v_isShared_2150_ = v_isSharedCheck_2154_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_a_2147_);
lean_dec(v___x_2123_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2154_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
lean_object* v___x_2152_; 
if (v_isShared_2150_ == 0)
{
v___x_2152_ = v___x_2149_;
goto v_reusejp_2151_;
}
else
{
lean_object* v_reuseFailAlloc_2153_; 
v_reuseFailAlloc_2153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2153_, 0, v_a_2147_);
v___x_2152_ = v_reuseFailAlloc_2153_;
goto v_reusejp_2151_;
}
v_reusejp_2151_:
{
return v___x_2152_;
}
}
}
}
else
{
lean_object* v_a_2155_; lean_object* v___x_2157_; uint8_t v_isShared_2158_; uint8_t v_isSharedCheck_2162_; 
lean_dec_ref(v___f_2110_);
lean_dec_ref(v_a_2109_);
lean_dec_ref(v___f_2107_);
lean_dec_ref(v_a_2106_);
lean_dec_ref(v___x_2105_);
v_a_2155_ = lean_ctor_get(v___x_2117_, 0);
v_isSharedCheck_2162_ = !lean_is_exclusive(v___x_2117_);
if (v_isSharedCheck_2162_ == 0)
{
v___x_2157_ = v___x_2117_;
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
else
{
lean_inc(v_a_2155_);
lean_dec(v___x_2117_);
v___x_2157_ = lean_box(0);
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
v_resetjp_2156_:
{
lean_object* v___x_2160_; 
if (v_isShared_2158_ == 0)
{
v___x_2160_ = v___x_2157_;
goto v_reusejp_2159_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v_a_2155_);
v___x_2160_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2159_;
}
v_reusejp_2159_:
{
return v___x_2160_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__6___boxed(lean_object* v___x_2163_, lean_object* v_a_2164_, lean_object* v___f_2165_, lean_object* v_hasTrace_2166_, lean_object* v_a_2167_, lean_object* v___f_2168_, lean_object* v_ext_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_){
_start:
{
uint8_t v_hasTrace_boxed_2175_; lean_object* v_res_2176_; 
v_hasTrace_boxed_2175_ = lean_unbox(v_hasTrace_2166_);
v_res_2176_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__6(v___x_2163_, v_a_2164_, v___f_2165_, v_hasTrace_boxed_2175_, v_a_2167_, v___f_2168_, v_ext_2169_, v___y_2170_, v___y_2171_, v___y_2172_, v___y_2173_);
lean_dec(v___y_2173_);
lean_dec_ref(v___y_2172_);
lean_dec(v___y_2171_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v_ext_2169_);
return v_res_2176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__7(lean_object* v___x_2177_, lean_object* v_a_2178_, lean_object* v___f_2179_, uint8_t v___x_2180_, lean_object* v_a_2181_, lean_object* v___f_2182_, lean_object* v_ext_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_){
_start:
{
lean_object* v___x_2189_; 
v___x_2189_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_ext_2183_, v___y_2187_);
if (lean_obj_tag(v___x_2189_) == 0)
{
lean_object* v_a_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; 
v_a_2190_ = lean_ctor_get(v___x_2189_, 0);
lean_inc(v_a_2190_);
lean_dec_ref_known(v___x_2189_, 1);
v___x_2191_ = lean_unsigned_to_nat(1u);
v___x_2192_ = lean_mk_empty_array_with_capacity(v___x_2191_);
v___x_2193_ = lean_array_push(v___x_2192_, v_a_2190_);
v___x_2194_ = l_Lean_Options_empty;
v___x_2195_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_2177_, v___x_2193_, v_a_2178_, v___x_2194_, v___y_2184_, v___y_2186_, v___y_2187_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_object* v_a_2196_; lean_object* v___x_2197_; lean_object* v___f_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___f_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; 
v_a_2196_ = lean_ctor_get(v___x_2195_, 0);
lean_inc(v_a_2196_);
lean_dec_ref_known(v___x_2195_, 1);
v___x_2197_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0));
v___f_2198_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed), 11, 2);
lean_closure_set(v___f_2198_, 0, v___x_2197_);
lean_closure_set(v___f_2198_, 1, v___f_2179_);
v___x_2199_ = lean_box(v___x_2180_);
v___x_2200_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed), 11, 2);
lean_closure_set(v___x_2200_, 0, v___x_2197_);
lean_closure_set(v___x_2200_, 1, v___x_2199_);
v___x_2201_ = lean_box(0);
v___f_2202_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__1));
lean_inc_ref(v_a_2181_);
v___x_2203_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2203_, 0, v_a_2181_);
lean_ctor_set(v___x_2203_, 1, v___x_2201_);
lean_ctor_set_uint8(v___x_2203_, sizeof(void*)*2, v___x_2180_);
v___x_2204_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2));
v___x_2205_ = lean_unsigned_to_nat(32u);
v___x_2206_ = lean_mk_empty_array_with_capacity(v___x_2205_);
lean_dec_ref(v___x_2206_);
v___x_2207_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9);
v___x_2208_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_2208_, 0, v___f_2198_);
lean_ctor_set(v___x_2208_, 1, v___x_2204_);
lean_ctor_set(v___x_2208_, 2, v___f_2202_);
lean_ctor_set(v___x_2208_, 3, v___f_2182_);
lean_ctor_set(v___x_2208_, 4, v___x_2200_);
lean_ctor_set_uint8(v___x_2208_, sizeof(void*)*5, v___x_2180_);
v___x_2209_ = l_Lean_Meta_Simp_main(v_a_2181_, v_a_2196_, v___x_2207_, v___x_2208_, v___y_2184_, v___y_2185_, v___y_2186_, v___y_2187_);
if (lean_obj_tag(v___x_2209_) == 0)
{
lean_object* v_a_2210_; lean_object* v_fst_2211_; lean_object* v___x_2212_; 
v_a_2210_ = lean_ctor_get(v___x_2209_, 0);
lean_inc(v_a_2210_);
lean_dec_ref_known(v___x_2209_, 1);
v_fst_2211_ = lean_ctor_get(v_a_2210_, 0);
lean_inc(v_fst_2211_);
lean_dec(v_a_2210_);
v___x_2212_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_2203_, v_fst_2211_, v___y_2184_, v___y_2185_, v___y_2186_, v___y_2187_);
return v___x_2212_;
}
else
{
lean_object* v_a_2213_; lean_object* v___x_2215_; uint8_t v_isShared_2216_; uint8_t v_isSharedCheck_2220_; 
lean_dec_ref_known(v___x_2203_, 2);
v_a_2213_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2220_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2220_ == 0)
{
v___x_2215_ = v___x_2209_;
v_isShared_2216_ = v_isSharedCheck_2220_;
goto v_resetjp_2214_;
}
else
{
lean_inc(v_a_2213_);
lean_dec(v___x_2209_);
v___x_2215_ = lean_box(0);
v_isShared_2216_ = v_isSharedCheck_2220_;
goto v_resetjp_2214_;
}
v_resetjp_2214_:
{
lean_object* v___x_2218_; 
if (v_isShared_2216_ == 0)
{
v___x_2218_ = v___x_2215_;
goto v_reusejp_2217_;
}
else
{
lean_object* v_reuseFailAlloc_2219_; 
v_reuseFailAlloc_2219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2219_, 0, v_a_2213_);
v___x_2218_ = v_reuseFailAlloc_2219_;
goto v_reusejp_2217_;
}
v_reusejp_2217_:
{
return v___x_2218_;
}
}
}
}
else
{
lean_object* v_a_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2228_; 
lean_dec_ref(v___f_2182_);
lean_dec_ref(v_a_2181_);
lean_dec_ref(v___f_2179_);
v_a_2221_ = lean_ctor_get(v___x_2195_, 0);
v_isSharedCheck_2228_ = !lean_is_exclusive(v___x_2195_);
if (v_isSharedCheck_2228_ == 0)
{
v___x_2223_ = v___x_2195_;
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_a_2221_);
lean_dec(v___x_2195_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2226_; 
if (v_isShared_2224_ == 0)
{
v___x_2226_ = v___x_2223_;
goto v_reusejp_2225_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v_a_2221_);
v___x_2226_ = v_reuseFailAlloc_2227_;
goto v_reusejp_2225_;
}
v_reusejp_2225_:
{
return v___x_2226_;
}
}
}
}
else
{
lean_object* v_a_2229_; lean_object* v___x_2231_; uint8_t v_isShared_2232_; uint8_t v_isSharedCheck_2236_; 
lean_dec_ref(v___f_2182_);
lean_dec_ref(v_a_2181_);
lean_dec_ref(v___f_2179_);
lean_dec_ref(v_a_2178_);
lean_dec_ref(v___x_2177_);
v_a_2229_ = lean_ctor_get(v___x_2189_, 0);
v_isSharedCheck_2236_ = !lean_is_exclusive(v___x_2189_);
if (v_isSharedCheck_2236_ == 0)
{
v___x_2231_ = v___x_2189_;
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
else
{
lean_inc(v_a_2229_);
lean_dec(v___x_2189_);
v___x_2231_ = lean_box(0);
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
v_resetjp_2230_:
{
lean_object* v___x_2234_; 
if (v_isShared_2232_ == 0)
{
v___x_2234_ = v___x_2231_;
goto v_reusejp_2233_;
}
else
{
lean_object* v_reuseFailAlloc_2235_; 
v_reuseFailAlloc_2235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2235_, 0, v_a_2229_);
v___x_2234_ = v_reuseFailAlloc_2235_;
goto v_reusejp_2233_;
}
v_reusejp_2233_:
{
return v___x_2234_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___lam__7___boxed(lean_object* v___x_2237_, lean_object* v_a_2238_, lean_object* v___f_2239_, lean_object* v___x_2240_, lean_object* v_a_2241_, lean_object* v___f_2242_, lean_object* v_ext_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_){
_start:
{
uint8_t v___x_22107__boxed_2249_; lean_object* v_res_2250_; 
v___x_22107__boxed_2249_ = lean_unbox(v___x_2240_);
v_res_2250_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__7(v___x_2237_, v_a_2238_, v___f_2239_, v___x_22107__boxed_2249_, v_a_2241_, v___f_2242_, v_ext_2243_, v___y_2244_, v___y_2245_, v___y_2246_, v___y_2247_);
lean_dec(v___y_2247_);
lean_dec_ref(v___y_2246_);
lean_dec(v___y_2245_);
lean_dec_ref(v___y_2244_);
lean_dec_ref(v_ext_2243_);
return v_res_2250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6(lean_object* v_opts_2251_, lean_object* v_opt_2252_){
_start:
{
lean_object* v_name_2253_; lean_object* v_defValue_2254_; lean_object* v_map_2255_; lean_object* v___x_2256_; 
v_name_2253_ = lean_ctor_get(v_opt_2252_, 0);
v_defValue_2254_ = lean_ctor_get(v_opt_2252_, 1);
v_map_2255_ = lean_ctor_get(v_opts_2251_, 0);
v___x_2256_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2255_, v_name_2253_);
if (lean_obj_tag(v___x_2256_) == 0)
{
lean_inc(v_defValue_2254_);
return v_defValue_2254_;
}
else
{
lean_object* v_val_2257_; 
v_val_2257_ = lean_ctor_get(v___x_2256_, 0);
lean_inc(v_val_2257_);
lean_dec_ref_known(v___x_2256_, 1);
if (lean_obj_tag(v_val_2257_) == 3)
{
lean_object* v_v_2258_; 
v_v_2258_ = lean_ctor_get(v_val_2257_, 0);
lean_inc(v_v_2258_);
lean_dec_ref_known(v_val_2257_, 1);
return v_v_2258_;
}
else
{
lean_dec(v_val_2257_);
lean_inc(v_defValue_2254_);
return v_defValue_2254_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6___boxed(lean_object* v_opts_2259_, lean_object* v_opt_2260_){
_start:
{
lean_object* v_res_2261_; 
v_res_2261_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6(v_opts_2259_, v_opt_2260_);
lean_dec_ref(v_opt_2260_);
lean_dec_ref(v_opts_2259_);
return v_res_2261_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5(lean_object* v_e_2262_){
_start:
{
if (lean_obj_tag(v_e_2262_) == 0)
{
uint8_t v___x_2263_; 
v___x_2263_ = 2;
return v___x_2263_;
}
else
{
uint8_t v___x_2264_; 
v___x_2264_ = 0;
return v___x_2264_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5___boxed(lean_object* v_e_2265_){
_start:
{
uint8_t v_res_2266_; lean_object* v_r_2267_; 
v_res_2266_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5(v_e_2265_);
lean_dec_ref(v_e_2265_);
v_r_2267_ = lean_box(v_res_2266_);
return v_r_2267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4(size_t v_sz_2268_, size_t v_i_2269_, lean_object* v_bs_2270_){
_start:
{
uint8_t v___x_2271_; 
v___x_2271_ = lean_usize_dec_lt(v_i_2269_, v_sz_2268_);
if (v___x_2271_ == 0)
{
return v_bs_2270_;
}
else
{
lean_object* v_v_2272_; lean_object* v_msg_2273_; lean_object* v___x_2274_; lean_object* v_bs_x27_2275_; size_t v___x_2276_; size_t v___x_2277_; lean_object* v___x_2278_; 
v_v_2272_ = lean_array_uget_borrowed(v_bs_2270_, v_i_2269_);
v_msg_2273_ = lean_ctor_get(v_v_2272_, 1);
lean_inc_ref(v_msg_2273_);
v___x_2274_ = lean_unsigned_to_nat(0u);
v_bs_x27_2275_ = lean_array_uset(v_bs_2270_, v_i_2269_, v___x_2274_);
v___x_2276_ = ((size_t)1ULL);
v___x_2277_ = lean_usize_add(v_i_2269_, v___x_2276_);
v___x_2278_ = lean_array_uset(v_bs_x27_2275_, v_i_2269_, v_msg_2273_);
v_i_2269_ = v___x_2277_;
v_bs_2270_ = v___x_2278_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4___boxed(lean_object* v_sz_2280_, lean_object* v_i_2281_, lean_object* v_bs_2282_){
_start:
{
size_t v_sz_boxed_2283_; size_t v_i_boxed_2284_; lean_object* v_res_2285_; 
v_sz_boxed_2283_ = lean_unbox_usize(v_sz_2280_);
lean_dec(v_sz_2280_);
v_i_boxed_2284_ = lean_unbox_usize(v_i_2281_);
lean_dec(v_i_2281_);
v_res_2285_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4(v_sz_boxed_2283_, v_i_boxed_2284_, v_bs_2282_);
return v_res_2285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3(lean_object* v_oldTraces_2286_, lean_object* v_data_2287_, lean_object* v_ref_2288_, lean_object* v_msg_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_){
_start:
{
lean_object* v_fileName_2295_; lean_object* v_fileMap_2296_; lean_object* v_options_2297_; lean_object* v_currRecDepth_2298_; lean_object* v_maxRecDepth_2299_; lean_object* v_ref_2300_; lean_object* v_currNamespace_2301_; lean_object* v_openDecls_2302_; lean_object* v_initHeartbeats_2303_; lean_object* v_maxHeartbeats_2304_; lean_object* v_quotContext_2305_; lean_object* v_currMacroScope_2306_; uint8_t v_diag_2307_; lean_object* v_cancelTk_x3f_2308_; uint8_t v_suppressElabErrors_2309_; lean_object* v_inheritedTraceOptions_2310_; lean_object* v___x_2311_; lean_object* v_traceState_2312_; lean_object* v_traces_2313_; lean_object* v_ref_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; size_t v_sz_2317_; size_t v___x_2318_; lean_object* v___x_2319_; lean_object* v_msg_2320_; lean_object* v___x_2321_; lean_object* v_a_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2359_; 
v_fileName_2295_ = lean_ctor_get(v___y_2292_, 0);
v_fileMap_2296_ = lean_ctor_get(v___y_2292_, 1);
v_options_2297_ = lean_ctor_get(v___y_2292_, 2);
v_currRecDepth_2298_ = lean_ctor_get(v___y_2292_, 3);
v_maxRecDepth_2299_ = lean_ctor_get(v___y_2292_, 4);
v_ref_2300_ = lean_ctor_get(v___y_2292_, 5);
v_currNamespace_2301_ = lean_ctor_get(v___y_2292_, 6);
v_openDecls_2302_ = lean_ctor_get(v___y_2292_, 7);
v_initHeartbeats_2303_ = lean_ctor_get(v___y_2292_, 8);
v_maxHeartbeats_2304_ = lean_ctor_get(v___y_2292_, 9);
v_quotContext_2305_ = lean_ctor_get(v___y_2292_, 10);
v_currMacroScope_2306_ = lean_ctor_get(v___y_2292_, 11);
v_diag_2307_ = lean_ctor_get_uint8(v___y_2292_, sizeof(void*)*14);
v_cancelTk_x3f_2308_ = lean_ctor_get(v___y_2292_, 12);
v_suppressElabErrors_2309_ = lean_ctor_get_uint8(v___y_2292_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2310_ = lean_ctor_get(v___y_2292_, 13);
v___x_2311_ = lean_st_ref_get(v___y_2293_);
v_traceState_2312_ = lean_ctor_get(v___x_2311_, 4);
lean_inc_ref(v_traceState_2312_);
lean_dec(v___x_2311_);
v_traces_2313_ = lean_ctor_get(v_traceState_2312_, 0);
lean_inc_ref(v_traces_2313_);
lean_dec_ref(v_traceState_2312_);
v_ref_2314_ = l_Lean_replaceRef(v_ref_2288_, v_ref_2300_);
lean_inc_ref(v_inheritedTraceOptions_2310_);
lean_inc(v_cancelTk_x3f_2308_);
lean_inc(v_currMacroScope_2306_);
lean_inc(v_quotContext_2305_);
lean_inc(v_maxHeartbeats_2304_);
lean_inc(v_initHeartbeats_2303_);
lean_inc(v_openDecls_2302_);
lean_inc(v_currNamespace_2301_);
lean_inc(v_maxRecDepth_2299_);
lean_inc(v_currRecDepth_2298_);
lean_inc_ref(v_options_2297_);
lean_inc_ref(v_fileMap_2296_);
lean_inc_ref(v_fileName_2295_);
v___x_2315_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2315_, 0, v_fileName_2295_);
lean_ctor_set(v___x_2315_, 1, v_fileMap_2296_);
lean_ctor_set(v___x_2315_, 2, v_options_2297_);
lean_ctor_set(v___x_2315_, 3, v_currRecDepth_2298_);
lean_ctor_set(v___x_2315_, 4, v_maxRecDepth_2299_);
lean_ctor_set(v___x_2315_, 5, v_ref_2314_);
lean_ctor_set(v___x_2315_, 6, v_currNamespace_2301_);
lean_ctor_set(v___x_2315_, 7, v_openDecls_2302_);
lean_ctor_set(v___x_2315_, 8, v_initHeartbeats_2303_);
lean_ctor_set(v___x_2315_, 9, v_maxHeartbeats_2304_);
lean_ctor_set(v___x_2315_, 10, v_quotContext_2305_);
lean_ctor_set(v___x_2315_, 11, v_currMacroScope_2306_);
lean_ctor_set(v___x_2315_, 12, v_cancelTk_x3f_2308_);
lean_ctor_set(v___x_2315_, 13, v_inheritedTraceOptions_2310_);
lean_ctor_set_uint8(v___x_2315_, sizeof(void*)*14, v_diag_2307_);
lean_ctor_set_uint8(v___x_2315_, sizeof(void*)*14 + 1, v_suppressElabErrors_2309_);
v___x_2316_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2313_);
lean_dec_ref(v_traces_2313_);
v_sz_2317_ = lean_array_size(v___x_2316_);
v___x_2318_ = ((size_t)0ULL);
v___x_2319_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3_spec__4(v_sz_2317_, v___x_2318_, v___x_2316_);
v_msg_2320_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2320_, 0, v_data_2287_);
lean_ctor_set(v_msg_2320_, 1, v_msg_2289_);
lean_ctor_set(v_msg_2320_, 2, v___x_2319_);
v___x_2321_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0_spec__0(v_msg_2320_, v___y_2290_, v___y_2291_, v___x_2315_, v___y_2293_);
lean_dec_ref_known(v___x_2315_, 14);
v_a_2322_ = lean_ctor_get(v___x_2321_, 0);
v_isSharedCheck_2359_ = !lean_is_exclusive(v___x_2321_);
if (v_isSharedCheck_2359_ == 0)
{
v___x_2324_ = v___x_2321_;
v_isShared_2325_ = v_isSharedCheck_2359_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_a_2322_);
lean_dec(v___x_2321_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2359_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
lean_object* v___x_2326_; lean_object* v_traceState_2327_; lean_object* v_env_2328_; lean_object* v_nextMacroScope_2329_; lean_object* v_ngen_2330_; lean_object* v_auxDeclNGen_2331_; lean_object* v_cache_2332_; lean_object* v_messages_2333_; lean_object* v_infoState_2334_; lean_object* v_snapshotTasks_2335_; lean_object* v___x_2337_; uint8_t v_isShared_2338_; uint8_t v_isSharedCheck_2358_; 
v___x_2326_ = lean_st_ref_take(v___y_2293_);
v_traceState_2327_ = lean_ctor_get(v___x_2326_, 4);
v_env_2328_ = lean_ctor_get(v___x_2326_, 0);
v_nextMacroScope_2329_ = lean_ctor_get(v___x_2326_, 1);
v_ngen_2330_ = lean_ctor_get(v___x_2326_, 2);
v_auxDeclNGen_2331_ = lean_ctor_get(v___x_2326_, 3);
v_cache_2332_ = lean_ctor_get(v___x_2326_, 5);
v_messages_2333_ = lean_ctor_get(v___x_2326_, 6);
v_infoState_2334_ = lean_ctor_get(v___x_2326_, 7);
v_snapshotTasks_2335_ = lean_ctor_get(v___x_2326_, 8);
v_isSharedCheck_2358_ = !lean_is_exclusive(v___x_2326_);
if (v_isSharedCheck_2358_ == 0)
{
v___x_2337_ = v___x_2326_;
v_isShared_2338_ = v_isSharedCheck_2358_;
goto v_resetjp_2336_;
}
else
{
lean_inc(v_snapshotTasks_2335_);
lean_inc(v_infoState_2334_);
lean_inc(v_messages_2333_);
lean_inc(v_cache_2332_);
lean_inc(v_traceState_2327_);
lean_inc(v_auxDeclNGen_2331_);
lean_inc(v_ngen_2330_);
lean_inc(v_nextMacroScope_2329_);
lean_inc(v_env_2328_);
lean_dec(v___x_2326_);
v___x_2337_ = lean_box(0);
v_isShared_2338_ = v_isSharedCheck_2358_;
goto v_resetjp_2336_;
}
v_resetjp_2336_:
{
uint64_t v_tid_2339_; lean_object* v___x_2341_; uint8_t v_isShared_2342_; uint8_t v_isSharedCheck_2356_; 
v_tid_2339_ = lean_ctor_get_uint64(v_traceState_2327_, sizeof(void*)*1);
v_isSharedCheck_2356_ = !lean_is_exclusive(v_traceState_2327_);
if (v_isSharedCheck_2356_ == 0)
{
lean_object* v_unused_2357_; 
v_unused_2357_ = lean_ctor_get(v_traceState_2327_, 0);
lean_dec(v_unused_2357_);
v___x_2341_ = v_traceState_2327_;
v_isShared_2342_ = v_isSharedCheck_2356_;
goto v_resetjp_2340_;
}
else
{
lean_dec(v_traceState_2327_);
v___x_2341_ = lean_box(0);
v_isShared_2342_ = v_isSharedCheck_2356_;
goto v_resetjp_2340_;
}
v_resetjp_2340_:
{
lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2346_; 
v___x_2343_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2343_, 0, v_ref_2288_);
lean_ctor_set(v___x_2343_, 1, v_a_2322_);
v___x_2344_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2286_, v___x_2343_);
if (v_isShared_2342_ == 0)
{
lean_ctor_set(v___x_2341_, 0, v___x_2344_);
v___x_2346_ = v___x_2341_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2355_; 
v_reuseFailAlloc_2355_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2355_, 0, v___x_2344_);
lean_ctor_set_uint64(v_reuseFailAlloc_2355_, sizeof(void*)*1, v_tid_2339_);
v___x_2346_ = v_reuseFailAlloc_2355_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
lean_object* v___x_2348_; 
if (v_isShared_2338_ == 0)
{
lean_ctor_set(v___x_2337_, 4, v___x_2346_);
v___x_2348_ = v___x_2337_;
goto v_reusejp_2347_;
}
else
{
lean_object* v_reuseFailAlloc_2354_; 
v_reuseFailAlloc_2354_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2354_, 0, v_env_2328_);
lean_ctor_set(v_reuseFailAlloc_2354_, 1, v_nextMacroScope_2329_);
lean_ctor_set(v_reuseFailAlloc_2354_, 2, v_ngen_2330_);
lean_ctor_set(v_reuseFailAlloc_2354_, 3, v_auxDeclNGen_2331_);
lean_ctor_set(v_reuseFailAlloc_2354_, 4, v___x_2346_);
lean_ctor_set(v_reuseFailAlloc_2354_, 5, v_cache_2332_);
lean_ctor_set(v_reuseFailAlloc_2354_, 6, v_messages_2333_);
lean_ctor_set(v_reuseFailAlloc_2354_, 7, v_infoState_2334_);
lean_ctor_set(v_reuseFailAlloc_2354_, 8, v_snapshotTasks_2335_);
v___x_2348_ = v_reuseFailAlloc_2354_;
goto v_reusejp_2347_;
}
v_reusejp_2347_:
{
lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2352_; 
v___x_2349_ = lean_st_ref_set(v___y_2293_, v___x_2348_);
v___x_2350_ = lean_box(0);
if (v_isShared_2325_ == 0)
{
lean_ctor_set(v___x_2324_, 0, v___x_2350_);
v___x_2352_ = v___x_2324_;
goto v_reusejp_2351_;
}
else
{
lean_object* v_reuseFailAlloc_2353_; 
v_reuseFailAlloc_2353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2353_, 0, v___x_2350_);
v___x_2352_ = v_reuseFailAlloc_2353_;
goto v_reusejp_2351_;
}
v_reusejp_2351_:
{
return v___x_2352_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3___boxed(lean_object* v_oldTraces_2360_, lean_object* v_data_2361_, lean_object* v_ref_2362_, lean_object* v_msg_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_){
_start:
{
lean_object* v_res_2369_; 
v_res_2369_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3(v_oldTraces_2360_, v_data_2361_, v_ref_2362_, v_msg_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
return v_res_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(lean_object* v_x_2370_){
_start:
{
if (lean_obj_tag(v_x_2370_) == 0)
{
lean_object* v_a_2372_; lean_object* v___x_2374_; uint8_t v_isShared_2375_; uint8_t v_isSharedCheck_2379_; 
v_a_2372_ = lean_ctor_get(v_x_2370_, 0);
v_isSharedCheck_2379_ = !lean_is_exclusive(v_x_2370_);
if (v_isSharedCheck_2379_ == 0)
{
v___x_2374_ = v_x_2370_;
v_isShared_2375_ = v_isSharedCheck_2379_;
goto v_resetjp_2373_;
}
else
{
lean_inc(v_a_2372_);
lean_dec(v_x_2370_);
v___x_2374_ = lean_box(0);
v_isShared_2375_ = v_isSharedCheck_2379_;
goto v_resetjp_2373_;
}
v_resetjp_2373_:
{
lean_object* v___x_2377_; 
if (v_isShared_2375_ == 0)
{
lean_ctor_set_tag(v___x_2374_, 1);
v___x_2377_ = v___x_2374_;
goto v_reusejp_2376_;
}
else
{
lean_object* v_reuseFailAlloc_2378_; 
v_reuseFailAlloc_2378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2378_, 0, v_a_2372_);
v___x_2377_ = v_reuseFailAlloc_2378_;
goto v_reusejp_2376_;
}
v_reusejp_2376_:
{
return v___x_2377_;
}
}
}
else
{
lean_object* v_a_2380_; lean_object* v___x_2382_; uint8_t v_isShared_2383_; uint8_t v_isSharedCheck_2387_; 
v_a_2380_ = lean_ctor_get(v_x_2370_, 0);
v_isSharedCheck_2387_ = !lean_is_exclusive(v_x_2370_);
if (v_isSharedCheck_2387_ == 0)
{
v___x_2382_ = v_x_2370_;
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
else
{
lean_inc(v_a_2380_);
lean_dec(v_x_2370_);
v___x_2382_ = lean_box(0);
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
v_resetjp_2381_:
{
lean_object* v___x_2385_; 
if (v_isShared_2383_ == 0)
{
lean_ctor_set_tag(v___x_2382_, 0);
v___x_2385_ = v___x_2382_;
goto v_reusejp_2384_;
}
else
{
lean_object* v_reuseFailAlloc_2386_; 
v_reuseFailAlloc_2386_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2386_, 0, v_a_2380_);
v___x_2385_ = v_reuseFailAlloc_2386_;
goto v_reusejp_2384_;
}
v_reusejp_2384_:
{
return v___x_2385_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg___boxed(lean_object* v_x_2388_, lean_object* v___y_2389_){
_start:
{
lean_object* v_res_2390_; 
v_res_2390_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(v_x_2388_);
return v_res_2390_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0(void){
_start:
{
lean_object* v___x_2391_; double v___x_2392_; 
v___x_2391_ = lean_unsigned_to_nat(0u);
v___x_2392_ = lean_float_of_nat(v___x_2391_);
return v___x_2392_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2(void){
_start:
{
lean_object* v___x_2394_; lean_object* v___x_2395_; 
v___x_2394_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__1));
v___x_2395_ = l_Lean_stringToMessageData(v___x_2394_);
return v___x_2395_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3(void){
_start:
{
lean_object* v___x_2396_; double v___x_2397_; 
v___x_2396_ = lean_unsigned_to_nat(1000u);
v___x_2397_ = lean_float_of_nat(v___x_2396_);
return v___x_2397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3(lean_object* v_cls_2398_, uint8_t v_collapsed_2399_, lean_object* v_tag_2400_, lean_object* v_opts_2401_, uint8_t v_clsEnabled_2402_, lean_object* v_oldTraces_2403_, lean_object* v_msg_2404_, lean_object* v_resStartStop_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_, lean_object* v___y_2408_, lean_object* v___y_2409_){
_start:
{
lean_object* v_fst_2411_; lean_object* v_snd_2412_; lean_object* v___y_2414_; lean_object* v___y_2415_; lean_object* v_data_2416_; lean_object* v_fst_2427_; lean_object* v_snd_2428_; lean_object* v___x_2429_; uint8_t v___x_2430_; lean_object* v___y_2432_; lean_object* v_a_2433_; uint8_t v___y_2448_; double v___y_2479_; 
v_fst_2411_ = lean_ctor_get(v_resStartStop_2405_, 0);
lean_inc(v_fst_2411_);
v_snd_2412_ = lean_ctor_get(v_resStartStop_2405_, 1);
lean_inc(v_snd_2412_);
lean_dec_ref(v_resStartStop_2405_);
v_fst_2427_ = lean_ctor_get(v_snd_2412_, 0);
lean_inc(v_fst_2427_);
v_snd_2428_ = lean_ctor_get(v_snd_2412_, 1);
lean_inc(v_snd_2428_);
lean_dec(v_snd_2412_);
v___x_2429_ = l_Lean_trace_profiler;
v___x_2430_ = lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(v_opts_2401_, v___x_2429_);
if (v___x_2430_ == 0)
{
v___y_2448_ = v___x_2430_;
goto v___jp_2447_;
}
else
{
lean_object* v___x_2484_; uint8_t v___x_2485_; 
v___x_2484_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2485_ = lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(v_opts_2401_, v___x_2484_);
if (v___x_2485_ == 0)
{
lean_object* v___x_2486_; lean_object* v___x_2487_; double v___x_2488_; double v___x_2489_; double v___x_2490_; 
v___x_2486_ = l_Lean_trace_profiler_threshold;
v___x_2487_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6(v_opts_2401_, v___x_2486_);
v___x_2488_ = lean_float_of_nat(v___x_2487_);
v___x_2489_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__3);
v___x_2490_ = lean_float_div(v___x_2488_, v___x_2489_);
v___y_2479_ = v___x_2490_;
goto v___jp_2478_;
}
else
{
lean_object* v___x_2491_; lean_object* v___x_2492_; double v___x_2493_; 
v___x_2491_ = l_Lean_trace_profiler_threshold;
v___x_2492_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__6(v_opts_2401_, v___x_2491_);
v___x_2493_ = lean_float_of_nat(v___x_2492_);
v___y_2479_ = v___x_2493_;
goto v___jp_2478_;
}
}
v___jp_2413_:
{
lean_object* v___x_2417_; 
lean_inc(v___y_2414_);
v___x_2417_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__3(v_oldTraces_2403_, v_data_2416_, v___y_2414_, v___y_2415_, v___y_2406_, v___y_2407_, v___y_2408_, v___y_2409_);
if (lean_obj_tag(v___x_2417_) == 0)
{
lean_object* v___x_2418_; 
lean_dec_ref_known(v___x_2417_, 1);
v___x_2418_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(v_fst_2411_);
return v___x_2418_;
}
else
{
lean_object* v_a_2419_; lean_object* v___x_2421_; uint8_t v_isShared_2422_; uint8_t v_isSharedCheck_2426_; 
lean_dec(v_fst_2411_);
v_a_2419_ = lean_ctor_get(v___x_2417_, 0);
v_isSharedCheck_2426_ = !lean_is_exclusive(v___x_2417_);
if (v_isSharedCheck_2426_ == 0)
{
v___x_2421_ = v___x_2417_;
v_isShared_2422_ = v_isSharedCheck_2426_;
goto v_resetjp_2420_;
}
else
{
lean_inc(v_a_2419_);
lean_dec(v___x_2417_);
v___x_2421_ = lean_box(0);
v_isShared_2422_ = v_isSharedCheck_2426_;
goto v_resetjp_2420_;
}
v_resetjp_2420_:
{
lean_object* v___x_2424_; 
if (v_isShared_2422_ == 0)
{
v___x_2424_ = v___x_2421_;
goto v_reusejp_2423_;
}
else
{
lean_object* v_reuseFailAlloc_2425_; 
v_reuseFailAlloc_2425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2425_, 0, v_a_2419_);
v___x_2424_ = v_reuseFailAlloc_2425_;
goto v_reusejp_2423_;
}
v_reusejp_2423_:
{
return v___x_2424_;
}
}
}
}
v___jp_2431_:
{
uint8_t v_result_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; double v___x_2437_; lean_object* v_data_2438_; 
v_result_2434_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__5(v_fst_2411_);
v___x_2435_ = lean_box(v_result_2434_);
v___x_2436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2436_, 0, v___x_2435_);
v___x_2437_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__0);
lean_inc_ref(v_tag_2400_);
lean_inc_ref(v___x_2436_);
lean_inc(v_cls_2398_);
v_data_2438_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2438_, 0, v_cls_2398_);
lean_ctor_set(v_data_2438_, 1, v___x_2436_);
lean_ctor_set(v_data_2438_, 2, v_tag_2400_);
lean_ctor_set_float(v_data_2438_, sizeof(void*)*3, v___x_2437_);
lean_ctor_set_float(v_data_2438_, sizeof(void*)*3 + 8, v___x_2437_);
lean_ctor_set_uint8(v_data_2438_, sizeof(void*)*3 + 16, v_collapsed_2399_);
if (v___x_2430_ == 0)
{
lean_dec_ref_known(v___x_2436_, 1);
lean_dec(v_snd_2428_);
lean_dec(v_fst_2427_);
lean_dec_ref(v_tag_2400_);
lean_dec(v_cls_2398_);
v___y_2414_ = v___y_2432_;
v___y_2415_ = v_a_2433_;
v_data_2416_ = v_data_2438_;
goto v___jp_2413_;
}
else
{
lean_object* v_data_2439_; double v___x_2440_; double v___x_2441_; 
lean_dec_ref_known(v_data_2438_, 3);
v_data_2439_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2439_, 0, v_cls_2398_);
lean_ctor_set(v_data_2439_, 1, v___x_2436_);
lean_ctor_set(v_data_2439_, 2, v_tag_2400_);
v___x_2440_ = lean_unbox_float(v_fst_2427_);
lean_dec(v_fst_2427_);
lean_ctor_set_float(v_data_2439_, sizeof(void*)*3, v___x_2440_);
v___x_2441_ = lean_unbox_float(v_snd_2428_);
lean_dec(v_snd_2428_);
lean_ctor_set_float(v_data_2439_, sizeof(void*)*3 + 8, v___x_2441_);
lean_ctor_set_uint8(v_data_2439_, sizeof(void*)*3 + 16, v_collapsed_2399_);
v___y_2414_ = v___y_2432_;
v___y_2415_ = v_a_2433_;
v_data_2416_ = v_data_2439_;
goto v___jp_2413_;
}
}
v___jp_2442_:
{
lean_object* v_ref_2443_; lean_object* v___x_2444_; 
v_ref_2443_ = lean_ctor_get(v___y_2408_, 5);
lean_inc(v___y_2409_);
lean_inc_ref(v___y_2408_);
lean_inc(v___y_2407_);
lean_inc_ref(v___y_2406_);
lean_inc(v_fst_2411_);
v___x_2444_ = lean_apply_6(v_msg_2404_, v_fst_2411_, v___y_2406_, v___y_2407_, v___y_2408_, v___y_2409_, lean_box(0));
if (lean_obj_tag(v___x_2444_) == 0)
{
lean_object* v_a_2445_; 
v_a_2445_ = lean_ctor_get(v___x_2444_, 0);
lean_inc(v_a_2445_);
lean_dec_ref_known(v___x_2444_, 1);
v___y_2432_ = v_ref_2443_;
v_a_2433_ = v_a_2445_;
goto v___jp_2431_;
}
else
{
lean_object* v___x_2446_; 
lean_dec_ref_known(v___x_2444_, 1);
v___x_2446_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___closed__2);
v___y_2432_ = v_ref_2443_;
v_a_2433_ = v___x_2446_;
goto v___jp_2431_;
}
}
v___jp_2447_:
{
if (v_clsEnabled_2402_ == 0)
{
if (v___y_2448_ == 0)
{
lean_object* v___x_2449_; lean_object* v_traceState_2450_; lean_object* v_env_2451_; lean_object* v_nextMacroScope_2452_; lean_object* v_ngen_2453_; lean_object* v_auxDeclNGen_2454_; lean_object* v_cache_2455_; lean_object* v_messages_2456_; lean_object* v_infoState_2457_; lean_object* v_snapshotTasks_2458_; lean_object* v___x_2460_; uint8_t v_isShared_2461_; uint8_t v_isSharedCheck_2477_; 
lean_dec(v_snd_2428_);
lean_dec(v_fst_2427_);
lean_dec_ref(v_msg_2404_);
lean_dec_ref(v_tag_2400_);
lean_dec(v_cls_2398_);
v___x_2449_ = lean_st_ref_take(v___y_2409_);
v_traceState_2450_ = lean_ctor_get(v___x_2449_, 4);
v_env_2451_ = lean_ctor_get(v___x_2449_, 0);
v_nextMacroScope_2452_ = lean_ctor_get(v___x_2449_, 1);
v_ngen_2453_ = lean_ctor_get(v___x_2449_, 2);
v_auxDeclNGen_2454_ = lean_ctor_get(v___x_2449_, 3);
v_cache_2455_ = lean_ctor_get(v___x_2449_, 5);
v_messages_2456_ = lean_ctor_get(v___x_2449_, 6);
v_infoState_2457_ = lean_ctor_get(v___x_2449_, 7);
v_snapshotTasks_2458_ = lean_ctor_get(v___x_2449_, 8);
v_isSharedCheck_2477_ = !lean_is_exclusive(v___x_2449_);
if (v_isSharedCheck_2477_ == 0)
{
v___x_2460_ = v___x_2449_;
v_isShared_2461_ = v_isSharedCheck_2477_;
goto v_resetjp_2459_;
}
else
{
lean_inc(v_snapshotTasks_2458_);
lean_inc(v_infoState_2457_);
lean_inc(v_messages_2456_);
lean_inc(v_cache_2455_);
lean_inc(v_traceState_2450_);
lean_inc(v_auxDeclNGen_2454_);
lean_inc(v_ngen_2453_);
lean_inc(v_nextMacroScope_2452_);
lean_inc(v_env_2451_);
lean_dec(v___x_2449_);
v___x_2460_ = lean_box(0);
v_isShared_2461_ = v_isSharedCheck_2477_;
goto v_resetjp_2459_;
}
v_resetjp_2459_:
{
uint64_t v_tid_2462_; lean_object* v_traces_2463_; lean_object* v___x_2465_; uint8_t v_isShared_2466_; uint8_t v_isSharedCheck_2476_; 
v_tid_2462_ = lean_ctor_get_uint64(v_traceState_2450_, sizeof(void*)*1);
v_traces_2463_ = lean_ctor_get(v_traceState_2450_, 0);
v_isSharedCheck_2476_ = !lean_is_exclusive(v_traceState_2450_);
if (v_isSharedCheck_2476_ == 0)
{
v___x_2465_ = v_traceState_2450_;
v_isShared_2466_ = v_isSharedCheck_2476_;
goto v_resetjp_2464_;
}
else
{
lean_inc(v_traces_2463_);
lean_dec(v_traceState_2450_);
v___x_2465_ = lean_box(0);
v_isShared_2466_ = v_isSharedCheck_2476_;
goto v_resetjp_2464_;
}
v_resetjp_2464_:
{
lean_object* v___x_2467_; lean_object* v___x_2469_; 
v___x_2467_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2403_, v_traces_2463_);
lean_dec_ref(v_traces_2463_);
if (v_isShared_2466_ == 0)
{
lean_ctor_set(v___x_2465_, 0, v___x_2467_);
v___x_2469_ = v___x_2465_;
goto v_reusejp_2468_;
}
else
{
lean_object* v_reuseFailAlloc_2475_; 
v_reuseFailAlloc_2475_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2475_, 0, v___x_2467_);
lean_ctor_set_uint64(v_reuseFailAlloc_2475_, sizeof(void*)*1, v_tid_2462_);
v___x_2469_ = v_reuseFailAlloc_2475_;
goto v_reusejp_2468_;
}
v_reusejp_2468_:
{
lean_object* v___x_2471_; 
if (v_isShared_2461_ == 0)
{
lean_ctor_set(v___x_2460_, 4, v___x_2469_);
v___x_2471_ = v___x_2460_;
goto v_reusejp_2470_;
}
else
{
lean_object* v_reuseFailAlloc_2474_; 
v_reuseFailAlloc_2474_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2474_, 0, v_env_2451_);
lean_ctor_set(v_reuseFailAlloc_2474_, 1, v_nextMacroScope_2452_);
lean_ctor_set(v_reuseFailAlloc_2474_, 2, v_ngen_2453_);
lean_ctor_set(v_reuseFailAlloc_2474_, 3, v_auxDeclNGen_2454_);
lean_ctor_set(v_reuseFailAlloc_2474_, 4, v___x_2469_);
lean_ctor_set(v_reuseFailAlloc_2474_, 5, v_cache_2455_);
lean_ctor_set(v_reuseFailAlloc_2474_, 6, v_messages_2456_);
lean_ctor_set(v_reuseFailAlloc_2474_, 7, v_infoState_2457_);
lean_ctor_set(v_reuseFailAlloc_2474_, 8, v_snapshotTasks_2458_);
v___x_2471_ = v_reuseFailAlloc_2474_;
goto v_reusejp_2470_;
}
v_reusejp_2470_:
{
lean_object* v___x_2472_; lean_object* v___x_2473_; 
v___x_2472_ = lean_st_ref_set(v___y_2409_, v___x_2471_);
v___x_2473_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(v_fst_2411_);
return v___x_2473_;
}
}
}
}
}
else
{
goto v___jp_2442_;
}
}
else
{
goto v___jp_2442_;
}
}
v___jp_2478_:
{
double v___x_2480_; double v___x_2481_; double v___x_2482_; uint8_t v___x_2483_; 
v___x_2480_ = lean_unbox_float(v_snd_2428_);
v___x_2481_ = lean_unbox_float(v_fst_2427_);
v___x_2482_ = lean_float_sub(v___x_2480_, v___x_2481_);
v___x_2483_ = lean_float_decLt(v___y_2479_, v___x_2482_);
v___y_2448_ = v___x_2483_;
goto v___jp_2447_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3___boxed(lean_object* v_cls_2494_, lean_object* v_collapsed_2495_, lean_object* v_tag_2496_, lean_object* v_opts_2497_, lean_object* v_clsEnabled_2498_, lean_object* v_oldTraces_2499_, lean_object* v_msg_2500_, lean_object* v_resStartStop_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_){
_start:
{
uint8_t v_collapsed_boxed_2507_; uint8_t v_clsEnabled_boxed_2508_; lean_object* v_res_2509_; 
v_collapsed_boxed_2507_ = lean_unbox(v_collapsed_2495_);
v_clsEnabled_boxed_2508_ = lean_unbox(v_clsEnabled_2498_);
v_res_2509_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3(v_cls_2494_, v_collapsed_boxed_2507_, v_tag_2496_, v_opts_2497_, v_clsEnabled_boxed_2508_, v_oldTraces_2499_, v_msg_2500_, v_resStartStop_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_);
lean_dec(v___y_2505_);
lean_dec_ref(v___y_2504_);
lean_dec(v___y_2503_);
lean_dec_ref(v___y_2502_);
lean_dec_ref(v_opts_2497_);
return v_res_2509_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__6(void){
_start:
{
lean_object* v___x_2521_; lean_object* v___x_2522_; 
v___x_2521_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__5));
v___x_2522_ = l_Lean_stringToMessageData(v___x_2521_);
return v___x_2522_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__11(void){
_start:
{
lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; 
v___x_2530_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__7));
v___x_2531_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__10));
v___x_2532_ = l_Lean_Name_append(v___x_2531_, v___x_2530_);
return v___x_2532_;
}
}
static double _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__12(void){
_start:
{
lean_object* v___x_2533_; double v___x_2534_; 
v___x_2533_ = lean_unsigned_to_nat(1000000000u);
v___x_2534_ = lean_float_of_nat(v___x_2533_);
return v___x_2534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive(uint8_t v_expensive_2535_, lean_object* v_e_2536_, lean_object* v_a_2537_, lean_object* v_a_2538_, lean_object* v_a_2539_, lean_object* v_a_2540_){
_start:
{
lean_object* v_options_2542_; uint8_t v_hasTrace_2543_; 
v_options_2542_ = lean_ctor_get(v_a_2539_, 2);
v_hasTrace_2543_ = lean_ctor_get_uint8(v_options_2542_, sizeof(void*)*1);
if (v_hasTrace_2543_ == 0)
{
lean_object* v___x_2544_; lean_object* v_a_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; uint8_t v___x_2548_; uint8_t v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2544_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_2536_, v_a_2538_);
v_a_2545_ = lean_ctor_get(v___x_2544_, 0);
lean_inc(v_a_2545_);
lean_dec_ref(v___x_2544_);
v___x_2546_ = lean_unsigned_to_nat(100000u);
v___x_2547_ = lean_unsigned_to_nat(2u);
v___x_2548_ = 1;
v___x_2549_ = 0;
v___x_2550_ = lean_box(0);
v___x_2551_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_2551_, 0, v___x_2546_);
lean_ctor_set(v___x_2551_, 1, v___x_2547_);
lean_ctor_set(v___x_2551_, 2, v___x_2550_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 1, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 2, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 3, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 4, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 5, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 6, v___x_2549_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 7, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 8, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 9, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 10, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 11, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 12, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 13, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 14, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 15, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 16, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 17, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 18, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 19, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 20, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 21, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 22, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 23, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 24, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 25, v___x_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 26, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 27, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*3 + 28, v_hasTrace_2543_);
v___x_2552_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_2540_);
if (lean_obj_tag(v___x_2552_) == 0)
{
lean_object* v_a_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; 
v_a_2553_ = lean_ctor_get(v___x_2552_, 0);
lean_inc(v_a_2553_);
lean_dec_ref_known(v___x_2552_, 1);
v___x_2554_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__1));
v___x_2555_ = l_Lean_Meta_getSimpExtension_x3f(v___x_2554_, v_a_2539_, v_a_2540_);
if (lean_obj_tag(v___x_2555_) == 0)
{
lean_object* v_a_2556_; lean_object* v___x_2557_; lean_object* v___f_2558_; lean_object* v___f_2559_; lean_object* v_ext_2561_; lean_object* v___y_2562_; lean_object* v___y_2563_; lean_object* v___y_2564_; lean_object* v___y_2565_; 
v_a_2556_ = lean_ctor_get(v___x_2555_, 0);
lean_inc(v_a_2556_);
lean_dec_ref_known(v___x_2555_, 1);
v___x_2557_ = lean_box(v_expensive_2535_);
v___f_2558_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__0___boxed), 11, 1);
lean_closure_set(v___f_2558_, 0, v___x_2557_);
v___f_2559_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__2));
if (lean_obj_tag(v_a_2556_) == 0)
{
lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v_a_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2619_; 
lean_dec_ref(v___f_2558_);
lean_dec(v_a_2553_);
lean_dec_ref_known(v___x_2551_, 3);
lean_dec(v_a_2545_);
v___x_2610_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__6, &lp_mathlib_Tactic_ReduceModChar_derive___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__6);
v___x_2611_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_2610_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
v_a_2612_ = lean_ctor_get(v___x_2611_, 0);
v_isSharedCheck_2619_ = !lean_is_exclusive(v___x_2611_);
if (v_isSharedCheck_2619_ == 0)
{
v___x_2614_ = v___x_2611_;
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_a_2612_);
lean_dec(v___x_2611_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2617_; 
if (v_isShared_2615_ == 0)
{
v___x_2617_ = v___x_2614_;
goto v_reusejp_2616_;
}
else
{
lean_object* v_reuseFailAlloc_2618_; 
v_reuseFailAlloc_2618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2618_, 0, v_a_2612_);
v___x_2617_ = v_reuseFailAlloc_2618_;
goto v_reusejp_2616_;
}
v_reusejp_2616_:
{
return v___x_2617_;
}
}
}
else
{
lean_object* v_val_2620_; 
v_val_2620_ = lean_ctor_get(v_a_2556_, 0);
lean_inc(v_val_2620_);
lean_dec_ref_known(v_a_2556_, 1);
v_ext_2561_ = v_val_2620_;
v___y_2562_ = v_a_2537_;
v___y_2563_ = v_a_2538_;
v___y_2564_ = v_a_2539_;
v___y_2565_ = v_a_2540_;
goto v___jp_2560_;
}
v___jp_2560_:
{
lean_object* v___x_2566_; 
v___x_2566_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_ext_2561_, v___y_2565_);
lean_dec_ref(v_ext_2561_);
if (lean_obj_tag(v___x_2566_) == 0)
{
lean_object* v_a_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; 
v_a_2567_ = lean_ctor_get(v___x_2566_, 0);
lean_inc(v_a_2567_);
lean_dec_ref_known(v___x_2566_, 1);
v___x_2568_ = lean_unsigned_to_nat(1u);
v___x_2569_ = lean_mk_empty_array_with_capacity(v___x_2568_);
v___x_2570_ = lean_array_push(v___x_2569_, v_a_2567_);
v___x_2571_ = l_Lean_Options_empty;
v___x_2572_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_2551_, v___x_2570_, v_a_2553_, v___x_2571_, v___y_2562_, v___y_2564_, v___y_2565_);
if (lean_obj_tag(v___x_2572_) == 0)
{
lean_object* v_a_2573_; lean_object* v___x_2574_; lean_object* v___f_2575_; lean_object* v___x_2576_; lean_object* v___f_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; 
v_a_2573_ = lean_ctor_get(v___x_2572_, 0);
lean_inc(v_a_2573_);
lean_dec_ref_known(v___x_2572_, 1);
v___x_2574_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0));
v___f_2575_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed), 11, 2);
lean_closure_set(v___f_2575_, 0, v___x_2574_);
lean_closure_set(v___f_2575_, 1, v___f_2558_);
v___x_2576_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__3));
v___f_2577_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__4));
lean_inc(v_a_2545_);
v___x_2578_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2578_, 0, v_a_2545_);
lean_ctor_set(v___x_2578_, 1, v___x_2550_);
lean_ctor_set_uint8(v___x_2578_, sizeof(void*)*2, v___x_2548_);
v___x_2579_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2));
v___x_2580_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9);
v___x_2581_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_2581_, 0, v___f_2575_);
lean_ctor_set(v___x_2581_, 1, v___x_2579_);
lean_ctor_set(v___x_2581_, 2, v___f_2577_);
lean_ctor_set(v___x_2581_, 3, v___f_2559_);
lean_ctor_set(v___x_2581_, 4, v___x_2576_);
lean_ctor_set_uint8(v___x_2581_, sizeof(void*)*5, v___x_2548_);
v___x_2582_ = l_Lean_Meta_Simp_main(v_a_2545_, v_a_2573_, v___x_2580_, v___x_2581_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_);
if (lean_obj_tag(v___x_2582_) == 0)
{
lean_object* v_a_2583_; lean_object* v_fst_2584_; lean_object* v___x_2585_; 
v_a_2583_ = lean_ctor_get(v___x_2582_, 0);
lean_inc(v_a_2583_);
lean_dec_ref_known(v___x_2582_, 1);
v_fst_2584_ = lean_ctor_get(v_a_2583_, 0);
lean_inc(v_fst_2584_);
lean_dec(v_a_2583_);
v___x_2585_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_2578_, v_fst_2584_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_);
return v___x_2585_;
}
else
{
lean_object* v_a_2586_; lean_object* v___x_2588_; uint8_t v_isShared_2589_; uint8_t v_isSharedCheck_2593_; 
lean_dec_ref_known(v___x_2578_, 2);
v_a_2586_ = lean_ctor_get(v___x_2582_, 0);
v_isSharedCheck_2593_ = !lean_is_exclusive(v___x_2582_);
if (v_isSharedCheck_2593_ == 0)
{
v___x_2588_ = v___x_2582_;
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
else
{
lean_inc(v_a_2586_);
lean_dec(v___x_2582_);
v___x_2588_ = lean_box(0);
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
v_resetjp_2587_:
{
lean_object* v___x_2591_; 
if (v_isShared_2589_ == 0)
{
v___x_2591_ = v___x_2588_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2592_; 
v_reuseFailAlloc_2592_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2592_, 0, v_a_2586_);
v___x_2591_ = v_reuseFailAlloc_2592_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
return v___x_2591_;
}
}
}
}
else
{
lean_object* v_a_2594_; lean_object* v___x_2596_; uint8_t v_isShared_2597_; uint8_t v_isSharedCheck_2601_; 
lean_dec_ref(v___f_2558_);
lean_dec(v_a_2545_);
v_a_2594_ = lean_ctor_get(v___x_2572_, 0);
v_isSharedCheck_2601_ = !lean_is_exclusive(v___x_2572_);
if (v_isSharedCheck_2601_ == 0)
{
v___x_2596_ = v___x_2572_;
v_isShared_2597_ = v_isSharedCheck_2601_;
goto v_resetjp_2595_;
}
else
{
lean_inc(v_a_2594_);
lean_dec(v___x_2572_);
v___x_2596_ = lean_box(0);
v_isShared_2597_ = v_isSharedCheck_2601_;
goto v_resetjp_2595_;
}
v_resetjp_2595_:
{
lean_object* v___x_2599_; 
if (v_isShared_2597_ == 0)
{
v___x_2599_ = v___x_2596_;
goto v_reusejp_2598_;
}
else
{
lean_object* v_reuseFailAlloc_2600_; 
v_reuseFailAlloc_2600_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2600_, 0, v_a_2594_);
v___x_2599_ = v_reuseFailAlloc_2600_;
goto v_reusejp_2598_;
}
v_reusejp_2598_:
{
return v___x_2599_;
}
}
}
}
else
{
lean_object* v_a_2602_; lean_object* v___x_2604_; uint8_t v_isShared_2605_; uint8_t v_isSharedCheck_2609_; 
lean_dec_ref(v___f_2558_);
lean_dec(v_a_2553_);
lean_dec_ref_known(v___x_2551_, 3);
lean_dec(v_a_2545_);
v_a_2602_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2609_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2609_ == 0)
{
v___x_2604_ = v___x_2566_;
v_isShared_2605_ = v_isSharedCheck_2609_;
goto v_resetjp_2603_;
}
else
{
lean_inc(v_a_2602_);
lean_dec(v___x_2566_);
v___x_2604_ = lean_box(0);
v_isShared_2605_ = v_isSharedCheck_2609_;
goto v_resetjp_2603_;
}
v_resetjp_2603_:
{
lean_object* v___x_2607_; 
if (v_isShared_2605_ == 0)
{
v___x_2607_ = v___x_2604_;
goto v_reusejp_2606_;
}
else
{
lean_object* v_reuseFailAlloc_2608_; 
v_reuseFailAlloc_2608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2608_, 0, v_a_2602_);
v___x_2607_ = v_reuseFailAlloc_2608_;
goto v_reusejp_2606_;
}
v_reusejp_2606_:
{
return v___x_2607_;
}
}
}
}
}
else
{
lean_object* v_a_2621_; lean_object* v___x_2623_; uint8_t v_isShared_2624_; uint8_t v_isSharedCheck_2628_; 
lean_dec(v_a_2553_);
lean_dec_ref_known(v___x_2551_, 3);
lean_dec(v_a_2545_);
v_a_2621_ = lean_ctor_get(v___x_2555_, 0);
v_isSharedCheck_2628_ = !lean_is_exclusive(v___x_2555_);
if (v_isSharedCheck_2628_ == 0)
{
v___x_2623_ = v___x_2555_;
v_isShared_2624_ = v_isSharedCheck_2628_;
goto v_resetjp_2622_;
}
else
{
lean_inc(v_a_2621_);
lean_dec(v___x_2555_);
v___x_2623_ = lean_box(0);
v_isShared_2624_ = v_isSharedCheck_2628_;
goto v_resetjp_2622_;
}
v_resetjp_2622_:
{
lean_object* v___x_2626_; 
if (v_isShared_2624_ == 0)
{
v___x_2626_ = v___x_2623_;
goto v_reusejp_2625_;
}
else
{
lean_object* v_reuseFailAlloc_2627_; 
v_reuseFailAlloc_2627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2627_, 0, v_a_2621_);
v___x_2626_ = v_reuseFailAlloc_2627_;
goto v_reusejp_2625_;
}
v_reusejp_2625_:
{
return v___x_2626_;
}
}
}
}
else
{
lean_object* v_a_2629_; lean_object* v___x_2631_; uint8_t v_isShared_2632_; uint8_t v_isSharedCheck_2636_; 
lean_dec_ref_known(v___x_2551_, 3);
lean_dec(v_a_2545_);
v_a_2629_ = lean_ctor_get(v___x_2552_, 0);
v_isSharedCheck_2636_ = !lean_is_exclusive(v___x_2552_);
if (v_isSharedCheck_2636_ == 0)
{
v___x_2631_ = v___x_2552_;
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
else
{
lean_inc(v_a_2629_);
lean_dec(v___x_2552_);
v___x_2631_ = lean_box(0);
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
v_resetjp_2630_:
{
lean_object* v___x_2634_; 
if (v_isShared_2632_ == 0)
{
v___x_2634_ = v___x_2631_;
goto v_reusejp_2633_;
}
else
{
lean_object* v_reuseFailAlloc_2635_; 
v_reuseFailAlloc_2635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2635_, 0, v_a_2629_);
v___x_2634_ = v_reuseFailAlloc_2635_;
goto v_reusejp_2633_;
}
v_reusejp_2633_:
{
return v___x_2634_;
}
}
}
}
else
{
lean_object* v_inheritedTraceOptions_2637_; lean_object* v___f_2638_; lean_object* v___x_2639_; lean_object* v___f_2640_; lean_object* v___f_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; uint8_t v___x_2645_; lean_object* v___y_2647_; lean_object* v___y_2648_; lean_object* v_a_2649_; lean_object* v___y_2659_; lean_object* v___y_2660_; lean_object* v_a_2661_; lean_object* v___y_2664_; lean_object* v___y_2665_; lean_object* v___y_2666_; lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v_a_2679_; lean_object* v___y_2692_; lean_object* v___y_2693_; lean_object* v_a_2694_; lean_object* v___y_2697_; lean_object* v___y_2698_; lean_object* v___y_2699_; 
v_inheritedTraceOptions_2637_ = lean_ctor_get(v_a_2539_, 13);
v___f_2638_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__2));
v___x_2639_ = lean_box(v_expensive_2535_);
v___f_2640_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__0___boxed), 11, 1);
lean_closure_set(v___f_2640_, 0, v___x_2639_);
lean_inc_ref(v_e_2536_);
v___f_2641_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__8___boxed), 7, 1);
lean_closure_set(v___f_2641_, 0, v_e_2536_);
v___x_2642_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__7));
v___x_2643_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__8));
v___x_2644_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__11, &lp_mathlib_Tactic_ReduceModChar_derive___closed__11_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__11);
v___x_2645_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2637_, v_options_2542_, v___x_2644_);
if (v___x_2645_ == 0)
{
lean_object* v___x_2755_; uint8_t v___x_2756_; 
v___x_2755_ = l_Lean_trace_profiler;
v___x_2756_ = lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(v_options_2542_, v___x_2755_);
if (v___x_2756_ == 0)
{
lean_object* v___x_2757_; lean_object* v_a_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; uint8_t v___x_2761_; lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; 
lean_dec_ref(v___f_2641_);
v___x_2757_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_2536_, v_a_2538_);
v_a_2758_ = lean_ctor_get(v___x_2757_, 0);
lean_inc(v_a_2758_);
lean_dec_ref(v___x_2757_);
v___x_2759_ = lean_unsigned_to_nat(100000u);
v___x_2760_ = lean_unsigned_to_nat(2u);
v___x_2761_ = 0;
v___x_2762_ = lean_box(0);
v___x_2763_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_2763_, 0, v___x_2759_);
lean_ctor_set(v___x_2763_, 1, v___x_2760_);
lean_ctor_set(v___x_2763_, 2, v___x_2762_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 1, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 2, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 3, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 4, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 5, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 6, v___x_2761_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 7, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 8, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 9, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 10, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 11, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 12, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 13, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 14, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 15, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 16, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 17, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 18, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 19, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 20, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 21, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 22, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 23, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 24, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 25, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 26, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 27, v___x_2756_);
lean_ctor_set_uint8(v___x_2763_, sizeof(void*)*3 + 28, v___x_2756_);
v___x_2764_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_2540_);
if (lean_obj_tag(v___x_2764_) == 0)
{
lean_object* v_a_2765_; lean_object* v___x_2766_; lean_object* v___x_2767_; 
v_a_2765_ = lean_ctor_get(v___x_2764_, 0);
lean_inc(v_a_2765_);
lean_dec_ref_known(v___x_2764_, 1);
v___x_2766_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__1));
v___x_2767_ = l_Lean_Meta_getSimpExtension_x3f(v___x_2766_, v_a_2539_, v_a_2540_);
if (lean_obj_tag(v___x_2767_) == 0)
{
lean_object* v_a_2768_; lean_object* v_ext_2770_; lean_object* v___y_2771_; lean_object* v___y_2772_; lean_object* v___y_2773_; lean_object* v___y_2774_; 
v_a_2768_ = lean_ctor_get(v___x_2767_, 0);
lean_inc(v_a_2768_);
lean_dec_ref_known(v___x_2767_, 1);
if (lean_obj_tag(v_a_2768_) == 0)
{
lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v_a_2822_; lean_object* v___x_2824_; uint8_t v_isShared_2825_; uint8_t v_isSharedCheck_2829_; 
lean_dec(v_a_2765_);
lean_dec_ref_known(v___x_2763_, 3);
lean_dec(v_a_2758_);
lean_dec_ref(v___f_2640_);
v___x_2820_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__6, &lp_mathlib_Tactic_ReduceModChar_derive___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__6);
v___x_2821_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_2820_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
v_a_2822_ = lean_ctor_get(v___x_2821_, 0);
v_isSharedCheck_2829_ = !lean_is_exclusive(v___x_2821_);
if (v_isSharedCheck_2829_ == 0)
{
v___x_2824_ = v___x_2821_;
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
else
{
lean_inc(v_a_2822_);
lean_dec(v___x_2821_);
v___x_2824_ = lean_box(0);
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
v_resetjp_2823_:
{
lean_object* v___x_2827_; 
if (v_isShared_2825_ == 0)
{
v___x_2827_ = v___x_2824_;
goto v_reusejp_2826_;
}
else
{
lean_object* v_reuseFailAlloc_2828_; 
v_reuseFailAlloc_2828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2828_, 0, v_a_2822_);
v___x_2827_ = v_reuseFailAlloc_2828_;
goto v_reusejp_2826_;
}
v_reusejp_2826_:
{
return v___x_2827_;
}
}
}
else
{
lean_object* v_val_2830_; 
v_val_2830_ = lean_ctor_get(v_a_2768_, 0);
lean_inc(v_val_2830_);
lean_dec_ref_known(v_a_2768_, 1);
v_ext_2770_ = v_val_2830_;
v___y_2771_ = v_a_2537_;
v___y_2772_ = v_a_2538_;
v___y_2773_ = v_a_2539_;
v___y_2774_ = v_a_2540_;
goto v___jp_2769_;
}
v___jp_2769_:
{
lean_object* v___x_2775_; 
v___x_2775_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_ext_2770_, v___y_2774_);
lean_dec_ref(v_ext_2770_);
if (lean_obj_tag(v___x_2775_) == 0)
{
lean_object* v_a_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; 
v_a_2776_ = lean_ctor_get(v___x_2775_, 0);
lean_inc(v_a_2776_);
lean_dec_ref_known(v___x_2775_, 1);
v___x_2777_ = lean_unsigned_to_nat(1u);
v___x_2778_ = lean_mk_empty_array_with_capacity(v___x_2777_);
v___x_2779_ = lean_array_push(v___x_2778_, v_a_2776_);
v___x_2780_ = l_Lean_Options_empty;
v___x_2781_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_2763_, v___x_2779_, v_a_2765_, v___x_2780_, v___y_2771_, v___y_2773_, v___y_2774_);
if (lean_obj_tag(v___x_2781_) == 0)
{
lean_object* v_a_2782_; lean_object* v___x_2783_; lean_object* v___f_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___f_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; 
v_a_2782_ = lean_ctor_get(v___x_2781_, 0);
lean_inc(v_a_2782_);
lean_dec_ref_known(v___x_2781_, 1);
v___x_2783_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__0));
v___f_2784_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__2___boxed), 11, 2);
lean_closure_set(v___f_2784_, 0, v___x_2783_);
lean_closure_set(v___f_2784_, 1, v___f_2640_);
v___x_2785_ = lean_box(v_hasTrace_2543_);
v___x_2786_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_NormNum_discharge___boxed), 11, 2);
lean_closure_set(v___x_2786_, 0, v___x_2783_);
lean_closure_set(v___x_2786_, 1, v___x_2785_);
v___f_2787_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__4));
lean_inc(v_a_2758_);
v___x_2788_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2788_, 0, v_a_2758_);
lean_ctor_set(v___x_2788_, 1, v___x_2762_);
lean_ctor_set_uint8(v___x_2788_, sizeof(void*)*2, v_hasTrace_2543_);
v___x_2789_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__2));
v___x_2790_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9, &lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___lam__6___closed__9);
v___x_2791_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_2791_, 0, v___f_2784_);
lean_ctor_set(v___x_2791_, 1, v___x_2789_);
lean_ctor_set(v___x_2791_, 2, v___f_2787_);
lean_ctor_set(v___x_2791_, 3, v___f_2638_);
lean_ctor_set(v___x_2791_, 4, v___x_2786_);
lean_ctor_set_uint8(v___x_2791_, sizeof(void*)*5, v_hasTrace_2543_);
v___x_2792_ = l_Lean_Meta_Simp_main(v_a_2758_, v_a_2782_, v___x_2790_, v___x_2791_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_);
if (lean_obj_tag(v___x_2792_) == 0)
{
lean_object* v_a_2793_; lean_object* v_fst_2794_; lean_object* v___x_2795_; 
v_a_2793_ = lean_ctor_get(v___x_2792_, 0);
lean_inc(v_a_2793_);
lean_dec_ref_known(v___x_2792_, 1);
v_fst_2794_ = lean_ctor_get(v_a_2793_, 0);
lean_inc(v_fst_2794_);
lean_dec(v_a_2793_);
v___x_2795_ = l_Lean_Meta_Simp_Result_mkEqTrans(v___x_2788_, v_fst_2794_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_);
return v___x_2795_;
}
else
{
lean_object* v_a_2796_; lean_object* v___x_2798_; uint8_t v_isShared_2799_; uint8_t v_isSharedCheck_2803_; 
lean_dec_ref_known(v___x_2788_, 2);
v_a_2796_ = lean_ctor_get(v___x_2792_, 0);
v_isSharedCheck_2803_ = !lean_is_exclusive(v___x_2792_);
if (v_isSharedCheck_2803_ == 0)
{
v___x_2798_ = v___x_2792_;
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
else
{
lean_inc(v_a_2796_);
lean_dec(v___x_2792_);
v___x_2798_ = lean_box(0);
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
v_resetjp_2797_:
{
lean_object* v___x_2801_; 
if (v_isShared_2799_ == 0)
{
v___x_2801_ = v___x_2798_;
goto v_reusejp_2800_;
}
else
{
lean_object* v_reuseFailAlloc_2802_; 
v_reuseFailAlloc_2802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2802_, 0, v_a_2796_);
v___x_2801_ = v_reuseFailAlloc_2802_;
goto v_reusejp_2800_;
}
v_reusejp_2800_:
{
return v___x_2801_;
}
}
}
}
else
{
lean_object* v_a_2804_; lean_object* v___x_2806_; uint8_t v_isShared_2807_; uint8_t v_isSharedCheck_2811_; 
lean_dec(v_a_2758_);
lean_dec_ref(v___f_2640_);
v_a_2804_ = lean_ctor_get(v___x_2781_, 0);
v_isSharedCheck_2811_ = !lean_is_exclusive(v___x_2781_);
if (v_isSharedCheck_2811_ == 0)
{
v___x_2806_ = v___x_2781_;
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
else
{
lean_inc(v_a_2804_);
lean_dec(v___x_2781_);
v___x_2806_ = lean_box(0);
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
v_resetjp_2805_:
{
lean_object* v___x_2809_; 
if (v_isShared_2807_ == 0)
{
v___x_2809_ = v___x_2806_;
goto v_reusejp_2808_;
}
else
{
lean_object* v_reuseFailAlloc_2810_; 
v_reuseFailAlloc_2810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2810_, 0, v_a_2804_);
v___x_2809_ = v_reuseFailAlloc_2810_;
goto v_reusejp_2808_;
}
v_reusejp_2808_:
{
return v___x_2809_;
}
}
}
}
else
{
lean_object* v_a_2812_; lean_object* v___x_2814_; uint8_t v_isShared_2815_; uint8_t v_isSharedCheck_2819_; 
lean_dec(v_a_2765_);
lean_dec_ref_known(v___x_2763_, 3);
lean_dec(v_a_2758_);
lean_dec_ref(v___f_2640_);
v_a_2812_ = lean_ctor_get(v___x_2775_, 0);
v_isSharedCheck_2819_ = !lean_is_exclusive(v___x_2775_);
if (v_isSharedCheck_2819_ == 0)
{
v___x_2814_ = v___x_2775_;
v_isShared_2815_ = v_isSharedCheck_2819_;
goto v_resetjp_2813_;
}
else
{
lean_inc(v_a_2812_);
lean_dec(v___x_2775_);
v___x_2814_ = lean_box(0);
v_isShared_2815_ = v_isSharedCheck_2819_;
goto v_resetjp_2813_;
}
v_resetjp_2813_:
{
lean_object* v___x_2817_; 
if (v_isShared_2815_ == 0)
{
v___x_2817_ = v___x_2814_;
goto v_reusejp_2816_;
}
else
{
lean_object* v_reuseFailAlloc_2818_; 
v_reuseFailAlloc_2818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2818_, 0, v_a_2812_);
v___x_2817_ = v_reuseFailAlloc_2818_;
goto v_reusejp_2816_;
}
v_reusejp_2816_:
{
return v___x_2817_;
}
}
}
}
}
else
{
lean_object* v_a_2831_; lean_object* v___x_2833_; uint8_t v_isShared_2834_; uint8_t v_isSharedCheck_2838_; 
lean_dec(v_a_2765_);
lean_dec_ref_known(v___x_2763_, 3);
lean_dec(v_a_2758_);
lean_dec_ref(v___f_2640_);
v_a_2831_ = lean_ctor_get(v___x_2767_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2767_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2833_ = v___x_2767_;
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
else
{
lean_inc(v_a_2831_);
lean_dec(v___x_2767_);
v___x_2833_ = lean_box(0);
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
v_resetjp_2832_:
{
lean_object* v___x_2836_; 
if (v_isShared_2834_ == 0)
{
v___x_2836_ = v___x_2833_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_a_2831_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
}
else
{
lean_object* v_a_2839_; lean_object* v___x_2841_; uint8_t v_isShared_2842_; uint8_t v_isSharedCheck_2846_; 
lean_dec_ref_known(v___x_2763_, 3);
lean_dec(v_a_2758_);
lean_dec_ref(v___f_2640_);
v_a_2839_ = lean_ctor_get(v___x_2764_, 0);
v_isSharedCheck_2846_ = !lean_is_exclusive(v___x_2764_);
if (v_isSharedCheck_2846_ == 0)
{
v___x_2841_ = v___x_2764_;
v_isShared_2842_ = v_isSharedCheck_2846_;
goto v_resetjp_2840_;
}
else
{
lean_inc(v_a_2839_);
lean_dec(v___x_2764_);
v___x_2841_ = lean_box(0);
v_isShared_2842_ = v_isSharedCheck_2846_;
goto v_resetjp_2840_;
}
v_resetjp_2840_:
{
lean_object* v___x_2844_; 
if (v_isShared_2842_ == 0)
{
v___x_2844_ = v___x_2841_;
goto v_reusejp_2843_;
}
else
{
lean_object* v_reuseFailAlloc_2845_; 
v_reuseFailAlloc_2845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2845_, 0, v_a_2839_);
v___x_2844_ = v_reuseFailAlloc_2845_;
goto v_reusejp_2843_;
}
v_reusejp_2843_:
{
return v___x_2844_;
}
}
}
}
else
{
goto v___jp_2709_;
}
}
else
{
goto v___jp_2709_;
}
v___jp_2646_:
{
lean_object* v___x_2650_; double v___x_2651_; double v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; 
v___x_2650_ = lean_io_get_num_heartbeats();
v___x_2651_ = lean_float_of_nat(v___y_2648_);
v___x_2652_ = lean_float_of_nat(v___x_2650_);
v___x_2653_ = lean_box_float(v___x_2651_);
v___x_2654_ = lean_box_float(v___x_2652_);
v___x_2655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2655_, 0, v___x_2653_);
lean_ctor_set(v___x_2655_, 1, v___x_2654_);
v___x_2656_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2656_, 0, v_a_2649_);
lean_ctor_set(v___x_2656_, 1, v___x_2655_);
v___x_2657_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3(v___x_2642_, v_hasTrace_2543_, v___x_2643_, v_options_2542_, v___x_2645_, v___y_2647_, v___f_2641_, v___x_2656_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
return v___x_2657_;
}
v___jp_2658_:
{
lean_object* v___x_2662_; 
v___x_2662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2662_, 0, v_a_2661_);
v___y_2647_ = v___y_2659_;
v___y_2648_ = v___y_2660_;
v_a_2649_ = v___x_2662_;
goto v___jp_2646_;
}
v___jp_2663_:
{
if (lean_obj_tag(v___y_2666_) == 0)
{
lean_object* v_a_2667_; lean_object* v___x_2669_; uint8_t v_isShared_2670_; uint8_t v_isSharedCheck_2674_; 
v_a_2667_ = lean_ctor_get(v___y_2666_, 0);
v_isSharedCheck_2674_ = !lean_is_exclusive(v___y_2666_);
if (v_isSharedCheck_2674_ == 0)
{
v___x_2669_ = v___y_2666_;
v_isShared_2670_ = v_isSharedCheck_2674_;
goto v_resetjp_2668_;
}
else
{
lean_inc(v_a_2667_);
lean_dec(v___y_2666_);
v___x_2669_ = lean_box(0);
v_isShared_2670_ = v_isSharedCheck_2674_;
goto v_resetjp_2668_;
}
v_resetjp_2668_:
{
lean_object* v___x_2672_; 
if (v_isShared_2670_ == 0)
{
lean_ctor_set_tag(v___x_2669_, 1);
v___x_2672_ = v___x_2669_;
goto v_reusejp_2671_;
}
else
{
lean_object* v_reuseFailAlloc_2673_; 
v_reuseFailAlloc_2673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2673_, 0, v_a_2667_);
v___x_2672_ = v_reuseFailAlloc_2673_;
goto v_reusejp_2671_;
}
v_reusejp_2671_:
{
v___y_2647_ = v___y_2664_;
v___y_2648_ = v___y_2665_;
v_a_2649_ = v___x_2672_;
goto v___jp_2646_;
}
}
}
else
{
lean_object* v_a_2675_; 
v_a_2675_ = lean_ctor_get(v___y_2666_, 0);
lean_inc(v_a_2675_);
lean_dec_ref_known(v___y_2666_, 1);
v___y_2659_ = v___y_2664_;
v___y_2660_ = v___y_2665_;
v_a_2661_ = v_a_2675_;
goto v___jp_2658_;
}
}
v___jp_2676_:
{
lean_object* v___x_2680_; double v___x_2681_; double v___x_2682_; double v___x_2683_; double v___x_2684_; double v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; 
v___x_2680_ = lean_io_mono_nanos_now();
v___x_2681_ = lean_float_of_nat(v___y_2678_);
v___x_2682_ = lean_float_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__12, &lp_mathlib_Tactic_ReduceModChar_derive___closed__12_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__12);
v___x_2683_ = lean_float_div(v___x_2681_, v___x_2682_);
v___x_2684_ = lean_float_of_nat(v___x_2680_);
v___x_2685_ = lean_float_div(v___x_2684_, v___x_2682_);
v___x_2686_ = lean_box_float(v___x_2683_);
v___x_2687_ = lean_box_float(v___x_2685_);
v___x_2688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2688_, 0, v___x_2686_);
lean_ctor_set(v___x_2688_, 1, v___x_2687_);
v___x_2689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2689_, 0, v_a_2679_);
lean_ctor_set(v___x_2689_, 1, v___x_2688_);
v___x_2690_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3(v___x_2642_, v_hasTrace_2543_, v___x_2643_, v_options_2542_, v___x_2645_, v___y_2677_, v___f_2641_, v___x_2689_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
return v___x_2690_;
}
v___jp_2691_:
{
lean_object* v___x_2695_; 
v___x_2695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2695_, 0, v_a_2694_);
v___y_2677_ = v___y_2692_;
v___y_2678_ = v___y_2693_;
v_a_2679_ = v___x_2695_;
goto v___jp_2676_;
}
v___jp_2696_:
{
if (lean_obj_tag(v___y_2699_) == 0)
{
lean_object* v_a_2700_; lean_object* v___x_2702_; uint8_t v_isShared_2703_; uint8_t v_isSharedCheck_2707_; 
v_a_2700_ = lean_ctor_get(v___y_2699_, 0);
v_isSharedCheck_2707_ = !lean_is_exclusive(v___y_2699_);
if (v_isSharedCheck_2707_ == 0)
{
v___x_2702_ = v___y_2699_;
v_isShared_2703_ = v_isSharedCheck_2707_;
goto v_resetjp_2701_;
}
else
{
lean_inc(v_a_2700_);
lean_dec(v___y_2699_);
v___x_2702_ = lean_box(0);
v_isShared_2703_ = v_isSharedCheck_2707_;
goto v_resetjp_2701_;
}
v_resetjp_2701_:
{
lean_object* v___x_2705_; 
if (v_isShared_2703_ == 0)
{
lean_ctor_set_tag(v___x_2702_, 1);
v___x_2705_ = v___x_2702_;
goto v_reusejp_2704_;
}
else
{
lean_object* v_reuseFailAlloc_2706_; 
v_reuseFailAlloc_2706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2706_, 0, v_a_2700_);
v___x_2705_ = v_reuseFailAlloc_2706_;
goto v_reusejp_2704_;
}
v_reusejp_2704_:
{
v___y_2677_ = v___y_2697_;
v___y_2678_ = v___y_2698_;
v_a_2679_ = v___x_2705_;
goto v___jp_2676_;
}
}
}
else
{
lean_object* v_a_2708_; 
v_a_2708_ = lean_ctor_get(v___y_2699_, 0);
lean_inc(v_a_2708_);
lean_dec_ref_known(v___y_2699_, 1);
v___y_2692_ = v___y_2697_;
v___y_2693_ = v___y_2698_;
v_a_2694_ = v_a_2708_;
goto v___jp_2691_;
}
}
v___jp_2709_:
{
lean_object* v___x_2710_; lean_object* v_a_2711_; lean_object* v___x_2712_; uint8_t v___x_2713_; 
v___x_2710_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Tactic_ReduceModChar_derive_spec__1___redArg(v_a_2540_);
v_a_2711_ = lean_ctor_get(v___x_2710_, 0);
lean_inc(v_a_2711_);
lean_dec_ref(v___x_2710_);
v___x_2712_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2713_ = lp_mathlib_Lean_Option_get___at___00Tactic_ReduceModChar_derive_spec__2(v_options_2542_, v___x_2712_);
if (v___x_2713_ == 0)
{
lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v_a_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; uint8_t v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; 
v___x_2714_ = lean_io_mono_nanos_now();
v___x_2715_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_2536_, v_a_2538_);
v_a_2716_ = lean_ctor_get(v___x_2715_, 0);
lean_inc(v_a_2716_);
lean_dec_ref(v___x_2715_);
v___x_2717_ = lean_unsigned_to_nat(100000u);
v___x_2718_ = lean_unsigned_to_nat(2u);
v___x_2719_ = 0;
v___x_2720_ = lean_box(0);
v___x_2721_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_2721_, 0, v___x_2717_);
lean_ctor_set(v___x_2721_, 1, v___x_2718_);
lean_ctor_set(v___x_2721_, 2, v___x_2720_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 1, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 2, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 3, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 4, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 5, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 6, v___x_2719_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 7, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 8, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 9, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 10, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 11, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 12, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 13, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 14, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 15, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 16, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 17, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 18, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 19, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 20, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 21, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 22, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 23, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 24, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 25, v_hasTrace_2543_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 26, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 27, v___x_2713_);
lean_ctor_set_uint8(v___x_2721_, sizeof(void*)*3 + 28, v___x_2713_);
v___x_2722_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_2540_);
if (lean_obj_tag(v___x_2722_) == 0)
{
lean_object* v_a_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; 
v_a_2723_ = lean_ctor_get(v___x_2722_, 0);
lean_inc(v_a_2723_);
lean_dec_ref_known(v___x_2722_, 1);
v___x_2724_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__1));
v___x_2725_ = l_Lean_Meta_getSimpExtension_x3f(v___x_2724_, v_a_2539_, v_a_2540_);
if (lean_obj_tag(v___x_2725_) == 0)
{
lean_object* v_a_2726_; 
v_a_2726_ = lean_ctor_get(v___x_2725_, 0);
lean_inc(v_a_2726_);
lean_dec_ref_known(v___x_2725_, 1);
if (lean_obj_tag(v_a_2726_) == 0)
{
lean_object* v___x_2727_; lean_object* v___x_2728_; lean_object* v_a_2729_; 
lean_dec(v_a_2723_);
lean_dec_ref_known(v___x_2721_, 3);
lean_dec(v_a_2716_);
lean_dec_ref(v___f_2640_);
v___x_2727_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__6, &lp_mathlib_Tactic_ReduceModChar_derive___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__6);
v___x_2728_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_2727_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
v_a_2729_ = lean_ctor_get(v___x_2728_, 0);
lean_inc(v_a_2729_);
lean_dec_ref(v___x_2728_);
v___y_2692_ = v_a_2711_;
v___y_2693_ = v___x_2714_;
v_a_2694_ = v_a_2729_;
goto v___jp_2691_;
}
else
{
lean_object* v_val_2730_; lean_object* v___x_2731_; 
v_val_2730_ = lean_ctor_get(v_a_2726_, 0);
lean_inc(v_val_2730_);
lean_dec_ref_known(v_a_2726_, 1);
v___x_2731_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__6(v___x_2721_, v_a_2723_, v___f_2640_, v_hasTrace_2543_, v_a_2716_, v___f_2638_, v_val_2730_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
lean_dec(v_val_2730_);
v___y_2697_ = v_a_2711_;
v___y_2698_ = v___x_2714_;
v___y_2699_ = v___x_2731_;
goto v___jp_2696_;
}
}
else
{
lean_object* v_a_2732_; 
lean_dec(v_a_2723_);
lean_dec_ref_known(v___x_2721_, 3);
lean_dec(v_a_2716_);
lean_dec_ref(v___f_2640_);
v_a_2732_ = lean_ctor_get(v___x_2725_, 0);
lean_inc(v_a_2732_);
lean_dec_ref_known(v___x_2725_, 1);
v___y_2692_ = v_a_2711_;
v___y_2693_ = v___x_2714_;
v_a_2694_ = v_a_2732_;
goto v___jp_2691_;
}
}
else
{
lean_object* v_a_2733_; 
lean_dec_ref_known(v___x_2721_, 3);
lean_dec(v_a_2716_);
lean_dec_ref(v___f_2640_);
v_a_2733_ = lean_ctor_get(v___x_2722_, 0);
lean_inc(v_a_2733_);
lean_dec_ref_known(v___x_2722_, 1);
v___y_2692_ = v_a_2711_;
v___y_2693_ = v___x_2714_;
v_a_2694_ = v_a_2733_;
goto v___jp_2691_;
}
}
else
{
lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v_a_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; uint8_t v___x_2739_; uint8_t v___x_2740_; lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; 
v___x_2734_ = lean_io_get_num_heartbeats();
v___x_2735_ = lp_mathlib_Lean_instantiateMVars___at___00Tactic_ReduceModChar_derive_spec__0___redArg(v_e_2536_, v_a_2538_);
v_a_2736_ = lean_ctor_get(v___x_2735_, 0);
lean_inc(v_a_2736_);
lean_dec_ref(v___x_2735_);
v___x_2737_ = lean_unsigned_to_nat(100000u);
v___x_2738_ = lean_unsigned_to_nat(2u);
v___x_2739_ = 0;
v___x_2740_ = 0;
v___x_2741_ = lean_box(0);
v___x_2742_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_2742_, 0, v___x_2737_);
lean_ctor_set(v___x_2742_, 1, v___x_2738_);
lean_ctor_set(v___x_2742_, 2, v___x_2741_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 1, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 2, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 3, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 4, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 5, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 6, v___x_2740_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 7, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 8, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 9, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 10, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 11, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 12, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 13, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 14, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 15, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 16, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 17, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 18, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 19, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 20, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 21, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 22, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 23, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 24, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 25, v___x_2713_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 26, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 27, v___x_2739_);
lean_ctor_set_uint8(v___x_2742_, sizeof(void*)*3 + 28, v___x_2739_);
v___x_2743_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_2540_);
if (lean_obj_tag(v___x_2743_) == 0)
{
lean_object* v_a_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; 
v_a_2744_ = lean_ctor_get(v___x_2743_, 0);
lean_inc(v_a_2744_);
lean_dec_ref_known(v___x_2743_, 1);
v___x_2745_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__1));
v___x_2746_ = l_Lean_Meta_getSimpExtension_x3f(v___x_2745_, v_a_2539_, v_a_2540_);
if (lean_obj_tag(v___x_2746_) == 0)
{
lean_object* v_a_2747_; 
v_a_2747_ = lean_ctor_get(v___x_2746_, 0);
lean_inc(v_a_2747_);
lean_dec_ref_known(v___x_2746_, 1);
if (lean_obj_tag(v_a_2747_) == 0)
{
lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v_a_2750_; 
lean_dec(v_a_2744_);
lean_dec_ref_known(v___x_2742_, 3);
lean_dec(v_a_2736_);
lean_dec_ref(v___f_2640_);
v___x_2748_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_derive___closed__6, &lp_mathlib_Tactic_ReduceModChar_derive___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_derive___closed__6);
v___x_2749_ = lp_mathlib_Lean_throwError___at___00Tactic_ReduceModChar_normBareNumeral_spec__0___redArg(v___x_2748_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
v_a_2750_ = lean_ctor_get(v___x_2749_, 0);
lean_inc(v_a_2750_);
lean_dec_ref(v___x_2749_);
v___y_2659_ = v_a_2711_;
v___y_2660_ = v___x_2734_;
v_a_2661_ = v_a_2750_;
goto v___jp_2658_;
}
else
{
lean_object* v_val_2751_; lean_object* v___x_2752_; 
v_val_2751_ = lean_ctor_get(v_a_2747_, 0);
lean_inc(v_val_2751_);
lean_dec_ref_known(v_a_2747_, 1);
v___x_2752_ = lp_mathlib_Tactic_ReduceModChar_derive___lam__7(v___x_2742_, v_a_2744_, v___f_2640_, v___x_2713_, v_a_2736_, v___f_2638_, v_val_2751_, v_a_2537_, v_a_2538_, v_a_2539_, v_a_2540_);
lean_dec(v_val_2751_);
v___y_2664_ = v_a_2711_;
v___y_2665_ = v___x_2734_;
v___y_2666_ = v___x_2752_;
goto v___jp_2663_;
}
}
else
{
lean_object* v_a_2753_; 
lean_dec(v_a_2744_);
lean_dec_ref_known(v___x_2742_, 3);
lean_dec(v_a_2736_);
lean_dec_ref(v___f_2640_);
v_a_2753_ = lean_ctor_get(v___x_2746_, 0);
lean_inc(v_a_2753_);
lean_dec_ref_known(v___x_2746_, 1);
v___y_2659_ = v_a_2711_;
v___y_2660_ = v___x_2734_;
v_a_2661_ = v_a_2753_;
goto v___jp_2658_;
}
}
else
{
lean_object* v_a_2754_; 
lean_dec_ref_known(v___x_2742_, 3);
lean_dec(v_a_2736_);
lean_dec_ref(v___f_2640_);
v_a_2754_ = lean_ctor_get(v___x_2743_, 0);
lean_inc(v_a_2754_);
lean_dec_ref_known(v___x_2743_, 1);
v___y_2659_ = v_a_2711_;
v___y_2660_ = v___x_2734_;
v_a_2661_ = v_a_2754_;
goto v___jp_2658_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar_derive___boxed(lean_object* v_expensive_2847_, lean_object* v_e_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_, lean_object* v_a_2852_, lean_object* v_a_2853_){
_start:
{
uint8_t v_expensive_boxed_2854_; lean_object* v_res_2855_; 
v_expensive_boxed_2854_ = lean_unbox(v_expensive_2847_);
v_res_2855_ = lp_mathlib_Tactic_ReduceModChar_derive(v_expensive_boxed_2854_, v_e_2848_, v_a_2849_, v_a_2850_, v_a_2851_, v_a_2852_);
lean_dec(v_a_2852_);
lean_dec_ref(v_a_2851_);
lean_dec(v_a_2850_);
lean_dec_ref(v_a_2849_);
return v_res_2855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4(lean_object* v_00_u03b1_2856_, lean_object* v_x_2857_, lean_object* v___y_2858_, lean_object* v___y_2859_, lean_object* v___y_2860_, lean_object* v___y_2861_){
_start:
{
lean_object* v___x_2863_; 
v___x_2863_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___redArg(v_x_2857_);
return v___x_2863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4___boxed(lean_object* v_00_u03b1_2864_, lean_object* v_x_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_, lean_object* v___y_2870_){
_start:
{
lean_object* v_res_2871_; 
v_res_2871_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Tactic_ReduceModChar_derive_spec__3_spec__4(v_00_u03b1_2864_, v_x_2865_, v___y_2866_, v___y_2867_, v___y_2868_, v___y_2869_);
lean_dec(v___y_2869_);
lean_dec_ref(v___y_2868_);
lean_dec(v___y_2867_);
lean_dec_ref(v___y_2866_);
return v_res_2871_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6(void){
_start:
{
lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; 
v___x_2885_ = l_Lean_Parser_Tactic_location;
v___x_2886_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__5));
v___x_2887_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2887_, 0, v___x_2886_);
lean_ctor_set(v___x_2887_, 1, v___x_2885_);
return v___x_2887_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7(void){
_start:
{
lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; 
v___x_2888_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6);
v___x_2889_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__3));
v___x_2890_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__2));
v___x_2891_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2890_);
lean_ctor_set(v___x_2891_, 1, v___x_2889_);
lean_ctor_set(v___x_2891_, 2, v___x_2888_);
return v___x_2891_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8(void){
_start:
{
lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; 
v___x_2892_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__7);
v___x_2893_ = lean_unsigned_to_nat(1022u);
v___x_2894_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0));
v___x_2895_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2895_, 0, v___x_2894_);
lean_ctor_set(v___x_2895_, 1, v___x_2893_);
lean_ctor_set(v___x_2895_, 2, v___x_2892_);
return v___x_2895_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char(void){
_start:
{
lean_object* v___x_2896_; 
v___x_2896_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__8);
return v___x_2896_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3(void){
_start:
{
lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2908_; 
v___x_2905_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__6);
v___x_2906_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__2));
v___x_2907_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__2));
v___x_2908_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2908_, 0, v___x_2907_);
lean_ctor_set(v___x_2908_, 1, v___x_2906_);
lean_ctor_set(v___x_2908_, 2, v___x_2905_);
return v___x_2908_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4(void){
_start:
{
lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; lean_object* v___x_2912_; 
v___x_2909_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__3);
v___x_2910_ = lean_unsigned_to_nat(1022u);
v___x_2911_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1));
v___x_2912_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2912_, 0, v___x_2911_);
lean_ctor_set(v___x_2912_, 1, v___x_2910_);
lean_ctor_set(v___x_2912_, 2, v___x_2909_);
return v___x_2912_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21(void){
_start:
{
lean_object* v___x_2913_; 
v___x_2913_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4, &lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4_once, _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__4);
return v___x_2913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0(lean_object* v_x_2914_, lean_object* v___y_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_, lean_object* v___y_2919_){
_start:
{
uint8_t v___x_2921_; lean_object* v___x_2922_; 
v___x_2921_ = 0;
v___x_2922_ = lp_mathlib_Tactic_ReduceModChar_derive(v___x_2921_, v_x_2914_, v___y_2916_, v___y_2917_, v___y_2918_, v___y_2919_);
return v___x_2922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0___boxed(lean_object* v_x_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_, lean_object* v___y_2928_, lean_object* v___y_2929_){
_start:
{
lean_object* v_res_2930_; 
v_res_2930_ = lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___lam__0(v_x_2923_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_, v___y_2928_);
lean_dec(v___y_2928_);
lean_dec_ref(v___y_2927_);
lean_dec(v___y_2926_);
lean_dec_ref(v___y_2925_);
lean_dec_ref(v___y_2924_);
return v_res_2930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1(lean_object* v_loc_2932_, lean_object* v_a_2933_, lean_object* v_a_2934_, lean_object* v_a_2935_, lean_object* v_a_2936_, lean_object* v_a_2937_, lean_object* v_a_2938_, lean_object* v_a_2939_, lean_object* v_a_2940_){
_start:
{
lean_object* v___f_2942_; lean_object* v___y_2944_; 
v___f_2942_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___closed__0));
if (lean_obj_tag(v_loc_2932_) == 0)
{
lean_object* v___x_2952_; 
v___x_2952_ = lean_box(0);
v___y_2944_ = v___x_2952_;
goto v___jp_2943_;
}
else
{
lean_object* v_val_2953_; lean_object* v___x_2955_; uint8_t v_isShared_2956_; uint8_t v_isSharedCheck_2960_; 
v_val_2953_ = lean_ctor_get(v_loc_2932_, 0);
v_isSharedCheck_2960_ = !lean_is_exclusive(v_loc_2932_);
if (v_isSharedCheck_2960_ == 0)
{
v___x_2955_ = v_loc_2932_;
v_isShared_2956_ = v_isSharedCheck_2960_;
goto v_resetjp_2954_;
}
else
{
lean_inc(v_val_2953_);
lean_dec(v_loc_2932_);
v___x_2955_ = lean_box(0);
v_isShared_2956_ = v_isSharedCheck_2960_;
goto v_resetjp_2954_;
}
v_resetjp_2954_:
{
lean_object* v___x_2958_; 
if (v_isShared_2956_ == 0)
{
v___x_2958_ = v___x_2955_;
goto v_reusejp_2957_;
}
else
{
lean_object* v_reuseFailAlloc_2959_; 
v_reuseFailAlloc_2959_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2959_, 0, v_val_2953_);
v___x_2958_ = v_reuseFailAlloc_2959_;
goto v_reusejp_2957_;
}
v_reusejp_2957_:
{
v___y_2944_ = v___x_2958_;
goto v___jp_2943_;
}
}
}
v___jp_2943_:
{
lean_object* v___x_2945_; lean_object* v_loc_2946_; lean_object* v___x_2947_; uint8_t v___x_2948_; uint8_t v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; 
v___x_2945_ = l_Lean_mkOptionalNode(v___y_2944_);
v_loc_2946_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_2945_);
lean_dec(v___x_2945_);
v___x_2947_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__0));
v___x_2948_ = 0;
v___x_2949_ = 0;
v___x_2950_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_2951_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_2942_, v___x_2947_, v_loc_2946_, v___x_2948_, v___x_2949_, v___x_2950_, v_a_2933_, v_a_2934_, v_a_2935_, v_a_2936_, v_a_2937_, v_a_2938_, v_a_2939_, v_a_2940_);
lean_dec(v_loc_2946_);
return v___x_2951_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___boxed(lean_object* v_loc_2961_, lean_object* v_a_2962_, lean_object* v_a_2963_, lean_object* v_a_2964_, lean_object* v_a_2965_, lean_object* v_a_2966_, lean_object* v_a_2967_, lean_object* v_a_2968_, lean_object* v_a_2969_, lean_object* v_a_2970_){
_start:
{
lean_object* v_res_2971_; 
v_res_2971_ = lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1(v_loc_2961_, v_a_2962_, v_a_2963_, v_a_2964_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_, v_a_2969_);
lean_dec(v_a_2969_);
lean_dec_ref(v_a_2968_);
lean_dec(v_a_2967_);
lean_dec_ref(v_a_2966_);
lean_dec(v_a_2965_);
lean_dec_ref(v_a_2964_);
lean_dec(v_a_2963_);
lean_dec_ref(v_a_2962_);
return v_res_2971_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2972_; lean_object* v___x_2973_; lean_object* v___x_2974_; 
v___x_2972_ = lean_box(0);
v___x_2973_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2974_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2974_, 0, v___x_2973_);
lean_ctor_set(v___x_2974_, 1, v___x_2972_);
return v___x_2974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2976_; lean_object* v___x_2977_; 
v___x_2976_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___closed__0);
v___x_2977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2977_, 0, v___x_2976_);
return v___x_2977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg___boxed(lean_object* v___y_2978_){
_start:
{
lean_object* v_res_2979_; 
v_res_2979_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v_res_2979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0(lean_object* v_00_u03b1_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_){
_start:
{
lean_object* v___x_2990_; 
v___x_2990_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v___x_2990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___boxed(lean_object* v_00_u03b1_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_){
_start:
{
lean_object* v_res_3001_; 
v_res_3001_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0(v_00_u03b1_2991_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_, v___y_2998_, v___y_2999_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
lean_dec(v___y_2997_);
lean_dec_ref(v___y_2996_);
lean_dec(v___y_2995_);
lean_dec_ref(v___y_2994_);
lean_dec(v___y_2993_);
lean_dec_ref(v___y_2992_);
return v_res_3001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0(uint8_t v___x_3002_, lean_object* v_x_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_){
_start:
{
lean_object* v___x_3010_; 
v___x_3010_ = lp_mathlib_Tactic_ReduceModChar_derive(v___x_3002_, v_x_3003_, v___y_3005_, v___y_3006_, v___y_3007_, v___y_3008_);
return v___x_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0___boxed(lean_object* v___x_3011_, lean_object* v_x_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_){
_start:
{
uint8_t v___x_721__boxed_3019_; lean_object* v_res_3020_; 
v___x_721__boxed_3019_ = lean_unbox(v___x_3011_);
v_res_3020_ = lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0(v___x_721__boxed_3019_, v_x_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_, v___y_3017_);
lean_dec(v___y_3017_);
lean_dec_ref(v___y_3016_);
lean_dec(v___y_3015_);
lean_dec_ref(v___y_3014_);
lean_dec_ref(v___y_3013_);
return v_res_3020_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0(void){
_start:
{
lean_object* v___x_3021_; lean_object* v___x_3022_; 
v___x_3021_ = lean_box(0);
v___x_3022_ = l_Lean_mkOptionalNode(v___x_3021_);
return v___x_3022_;
}
}
static lean_object* _init_lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1(void){
_start:
{
lean_object* v___x_3023_; lean_object* v_loc_3024_; 
v___x_3023_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0, &lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0_once, _init_lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__0);
v_loc_3024_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_3023_);
return v_loc_3024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1(lean_object* v_x_3025_, lean_object* v_a_3026_, lean_object* v_a_3027_, lean_object* v_a_3028_, lean_object* v_a_3029_, lean_object* v_a_3030_, lean_object* v_a_3031_, lean_object* v_a_3032_, lean_object* v_a_3033_){
_start:
{
lean_object* v___x_3035_; lean_object* v___x_3036_; uint8_t v___x_3037_; 
v___x_3035_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__0));
v___x_3036_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char___closed__0));
lean_inc(v_x_3025_);
v___x_3037_ = l_Lean_Syntax_isOfKind(v_x_3025_, v___x_3036_);
if (v___x_3037_ == 0)
{
lean_object* v___x_3038_; 
lean_dec(v_x_3025_);
v___x_3038_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v___x_3038_;
}
else
{
lean_object* v___x_3039_; lean_object* v___x_3040_; uint8_t v___x_3041_; 
v___x_3039_ = lean_unsigned_to_nat(1u);
v___x_3040_ = l_Lean_Syntax_getArg(v_x_3025_, v___x_3039_);
lean_dec(v_x_3025_);
v___x_3041_ = l_Lean_Syntax_isNone(v___x_3040_);
if (v___x_3041_ == 0)
{
uint8_t v___x_3042_; 
lean_inc(v___x_3040_);
v___x_3042_ = l_Lean_Syntax_matchesNull(v___x_3040_, v___x_3039_);
if (v___x_3042_ == 0)
{
lean_object* v___x_3043_; 
lean_dec(v___x_3040_);
v___x_3043_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v___x_3043_;
}
else
{
lean_object* v___x_3044_; lean_object* v___f_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; lean_object* v_loc_3050_; uint8_t v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; 
v___x_3044_ = lean_box(v___x_3041_);
v___f_3045_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_3045_, 0, v___x_3044_);
v___x_3046_ = lean_unsigned_to_nat(0u);
v___x_3047_ = l_Lean_Syntax_getArg(v___x_3040_, v___x_3046_);
lean_dec(v___x_3040_);
v___x_3048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3048_, 0, v___x_3047_);
v___x_3049_ = l_Lean_mkOptionalNode(v___x_3048_);
v_loc_3050_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_3049_);
lean_dec(v___x_3049_);
v___x_3051_ = 0;
v___x_3052_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3053_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_3045_, v___x_3035_, v_loc_3050_, v___x_3051_, v___x_3041_, v___x_3052_, v_a_3026_, v_a_3027_, v_a_3028_, v_a_3029_, v_a_3030_, v_a_3031_, v_a_3032_, v_a_3033_);
lean_dec(v_loc_3050_);
return v___x_3053_;
}
}
else
{
lean_object* v___f_3054_; lean_object* v_loc_3055_; uint8_t v___x_3056_; uint8_t v___x_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; 
lean_dec(v___x_3040_);
v___f_3054_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_unsafe__1___closed__0));
v_loc_3055_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1, &lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1_once, _init_lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1);
v___x_3056_ = 0;
v___x_3057_ = 0;
v___x_3058_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3059_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_3054_, v___x_3035_, v_loc_3055_, v___x_3056_, v___x_3057_, v___x_3058_, v_a_3026_, v_a_3027_, v_a_3028_, v_a_3029_, v_a_3030_, v_a_3031_, v_a_3032_, v_a_3033_);
return v___x_3059_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___boxed(lean_object* v_x_3060_, lean_object* v_a_3061_, lean_object* v_a_3062_, lean_object* v_a_3063_, lean_object* v_a_3064_, lean_object* v_a_3065_, lean_object* v_a_3066_, lean_object* v_a_3067_, lean_object* v_a_3068_, lean_object* v_a_3069_){
_start:
{
lean_object* v_res_3070_; 
v_res_3070_ = lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1(v_x_3060_, v_a_3061_, v_a_3062_, v_a_3063_, v_a_3064_, v_a_3065_, v_a_3066_, v_a_3067_, v_a_3068_);
lean_dec(v_a_3068_);
lean_dec_ref(v_a_3067_);
lean_dec(v_a_3066_);
lean_dec_ref(v_a_3065_);
lean_dec(v_a_3064_);
lean_dec_ref(v_a_3063_);
lean_dec(v_a_3062_);
lean_dec_ref(v_a_3061_);
return v_res_3070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0(lean_object* v_x_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_){
_start:
{
uint8_t v___x_3078_; lean_object* v___x_3079_; 
v___x_3078_ = 1;
v___x_3079_ = lp_mathlib_Tactic_ReduceModChar_derive(v___x_3078_, v_x_3071_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_);
return v___x_3079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0___boxed(lean_object* v_x_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_){
_start:
{
lean_object* v_res_3087_; 
v_res_3087_ = lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___lam__0(v_x_3080_, v___y_3081_, v___y_3082_, v___y_3083_, v___y_3084_, v___y_3085_);
lean_dec(v___y_3085_);
lean_dec_ref(v___y_3084_);
lean_dec(v___y_3083_);
lean_dec_ref(v___y_3082_);
lean_dec_ref(v___y_3081_);
return v_res_3087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1(lean_object* v_loc_3089_, lean_object* v_a_3090_, lean_object* v_a_3091_, lean_object* v_a_3092_, lean_object* v_a_3093_, lean_object* v_a_3094_, lean_object* v_a_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_){
_start:
{
lean_object* v___f_3099_; lean_object* v___y_3101_; 
v___f_3099_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___closed__0));
if (lean_obj_tag(v_loc_3089_) == 0)
{
lean_object* v___x_3109_; 
v___x_3109_ = lean_box(0);
v___y_3101_ = v___x_3109_;
goto v___jp_3100_;
}
else
{
lean_object* v_val_3110_; lean_object* v___x_3112_; uint8_t v_isShared_3113_; uint8_t v_isSharedCheck_3117_; 
v_val_3110_ = lean_ctor_get(v_loc_3089_, 0);
v_isSharedCheck_3117_ = !lean_is_exclusive(v_loc_3089_);
if (v_isSharedCheck_3117_ == 0)
{
v___x_3112_ = v_loc_3089_;
v_isShared_3113_ = v_isSharedCheck_3117_;
goto v_resetjp_3111_;
}
else
{
lean_inc(v_val_3110_);
lean_dec(v_loc_3089_);
v___x_3112_ = lean_box(0);
v_isShared_3113_ = v_isSharedCheck_3117_;
goto v_resetjp_3111_;
}
v_resetjp_3111_:
{
lean_object* v___x_3115_; 
if (v_isShared_3113_ == 0)
{
v___x_3115_ = v___x_3112_;
goto v_reusejp_3114_;
}
else
{
lean_object* v_reuseFailAlloc_3116_; 
v_reuseFailAlloc_3116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3116_, 0, v_val_3110_);
v___x_3115_ = v_reuseFailAlloc_3116_;
goto v_reusejp_3114_;
}
v_reusejp_3114_:
{
v___y_3101_ = v___x_3115_;
goto v___jp_3100_;
}
}
}
v___jp_3100_:
{
lean_object* v___x_3102_; lean_object* v_loc_3103_; lean_object* v___x_3104_; uint8_t v___x_3105_; uint8_t v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; 
v___x_3102_ = l_Lean_mkOptionalNode(v___y_3101_);
v_loc_3103_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_3102_);
lean_dec(v___x_3102_);
v___x_3104_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__0));
v___x_3105_ = 0;
v___x_3106_ = 0;
v___x_3107_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3108_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_3099_, v___x_3104_, v_loc_3103_, v___x_3105_, v___x_3106_, v___x_3107_, v_a_3090_, v_a_3091_, v_a_3092_, v_a_3093_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_);
lean_dec(v_loc_3103_);
return v___x_3108_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1___boxed(lean_object* v_loc_3118_, lean_object* v_a_3119_, lean_object* v_a_3120_, lean_object* v_a_3121_, lean_object* v_a_3122_, lean_object* v_a_3123_, lean_object* v_a_3124_, lean_object* v_a_3125_, lean_object* v_a_3126_, lean_object* v_a_3127_){
_start:
{
lean_object* v_res_3128_; 
v_res_3128_ = lp_mathlib___private_Mathlib_Tactic_ReduceModChar_0__Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1_unsafe__1(v_loc_3118_, v_a_3119_, v_a_3120_, v_a_3121_, v_a_3122_, v_a_3123_, v_a_3124_, v_a_3125_, v_a_3126_);
lean_dec(v_a_3126_);
lean_dec_ref(v_a_3125_);
lean_dec(v_a_3124_);
lean_dec_ref(v_a_3123_);
lean_dec(v_a_3122_);
lean_dec_ref(v_a_3121_);
lean_dec(v_a_3120_);
lean_dec_ref(v_a_3119_);
return v_res_3128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1(lean_object* v_x_3129_, lean_object* v_a_3130_, lean_object* v_a_3131_, lean_object* v_a_3132_, lean_object* v_a_3133_, lean_object* v_a_3134_, lean_object* v_a_3135_, lean_object* v_a_3136_, lean_object* v_a_3137_){
_start:
{
lean_object* v___x_3139_; uint8_t v___x_3140_; 
v___x_3139_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21___closed__1));
lean_inc(v_x_3129_);
v___x_3140_ = l_Lean_Syntax_isOfKind(v_x_3129_, v___x_3139_);
if (v___x_3140_ == 0)
{
lean_object* v___x_3141_; 
lean_dec(v_x_3129_);
v___x_3141_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v___x_3141_;
}
else
{
lean_object* v___x_3142_; lean_object* v___x_3143_; uint8_t v___x_3144_; 
v___x_3142_ = lean_unsigned_to_nat(1u);
v___x_3143_ = l_Lean_Syntax_getArg(v_x_3129_, v___x_3142_);
lean_dec(v_x_3129_);
v___x_3144_ = l_Lean_Syntax_isNone(v___x_3143_);
if (v___x_3144_ == 0)
{
uint8_t v___x_3145_; 
lean_inc(v___x_3143_);
v___x_3145_ = l_Lean_Syntax_matchesNull(v___x_3143_, v___x_3142_);
if (v___x_3145_ == 0)
{
lean_object* v___x_3146_; 
lean_dec(v___x_3143_);
v___x_3146_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1_spec__0___redArg();
return v___x_3146_;
}
else
{
lean_object* v___x_3147_; lean_object* v___f_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v___x_3151_; lean_object* v___x_3152_; lean_object* v_loc_3153_; lean_object* v___x_3154_; uint8_t v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; 
v___x_3147_ = lean_box(v___x_3145_);
v___f_3148_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_3148_, 0, v___x_3147_);
v___x_3149_ = lean_unsigned_to_nat(0u);
v___x_3150_ = l_Lean_Syntax_getArg(v___x_3143_, v___x_3149_);
lean_dec(v___x_3143_);
v___x_3151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3151_, 0, v___x_3150_);
v___x_3152_ = l_Lean_mkOptionalNode(v___x_3151_);
v_loc_3153_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_3152_);
lean_dec(v___x_3152_);
v___x_3154_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__0));
v___x_3155_ = 0;
v___x_3156_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3157_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_3148_, v___x_3154_, v_loc_3153_, v___x_3155_, v___x_3144_, v___x_3156_, v_a_3130_, v_a_3131_, v_a_3132_, v_a_3133_, v_a_3134_, v_a_3135_, v_a_3136_, v_a_3137_);
lean_dec(v_loc_3153_);
return v___x_3157_;
}
}
else
{
lean_object* v___x_3158_; lean_object* v___f_3159_; lean_object* v_loc_3160_; lean_object* v___x_3161_; uint8_t v___x_3162_; uint8_t v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; 
lean_dec(v___x_3143_);
v___x_3158_ = lean_box(v___x_3144_);
v___f_3159_ = lean_alloc_closure((void*)(lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_3159_, 0, v___x_3158_);
v_loc_3160_ = lean_obj_once(&lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1, &lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1_once, _init_lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char__1___closed__1);
v___x_3161_ = ((lean_object*)(lp_mathlib_Tactic_ReduceModChar_derive___closed__0));
v___x_3162_ = 0;
v___x_3163_ = 0;
v___x_3164_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_3165_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___f_3159_, v___x_3161_, v_loc_3160_, v___x_3162_, v___x_3163_, v___x_3164_, v_a_3130_, v_a_3131_, v_a_3132_, v_a_3133_, v_a_3134_, v_a_3135_, v_a_3136_, v_a_3137_);
return v___x_3165_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1___boxed(lean_object* v_x_3166_, lean_object* v_a_3167_, lean_object* v_a_3168_, lean_object* v_a_3169_, lean_object* v_a_3170_, lean_object* v_a_3171_, lean_object* v_a_3172_, lean_object* v_a_3173_, lean_object* v_a_3174_, lean_object* v_a_3175_){
_start:
{
lean_object* v_res_3176_; 
v_res_3176_ = lp_mathlib_Tactic_ReduceModChar___aux__Mathlib__Tactic__ReduceModChar______elabRules__Tactic__ReduceModChar__reduce__mod__char_x21__1(v_x_3166_, v_a_3167_, v_a_3168_, v_a_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
lean_dec(v_a_3174_);
lean_dec_ref(v_a_3173_);
lean_dec(v_a_3172_);
lean_dec_ref(v_a_3171_);
lean_dec(v_a_3170_);
lean_dec_ref(v_a_3169_);
lean_dec(v_a_3168_);
lean_dec_ref(v_a_3167_);
return v_res_3176_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ReduceModChar_Ext(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ReduceModChar(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ReduceModChar_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ReduceModChar(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Tactic_ReduceModChar_reduce__mod__char = _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char();
lean_mark_persistent(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char);
lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21 = _init_lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21();
lean_mark_persistent(lp_mathlib_Tactic_ReduceModChar_reduce__mod__char_x21);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ReduceModChar_Ext(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ReduceModChar(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_PowMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ReduceModChar_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ReduceModChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ReduceModChar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ReduceModChar(builtin);
}
#ifdef __cplusplus
}
#endif
