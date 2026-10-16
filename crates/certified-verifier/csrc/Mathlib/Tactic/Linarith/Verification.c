// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Verification
// Imports: public import Init public meta import Init public meta import Mathlib.Util.Qq public meta import Mathlib.Tactic.Linarith.Datatypes public import Mathlib.Tactic.Linarith.Parsing
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_decEq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lp_mathlib_Lean_Expr_ineq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MVarId_rewrite(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_eraseDupsBy___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* lp_mathlib_synthesizeUsing_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipIdxTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Ineq_toString(uint8_t);
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__0 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__0_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__1 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__1_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__1_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__2 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__2_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__3 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__3_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__4;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__5 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__5_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__6 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__6_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__5_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__7 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__7_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__8 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__8_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__9 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__9_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__9_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__10 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__10_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "instMulZeroClassOfSemiring"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__11 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__11_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__11_value),LEAN_SCALAR_PTR_LITERAL(31, 133, 13, 57, 152, 228, 72, 248)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__12 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__12_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__13 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__13_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__14;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__15 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__15_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__16 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__16_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__15_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__16_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__17 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__17_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__18 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__18_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__19 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__19_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__18_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__19_value),LEAN_SCALAR_PTR_LITERAL(52, 219, 71, 246, 148, 114, 208, 126)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__20 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__20_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__21 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__21_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__22 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__22_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__21_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__22_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__23 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__23_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__24 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__24_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__25 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__25_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__24_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__25_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__26 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__26_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__27 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__27_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__28 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__28_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__27_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__28_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__29 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__29_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "instOfNatAtLeastTwo"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__30 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__30_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__30_value),LEAN_SCALAR_PTR_LITERAL(223, 182, 28, 70, 145, 92, 58, 230)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__31 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__31_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__32 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__32_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__18_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__32_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__33 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__33_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__34 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__34_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "instAtLeastTwoHAddOfNat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__35 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__35_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__34_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__35_value),LEAN_SCALAR_PTR_LITERAL(33, 122, 138, 72, 117, 62, 160, 52)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__36 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__36_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__37;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__38 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__38_value;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__39 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__39_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__38_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__39_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__40 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__40_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__41 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__41_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__41_value)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__42 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__42_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__42_value)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__43 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__43_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__44;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__34_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__45 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__45_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__46;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__47;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__48;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__49;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__50 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__50_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__50_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__51 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__51_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__52;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__53;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instAddNat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__54 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__54_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__54_value),LEAN_SCALAR_PTR_LITERAL(228, 164, 175, 25, 228, 165, 175, 183)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__55 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__55_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__56;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__57;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__58;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__59;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__60;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__61;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__62 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__62_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__62_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__63 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__63_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__64;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__65;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__66;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "instNeZeroNatHAdd_1"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__67 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__67_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__67_value),LEAN_SCALAR_PTR_LITERAL(134, 229, 97, 126, 81, 20, 155, 10)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__68 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__68_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__69;
static const lean_string_object lp_mathlib_Qq_ofNatQ___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instNeZeroSucc"};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__70 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__70_value;
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__34_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Qq_ofNatQ___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__71_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__70_value),LEAN_SCALAR_PTR_LITERAL(163, 205, 35, 215, 215, 220, 7, 150)}};
static const lean_object* lp_mathlib_Qq_ofNatQ___closed__71 = (const lean_object*)&lp_mathlib_Qq_ofNatQ___closed__71_value;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__72;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__73;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__74;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__75;
static lean_once_cell_t lp_mathlib_Qq_ofNatQ___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_ofNatQ___closed__76;
LEAN_EXPORT lean_object* lp_mathlib_Qq_ofNatQ(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__1_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__8_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__27_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddSemigroup"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toAdd"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 220, 39, 90, 153, 217, 67, 109)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__1_value),LEAN_SCALAR_PTR_LITERAL(229, 213, 77, 96, 176, 6, 135, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddMonoid"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddSemigroup"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__4_value),LEAN_SCALAR_PTR_LITERAL(85, 99, 140, 61, 118, 63, 153, 27)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "AddZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(171, 135, 49, 0, 6, 244, 57, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__9_value),LEAN_SCALAR_PTR_LITERAL(87, 27, 84, 210, 142, 102, 48, 129)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toAddZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(157, 204, 59, 233, 207, 78, 141, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__3_value),LEAN_SCALAR_PTR_LITERAL(64, 236, 134, 119, 35, 182, 73, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 217, 102, 131, 1, 241, 19, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addExprs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__3_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addExprs___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "eq_of_eq_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__3_value),LEAN_SCALAR_PTR_LITERAL(136, 216, 176, 65, 1, 189, 46, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "le_of_eq_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 124, 126, 6, 224, 71, 46, 181)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "lt_of_eq_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__7_value),LEAN_SCALAR_PTR_LITERAL(143, 171, 112, 109, 186, 14, 142, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "le_of_le_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__9_value),LEAN_SCALAR_PTR_LITERAL(128, 130, 121, 55, 104, 142, 71, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "add_nonpos"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__11_value),LEAN_SCALAR_PTR_LITERAL(72, 247, 125, 119, 168, 109, 211, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "add_lt_of_le_of_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(35, 71, 228, 114, 30, 117, 205, 193)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "lt_of_lt_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__15_value),LEAN_SCALAR_PTR_LITERAL(45, 208, 247, 204, 42, 123, 116, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "add_lt_of_neg_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__17_value),LEAN_SCALAR_PTR_LITERAL(112, 142, 30, 46, 249, 159, 150, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "add_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__19_value),LEAN_SCALAR_PTR_LITERAL(65, 192, 248, 12, 140, 133, 173, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "no linear hypotheses found"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "zero_lt_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__0_value),LEAN_SCALAR_PTR_LITERAL(242, 73, 78, 209, 90, 120, 122, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "neg_neg_of_pos"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 190, 95, 20, 204, 168, 75, 189)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "neg_eq_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(36, 31, 137, 23, 233, 15, 148, 79)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(14, 81, 9, 215, 230, 198, 87, 3)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_ofNatQ___closed__5_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__1_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0;
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linarith"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "detail"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(29, 12, 183, 160, 66, 250, 13, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decEq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8___closed__0 = (const lean_object*)&lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = " Invoking oracle"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = " Building final expression"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "linarith failed to find a contradiction"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__1_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 66, 254, 161, 210, 133, 94, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "lt_irrefl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__17(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__0_value;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__0_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1_value;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__0_value;
static const lean_string_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__1_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__0_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5 = (const lean_object*)&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__0 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__0_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__0_value)}};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__1 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__1_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__2 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__2_value;
static const lean_string_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__3 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__3_value;
static const lean_string_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__4 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__4_value;
static lean_once_cell_t lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5;
static lean_once_cell_t lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6;
static const lean_ctor_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__3_value)}};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__7 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__7_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__4_value)}};
static const lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__8 = (const lean_object*)&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14(lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__0_value;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__0_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___closed__0 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "no args to linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "proveFalseByLinarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__2_value),LEAN_SCALAR_PTR_LITERAL(16, 8, 30, 179, 72, 55, 139, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "proveEqZeroUsing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "mkLTZeroProof"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Linarith.lt_irrefl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "addNegEqProofs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "mkNegOneLtZeroProof"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__9_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__10_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "\nshould be both 0 and negative"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "found a contradiction: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "linearFormsAndMaxVar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "comps:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "inputs:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__18(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__4(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__3));
v___x_9_ = l_Lean_Expr_lit___override(v___x_8_);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__14(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__13));
v___x_26_ = l_Lean_Expr_lit___override(v___x_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__37(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = lean_box(0);
v___x_65_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__36));
v___x_66_ = l_Lean_Expr_const___override(v___x_65_, v___x_64_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__44(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__43));
v___x_82_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__40));
v___x_83_ = l_Lean_Expr_const___override(v___x_82_, v___x_81_);
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__46(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_86_ = lean_box(0);
v___x_87_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__45));
v___x_88_ = l_Lean_Expr_const___override(v___x_87_, v___x_86_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__47(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__46, &lp_mathlib_Qq_ofNatQ___closed__46_once, _init_lp_mathlib_Qq_ofNatQ___closed__46);
v___x_90_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__44, &lp_mathlib_Qq_ofNatQ___closed__44_once, _init_lp_mathlib_Qq_ofNatQ___closed__44);
v___x_91_ = l_Lean_Expr_app___override(v___x_90_, v___x_89_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__48(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__46, &lp_mathlib_Qq_ofNatQ___closed__46_once, _init_lp_mathlib_Qq_ofNatQ___closed__46);
v___x_93_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__47, &lp_mathlib_Qq_ofNatQ___closed__47_once, _init_lp_mathlib_Qq_ofNatQ___closed__47);
v___x_94_ = l_Lean_Expr_app___override(v___x_93_, v___x_92_);
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__49(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__46, &lp_mathlib_Qq_ofNatQ___closed__46_once, _init_lp_mathlib_Qq_ofNatQ___closed__46);
v___x_96_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__48, &lp_mathlib_Qq_ofNatQ___closed__48_once, _init_lp_mathlib_Qq_ofNatQ___closed__48);
v___x_97_ = l_Lean_Expr_app___override(v___x_96_, v___x_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__52(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_101_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__41));
v___x_102_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__51));
v___x_103_ = l_Lean_Expr_const___override(v___x_102_, v___x_101_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__53(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__46, &lp_mathlib_Qq_ofNatQ___closed__46_once, _init_lp_mathlib_Qq_ofNatQ___closed__46);
v___x_105_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__52, &lp_mathlib_Qq_ofNatQ___closed__52_once, _init_lp_mathlib_Qq_ofNatQ___closed__52);
v___x_106_ = l_Lean_Expr_app___override(v___x_105_, v___x_104_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__56(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_box(0);
v___x_111_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__55));
v___x_112_ = l_Lean_Expr_const___override(v___x_111_, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__57(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__56, &lp_mathlib_Qq_ofNatQ___closed__56_once, _init_lp_mathlib_Qq_ofNatQ___closed__56);
v___x_114_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__53, &lp_mathlib_Qq_ofNatQ___closed__53_once, _init_lp_mathlib_Qq_ofNatQ___closed__53);
v___x_115_ = l_Lean_Expr_app___override(v___x_114_, v___x_113_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__58(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_116_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__57, &lp_mathlib_Qq_ofNatQ___closed__57_once, _init_lp_mathlib_Qq_ofNatQ___closed__57);
v___x_117_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__49, &lp_mathlib_Qq_ofNatQ___closed__49_once, _init_lp_mathlib_Qq_ofNatQ___closed__49);
v___x_118_ = l_Lean_Expr_app___override(v___x_117_, v___x_116_);
return v___x_118_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__59(void){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_119_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__41));
v___x_120_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
v___x_121_ = l_Lean_Expr_const___override(v___x_120_, v___x_119_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__60(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_122_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__46, &lp_mathlib_Qq_ofNatQ___closed__46_once, _init_lp_mathlib_Qq_ofNatQ___closed__46);
v___x_123_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__59, &lp_mathlib_Qq_ofNatQ___closed__59_once, _init_lp_mathlib_Qq_ofNatQ___closed__59);
v___x_124_ = l_Lean_Expr_app___override(v___x_123_, v___x_122_);
return v___x_124_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__61(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__14, &lp_mathlib_Qq_ofNatQ___closed__14_once, _init_lp_mathlib_Qq_ofNatQ___closed__14);
v___x_126_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__60, &lp_mathlib_Qq_ofNatQ___closed__60_once, _init_lp_mathlib_Qq_ofNatQ___closed__60);
v___x_127_ = l_Lean_Expr_app___override(v___x_126_, v___x_125_);
return v___x_127_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__64(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = lean_box(0);
v___x_132_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__63));
v___x_133_ = l_Lean_Expr_const___override(v___x_132_, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__65(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__14, &lp_mathlib_Qq_ofNatQ___closed__14_once, _init_lp_mathlib_Qq_ofNatQ___closed__14);
v___x_135_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__64, &lp_mathlib_Qq_ofNatQ___closed__64_once, _init_lp_mathlib_Qq_ofNatQ___closed__64);
v___x_136_ = l_Lean_Expr_app___override(v___x_135_, v___x_134_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__66(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__65, &lp_mathlib_Qq_ofNatQ___closed__65_once, _init_lp_mathlib_Qq_ofNatQ___closed__65);
v___x_138_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__61, &lp_mathlib_Qq_ofNatQ___closed__61_once, _init_lp_mathlib_Qq_ofNatQ___closed__61);
v___x_139_ = l_Lean_Expr_app___override(v___x_138_, v___x_137_);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__69(void){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_143_ = lean_box(0);
v___x_144_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__68));
v___x_145_ = l_Lean_Expr_const___override(v___x_144_, v___x_143_);
return v___x_145_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__72(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_150_ = lean_box(0);
v___x_151_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__71));
v___x_152_ = l_Lean_Expr_const___override(v___x_151_, v___x_150_);
return v___x_152_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__73(void){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_153_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__4, &lp_mathlib_Qq_ofNatQ___closed__4_once, _init_lp_mathlib_Qq_ofNatQ___closed__4);
v___x_154_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__60, &lp_mathlib_Qq_ofNatQ___closed__60_once, _init_lp_mathlib_Qq_ofNatQ___closed__60);
v___x_155_ = l_Lean_Expr_app___override(v___x_154_, v___x_153_);
return v___x_155_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__74(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_156_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__4, &lp_mathlib_Qq_ofNatQ___closed__4_once, _init_lp_mathlib_Qq_ofNatQ___closed__4);
v___x_157_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__64, &lp_mathlib_Qq_ofNatQ___closed__64_once, _init_lp_mathlib_Qq_ofNatQ___closed__64);
v___x_158_ = l_Lean_Expr_app___override(v___x_157_, v___x_156_);
return v___x_158_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__75(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__74, &lp_mathlib_Qq_ofNatQ___closed__74_once, _init_lp_mathlib_Qq_ofNatQ___closed__74);
v___x_160_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__73, &lp_mathlib_Qq_ofNatQ___closed__73_once, _init_lp_mathlib_Qq_ofNatQ___closed__73);
v___x_161_ = l_Lean_Expr_app___override(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Qq_ofNatQ___closed__76(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__75, &lp_mathlib_Qq_ofNatQ___closed__75_once, _init_lp_mathlib_Qq_ofNatQ___closed__75);
v___x_163_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__72, &lp_mathlib_Qq_ofNatQ___closed__72_once, _init_lp_mathlib_Qq_ofNatQ___closed__72);
v___x_164_ = l_Lean_Expr_app___override(v___x_163_, v___x_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_ofNatQ(lean_object* v_u_165_, lean_object* v_00_u03b1_166_, lean_object* v_x_167_, lean_object* v_n_168_){
_start:
{
lean_object* v_zero_169_; uint8_t v_isZero_170_; 
v_zero_169_ = lean_unsigned_to_nat(0u);
v_isZero_170_ = lean_nat_dec_eq(v_n_168_, v_zero_169_);
if (v_isZero_170_ == 1)
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
lean_dec(v_n_168_);
v___x_171_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
v___x_172_ = lean_box(0);
v___x_173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_173_, 0, v_u_165_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
lean_inc_ref_n(v___x_173_, 3);
v___x_174_ = l_Lean_Expr_const___override(v___x_171_, v___x_173_);
lean_inc_ref_n(v_00_u03b1_166_, 3);
v___x_175_ = l_Lean_Expr_app___override(v___x_174_, v_00_u03b1_166_);
v___x_176_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__4, &lp_mathlib_Qq_ofNatQ___closed__4_once, _init_lp_mathlib_Qq_ofNatQ___closed__4);
v___x_177_ = l_Lean_Expr_app___override(v___x_175_, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__7));
v___x_179_ = l_Lean_Expr_const___override(v___x_178_, v___x_173_);
v___x_180_ = l_Lean_Expr_app___override(v___x_179_, v_00_u03b1_166_);
v___x_181_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__10));
v___x_182_ = l_Lean_Expr_const___override(v___x_181_, v___x_173_);
v___x_183_ = l_Lean_Expr_app___override(v___x_182_, v_00_u03b1_166_);
v___x_184_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__12));
v___x_185_ = l_Lean_Expr_const___override(v___x_184_, v___x_173_);
v___x_186_ = l_Lean_Expr_app___override(v___x_185_, v_00_u03b1_166_);
v___x_187_ = l_Lean_Expr_app___override(v___x_186_, v_x_167_);
v___x_188_ = l_Lean_Expr_app___override(v___x_183_, v___x_187_);
v___x_189_ = l_Lean_Expr_app___override(v___x_180_, v___x_188_);
v___x_190_ = l_Lean_Expr_app___override(v___x_177_, v___x_189_);
return v___x_190_;
}
else
{
lean_object* v_one_191_; lean_object* v_n_192_; uint8_t v_isZero_193_; 
v_one_191_ = lean_unsigned_to_nat(1u);
v_n_192_ = lean_nat_sub(v_n_168_, v_one_191_);
v_isZero_193_ = lean_nat_dec_eq(v_n_192_, v_zero_169_);
if (v_isZero_193_ == 1)
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
lean_dec(v_n_192_);
lean_dec(v_n_168_);
v___x_194_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
v___x_195_ = lean_box(0);
v___x_196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_196_, 0, v_u_165_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
lean_inc_ref_n(v___x_196_, 5);
v___x_197_ = l_Lean_Expr_const___override(v___x_194_, v___x_196_);
lean_inc_ref_n(v_00_u03b1_166_, 5);
v___x_198_ = l_Lean_Expr_app___override(v___x_197_, v_00_u03b1_166_);
v___x_199_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__14, &lp_mathlib_Qq_ofNatQ___closed__14_once, _init_lp_mathlib_Qq_ofNatQ___closed__14);
v___x_200_ = l_Lean_Expr_app___override(v___x_198_, v___x_199_);
v___x_201_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__17));
v___x_202_ = l_Lean_Expr_const___override(v___x_201_, v___x_196_);
v___x_203_ = l_Lean_Expr_app___override(v___x_202_, v_00_u03b1_166_);
v___x_204_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__20));
v___x_205_ = l_Lean_Expr_const___override(v___x_204_, v___x_196_);
v___x_206_ = l_Lean_Expr_app___override(v___x_205_, v_00_u03b1_166_);
v___x_207_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__23));
v___x_208_ = l_Lean_Expr_const___override(v___x_207_, v___x_196_);
v___x_209_ = l_Lean_Expr_app___override(v___x_208_, v_00_u03b1_166_);
v___x_210_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__26));
v___x_211_ = l_Lean_Expr_const___override(v___x_210_, v___x_196_);
v___x_212_ = l_Lean_Expr_app___override(v___x_211_, v_00_u03b1_166_);
v___x_213_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__29));
v___x_214_ = l_Lean_Expr_const___override(v___x_213_, v___x_196_);
v___x_215_ = l_Lean_Expr_app___override(v___x_214_, v_00_u03b1_166_);
v___x_216_ = l_Lean_Expr_app___override(v___x_215_, v_x_167_);
v___x_217_ = l_Lean_Expr_app___override(v___x_212_, v___x_216_);
v___x_218_ = l_Lean_Expr_app___override(v___x_209_, v___x_217_);
v___x_219_ = l_Lean_Expr_app___override(v___x_206_, v___x_218_);
v___x_220_ = l_Lean_Expr_app___override(v___x_203_, v___x_219_);
v___x_221_ = l_Lean_Expr_app___override(v___x_200_, v___x_220_);
return v___x_221_;
}
else
{
lean_object* v_n_222_; lean_object* v_lit_223_; lean_object* v_k_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v_n_222_ = lean_nat_sub(v_n_192_, v_one_191_);
lean_dec(v_n_192_);
v_lit_223_ = l_Lean_mkRawNatLit(v_n_168_);
v_k_224_ = l_Lean_mkRawNatLit(v_n_222_);
v___x_225_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
v___x_226_ = lean_box(0);
v___x_227_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_227_, 0, v_u_165_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
lean_inc_ref_n(v___x_227_, 5);
v___x_228_ = l_Lean_Expr_const___override(v___x_225_, v___x_227_);
lean_inc_ref_n(v_00_u03b1_166_, 5);
v___x_229_ = l_Lean_Expr_app___override(v___x_228_, v_00_u03b1_166_);
lean_inc_ref(v_lit_223_);
v___x_230_ = l_Lean_Expr_app___override(v___x_229_, v_lit_223_);
v___x_231_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__31));
v___x_232_ = l_Lean_Expr_const___override(v___x_231_, v___x_227_);
v___x_233_ = l_Lean_Expr_app___override(v___x_232_, v_00_u03b1_166_);
v___x_234_ = l_Lean_Expr_app___override(v___x_233_, v_lit_223_);
v___x_235_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__33));
v___x_236_ = l_Lean_Expr_const___override(v___x_235_, v___x_227_);
v___x_237_ = l_Lean_Expr_app___override(v___x_236_, v_00_u03b1_166_);
v___x_238_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__23));
v___x_239_ = l_Lean_Expr_const___override(v___x_238_, v___x_227_);
v___x_240_ = l_Lean_Expr_app___override(v___x_239_, v_00_u03b1_166_);
v___x_241_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__26));
v___x_242_ = l_Lean_Expr_const___override(v___x_241_, v___x_227_);
v___x_243_ = l_Lean_Expr_app___override(v___x_242_, v_00_u03b1_166_);
v___x_244_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__29));
v___x_245_ = l_Lean_Expr_const___override(v___x_244_, v___x_227_);
v___x_246_ = l_Lean_Expr_app___override(v___x_245_, v_00_u03b1_166_);
v___x_247_ = l_Lean_Expr_app___override(v___x_246_, v_x_167_);
v___x_248_ = l_Lean_Expr_app___override(v___x_243_, v___x_247_);
v___x_249_ = l_Lean_Expr_app___override(v___x_240_, v___x_248_);
v___x_250_ = l_Lean_Expr_app___override(v___x_237_, v___x_249_);
v___x_251_ = l_Lean_Expr_app___override(v___x_234_, v___x_250_);
v___x_252_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__37, &lp_mathlib_Qq_ofNatQ___closed__37_once, _init_lp_mathlib_Qq_ofNatQ___closed__37);
v___x_253_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__58, &lp_mathlib_Qq_ofNatQ___closed__58_once, _init_lp_mathlib_Qq_ofNatQ___closed__58);
lean_inc_ref(v_k_224_);
v___x_254_ = l_Lean_Expr_app___override(v___x_253_, v_k_224_);
v___x_255_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__66, &lp_mathlib_Qq_ofNatQ___closed__66_once, _init_lp_mathlib_Qq_ofNatQ___closed__66);
v___x_256_ = l_Lean_Expr_app___override(v___x_254_, v___x_255_);
v___x_257_ = l_Lean_Expr_app___override(v___x_252_, v___x_256_);
v___x_258_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__69, &lp_mathlib_Qq_ofNatQ___closed__69_once, _init_lp_mathlib_Qq_ofNatQ___closed__69);
v___x_259_ = l_Lean_Expr_app___override(v___x_258_, v_k_224_);
v___x_260_ = l_Lean_Expr_app___override(v___x_259_, v___x_255_);
v___x_261_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__76, &lp_mathlib_Qq_ofNatQ___closed__76_once, _init_lp_mathlib_Qq_ofNatQ___closed__76);
v___x_262_ = l_Lean_Expr_app___override(v___x_260_, v___x_261_);
v___x_263_ = l_Lean_Expr_app___override(v___x_257_, v___x_262_);
v___x_264_ = l_Lean_Expr_app___override(v___x_251_, v___x_263_);
v___x_265_ = l_Lean_Expr_app___override(v___x_230_, v___x_264_);
return v___x_265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27(lean_object* v_u_282_, lean_object* v_n_283_, lean_object* v_00_u03b1_284_, lean_object* v_inst_285_, lean_object* v_e_286_){
_start:
{
lean_object* v___x_287_; uint8_t v___x_288_; 
v___x_287_ = lean_unsigned_to_nat(1u);
v___x_288_ = lean_nat_dec_eq(v_n_283_, v___x_287_);
if (v___x_288_ == 0)
{
lean_object* v_n_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
lean_inc_ref(v_inst_285_);
lean_inc_ref_n(v_00_u03b1_284_, 6);
lean_inc_n(v_u_282_, 3);
v_n_289_ = lp_mathlib_Qq_ofNatQ(v_u_282_, v_00_u03b1_284_, v_inst_285_, v_n_283_);
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__2));
v___x_291_ = lean_box(0);
v___x_292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_292_, 0, v_u_282_);
lean_ctor_set(v___x_292_, 1, v___x_291_);
lean_inc_ref_n(v___x_292_, 3);
v___x_293_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_293_, 0, v_u_282_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
v___x_294_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_294_, 0, v_u_282_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = l_Lean_Expr_const___override(v___x_290_, v___x_294_);
v___x_296_ = l_Lean_Expr_app___override(v___x_295_, v_00_u03b1_284_);
v___x_297_ = l_Lean_Expr_app___override(v___x_296_, v_00_u03b1_284_);
v___x_298_ = l_Lean_Expr_app___override(v___x_297_, v_00_u03b1_284_);
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__4));
v___x_300_ = l_Lean_Expr_const___override(v___x_299_, v___x_292_);
v___x_301_ = l_Lean_Expr_app___override(v___x_300_, v_00_u03b1_284_);
v___x_302_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__7));
v___x_303_ = l_Lean_Expr_const___override(v___x_302_, v___x_292_);
v___x_304_ = l_Lean_Expr_app___override(v___x_303_, v_00_u03b1_284_);
v___x_305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27___closed__9));
v___x_306_ = l_Lean_Expr_const___override(v___x_305_, v___x_292_);
v___x_307_ = l_Lean_Expr_app___override(v___x_306_, v_00_u03b1_284_);
v___x_308_ = l_Lean_Expr_app___override(v___x_307_, v_inst_285_);
v___x_309_ = l_Lean_Expr_app___override(v___x_304_, v___x_308_);
v___x_310_ = l_Lean_Expr_app___override(v___x_301_, v___x_309_);
v___x_311_ = l_Lean_Expr_app___override(v___x_298_, v___x_310_);
v___x_312_ = l_Lean_Expr_app___override(v___x_311_, v_n_289_);
v___x_313_ = l_Lean_Expr_app___override(v___x_312_, v_e_286_);
return v___x_313_;
}
else
{
lean_dec_ref(v_inst_285_);
lean_dec_ref(v_00_u03b1_284_);
lean_dec(v_n_283_);
lean_dec(v_u_282_);
return v_e_286_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr(lean_object* v_n_316_, lean_object* v_e_317_, lean_object* v_a_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_Qq_inferTypeQ_x27(v_e_317_, v_a_318_, v_a_319_, v_a_320_, v_a_321_);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v_a_324_; lean_object* v_snd_325_; lean_object* v_fst_326_; lean_object* v_fst_327_; lean_object* v_snd_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_349_; 
v_a_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc(v_a_324_);
lean_dec_ref_known(v___x_323_, 1);
v_snd_325_ = lean_ctor_get(v_a_324_, 1);
lean_inc(v_snd_325_);
v_fst_326_ = lean_ctor_get(v_a_324_, 0);
lean_inc(v_fst_326_);
lean_dec(v_a_324_);
v_fst_327_ = lean_ctor_get(v_snd_325_, 0);
v_snd_328_ = lean_ctor_get(v_snd_325_, 1);
v_isSharedCheck_349_ = !lean_is_exclusive(v_snd_325_);
if (v_isSharedCheck_349_ == 0)
{
v___x_330_ = v_snd_325_;
v_isShared_331_ = v_isSharedCheck_349_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_snd_328_);
lean_inc(v_fst_327_);
lean_dec(v_snd_325_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_349_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___closed__0));
v___x_333_ = lean_box(0);
lean_inc(v_fst_326_);
if (v_isShared_331_ == 0)
{
lean_ctor_set_tag(v___x_330_, 1);
lean_ctor_set(v___x_330_, 1, v___x_333_);
lean_ctor_set(v___x_330_, 0, v_fst_326_);
v___x_335_ = v___x_330_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v_fst_326_);
lean_ctor_set(v_reuseFailAlloc_348_, 1, v___x_333_);
v___x_335_ = v_reuseFailAlloc_348_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_336_ = l_Lean_Expr_const___override(v___x_332_, v___x_335_);
lean_inc(v_fst_327_);
v___x_337_ = l_Lean_Expr_app___override(v___x_336_, v_fst_327_);
v___x_338_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_337_, v_a_318_, v_a_319_, v_a_320_, v_a_321_);
if (lean_obj_tag(v___x_338_) == 0)
{
lean_object* v_a_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_347_; 
v_a_339_ = lean_ctor_get(v___x_338_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v___x_338_);
if (v_isSharedCheck_347_ == 0)
{
v___x_341_ = v___x_338_;
v_isShared_342_ = v_isSharedCheck_347_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_a_339_);
lean_dec(v___x_338_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_347_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_343_; lean_object* v___x_345_; 
v___x_343_ = lp_mathlib_Mathlib_Tactic_Linarith_mulExpr_x27(v_fst_326_, v_n_316_, v_fst_327_, v_a_339_, v_snd_328_);
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 0, v___x_343_);
v___x_345_ = v___x_341_;
goto v_reusejp_344_;
}
else
{
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v___x_343_);
v___x_345_ = v_reuseFailAlloc_346_;
goto v_reusejp_344_;
}
v_reusejp_344_:
{
return v___x_345_;
}
}
}
else
{
lean_dec(v_snd_328_);
lean_dec(v_fst_327_);
lean_dec(v_fst_326_);
lean_dec(v_n_316_);
return v___x_338_;
}
}
}
}
else
{
lean_object* v_a_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_357_; 
lean_dec(v_n_316_);
v_a_350_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_357_ == 0)
{
v___x_352_ = v___x_323_;
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_a_350_);
lean_dec(v___x_323_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v___x_355_; 
if (v_isShared_353_ == 0)
{
v___x_355_ = v___x_352_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v_a_350_);
v___x_355_ = v_reuseFailAlloc_356_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
return v___x_355_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mulExpr___boxed(lean_object* v_n_358_, lean_object* v_e_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_Mathlib_Tactic_Linarith_mulExpr(v_n_358_, v_e_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_);
lean_dec(v_a_363_);
lean_dec_ref(v_a_362_);
lean_dec(v_a_361_);
lean_dec_ref(v_a_360_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go(lean_object* v_u_376_, lean_object* v_00_u03b1_377_, lean_object* v___inst_378_, lean_object* v_p_379_, lean_object* v_a_380_){
_start:
{
if (lean_obj_tag(v_a_380_) == 0)
{
lean_dec_ref(v___inst_378_);
lean_dec_ref(v_00_u03b1_377_);
lean_dec(v_u_376_);
return v_p_379_;
}
else
{
lean_object* v_tail_381_; 
v_tail_381_ = lean_ctor_get(v_a_380_, 1);
if (lean_obj_tag(v_tail_381_) == 0)
{
lean_object* v_head_382_; lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_412_; 
v_head_382_ = lean_ctor_get(v_a_380_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v_a_380_);
if (v_isSharedCheck_412_ == 0)
{
lean_object* v_unused_413_; 
v_unused_413_ = lean_ctor_get(v_a_380_, 1);
lean_dec(v_unused_413_);
v___x_384_ = v_a_380_;
v_isShared_385_ = v_isSharedCheck_412_;
goto v_resetjp_383_;
}
else
{
lean_inc(v_head_382_);
lean_dec(v_a_380_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_412_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_389_; 
v___x_386_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__40));
v___x_387_ = lean_box(0);
lean_inc(v_u_376_);
if (v_isShared_385_ == 0)
{
lean_ctor_set(v___x_384_, 1, v___x_387_);
lean_ctor_set(v___x_384_, 0, v_u_376_);
v___x_389_ = v___x_384_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_u_376_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v___x_387_);
v___x_389_ = v_reuseFailAlloc_411_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
lean_inc_ref_n(v___x_389_, 3);
lean_inc(v_u_376_);
v___x_390_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_390_, 0, v_u_376_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_391_, 0, v_u_376_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = l_Lean_Expr_const___override(v___x_386_, v___x_391_);
lean_inc_ref_n(v_00_u03b1_377_, 5);
v___x_393_ = l_Lean_Expr_app___override(v___x_392_, v_00_u03b1_377_);
v___x_394_ = l_Lean_Expr_app___override(v___x_393_, v_00_u03b1_377_);
v___x_395_ = l_Lean_Expr_app___override(v___x_394_, v_00_u03b1_377_);
v___x_396_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__51));
v___x_397_ = l_Lean_Expr_const___override(v___x_396_, v___x_389_);
v___x_398_ = l_Lean_Expr_app___override(v___x_397_, v_00_u03b1_377_);
v___x_399_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2));
v___x_400_ = l_Lean_Expr_const___override(v___x_399_, v___x_389_);
v___x_401_ = l_Lean_Expr_app___override(v___x_400_, v_00_u03b1_377_);
v___x_402_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5));
v___x_403_ = l_Lean_Expr_const___override(v___x_402_, v___x_389_);
v___x_404_ = l_Lean_Expr_app___override(v___x_403_, v_00_u03b1_377_);
v___x_405_ = l_Lean_Expr_app___override(v___x_404_, v___inst_378_);
v___x_406_ = l_Lean_Expr_app___override(v___x_401_, v___x_405_);
v___x_407_ = l_Lean_Expr_app___override(v___x_398_, v___x_406_);
v___x_408_ = l_Lean_Expr_app___override(v___x_395_, v___x_407_);
v___x_409_ = l_Lean_Expr_app___override(v___x_408_, v_p_379_);
v___x_410_ = l_Lean_Expr_app___override(v___x_409_, v_head_382_);
return v___x_410_;
}
}
}
else
{
lean_object* v_head_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_445_; 
lean_inc(v_tail_381_);
v_head_414_ = lean_ctor_get(v_a_380_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v_a_380_);
if (v_isSharedCheck_445_ == 0)
{
lean_object* v_unused_446_; 
v_unused_446_ = lean_ctor_get(v_a_380_, 1);
lean_dec(v_unused_446_);
v___x_416_ = v_a_380_;
v_isShared_417_ = v_isSharedCheck_445_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_head_414_);
lean_dec(v_a_380_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_445_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_421_; 
v___x_418_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__40));
v___x_419_ = lean_box(0);
lean_inc(v_u_376_);
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 1, v___x_419_);
lean_ctor_set(v___x_416_, 0, v_u_376_);
v___x_421_ = v___x_416_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v_u_376_);
lean_ctor_set(v_reuseFailAlloc_444_, 1, v___x_419_);
v___x_421_ = v_reuseFailAlloc_444_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
lean_inc_ref_n(v___x_421_, 3);
lean_inc_n(v_u_376_, 2);
v___x_422_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_422_, 0, v_u_376_);
lean_ctor_set(v___x_422_, 1, v___x_421_);
v___x_423_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_423_, 0, v_u_376_);
lean_ctor_set(v___x_423_, 1, v___x_422_);
v___x_424_ = l_Lean_Expr_const___override(v___x_418_, v___x_423_);
lean_inc_ref_n(v_00_u03b1_377_, 6);
v___x_425_ = l_Lean_Expr_app___override(v___x_424_, v_00_u03b1_377_);
v___x_426_ = l_Lean_Expr_app___override(v___x_425_, v_00_u03b1_377_);
v___x_427_ = l_Lean_Expr_app___override(v___x_426_, v_00_u03b1_377_);
v___x_428_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__51));
v___x_429_ = l_Lean_Expr_const___override(v___x_428_, v___x_421_);
v___x_430_ = l_Lean_Expr_app___override(v___x_429_, v_00_u03b1_377_);
v___x_431_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__2));
v___x_432_ = l_Lean_Expr_const___override(v___x_431_, v___x_421_);
v___x_433_ = l_Lean_Expr_app___override(v___x_432_, v_00_u03b1_377_);
v___x_434_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go___closed__5));
v___x_435_ = l_Lean_Expr_const___override(v___x_434_, v___x_421_);
v___x_436_ = l_Lean_Expr_app___override(v___x_435_, v_00_u03b1_377_);
lean_inc_ref(v___inst_378_);
v___x_437_ = l_Lean_Expr_app___override(v___x_436_, v___inst_378_);
v___x_438_ = l_Lean_Expr_app___override(v___x_433_, v___x_437_);
v___x_439_ = l_Lean_Expr_app___override(v___x_430_, v___x_438_);
v___x_440_ = l_Lean_Expr_app___override(v___x_427_, v___x_439_);
v___x_441_ = l_Lean_Expr_app___override(v___x_440_, v_p_379_);
v___x_442_ = l_Lean_Expr_app___override(v___x_441_, v_head_414_);
v_p_379_ = v___x_442_;
v_a_380_ = v_tail_381_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27(lean_object* v_u_460_, lean_object* v_00_u03b1_461_, lean_object* v___inst_462_, lean_object* v_x_463_){
_start:
{
if (lean_obj_tag(v_x_463_) == 0)
{
lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_464_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
v___x_465_ = lean_box(0);
v___x_466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_466_, 0, v_u_460_);
lean_ctor_set(v___x_466_, 1, v___x_465_);
lean_inc_ref_n(v___x_466_, 4);
v___x_467_ = l_Lean_Expr_const___override(v___x_464_, v___x_466_);
lean_inc_ref_n(v_00_u03b1_461_, 4);
v___x_468_ = l_Lean_Expr_app___override(v___x_467_, v_00_u03b1_461_);
v___x_469_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__4, &lp_mathlib_Qq_ofNatQ___closed__4_once, _init_lp_mathlib_Qq_ofNatQ___closed__4);
v___x_470_ = l_Lean_Expr_app___override(v___x_468_, v___x_469_);
v___x_471_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__7));
v___x_472_ = l_Lean_Expr_const___override(v___x_471_, v___x_466_);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v_00_u03b1_461_);
v___x_474_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__1));
v___x_475_ = l_Lean_Expr_const___override(v___x_474_, v___x_466_);
v___x_476_ = l_Lean_Expr_app___override(v___x_475_, v_00_u03b1_461_);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__4));
v___x_478_ = l_Lean_Expr_const___override(v___x_477_, v___x_466_);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_00_u03b1_461_);
v___x_480_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27___closed__6));
v___x_481_ = l_Lean_Expr_const___override(v___x_480_, v___x_466_);
v___x_482_ = l_Lean_Expr_app___override(v___x_481_, v_00_u03b1_461_);
v___x_483_ = l_Lean_Expr_app___override(v___x_482_, v___inst_462_);
v___x_484_ = l_Lean_Expr_app___override(v___x_479_, v___x_483_);
v___x_485_ = l_Lean_Expr_app___override(v___x_476_, v___x_484_);
v___x_486_ = l_Lean_Expr_app___override(v___x_473_, v___x_485_);
v___x_487_ = l_Lean_Expr_app___override(v___x_470_, v___x_486_);
return v___x_487_;
}
else
{
lean_object* v_head_488_; lean_object* v_tail_489_; lean_object* v___x_490_; 
v_head_488_ = lean_ctor_get(v_x_463_, 0);
lean_inc(v_head_488_);
v_tail_489_ = lean_ctor_get(v_x_463_, 1);
lean_inc(v_tail_489_);
lean_dec_ref_known(v_x_463_, 2);
v___x_490_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_addExprs_x27_go(v_u_460_, v_00_u03b1_461_, v___inst_462_, v_head_488_, v_tail_489_);
return v___x_490_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs(lean_object* v_x_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_){
_start:
{
if (lean_obj_tag(v_x_493_) == 0)
{
lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_499_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__75, &lp_mathlib_Qq_ofNatQ___closed__75_once, _init_lp_mathlib_Qq_ofNatQ___closed__75);
v___x_500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
return v___x_500_;
}
else
{
lean_object* v_head_501_; lean_object* v___x_502_; 
v_head_501_ = lean_ctor_get(v_x_493_, 0);
lean_inc(v_head_501_);
v___x_502_ = lp_mathlib_Qq_inferTypeQ_x27(v_head_501_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
if (lean_obj_tag(v___x_502_) == 0)
{
lean_object* v_a_503_; lean_object* v_snd_504_; lean_object* v_fst_505_; lean_object* v_fst_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_527_; 
v_a_503_ = lean_ctor_get(v___x_502_, 0);
lean_inc(v_a_503_);
lean_dec_ref_known(v___x_502_, 1);
v_snd_504_ = lean_ctor_get(v_a_503_, 1);
lean_inc(v_snd_504_);
v_fst_505_ = lean_ctor_get(v_a_503_, 0);
lean_inc(v_fst_505_);
lean_dec(v_a_503_);
v_fst_506_ = lean_ctor_get(v_snd_504_, 0);
v_isSharedCheck_527_ = !lean_is_exclusive(v_snd_504_);
if (v_isSharedCheck_527_ == 0)
{
lean_object* v_unused_528_; 
v_unused_528_ = lean_ctor_get(v_snd_504_, 1);
lean_dec(v_unused_528_);
v___x_508_ = v_snd_504_;
v_isShared_509_ = v_isSharedCheck_527_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_fst_506_);
lean_dec(v_snd_504_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_527_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_513_; 
v___x_510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addExprs___closed__0));
v___x_511_ = lean_box(0);
lean_inc(v_fst_505_);
if (v_isShared_509_ == 0)
{
lean_ctor_set_tag(v___x_508_, 1);
lean_ctor_set(v___x_508_, 1, v___x_511_);
lean_ctor_set(v___x_508_, 0, v_fst_505_);
v___x_513_ = v___x_508_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v_fst_505_);
lean_ctor_set(v_reuseFailAlloc_526_, 1, v___x_511_);
v___x_513_ = v_reuseFailAlloc_526_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; 
v___x_514_ = l_Lean_Expr_const___override(v___x_510_, v___x_513_);
lean_inc(v_fst_506_);
v___x_515_ = l_Lean_Expr_app___override(v___x_514_, v_fst_506_);
v___x_516_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_515_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
if (lean_obj_tag(v___x_516_) == 0)
{
lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_525_; 
v_a_517_ = lean_ctor_get(v___x_516_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_516_);
if (v_isSharedCheck_525_ == 0)
{
v___x_519_ = v___x_516_;
v_isShared_520_ = v_isSharedCheck_525_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_516_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_525_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_521_; lean_object* v___x_523_; 
v___x_521_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs_x27(v_fst_505_, v_fst_506_, v_a_517_, v_x_493_);
if (v_isShared_520_ == 0)
{
lean_ctor_set(v___x_519_, 0, v___x_521_);
v___x_523_ = v___x_519_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v___x_521_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
else
{
lean_dec(v_fst_506_);
lean_dec(v_fst_505_);
lean_dec_ref_known(v_x_493_, 2);
return v___x_516_;
}
}
}
}
else
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
lean_dec_ref_known(v_x_493_, 2);
v_a_529_ = lean_ctor_get(v___x_502_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_502_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_502_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_502_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addExprs___boxed(lean_object* v_x_537_, lean_object* v_a_538_, lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v_a_541_, lean_object* v_a_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs(v_x_537_, v_a_538_, v_a_539_, v_a_540_, v_a_541_);
lean_dec(v_a_541_);
lean_dec_ref(v_a_540_);
lean_dec(v_a_539_);
lean_dec_ref(v_a_538_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq(uint8_t v_x_601_, uint8_t v_x_602_){
_start:
{
switch(v_x_601_)
{
case 0:
{
switch(v_x_602_)
{
case 0:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_603_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__4));
v___x_604_ = lean_box(v_x_602_);
v___x_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_603_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
return v___x_605_;
}
case 1:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_606_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__6));
v___x_607_ = lean_box(v_x_602_);
v___x_608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_608_, 0, v___x_606_);
lean_ctor_set(v___x_608_, 1, v___x_607_);
return v___x_608_;
}
default: 
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__8));
v___x_610_ = lean_box(v_x_602_);
v___x_611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_609_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
return v___x_611_;
}
}
}
case 1:
{
switch(v_x_602_)
{
case 0:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__10));
v___x_613_ = lean_box(v_x_601_);
v___x_614_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_614_, 0, v___x_612_);
lean_ctor_set(v___x_614_, 1, v___x_613_);
return v___x_614_;
}
case 1:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__12));
v___x_616_ = lean_box(v_x_602_);
v___x_617_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_615_);
lean_ctor_set(v___x_617_, 1, v___x_616_);
return v___x_617_;
}
default: 
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__14));
v___x_619_ = lean_box(v_x_602_);
v___x_620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_618_);
lean_ctor_set(v___x_620_, 1, v___x_619_);
return v___x_620_;
}
}
}
default: 
{
switch(v_x_602_)
{
case 0:
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_621_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__16));
v___x_622_ = lean_box(v_x_601_);
v___x_623_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_623_, 0, v___x_621_);
lean_ctor_set(v___x_623_, 1, v___x_622_);
return v___x_623_;
}
case 1:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_624_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__18));
v___x_625_ = lean_box(v_x_601_);
v___x_626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_624_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
return v___x_626_;
}
default: 
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_627_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__20));
v___x_628_ = lean_box(v_x_602_);
v___x_629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_629_, 0, v___x_627_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
return v___x_629_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addIneq___boxed(lean_object* v_x_630_, lean_object* v_x_631_){
_start:
{
uint8_t v_x_317__boxed_632_; uint8_t v_x_318__boxed_633_; lean_object* v_res_634_; 
v_x_317__boxed_632_ = lean_unbox(v_x_630_);
v_x_318__boxed_633_ = lean_unbox(v_x_631_);
v_res_634_ = lp_mathlib_Mathlib_Tactic_Linarith_addIneq(v_x_317__boxed_632_, v_x_318__boxed_633_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step(uint8_t v_c_635_, lean_object* v_pf_636_, lean_object* v_npf_637_, lean_object* v_coeff_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(v_coeff_638_, v_npf_637_, v_a_639_, v_a_640_, v_a_641_, v_a_642_);
if (lean_obj_tag(v___x_644_) == 0)
{
lean_object* v_a_645_; lean_object* v_fst_646_; lean_object* v_snd_647_; uint8_t v___x_648_; lean_object* v___x_649_; lean_object* v_fst_650_; lean_object* v_snd_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_679_; 
v_a_645_ = lean_ctor_get(v___x_644_, 0);
lean_inc(v_a_645_);
lean_dec_ref_known(v___x_644_, 1);
v_fst_646_ = lean_ctor_get(v_a_645_, 0);
lean_inc(v_fst_646_);
v_snd_647_ = lean_ctor_get(v_a_645_, 1);
lean_inc(v_snd_647_);
lean_dec(v_a_645_);
v___x_648_ = lean_unbox(v_fst_646_);
lean_dec(v_fst_646_);
v___x_649_ = lp_mathlib_Mathlib_Tactic_Linarith_addIneq(v_c_635_, v___x_648_);
v_fst_650_ = lean_ctor_get(v___x_649_, 0);
v_snd_651_ = lean_ctor_get(v___x_649_, 1);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_679_ == 0)
{
v___x_653_ = v___x_649_;
v_isShared_654_ = v_isSharedCheck_679_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_snd_651_);
lean_inc(v_fst_650_);
lean_dec(v___x_649_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_679_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_655_ = lean_unsigned_to_nat(2u);
v___x_656_ = lean_mk_empty_array_with_capacity(v___x_655_);
v___x_657_ = lean_array_push(v___x_656_, v_pf_636_);
v___x_658_ = lean_array_push(v___x_657_, v_snd_647_);
v___x_659_ = l_Lean_Meta_mkAppM(v_fst_650_, v___x_658_, v_a_639_, v_a_640_, v_a_641_, v_a_642_);
if (lean_obj_tag(v___x_659_) == 0)
{
lean_object* v_a_660_; lean_object* v___x_662_; uint8_t v_isShared_663_; uint8_t v_isSharedCheck_670_; 
v_a_660_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_670_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_670_ == 0)
{
v___x_662_ = v___x_659_;
v_isShared_663_ = v_isSharedCheck_670_;
goto v_resetjp_661_;
}
else
{
lean_inc(v_a_660_);
lean_dec(v___x_659_);
v___x_662_ = lean_box(0);
v_isShared_663_ = v_isSharedCheck_670_;
goto v_resetjp_661_;
}
v_resetjp_661_:
{
lean_object* v___x_665_; 
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 1, v_a_660_);
lean_ctor_set(v___x_653_, 0, v_snd_651_);
v___x_665_ = v___x_653_;
goto v_reusejp_664_;
}
else
{
lean_object* v_reuseFailAlloc_669_; 
v_reuseFailAlloc_669_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_669_, 0, v_snd_651_);
lean_ctor_set(v_reuseFailAlloc_669_, 1, v_a_660_);
v___x_665_ = v_reuseFailAlloc_669_;
goto v_reusejp_664_;
}
v_reusejp_664_:
{
lean_object* v___x_667_; 
if (v_isShared_663_ == 0)
{
lean_ctor_set(v___x_662_, 0, v___x_665_);
v___x_667_ = v___x_662_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v___x_665_);
v___x_667_ = v_reuseFailAlloc_668_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
return v___x_667_;
}
}
}
}
else
{
lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
lean_del_object(v___x_653_);
lean_dec(v_snd_651_);
v_a_671_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_659_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_dec(v___x_659_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
else
{
lean_dec_ref(v_pf_636_);
return v___x_644_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step___boxed(lean_object* v_c_680_, lean_object* v_pf_681_, lean_object* v_npf_682_, lean_object* v_coeff_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_){
_start:
{
uint8_t v_c_boxed_689_; lean_object* v_res_690_; 
v_c_boxed_689_ = lean_unbox(v_c_680_);
v_res_690_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step(v_c_boxed_689_, v_pf_681_, v_npf_682_, v_coeff_683_, v_a_684_, v_a_685_, v_a_686_, v_a_687_);
lean_dec(v_a_687_);
lean_dec_ref(v_a_686_);
lean_dec(v_a_685_);
lean_dec_ref(v_a_684_);
return v_res_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(lean_object* v_msgData_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
lean_object* v___x_697_; lean_object* v_env_698_; lean_object* v___x_699_; lean_object* v_mctx_700_; lean_object* v_lctx_701_; lean_object* v_options_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_697_ = lean_st_ref_get(v___y_695_);
v_env_698_ = lean_ctor_get(v___x_697_, 0);
lean_inc_ref(v_env_698_);
lean_dec(v___x_697_);
v___x_699_ = lean_st_ref_get(v___y_693_);
v_mctx_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc_ref(v_mctx_700_);
lean_dec(v___x_699_);
v_lctx_701_ = lean_ctor_get(v___y_692_, 2);
v_options_702_ = lean_ctor_get(v___y_694_, 2);
lean_inc_ref(v_options_702_);
lean_inc_ref(v_lctx_701_);
v___x_703_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_703_, 0, v_env_698_);
lean_ctor_set(v___x_703_, 1, v_mctx_700_);
lean_ctor_set(v___x_703_, 2, v_lctx_701_);
lean_ctor_set(v___x_703_, 3, v_options_702_);
v___x_704_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_704_, 0, v___x_703_);
lean_ctor_set(v___x_704_, 1, v_msgData_691_);
v___x_705_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0___boxed(lean_object* v_msgData_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(v_msgData_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(lean_object* v_msg_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_){
_start:
{
lean_object* v_ref_719_; lean_object* v___x_720_; lean_object* v_a_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_729_; 
v_ref_719_ = lean_ctor_get(v___y_716_, 5);
v___x_720_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(v_msg_713_, v___y_714_, v___y_715_, v___y_716_, v___y_717_);
v_a_721_ = lean_ctor_get(v___x_720_, 0);
v_isSharedCheck_729_ = !lean_is_exclusive(v___x_720_);
if (v_isSharedCheck_729_ == 0)
{
v___x_723_ = v___x_720_;
v_isShared_724_ = v_isSharedCheck_729_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_a_721_);
lean_dec(v___x_720_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_729_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_725_; lean_object* v___x_727_; 
lean_inc(v_ref_719_);
v___x_725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_725_, 0, v_ref_719_);
lean_ctor_set(v___x_725_, 1, v_a_721_);
if (v_isShared_724_ == 0)
{
lean_ctor_set_tag(v___x_723_, 1);
lean_ctor_set(v___x_723_, 0, v___x_725_);
v___x_727_ = v___x_723_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v___x_725_);
v___x_727_ = v_reuseFailAlloc_728_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
return v___x_727_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg___boxed(lean_object* v_msg_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_){
_start:
{
lean_object* v_res_736_; 
v_res_736_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v_msg_730_, v___y_731_, v___y_732_, v___y_733_, v___y_734_);
lean_dec(v___y_734_);
lean_dec_ref(v___y_733_);
lean_dec(v___y_732_);
lean_dec_ref(v___y_731_);
return v_res_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1(lean_object* v_x_737_, lean_object* v_x_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_){
_start:
{
if (lean_obj_tag(v_x_738_) == 0)
{
lean_object* v___x_744_; 
v___x_744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_744_, 0, v_x_737_);
return v___x_744_;
}
else
{
lean_object* v_head_745_; lean_object* v_tail_746_; lean_object* v_fst_747_; lean_object* v_snd_748_; lean_object* v_fst_749_; lean_object* v_snd_750_; uint8_t v___x_751_; lean_object* v___x_752_; 
v_head_745_ = lean_ctor_get(v_x_738_, 0);
lean_inc(v_head_745_);
v_tail_746_ = lean_ctor_get(v_x_738_, 1);
lean_inc(v_tail_746_);
lean_dec_ref_known(v_x_738_, 2);
v_fst_747_ = lean_ctor_get(v_x_737_, 0);
lean_inc(v_fst_747_);
v_snd_748_ = lean_ctor_get(v_x_737_, 1);
lean_inc(v_snd_748_);
lean_dec_ref(v_x_737_);
v_fst_749_ = lean_ctor_get(v_head_745_, 0);
lean_inc(v_fst_749_);
v_snd_750_ = lean_ctor_get(v_head_745_, 1);
lean_inc(v_snd_750_);
lean_dec(v_head_745_);
v___x_751_ = lean_unbox(v_fst_747_);
lean_dec(v_fst_747_);
v___x_752_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_mkLTZeroProof_step(v___x_751_, v_snd_748_, v_fst_749_, v_snd_750_, v___y_739_, v___y_740_, v___y_741_, v___y_742_);
if (lean_obj_tag(v___x_752_) == 0)
{
lean_object* v_a_753_; 
v_a_753_ = lean_ctor_get(v___x_752_, 0);
lean_inc(v_a_753_);
lean_dec_ref_known(v___x_752_, 1);
v_x_737_ = v_a_753_;
v_x_738_ = v_tail_746_;
goto _start;
}
else
{
lean_dec(v_tail_746_);
return v___x_752_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1___boxed(lean_object* v_x_755_, lean_object* v_x_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_){
_start:
{
lean_object* v_res_762_; 
v_res_762_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1(v_x_755_, v_x_756_, v___y_757_, v___y_758_, v___y_759_, v___y_760_);
lean_dec(v___y_760_);
lean_dec_ref(v___y_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
return v_res_762_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1(void){
_start:
{
lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__0));
v___x_765_ = l_Lean_stringToMessageData(v___x_764_);
return v___x_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof(lean_object* v_x_766_, lean_object* v_a_767_, lean_object* v_a_768_, lean_object* v_a_769_, lean_object* v_a_770_){
_start:
{
if (lean_obj_tag(v_x_766_) == 0)
{
lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_772_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___closed__1);
v___x_773_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v___x_772_, v_a_767_, v_a_768_, v_a_769_, v_a_770_);
return v___x_773_;
}
else
{
lean_object* v_head_774_; lean_object* v_tail_775_; 
v_head_774_ = lean_ctor_get(v_x_766_, 0);
lean_inc(v_head_774_);
v_tail_775_ = lean_ctor_get(v_x_766_, 1);
lean_inc(v_tail_775_);
lean_dec_ref_known(v_x_766_, 2);
if (lean_obj_tag(v_tail_775_) == 0)
{
lean_object* v_fst_776_; lean_object* v_snd_777_; lean_object* v___x_778_; 
v_fst_776_ = lean_ctor_get(v_head_774_, 0);
lean_inc(v_fst_776_);
v_snd_777_ = lean_ctor_get(v_head_774_, 1);
lean_inc(v_snd_777_);
lean_dec(v_head_774_);
v___x_778_ = lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(v_snd_777_, v_fst_776_, v_a_767_, v_a_768_, v_a_769_, v_a_770_);
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_787_; 
v_a_779_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_787_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_787_ == 0)
{
v___x_781_ = v___x_778_;
v_isShared_782_ = v_isSharedCheck_787_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v___x_778_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_787_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v_snd_783_; lean_object* v___x_785_; 
v_snd_783_ = lean_ctor_get(v_a_779_, 1);
lean_inc(v_snd_783_);
lean_dec(v_a_779_);
if (v_isShared_782_ == 0)
{
lean_ctor_set(v___x_781_, 0, v_snd_783_);
v___x_785_ = v___x_781_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_snd_783_);
v___x_785_ = v_reuseFailAlloc_786_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
return v___x_785_;
}
}
}
else
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
v_a_788_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_778_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_778_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
v___x_793_ = v___x_790_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_a_788_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
else
{
lean_object* v_fst_796_; lean_object* v_snd_797_; lean_object* v___x_798_; 
v_fst_796_ = lean_ctor_get(v_head_774_, 0);
lean_inc(v_fst_796_);
v_snd_797_ = lean_ctor_get(v_head_774_, 1);
lean_inc(v_snd_797_);
lean_dec(v_head_774_);
v___x_798_ = lp_mathlib_Mathlib_Tactic_Linarith_mkSingleCompZeroOf(v_snd_797_, v_fst_796_, v_a_767_, v_a_768_, v_a_769_, v_a_770_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_a_799_; lean_object* v___x_800_; 
v_a_799_ = lean_ctor_get(v___x_798_, 0);
lean_inc(v_a_799_);
lean_dec_ref_known(v___x_798_, 1);
v___x_800_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__1(v_a_799_, v_tail_775_, v_a_767_, v_a_768_, v_a_769_, v_a_770_);
if (lean_obj_tag(v___x_800_) == 0)
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_809_; 
v_a_801_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_809_ == 0)
{
v___x_803_ = v___x_800_;
v_isShared_804_ = v_isSharedCheck_809_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_800_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_809_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v_snd_805_; lean_object* v___x_807_; 
v_snd_805_ = lean_ctor_get(v_a_801_, 1);
lean_inc(v_snd_805_);
lean_dec(v_a_801_);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v_snd_805_);
v___x_807_ = v___x_803_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v_snd_805_);
v___x_807_ = v_reuseFailAlloc_808_;
goto v_reusejp_806_;
}
v_reusejp_806_:
{
return v___x_807_;
}
}
}
else
{
lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_817_; 
v_a_810_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_817_ == 0)
{
v___x_812_ = v___x_800_;
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_800_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_815_; 
if (v_isShared_813_ == 0)
{
v___x_815_ = v___x_812_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v_a_810_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
}
}
else
{
lean_object* v_a_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_825_; 
lean_dec(v_tail_775_);
v_a_818_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_825_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_825_ == 0)
{
v___x_820_ = v___x_798_;
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_a_818_);
lean_dec(v___x_798_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_823_; 
if (v_isShared_821_ == 0)
{
v___x_823_ = v___x_820_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v_a_818_);
v___x_823_ = v_reuseFailAlloc_824_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
return v___x_823_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___boxed(lean_object* v_x_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_, lean_object* v_a_830_, lean_object* v_a_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof(v_x_826_, v_a_827_, v_a_828_, v_a_829_, v_a_830_);
lean_dec(v_a_830_);
lean_dec_ref(v_a_829_);
lean_dec(v_a_828_);
lean_dec_ref(v_a_827_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0(lean_object* v_00_u03b1_833_, lean_object* v_msg_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v_msg_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___boxed(lean_object* v_00_u03b1_841_, lean_object* v_msg_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0(v_00_u03b1_841_, v_msg_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof(lean_object* v_prf_849_, lean_object* v_a_850_, lean_object* v_a_851_, lean_object* v_a_852_, lean_object* v_a_853_){
_start:
{
lean_object* v___x_855_; 
lean_inc(v_a_853_);
lean_inc_ref(v_a_852_);
lean_inc(v_a_851_);
lean_inc_ref(v_a_850_);
v___x_855_ = lean_infer_type(v_prf_849_, v_a_850_, v_a_851_, v_a_852_, v_a_853_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_a_856_; lean_object* v___x_857_; 
v_a_856_ = lean_ctor_get(v___x_855_, 0);
lean_inc(v_a_856_);
lean_dec_ref_known(v___x_855_, 1);
v___x_857_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_856_, v_a_850_, v_a_851_, v_a_852_, v_a_853_);
if (lean_obj_tag(v___x_857_) == 0)
{
lean_object* v_a_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_868_; 
v_a_858_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_868_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_868_ == 0)
{
v___x_860_ = v___x_857_;
v_isShared_861_ = v_isSharedCheck_868_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_a_858_);
lean_dec(v___x_857_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_868_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v_snd_862_; lean_object* v_snd_863_; lean_object* v_fst_864_; lean_object* v___x_866_; 
v_snd_862_ = lean_ctor_get(v_a_858_, 1);
lean_inc(v_snd_862_);
lean_dec(v_a_858_);
v_snd_863_ = lean_ctor_get(v_snd_862_, 1);
lean_inc(v_snd_863_);
lean_dec(v_snd_862_);
v_fst_864_ = lean_ctor_get(v_snd_863_, 0);
lean_inc(v_fst_864_);
lean_dec(v_snd_863_);
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 0, v_fst_864_);
v___x_866_ = v___x_860_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_867_; 
v_reuseFailAlloc_867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_867_, 0, v_fst_864_);
v___x_866_ = v_reuseFailAlloc_867_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
return v___x_866_;
}
}
}
else
{
lean_object* v_a_869_; lean_object* v___x_871_; uint8_t v_isShared_872_; uint8_t v_isSharedCheck_876_; 
v_a_869_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_876_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_876_ == 0)
{
v___x_871_ = v___x_857_;
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
else
{
lean_inc(v_a_869_);
lean_dec(v___x_857_);
v___x_871_ = lean_box(0);
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
v_resetjp_870_:
{
lean_object* v___x_874_; 
if (v_isShared_872_ == 0)
{
v___x_874_ = v___x_871_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v_a_869_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
return v___x_874_;
}
}
}
}
else
{
return v___x_855_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof___boxed(lean_object* v_prf_877_, lean_object* v_a_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_, lean_object* v_a_882_){
_start:
{
lean_object* v_res_883_; 
v_res_883_ = lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof(v_prf_877_, v_a_878_, v_a_879_, v_a_880_, v_a_881_);
lean_dec(v_a_881_);
lean_dec_ref(v_a_880_);
lean_dec(v_a_879_);
lean_dec_ref(v_a_878_);
return v_res_883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof(lean_object* v_prf_884_, lean_object* v_a_885_, lean_object* v_a_886_, lean_object* v_a_887_, lean_object* v_a_888_){
_start:
{
lean_object* v___x_890_; 
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
lean_inc(v_a_886_);
lean_inc_ref(v_a_885_);
v___x_890_ = lean_infer_type(v_prf_884_, v_a_885_, v_a_886_, v_a_887_, v_a_888_);
if (lean_obj_tag(v___x_890_) == 0)
{
lean_object* v_a_891_; lean_object* v___x_892_; 
v_a_891_ = lean_ctor_get(v___x_890_, 0);
lean_inc(v_a_891_);
lean_dec_ref_known(v___x_890_, 1);
v___x_892_ = lp_mathlib_Lean_Expr_ineq_x3f(v_a_891_, v_a_885_, v_a_886_, v_a_887_, v_a_888_);
if (lean_obj_tag(v___x_892_) == 0)
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_902_; 
v_a_893_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_902_ == 0)
{
v___x_895_ = v___x_892_;
v_isShared_896_ = v_isSharedCheck_902_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_892_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_902_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v_snd_897_; lean_object* v_fst_898_; lean_object* v___x_900_; 
v_snd_897_ = lean_ctor_get(v_a_893_, 1);
lean_inc(v_snd_897_);
lean_dec(v_a_893_);
v_fst_898_ = lean_ctor_get(v_snd_897_, 0);
lean_inc(v_fst_898_);
lean_dec(v_snd_897_);
if (v_isShared_896_ == 0)
{
lean_ctor_set(v___x_895_, 0, v_fst_898_);
v___x_900_ = v___x_895_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_fst_898_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
else
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_910_; 
v_a_903_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_910_ == 0)
{
v___x_905_ = v___x_892_;
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_892_);
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
else
{
return v___x_890_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof___boxed(lean_object* v_prf_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_, lean_object* v_a_916_){
_start:
{
lean_object* v_res_917_; 
v_res_917_ = lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof(v_prf_911_, v_a_912_, v_a_913_, v_a_914_, v_a_915_);
lean_dec(v_a_915_);
lean_dec_ref(v_a_914_);
lean_dec(v_a_913_);
lean_dec_ref(v_a_912_);
return v_res_917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof(lean_object* v_tp_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_, lean_object* v_a_931_){
_start:
{
lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; 
v___x_933_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__1));
v___x_934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_934_, 0, v_tp_927_);
v___x_935_ = lean_box(0);
v___x_936_ = lean_unsigned_to_nat(4u);
v___x_937_ = lean_mk_empty_array_with_capacity(v___x_936_);
v___x_938_ = lean_array_push(v___x_937_, v___x_934_);
v___x_939_ = lean_array_push(v___x_938_, v___x_935_);
v___x_940_ = lean_array_push(v___x_939_, v___x_935_);
v___x_941_ = lean_array_push(v___x_940_, v___x_935_);
v___x_942_ = l_Lean_Meta_mkAppOptM(v___x_933_, v___x_941_, v_a_928_, v_a_929_, v_a_930_, v_a_931_);
if (lean_obj_tag(v___x_942_) == 0)
{
lean_object* v_a_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v_a_943_ = lean_ctor_get(v___x_942_, 0);
lean_inc(v_a_943_);
lean_dec_ref_known(v___x_942_, 1);
v___x_944_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___closed__3));
v___x_945_ = lean_unsigned_to_nat(1u);
v___x_946_ = lean_mk_empty_array_with_capacity(v___x_945_);
v___x_947_ = lean_array_push(v___x_946_, v_a_943_);
v___x_948_ = l_Lean_Meta_mkAppM(v___x_944_, v___x_947_, v_a_928_, v_a_929_, v_a_930_, v_a_931_);
return v___x_948_;
}
else
{
return v___x_942_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof___boxed(lean_object* v_tp_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_, lean_object* v_a_953_, lean_object* v_a_954_){
_start:
{
lean_object* v_res_955_; 
v_res_955_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof(v_tp_949_, v_a_950_, v_a_951_, v_a_952_, v_a_953_);
lean_dec(v_a_953_);
lean_dec_ref(v_a_952_);
lean_dec(v_a_951_);
lean_dec_ref(v_a_950_);
return v_res_955_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2(void){
_start:
{
lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; 
v___x_959_ = lean_box(0);
v___x_960_ = lean_unsigned_to_nat(3u);
v___x_961_ = lean_mk_empty_array_with_capacity(v___x_960_);
v___x_962_ = lean_array_push(v___x_961_, v___x_959_);
return v___x_962_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3(void){
_start:
{
lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_963_ = lean_box(0);
v___x_964_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__2);
v___x_965_ = lean_array_push(v___x_964_, v___x_963_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx(lean_object* v_x_971_, lean_object* v_a_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_){
_start:
{
if (lean_obj_tag(v_x_971_) == 0)
{
lean_object* v___x_977_; 
v___x_977_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_977_, 0, v_x_971_);
return v___x_977_;
}
else
{
lean_object* v_head_978_; lean_object* v_tail_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_1071_; 
v_head_978_ = lean_ctor_get(v_x_971_, 0);
v_tail_979_ = lean_ctor_get(v_x_971_, 1);
v_isSharedCheck_1071_ = !lean_is_exclusive(v_x_971_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_981_ = v_x_971_;
v_isShared_982_ = v_isSharedCheck_1071_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_tail_979_);
lean_inc(v_head_978_);
lean_dec(v_x_971_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_1071_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v_fst_983_; lean_object* v_snd_984_; lean_object* v___x_985_; 
v_fst_983_ = lean_ctor_get(v_head_978_, 0);
v_snd_984_ = lean_ctor_get(v_head_978_, 1);
lean_inc(v_a_975_);
lean_inc_ref(v_a_974_);
lean_inc(v_a_973_);
lean_inc_ref(v_a_972_);
lean_inc(v_fst_983_);
v___x_985_ = lean_infer_type(v_fst_983_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; lean_object* v___x_987_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
v___x_987_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_a_986_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; lean_object* v_fst_989_; uint8_t v___x_990_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
lean_inc(v_a_988_);
lean_dec_ref_known(v___x_987_, 1);
v_fst_989_ = lean_ctor_get(v_a_988_, 0);
v___x_990_ = lean_unbox(v_fst_989_);
if (v___x_990_ == 0)
{
lean_object* v_snd_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_1041_; 
v_snd_991_ = lean_ctor_get(v_a_988_, 1);
v_isSharedCheck_1041_ = !lean_is_exclusive(v_a_988_);
if (v_isSharedCheck_1041_ == 0)
{
lean_object* v_unused_1042_; 
v_unused_1042_ = lean_ctor_get(v_a_988_, 0);
lean_dec(v_unused_1042_);
v___x_993_ = v_a_988_;
v_isShared_994_ = v_isSharedCheck_1041_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_snd_991_);
lean_dec(v_a_988_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_1041_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; 
v___x_995_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__1));
v___x_996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_996_, 0, v_snd_991_);
v___x_997_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3, &lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__3);
v___x_998_ = lean_array_push(v___x_997_, v___x_996_);
v___x_999_ = l_Lean_Meta_mkAppOptM(v___x_995_, v___x_998_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_999_) == 0)
{
lean_object* v_a_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; 
v_a_1000_ = lean_ctor_get(v___x_999_, 0);
lean_inc(v_a_1000_);
lean_dec_ref_known(v___x_999_, 1);
v___x_1001_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___closed__6));
v___x_1002_ = lean_unsigned_to_nat(1u);
v___x_1003_ = lean_mk_empty_array_with_capacity(v___x_1002_);
lean_inc_ref(v___x_1003_);
v___x_1004_ = lean_array_push(v___x_1003_, v_a_1000_);
v___x_1005_ = l_Lean_Meta_mkAppM(v___x_1001_, v___x_1004_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_1005_) == 0)
{
lean_object* v_a_1006_; lean_object* v___x_1007_; 
v_a_1006_ = lean_ctor_get(v___x_1005_, 0);
lean_inc(v_a_1006_);
lean_dec_ref_known(v___x_1005_, 1);
v___x_1007_ = lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx(v_tail_979_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_1007_) == 0)
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1024_; 
v_a_1008_ = lean_ctor_get(v___x_1007_, 0);
v_isSharedCheck_1024_ = !lean_is_exclusive(v___x_1007_);
if (v_isSharedCheck_1024_ == 0)
{
v___x_1010_ = v___x_1007_;
v_isShared_1011_ = v_isSharedCheck_1024_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_1007_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1024_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1015_; 
lean_inc(v_fst_983_);
v___x_1012_ = lean_array_push(v___x_1003_, v_fst_983_);
v___x_1013_ = l_Lean_mkAppN(v_a_1006_, v___x_1012_);
lean_dec_ref(v___x_1012_);
lean_inc(v_snd_984_);
if (v_isShared_994_ == 0)
{
lean_ctor_set(v___x_993_, 1, v_snd_984_);
lean_ctor_set(v___x_993_, 0, v___x_1013_);
v___x_1015_ = v___x_993_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_1013_);
lean_ctor_set(v_reuseFailAlloc_1023_, 1, v_snd_984_);
v___x_1015_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
lean_object* v___x_1017_; 
if (v_isShared_982_ == 0)
{
lean_ctor_set(v___x_981_, 1, v_a_1008_);
lean_ctor_set(v___x_981_, 0, v___x_1015_);
v___x_1017_ = v___x_981_;
goto v_reusejp_1016_;
}
else
{
lean_object* v_reuseFailAlloc_1022_; 
v_reuseFailAlloc_1022_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1022_, 0, v___x_1015_);
lean_ctor_set(v_reuseFailAlloc_1022_, 1, v_a_1008_);
v___x_1017_ = v_reuseFailAlloc_1022_;
goto v_reusejp_1016_;
}
v_reusejp_1016_:
{
lean_object* v___x_1018_; lean_object* v___x_1020_; 
v___x_1018_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1018_, 0, v_head_978_);
lean_ctor_set(v___x_1018_, 1, v___x_1017_);
if (v_isShared_1011_ == 0)
{
lean_ctor_set(v___x_1010_, 0, v___x_1018_);
v___x_1020_ = v___x_1010_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1021_; 
v_reuseFailAlloc_1021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1021_, 0, v___x_1018_);
v___x_1020_ = v_reuseFailAlloc_1021_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
return v___x_1020_;
}
}
}
}
}
else
{
lean_dec(v_a_1006_);
lean_dec_ref(v___x_1003_);
lean_del_object(v___x_993_);
lean_del_object(v___x_981_);
lean_dec(v_head_978_);
return v___x_1007_;
}
}
else
{
lean_object* v_a_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1032_; 
lean_dec_ref(v___x_1003_);
lean_del_object(v___x_993_);
lean_del_object(v___x_981_);
lean_dec(v_tail_979_);
lean_dec(v_head_978_);
v_a_1025_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1032_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1027_ = v___x_1005_;
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_a_1025_);
lean_dec(v___x_1005_);
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
else
{
lean_object* v_a_1033_; lean_object* v___x_1035_; uint8_t v_isShared_1036_; uint8_t v_isSharedCheck_1040_; 
lean_del_object(v___x_993_);
lean_del_object(v___x_981_);
lean_dec(v_tail_979_);
lean_dec(v_head_978_);
v_a_1033_ = lean_ctor_get(v___x_999_, 0);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_999_);
if (v_isSharedCheck_1040_ == 0)
{
v___x_1035_ = v___x_999_;
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
else
{
lean_inc(v_a_1033_);
lean_dec(v___x_999_);
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
}
else
{
lean_object* v___x_1043_; 
lean_dec(v_a_988_);
v___x_1043_ = lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx(v_tail_979_, v_a_972_, v_a_973_, v_a_974_, v_a_975_);
if (lean_obj_tag(v___x_1043_) == 0)
{
lean_object* v_a_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1054_; 
v_a_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1054_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1054_ == 0)
{
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1054_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_a_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1054_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v___x_1049_; 
if (v_isShared_982_ == 0)
{
lean_ctor_set(v___x_981_, 1, v_a_1044_);
v___x_1049_ = v___x_981_;
goto v_reusejp_1048_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v_head_978_);
lean_ctor_set(v_reuseFailAlloc_1053_, 1, v_a_1044_);
v___x_1049_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1048_;
}
v_reusejp_1048_:
{
lean_object* v___x_1051_; 
if (v_isShared_1047_ == 0)
{
lean_ctor_set(v___x_1046_, 0, v___x_1049_);
v___x_1051_ = v___x_1046_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v___x_1049_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
else
{
lean_del_object(v___x_981_);
lean_dec(v_head_978_);
return v___x_1043_;
}
}
}
else
{
lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1062_; 
lean_del_object(v___x_981_);
lean_dec(v_tail_979_);
lean_dec(v_head_978_);
v_a_1055_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1062_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1062_ == 0)
{
v___x_1057_ = v___x_987_;
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_987_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
lean_object* v___x_1060_; 
if (v_isShared_1058_ == 0)
{
v___x_1060_ = v___x_1057_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v_a_1055_);
v___x_1060_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
return v___x_1060_;
}
}
}
}
else
{
lean_object* v_a_1063_; lean_object* v___x_1065_; uint8_t v_isShared_1066_; uint8_t v_isSharedCheck_1070_; 
lean_del_object(v___x_981_);
lean_dec(v_tail_979_);
lean_dec(v_head_978_);
v_a_1063_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1065_ = v___x_985_;
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
else
{
lean_inc(v_a_1063_);
lean_dec(v___x_985_);
v___x_1065_ = lean_box(0);
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
v_resetjp_1064_:
{
lean_object* v___x_1068_; 
if (v_isShared_1066_ == 0)
{
v___x_1068_ = v___x_1065_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_a_1063_);
v___x_1068_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
return v___x_1068_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___boxed(lean_object* v_x_1072_, lean_object* v_a_1073_, lean_object* v_a_1074_, lean_object* v_a_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_){
_start:
{
lean_object* v_res_1078_; 
v_res_1078_ = lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx(v_x_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_);
lean_dec(v_a_1076_);
lean_dec_ref(v_a_1075_);
lean_dec(v_a_1074_);
lean_dec_ref(v_a_1073_);
return v_res_1078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing(lean_object* v_tac_1084_, lean_object* v_e_1085_, lean_object* v_a_1086_, lean_object* v_a_1087_, lean_object* v_a_1088_, lean_object* v_a_1089_){
_start:
{
lean_object* v___x_1091_; 
v___x_1091_ = lp_mathlib_Qq_inferTypeQ_x27(v_e_1085_, v_a_1086_, v_a_1087_, v_a_1088_, v_a_1089_);
if (lean_obj_tag(v___x_1091_) == 0)
{
lean_object* v_a_1092_; lean_object* v_snd_1093_; lean_object* v_fst_1094_; lean_object* v___x_1096_; uint8_t v_isShared_1097_; uint8_t v_isSharedCheck_1133_; 
v_a_1092_ = lean_ctor_get(v___x_1091_, 0);
lean_inc(v_a_1092_);
lean_dec_ref_known(v___x_1091_, 1);
v_snd_1093_ = lean_ctor_get(v_a_1092_, 1);
v_fst_1094_ = lean_ctor_get(v_a_1092_, 0);
v_isSharedCheck_1133_ = !lean_is_exclusive(v_a_1092_);
if (v_isSharedCheck_1133_ == 0)
{
v___x_1096_ = v_a_1092_;
v_isShared_1097_ = v_isSharedCheck_1133_;
goto v_resetjp_1095_;
}
else
{
lean_inc(v_snd_1093_);
lean_inc(v_fst_1094_);
lean_dec(v_a_1092_);
v___x_1096_ = lean_box(0);
v_isShared_1097_ = v_isSharedCheck_1133_;
goto v_resetjp_1095_;
}
v_resetjp_1095_:
{
lean_object* v_fst_1098_; lean_object* v_snd_1099_; lean_object* v___x_1101_; uint8_t v_isShared_1102_; uint8_t v_isSharedCheck_1132_; 
v_fst_1098_ = lean_ctor_get(v_snd_1093_, 0);
v_snd_1099_ = lean_ctor_get(v_snd_1093_, 1);
v_isSharedCheck_1132_ = !lean_is_exclusive(v_snd_1093_);
if (v_isSharedCheck_1132_ == 0)
{
v___x_1101_ = v_snd_1093_;
v_isShared_1102_ = v_isSharedCheck_1132_;
goto v_resetjp_1100_;
}
else
{
lean_inc(v_snd_1099_);
lean_inc(v_fst_1098_);
lean_dec(v_snd_1093_);
v___x_1101_ = lean_box(0);
v_isShared_1102_ = v_isSharedCheck_1132_;
goto v_resetjp_1100_;
}
v_resetjp_1100_:
{
lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1107_; 
lean_inc(v_fst_1094_);
v___x_1103_ = l_Lean_Level_succ___override(v_fst_1094_);
v___x_1104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__0));
v___x_1105_ = lean_box(0);
if (v_isShared_1102_ == 0)
{
lean_ctor_set_tag(v___x_1101_, 1);
lean_ctor_set(v___x_1101_, 1, v___x_1105_);
lean_ctor_set(v___x_1101_, 0, v_fst_1094_);
v___x_1107_ = v___x_1101_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v_fst_1094_);
lean_ctor_set(v_reuseFailAlloc_1131_, 1, v___x_1105_);
v___x_1107_ = v_reuseFailAlloc_1131_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; 
lean_inc_ref(v___x_1107_);
v___x_1108_ = l_Lean_Expr_const___override(v___x_1104_, v___x_1107_);
lean_inc(v_fst_1098_);
v___x_1109_ = l_Lean_Expr_app___override(v___x_1108_, v_fst_1098_);
v___x_1110_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1109_, v_a_1086_, v_a_1087_, v_a_1088_, v_a_1089_);
if (lean_obj_tag(v___x_1110_) == 0)
{
lean_object* v_a_1111_; lean_object* v___x_1112_; lean_object* v___x_1114_; 
v_a_1111_ = lean_ctor_get(v___x_1110_, 0);
lean_inc(v_a_1111_);
lean_dec_ref_known(v___x_1110_, 1);
v___x_1112_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___closed__2));
if (v_isShared_1097_ == 0)
{
lean_ctor_set_tag(v___x_1096_, 1);
lean_ctor_set(v___x_1096_, 1, v___x_1105_);
lean_ctor_set(v___x_1096_, 0, v___x_1103_);
v___x_1114_ = v___x_1096_;
goto v_reusejp_1113_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v___x_1103_);
lean_ctor_set(v_reuseFailAlloc_1130_, 1, v___x_1105_);
v___x_1114_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1113_;
}
v_reusejp_1113_:
{
lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; 
v___x_1115_ = l_Lean_Expr_const___override(v___x_1112_, v___x_1114_);
lean_inc_n(v_fst_1098_, 2);
v___x_1116_ = l_Lean_Expr_app___override(v___x_1115_, v_fst_1098_);
v___x_1117_ = l_Lean_Expr_app___override(v___x_1116_, v_snd_1099_);
v___x_1118_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__2));
lean_inc_ref(v___x_1107_);
v___x_1119_ = l_Lean_Expr_const___override(v___x_1118_, v___x_1107_);
v___x_1120_ = l_Lean_Expr_app___override(v___x_1119_, v_fst_1098_);
v___x_1121_ = lean_obj_once(&lp_mathlib_Qq_ofNatQ___closed__4, &lp_mathlib_Qq_ofNatQ___closed__4_once, _init_lp_mathlib_Qq_ofNatQ___closed__4);
v___x_1122_ = l_Lean_Expr_app___override(v___x_1120_, v___x_1121_);
v___x_1123_ = ((lean_object*)(lp_mathlib_Qq_ofNatQ___closed__7));
v___x_1124_ = l_Lean_Expr_const___override(v___x_1123_, v___x_1107_);
v___x_1125_ = l_Lean_Expr_app___override(v___x_1124_, v_fst_1098_);
v___x_1126_ = l_Lean_Expr_app___override(v___x_1125_, v_a_1111_);
v___x_1127_ = l_Lean_Expr_app___override(v___x_1122_, v___x_1126_);
v___x_1128_ = l_Lean_Expr_app___override(v___x_1117_, v___x_1127_);
v___x_1129_ = lp_mathlib_synthesizeUsing_x27___redArg(v___x_1128_, v_tac_1084_, v_a_1086_, v_a_1087_, v_a_1088_, v_a_1089_);
return v___x_1129_;
}
}
else
{
lean_dec_ref(v___x_1107_);
lean_dec(v___x_1103_);
lean_dec(v_snd_1099_);
lean_dec(v_fst_1098_);
lean_del_object(v___x_1096_);
lean_dec_ref(v_tac_1084_);
return v___x_1110_;
}
}
}
}
}
else
{
lean_object* v_a_1134_; lean_object* v___x_1136_; uint8_t v_isShared_1137_; uint8_t v_isSharedCheck_1141_; 
lean_dec_ref(v_tac_1084_);
v_a_1134_ = lean_ctor_get(v___x_1091_, 0);
v_isSharedCheck_1141_ = !lean_is_exclusive(v___x_1091_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1136_ = v___x_1091_;
v_isShared_1137_ = v_isSharedCheck_1141_;
goto v_resetjp_1135_;
}
else
{
lean_inc(v_a_1134_);
lean_dec(v___x_1091_);
v___x_1136_ = lean_box(0);
v_isShared_1137_ = v_isSharedCheck_1141_;
goto v_resetjp_1135_;
}
v_resetjp_1135_:
{
lean_object* v___x_1139_; 
if (v_isShared_1137_ == 0)
{
v___x_1139_ = v___x_1136_;
goto v_reusejp_1138_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v_a_1134_);
v___x_1139_ = v_reuseFailAlloc_1140_;
goto v_reusejp_1138_;
}
v_reusejp_1138_:
{
return v___x_1139_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___boxed(lean_object* v_tac_1142_, lean_object* v_e_1143_, lean_object* v_a_1144_, lean_object* v_a_1145_, lean_object* v_a_1146_, lean_object* v_a_1147_, lean_object* v_a_1148_){
_start:
{
lean_object* v_res_1149_; 
v_res_1149_ = lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing(v_tac_1142_, v_e_1143_, v_a_1144_, v_a_1145_, v_a_1146_, v_a_1147_);
lean_dec(v_a_1147_);
lean_dec_ref(v_a_1146_);
lean_dec(v_a_1145_);
lean_dec_ref(v_a_1144_);
return v_res_1149_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; 
v___x_1150_ = lean_unsigned_to_nat(32u);
v___x_1151_ = lean_mk_empty_array_with_capacity(v___x_1150_);
v___x_1152_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1152_, 0, v___x_1151_);
return v___x_1152_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1(void){
_start:
{
size_t v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1153_ = ((size_t)5ULL);
v___x_1154_ = lean_unsigned_to_nat(0u);
v___x_1155_ = lean_unsigned_to_nat(32u);
v___x_1156_ = lean_mk_empty_array_with_capacity(v___x_1155_);
v___x_1157_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__0);
v___x_1158_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1158_, 0, v___x_1157_);
lean_ctor_set(v___x_1158_, 1, v___x_1156_);
lean_ctor_set(v___x_1158_, 2, v___x_1154_);
lean_ctor_set(v___x_1158_, 3, v___x_1154_);
lean_ctor_set_usize(v___x_1158_, 4, v___x_1153_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(lean_object* v___y_1159_){
_start:
{
lean_object* v___x_1161_; lean_object* v_traceState_1162_; lean_object* v_traces_1163_; lean_object* v___x_1164_; lean_object* v_traceState_1165_; lean_object* v_env_1166_; lean_object* v_nextMacroScope_1167_; lean_object* v_ngen_1168_; lean_object* v_auxDeclNGen_1169_; lean_object* v_cache_1170_; lean_object* v_messages_1171_; lean_object* v_infoState_1172_; lean_object* v_snapshotTasks_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1192_; 
v___x_1161_ = lean_st_ref_get(v___y_1159_);
v_traceState_1162_ = lean_ctor_get(v___x_1161_, 4);
lean_inc_ref(v_traceState_1162_);
lean_dec(v___x_1161_);
v_traces_1163_ = lean_ctor_get(v_traceState_1162_, 0);
lean_inc_ref(v_traces_1163_);
lean_dec_ref(v_traceState_1162_);
v___x_1164_ = lean_st_ref_take(v___y_1159_);
v_traceState_1165_ = lean_ctor_get(v___x_1164_, 4);
v_env_1166_ = lean_ctor_get(v___x_1164_, 0);
v_nextMacroScope_1167_ = lean_ctor_get(v___x_1164_, 1);
v_ngen_1168_ = lean_ctor_get(v___x_1164_, 2);
v_auxDeclNGen_1169_ = lean_ctor_get(v___x_1164_, 3);
v_cache_1170_ = lean_ctor_get(v___x_1164_, 5);
v_messages_1171_ = lean_ctor_get(v___x_1164_, 6);
v_infoState_1172_ = lean_ctor_get(v___x_1164_, 7);
v_snapshotTasks_1173_ = lean_ctor_get(v___x_1164_, 8);
v_isSharedCheck_1192_ = !lean_is_exclusive(v___x_1164_);
if (v_isSharedCheck_1192_ == 0)
{
v___x_1175_ = v___x_1164_;
v_isShared_1176_ = v_isSharedCheck_1192_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_snapshotTasks_1173_);
lean_inc(v_infoState_1172_);
lean_inc(v_messages_1171_);
lean_inc(v_cache_1170_);
lean_inc(v_traceState_1165_);
lean_inc(v_auxDeclNGen_1169_);
lean_inc(v_ngen_1168_);
lean_inc(v_nextMacroScope_1167_);
lean_inc(v_env_1166_);
lean_dec(v___x_1164_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1192_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
uint64_t v_tid_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1190_; 
v_tid_1177_ = lean_ctor_get_uint64(v_traceState_1165_, sizeof(void*)*1);
v_isSharedCheck_1190_ = !lean_is_exclusive(v_traceState_1165_);
if (v_isSharedCheck_1190_ == 0)
{
lean_object* v_unused_1191_; 
v_unused_1191_ = lean_ctor_get(v_traceState_1165_, 0);
lean_dec(v_unused_1191_);
v___x_1179_ = v_traceState_1165_;
v_isShared_1180_ = v_isSharedCheck_1190_;
goto v_resetjp_1178_;
}
else
{
lean_dec(v_traceState_1165_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1190_;
goto v_resetjp_1178_;
}
v_resetjp_1178_:
{
lean_object* v___x_1181_; lean_object* v___x_1183_; 
v___x_1181_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___closed__1);
if (v_isShared_1180_ == 0)
{
lean_ctor_set(v___x_1179_, 0, v___x_1181_);
v___x_1183_ = v___x_1179_;
goto v_reusejp_1182_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v___x_1181_);
lean_ctor_set_uint64(v_reuseFailAlloc_1189_, sizeof(void*)*1, v_tid_1177_);
v___x_1183_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1182_;
}
v_reusejp_1182_:
{
lean_object* v___x_1185_; 
if (v_isShared_1176_ == 0)
{
lean_ctor_set(v___x_1175_, 4, v___x_1183_);
v___x_1185_ = v___x_1175_;
goto v_reusejp_1184_;
}
else
{
lean_object* v_reuseFailAlloc_1188_; 
v_reuseFailAlloc_1188_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1188_, 0, v_env_1166_);
lean_ctor_set(v_reuseFailAlloc_1188_, 1, v_nextMacroScope_1167_);
lean_ctor_set(v_reuseFailAlloc_1188_, 2, v_ngen_1168_);
lean_ctor_set(v_reuseFailAlloc_1188_, 3, v_auxDeclNGen_1169_);
lean_ctor_set(v_reuseFailAlloc_1188_, 4, v___x_1183_);
lean_ctor_set(v_reuseFailAlloc_1188_, 5, v_cache_1170_);
lean_ctor_set(v_reuseFailAlloc_1188_, 6, v_messages_1171_);
lean_ctor_set(v_reuseFailAlloc_1188_, 7, v_infoState_1172_);
lean_ctor_set(v_reuseFailAlloc_1188_, 8, v_snapshotTasks_1173_);
v___x_1185_ = v_reuseFailAlloc_1188_;
goto v_reusejp_1184_;
}
v_reusejp_1184_:
{
lean_object* v___x_1186_; lean_object* v___x_1187_; 
v___x_1186_ = lean_st_ref_set(v___y_1159_, v___x_1185_);
v___x_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1187_, 0, v_traces_1163_);
return v___x_1187_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg___boxed(lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
lean_object* v_res_1195_; 
v_res_1195_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(v___y_1193_);
lean_dec(v___y_1193_);
return v_res_1195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0(lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_){
_start:
{
lean_object* v___x_1201_; 
v___x_1201_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(v___y_1199_);
return v___x_1201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___boxed(lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0(v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
return v_res_1207_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(lean_object* v_opts_1208_, lean_object* v_opt_1209_){
_start:
{
lean_object* v_name_1210_; lean_object* v_defValue_1211_; lean_object* v_map_1212_; lean_object* v___x_1213_; 
v_name_1210_ = lean_ctor_get(v_opt_1209_, 0);
v_defValue_1211_ = lean_ctor_get(v_opt_1209_, 1);
v_map_1212_ = lean_ctor_get(v_opts_1208_, 0);
v___x_1213_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1212_, v_name_1210_);
if (lean_obj_tag(v___x_1213_) == 0)
{
uint8_t v___x_1214_; 
v___x_1214_ = lean_unbox(v_defValue_1211_);
return v___x_1214_;
}
else
{
lean_object* v_val_1215_; 
v_val_1215_ = lean_ctor_get(v___x_1213_, 0);
lean_inc(v_val_1215_);
lean_dec_ref_known(v___x_1213_, 1);
if (lean_obj_tag(v_val_1215_) == 1)
{
uint8_t v_v_1216_; 
v_v_1216_ = lean_ctor_get_uint8(v_val_1215_, 0);
lean_dec_ref_known(v_val_1215_, 0);
return v_v_1216_;
}
else
{
uint8_t v___x_1217_; 
lean_dec(v_val_1215_);
v___x_1217_ = lean_unbox(v_defValue_1211_);
return v___x_1217_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1___boxed(lean_object* v_opts_1218_, lean_object* v_opt_1219_){
_start:
{
uint8_t v_res_1220_; lean_object* v_r_1221_; 
v_res_1220_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_1218_, v_opt_1219_);
lean_dec_ref(v_opt_1219_);
lean_dec_ref(v_opts_1218_);
v_r_1221_ = lean_box(v_res_1220_);
return v_r_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0(lean_object* v_s_1222_, lean_object* v_x_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_){
_start:
{
lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___x_1229_ = l_Lean_stringToMessageData(v_s_1222_);
v___x_1230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1230_, 0, v___x_1229_);
return v___x_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0___boxed(lean_object* v_s_1231_, lean_object* v_x_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_){
_start:
{
lean_object* v_res_1238_; 
v_res_1238_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0(v_s_1231_, v_x_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_);
lean_dec(v___y_1236_);
lean_dec_ref(v___y_1235_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
lean_dec_ref(v_x_1232_);
return v_res_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(lean_object* v_x_1239_){
_start:
{
if (lean_obj_tag(v_x_1239_) == 0)
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
v_a_1241_ = lean_ctor_get(v_x_1239_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v_x_1239_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v_x_1239_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v_x_1239_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
lean_ctor_set_tag(v___x_1243_, 1);
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_a_1241_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
else
{
lean_object* v_a_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1256_; 
v_a_1249_ = lean_ctor_get(v_x_1239_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v_x_1239_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v_x_1239_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v_x_1239_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1254_; 
if (v_isShared_1252_ == 0)
{
lean_ctor_set_tag(v___x_1251_, 0);
v___x_1254_ = v___x_1251_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(0, 1, 0);
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
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg___boxed(lean_object* v_x_1257_, lean_object* v___y_1258_){
_start:
{
lean_object* v_res_1259_; 
v_res_1259_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_x_1257_);
return v_res_1259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3(size_t v_sz_1260_, size_t v_i_1261_, lean_object* v_bs_1262_){
_start:
{
uint8_t v___x_1263_; 
v___x_1263_ = lean_usize_dec_lt(v_i_1261_, v_sz_1260_);
if (v___x_1263_ == 0)
{
return v_bs_1262_;
}
else
{
lean_object* v_v_1264_; lean_object* v_msg_1265_; lean_object* v___x_1266_; lean_object* v_bs_x27_1267_; size_t v___x_1268_; size_t v___x_1269_; lean_object* v___x_1270_; 
v_v_1264_ = lean_array_uget_borrowed(v_bs_1262_, v_i_1261_);
v_msg_1265_ = lean_ctor_get(v_v_1264_, 1);
lean_inc_ref(v_msg_1265_);
v___x_1266_ = lean_unsigned_to_nat(0u);
v_bs_x27_1267_ = lean_array_uset(v_bs_1262_, v_i_1261_, v___x_1266_);
v___x_1268_ = ((size_t)1ULL);
v___x_1269_ = lean_usize_add(v_i_1261_, v___x_1268_);
v___x_1270_ = lean_array_uset(v_bs_x27_1267_, v_i_1261_, v_msg_1265_);
v_i_1261_ = v___x_1269_;
v_bs_1262_ = v___x_1270_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3___boxed(lean_object* v_sz_1272_, lean_object* v_i_1273_, lean_object* v_bs_1274_){
_start:
{
size_t v_sz_boxed_1275_; size_t v_i_boxed_1276_; lean_object* v_res_1277_; 
v_sz_boxed_1275_ = lean_unbox_usize(v_sz_1272_);
lean_dec(v_sz_1272_);
v_i_boxed_1276_ = lean_unbox_usize(v_i_1273_);
lean_dec(v_i_1273_);
v_res_1277_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3(v_sz_boxed_1275_, v_i_boxed_1276_, v_bs_1274_);
return v_res_1277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(lean_object* v_oldTraces_1278_, lean_object* v_data_1279_, lean_object* v_ref_1280_, lean_object* v_msg_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_){
_start:
{
lean_object* v_fileName_1287_; lean_object* v_fileMap_1288_; lean_object* v_options_1289_; lean_object* v_currRecDepth_1290_; lean_object* v_maxRecDepth_1291_; lean_object* v_ref_1292_; lean_object* v_currNamespace_1293_; lean_object* v_openDecls_1294_; lean_object* v_initHeartbeats_1295_; lean_object* v_maxHeartbeats_1296_; lean_object* v_quotContext_1297_; lean_object* v_currMacroScope_1298_; uint8_t v_diag_1299_; lean_object* v_cancelTk_x3f_1300_; uint8_t v_suppressElabErrors_1301_; lean_object* v_inheritedTraceOptions_1302_; lean_object* v___x_1303_; lean_object* v_traceState_1304_; lean_object* v_traces_1305_; lean_object* v_ref_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; size_t v_sz_1309_; size_t v___x_1310_; lean_object* v___x_1311_; lean_object* v_msg_1312_; lean_object* v___x_1313_; lean_object* v_a_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1351_; 
v_fileName_1287_ = lean_ctor_get(v___y_1284_, 0);
v_fileMap_1288_ = lean_ctor_get(v___y_1284_, 1);
v_options_1289_ = lean_ctor_get(v___y_1284_, 2);
v_currRecDepth_1290_ = lean_ctor_get(v___y_1284_, 3);
v_maxRecDepth_1291_ = lean_ctor_get(v___y_1284_, 4);
v_ref_1292_ = lean_ctor_get(v___y_1284_, 5);
v_currNamespace_1293_ = lean_ctor_get(v___y_1284_, 6);
v_openDecls_1294_ = lean_ctor_get(v___y_1284_, 7);
v_initHeartbeats_1295_ = lean_ctor_get(v___y_1284_, 8);
v_maxHeartbeats_1296_ = lean_ctor_get(v___y_1284_, 9);
v_quotContext_1297_ = lean_ctor_get(v___y_1284_, 10);
v_currMacroScope_1298_ = lean_ctor_get(v___y_1284_, 11);
v_diag_1299_ = lean_ctor_get_uint8(v___y_1284_, sizeof(void*)*14);
v_cancelTk_x3f_1300_ = lean_ctor_get(v___y_1284_, 12);
v_suppressElabErrors_1301_ = lean_ctor_get_uint8(v___y_1284_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1302_ = lean_ctor_get(v___y_1284_, 13);
v___x_1303_ = lean_st_ref_get(v___y_1285_);
v_traceState_1304_ = lean_ctor_get(v___x_1303_, 4);
lean_inc_ref(v_traceState_1304_);
lean_dec(v___x_1303_);
v_traces_1305_ = lean_ctor_get(v_traceState_1304_, 0);
lean_inc_ref(v_traces_1305_);
lean_dec_ref(v_traceState_1304_);
v_ref_1306_ = l_Lean_replaceRef(v_ref_1280_, v_ref_1292_);
lean_inc_ref(v_inheritedTraceOptions_1302_);
lean_inc(v_cancelTk_x3f_1300_);
lean_inc(v_currMacroScope_1298_);
lean_inc(v_quotContext_1297_);
lean_inc(v_maxHeartbeats_1296_);
lean_inc(v_initHeartbeats_1295_);
lean_inc(v_openDecls_1294_);
lean_inc(v_currNamespace_1293_);
lean_inc(v_maxRecDepth_1291_);
lean_inc(v_currRecDepth_1290_);
lean_inc_ref(v_options_1289_);
lean_inc_ref(v_fileMap_1288_);
lean_inc_ref(v_fileName_1287_);
v___x_1307_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1307_, 0, v_fileName_1287_);
lean_ctor_set(v___x_1307_, 1, v_fileMap_1288_);
lean_ctor_set(v___x_1307_, 2, v_options_1289_);
lean_ctor_set(v___x_1307_, 3, v_currRecDepth_1290_);
lean_ctor_set(v___x_1307_, 4, v_maxRecDepth_1291_);
lean_ctor_set(v___x_1307_, 5, v_ref_1306_);
lean_ctor_set(v___x_1307_, 6, v_currNamespace_1293_);
lean_ctor_set(v___x_1307_, 7, v_openDecls_1294_);
lean_ctor_set(v___x_1307_, 8, v_initHeartbeats_1295_);
lean_ctor_set(v___x_1307_, 9, v_maxHeartbeats_1296_);
lean_ctor_set(v___x_1307_, 10, v_quotContext_1297_);
lean_ctor_set(v___x_1307_, 11, v_currMacroScope_1298_);
lean_ctor_set(v___x_1307_, 12, v_cancelTk_x3f_1300_);
lean_ctor_set(v___x_1307_, 13, v_inheritedTraceOptions_1302_);
lean_ctor_set_uint8(v___x_1307_, sizeof(void*)*14, v_diag_1299_);
lean_ctor_set_uint8(v___x_1307_, sizeof(void*)*14 + 1, v_suppressElabErrors_1301_);
v___x_1308_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1305_);
lean_dec_ref(v_traces_1305_);
v_sz_1309_ = lean_array_size(v___x_1308_);
v___x_1310_ = ((size_t)0ULL);
v___x_1311_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2_spec__3(v_sz_1309_, v___x_1310_, v___x_1308_);
v_msg_1312_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1312_, 0, v_data_1279_);
lean_ctor_set(v_msg_1312_, 1, v_msg_1281_);
lean_ctor_set(v_msg_1312_, 2, v___x_1311_);
v___x_1313_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(v_msg_1312_, v___y_1282_, v___y_1283_, v___x_1307_, v___y_1285_);
lean_dec_ref_known(v___x_1307_, 14);
v_a_1314_ = lean_ctor_get(v___x_1313_, 0);
v_isSharedCheck_1351_ = !lean_is_exclusive(v___x_1313_);
if (v_isSharedCheck_1351_ == 0)
{
v___x_1316_ = v___x_1313_;
v_isShared_1317_ = v_isSharedCheck_1351_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_a_1314_);
lean_dec(v___x_1313_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1351_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___x_1318_; lean_object* v_traceState_1319_; lean_object* v_env_1320_; lean_object* v_nextMacroScope_1321_; lean_object* v_ngen_1322_; lean_object* v_auxDeclNGen_1323_; lean_object* v_cache_1324_; lean_object* v_messages_1325_; lean_object* v_infoState_1326_; lean_object* v_snapshotTasks_1327_; lean_object* v___x_1329_; uint8_t v_isShared_1330_; uint8_t v_isSharedCheck_1350_; 
v___x_1318_ = lean_st_ref_take(v___y_1285_);
v_traceState_1319_ = lean_ctor_get(v___x_1318_, 4);
v_env_1320_ = lean_ctor_get(v___x_1318_, 0);
v_nextMacroScope_1321_ = lean_ctor_get(v___x_1318_, 1);
v_ngen_1322_ = lean_ctor_get(v___x_1318_, 2);
v_auxDeclNGen_1323_ = lean_ctor_get(v___x_1318_, 3);
v_cache_1324_ = lean_ctor_get(v___x_1318_, 5);
v_messages_1325_ = lean_ctor_get(v___x_1318_, 6);
v_infoState_1326_ = lean_ctor_get(v___x_1318_, 7);
v_snapshotTasks_1327_ = lean_ctor_get(v___x_1318_, 8);
v_isSharedCheck_1350_ = !lean_is_exclusive(v___x_1318_);
if (v_isSharedCheck_1350_ == 0)
{
v___x_1329_ = v___x_1318_;
v_isShared_1330_ = v_isSharedCheck_1350_;
goto v_resetjp_1328_;
}
else
{
lean_inc(v_snapshotTasks_1327_);
lean_inc(v_infoState_1326_);
lean_inc(v_messages_1325_);
lean_inc(v_cache_1324_);
lean_inc(v_traceState_1319_);
lean_inc(v_auxDeclNGen_1323_);
lean_inc(v_ngen_1322_);
lean_inc(v_nextMacroScope_1321_);
lean_inc(v_env_1320_);
lean_dec(v___x_1318_);
v___x_1329_ = lean_box(0);
v_isShared_1330_ = v_isSharedCheck_1350_;
goto v_resetjp_1328_;
}
v_resetjp_1328_:
{
uint64_t v_tid_1331_; lean_object* v___x_1333_; uint8_t v_isShared_1334_; uint8_t v_isSharedCheck_1348_; 
v_tid_1331_ = lean_ctor_get_uint64(v_traceState_1319_, sizeof(void*)*1);
v_isSharedCheck_1348_ = !lean_is_exclusive(v_traceState_1319_);
if (v_isSharedCheck_1348_ == 0)
{
lean_object* v_unused_1349_; 
v_unused_1349_ = lean_ctor_get(v_traceState_1319_, 0);
lean_dec(v_unused_1349_);
v___x_1333_ = v_traceState_1319_;
v_isShared_1334_ = v_isSharedCheck_1348_;
goto v_resetjp_1332_;
}
else
{
lean_dec(v_traceState_1319_);
v___x_1333_ = lean_box(0);
v_isShared_1334_ = v_isSharedCheck_1348_;
goto v_resetjp_1332_;
}
v_resetjp_1332_:
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1338_; 
v___x_1335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1335_, 0, v_ref_1280_);
lean_ctor_set(v___x_1335_, 1, v_a_1314_);
v___x_1336_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1278_, v___x_1335_);
if (v_isShared_1334_ == 0)
{
lean_ctor_set(v___x_1333_, 0, v___x_1336_);
v___x_1338_ = v___x_1333_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1347_; 
v_reuseFailAlloc_1347_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1347_, 0, v___x_1336_);
lean_ctor_set_uint64(v_reuseFailAlloc_1347_, sizeof(void*)*1, v_tid_1331_);
v___x_1338_ = v_reuseFailAlloc_1347_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
lean_object* v___x_1340_; 
if (v_isShared_1330_ == 0)
{
lean_ctor_set(v___x_1329_, 4, v___x_1338_);
v___x_1340_ = v___x_1329_;
goto v_reusejp_1339_;
}
else
{
lean_object* v_reuseFailAlloc_1346_; 
v_reuseFailAlloc_1346_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1346_, 0, v_env_1320_);
lean_ctor_set(v_reuseFailAlloc_1346_, 1, v_nextMacroScope_1321_);
lean_ctor_set(v_reuseFailAlloc_1346_, 2, v_ngen_1322_);
lean_ctor_set(v_reuseFailAlloc_1346_, 3, v_auxDeclNGen_1323_);
lean_ctor_set(v_reuseFailAlloc_1346_, 4, v___x_1338_);
lean_ctor_set(v_reuseFailAlloc_1346_, 5, v_cache_1324_);
lean_ctor_set(v_reuseFailAlloc_1346_, 6, v_messages_1325_);
lean_ctor_set(v_reuseFailAlloc_1346_, 7, v_infoState_1326_);
lean_ctor_set(v_reuseFailAlloc_1346_, 8, v_snapshotTasks_1327_);
v___x_1340_ = v_reuseFailAlloc_1346_;
goto v_reusejp_1339_;
}
v_reusejp_1339_:
{
lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1344_; 
v___x_1341_ = lean_st_ref_set(v___y_1285_, v___x_1340_);
v___x_1342_ = lean_box(0);
if (v_isShared_1317_ == 0)
{
lean_ctor_set(v___x_1316_, 0, v___x_1342_);
v___x_1344_ = v___x_1316_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v___x_1342_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
return v___x_1344_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2___boxed(lean_object* v_oldTraces_1352_, lean_object* v_data_1353_, lean_object* v_ref_1354_, lean_object* v_msg_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_){
_start:
{
lean_object* v_res_1361_; 
v_res_1361_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(v_oldTraces_1352_, v_data_1353_, v_ref_1354_, v_msg_1355_, v___y_1356_, v___y_1357_, v___y_1358_, v___y_1359_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
lean_dec(v___y_1357_);
lean_dec_ref(v___y_1356_);
return v_res_1361_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg(lean_object* v_e_1362_){
_start:
{
if (lean_obj_tag(v_e_1362_) == 0)
{
uint8_t v___x_1363_; 
v___x_1363_ = 2;
return v___x_1363_;
}
else
{
uint8_t v___x_1364_; 
v___x_1364_ = 0;
return v___x_1364_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg___boxed(lean_object* v_e_1365_){
_start:
{
uint8_t v_res_1366_; lean_object* v_r_1367_; 
v_res_1366_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg(v_e_1365_);
lean_dec_ref(v_e_1365_);
v_r_1367_ = lean_box(v_res_1366_);
return v_r_1367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(lean_object* v_opts_1368_, lean_object* v_opt_1369_){
_start:
{
lean_object* v_name_1370_; lean_object* v_defValue_1371_; lean_object* v_map_1372_; lean_object* v___x_1373_; 
v_name_1370_ = lean_ctor_get(v_opt_1369_, 0);
v_defValue_1371_ = lean_ctor_get(v_opt_1369_, 1);
v_map_1372_ = lean_ctor_get(v_opts_1368_, 0);
v___x_1373_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1372_, v_name_1370_);
if (lean_obj_tag(v___x_1373_) == 0)
{
lean_inc(v_defValue_1371_);
return v_defValue_1371_;
}
else
{
lean_object* v_val_1374_; 
v_val_1374_ = lean_ctor_get(v___x_1373_, 0);
lean_inc(v_val_1374_);
lean_dec_ref_known(v___x_1373_, 1);
if (lean_obj_tag(v_val_1374_) == 3)
{
lean_object* v_v_1375_; 
v_v_1375_ = lean_ctor_get(v_val_1374_, 0);
lean_inc(v_v_1375_);
lean_dec_ref_known(v_val_1374_, 1);
return v_v_1375_;
}
else
{
lean_dec(v_val_1374_);
lean_inc(v_defValue_1371_);
return v_defValue_1371_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5___boxed(lean_object* v_opts_1376_, lean_object* v_opt_1377_){
_start:
{
lean_object* v_res_1378_; 
v_res_1378_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_1376_, v_opt_1377_);
lean_dec_ref(v_opt_1377_);
lean_dec_ref(v_opts_1376_);
return v_res_1378_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1379_; double v___x_1380_; 
v___x_1379_ = lean_unsigned_to_nat(0u);
v___x_1380_ = lean_float_of_nat(v___x_1379_);
return v___x_1380_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; 
v___x_1382_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__1));
v___x_1383_ = l_Lean_stringToMessageData(v___x_1382_);
return v___x_1383_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1384_; double v___x_1385_; 
v___x_1384_ = lean_unsigned_to_nat(1000u);
v___x_1385_ = lean_float_of_nat(v___x_1384_);
return v___x_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(lean_object* v_cls_1386_, uint8_t v_collapsed_1387_, lean_object* v_tag_1388_, lean_object* v_opts_1389_, uint8_t v_clsEnabled_1390_, lean_object* v_oldTraces_1391_, lean_object* v_msg_1392_, lean_object* v_resStartStop_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_){
_start:
{
lean_object* v_fst_1399_; lean_object* v_snd_1400_; lean_object* v___y_1402_; lean_object* v___y_1403_; lean_object* v_data_1404_; lean_object* v_fst_1415_; lean_object* v_snd_1416_; lean_object* v___x_1417_; uint8_t v___x_1418_; lean_object* v___y_1420_; lean_object* v_a_1421_; uint8_t v___y_1436_; double v___y_1467_; 
v_fst_1399_ = lean_ctor_get(v_resStartStop_1393_, 0);
lean_inc(v_fst_1399_);
v_snd_1400_ = lean_ctor_get(v_resStartStop_1393_, 1);
lean_inc(v_snd_1400_);
lean_dec_ref(v_resStartStop_1393_);
v_fst_1415_ = lean_ctor_get(v_snd_1400_, 0);
lean_inc(v_fst_1415_);
v_snd_1416_ = lean_ctor_get(v_snd_1400_, 1);
lean_inc(v_snd_1416_);
lean_dec(v_snd_1400_);
v___x_1417_ = l_Lean_trace_profiler;
v___x_1418_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_1389_, v___x_1417_);
if (v___x_1418_ == 0)
{
v___y_1436_ = v___x_1418_;
goto v___jp_1435_;
}
else
{
lean_object* v___x_1472_; uint8_t v___x_1473_; 
v___x_1472_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1473_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_1389_, v___x_1472_);
if (v___x_1473_ == 0)
{
lean_object* v___x_1474_; lean_object* v___x_1475_; double v___x_1476_; double v___x_1477_; double v___x_1478_; 
v___x_1474_ = l_Lean_trace_profiler_threshold;
v___x_1475_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_1389_, v___x_1474_);
v___x_1476_ = lean_float_of_nat(v___x_1475_);
v___x_1477_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3);
v___x_1478_ = lean_float_div(v___x_1476_, v___x_1477_);
v___y_1467_ = v___x_1478_;
goto v___jp_1466_;
}
else
{
lean_object* v___x_1479_; lean_object* v___x_1480_; double v___x_1481_; 
v___x_1479_ = l_Lean_trace_profiler_threshold;
v___x_1480_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_1389_, v___x_1479_);
v___x_1481_ = lean_float_of_nat(v___x_1480_);
v___y_1467_ = v___x_1481_;
goto v___jp_1466_;
}
}
v___jp_1401_:
{
lean_object* v___x_1405_; 
lean_inc(v___y_1403_);
v___x_1405_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(v_oldTraces_1391_, v_data_1404_, v___y_1403_, v___y_1402_, v___y_1394_, v___y_1395_, v___y_1396_, v___y_1397_);
if (lean_obj_tag(v___x_1405_) == 0)
{
lean_object* v___x_1406_; 
lean_dec_ref_known(v___x_1405_, 1);
v___x_1406_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_1399_);
return v___x_1406_;
}
else
{
lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1414_; 
lean_dec(v_fst_1399_);
v_a_1407_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1414_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1414_ == 0)
{
v___x_1409_ = v___x_1405_;
v_isShared_1410_ = v_isSharedCheck_1414_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___x_1405_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1414_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
lean_object* v___x_1412_; 
if (v_isShared_1410_ == 0)
{
v___x_1412_ = v___x_1409_;
goto v_reusejp_1411_;
}
else
{
lean_object* v_reuseFailAlloc_1413_; 
v_reuseFailAlloc_1413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1413_, 0, v_a_1407_);
v___x_1412_ = v_reuseFailAlloc_1413_;
goto v_reusejp_1411_;
}
v_reusejp_1411_:
{
return v___x_1412_;
}
}
}
}
v___jp_1419_:
{
uint8_t v_result_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; double v___x_1425_; lean_object* v_data_1426_; 
v_result_1422_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg(v_fst_1399_);
v___x_1423_ = lean_box(v_result_1422_);
v___x_1424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1424_, 0, v___x_1423_);
v___x_1425_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0);
lean_inc_ref(v_tag_1388_);
lean_inc_ref(v___x_1424_);
lean_inc(v_cls_1386_);
v_data_1426_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1426_, 0, v_cls_1386_);
lean_ctor_set(v_data_1426_, 1, v___x_1424_);
lean_ctor_set(v_data_1426_, 2, v_tag_1388_);
lean_ctor_set_float(v_data_1426_, sizeof(void*)*3, v___x_1425_);
lean_ctor_set_float(v_data_1426_, sizeof(void*)*3 + 8, v___x_1425_);
lean_ctor_set_uint8(v_data_1426_, sizeof(void*)*3 + 16, v_collapsed_1387_);
if (v___x_1418_ == 0)
{
lean_dec_ref_known(v___x_1424_, 1);
lean_dec(v_snd_1416_);
lean_dec(v_fst_1415_);
lean_dec_ref(v_tag_1388_);
lean_dec(v_cls_1386_);
v___y_1402_ = v_a_1421_;
v___y_1403_ = v___y_1420_;
v_data_1404_ = v_data_1426_;
goto v___jp_1401_;
}
else
{
lean_object* v_data_1427_; double v___x_1428_; double v___x_1429_; 
lean_dec_ref_known(v_data_1426_, 3);
v_data_1427_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1427_, 0, v_cls_1386_);
lean_ctor_set(v_data_1427_, 1, v___x_1424_);
lean_ctor_set(v_data_1427_, 2, v_tag_1388_);
v___x_1428_ = lean_unbox_float(v_fst_1415_);
lean_dec(v_fst_1415_);
lean_ctor_set_float(v_data_1427_, sizeof(void*)*3, v___x_1428_);
v___x_1429_ = lean_unbox_float(v_snd_1416_);
lean_dec(v_snd_1416_);
lean_ctor_set_float(v_data_1427_, sizeof(void*)*3 + 8, v___x_1429_);
lean_ctor_set_uint8(v_data_1427_, sizeof(void*)*3 + 16, v_collapsed_1387_);
v___y_1402_ = v_a_1421_;
v___y_1403_ = v___y_1420_;
v_data_1404_ = v_data_1427_;
goto v___jp_1401_;
}
}
v___jp_1430_:
{
lean_object* v_ref_1431_; lean_object* v___x_1432_; 
v_ref_1431_ = lean_ctor_get(v___y_1396_, 5);
lean_inc(v___y_1397_);
lean_inc_ref(v___y_1396_);
lean_inc(v___y_1395_);
lean_inc_ref(v___y_1394_);
lean_inc(v_fst_1399_);
v___x_1432_ = lean_apply_6(v_msg_1392_, v_fst_1399_, v___y_1394_, v___y_1395_, v___y_1396_, v___y_1397_, lean_box(0));
if (lean_obj_tag(v___x_1432_) == 0)
{
lean_object* v_a_1433_; 
v_a_1433_ = lean_ctor_get(v___x_1432_, 0);
lean_inc(v_a_1433_);
lean_dec_ref_known(v___x_1432_, 1);
v___y_1420_ = v_ref_1431_;
v_a_1421_ = v_a_1433_;
goto v___jp_1419_;
}
else
{
lean_object* v___x_1434_; 
lean_dec_ref_known(v___x_1432_, 1);
v___x_1434_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2);
v___y_1420_ = v_ref_1431_;
v_a_1421_ = v___x_1434_;
goto v___jp_1419_;
}
}
v___jp_1435_:
{
if (v_clsEnabled_1390_ == 0)
{
if (v___y_1436_ == 0)
{
lean_object* v___x_1437_; lean_object* v_traceState_1438_; lean_object* v_env_1439_; lean_object* v_nextMacroScope_1440_; lean_object* v_ngen_1441_; lean_object* v_auxDeclNGen_1442_; lean_object* v_cache_1443_; lean_object* v_messages_1444_; lean_object* v_infoState_1445_; lean_object* v_snapshotTasks_1446_; lean_object* v___x_1448_; uint8_t v_isShared_1449_; uint8_t v_isSharedCheck_1465_; 
lean_dec(v_snd_1416_);
lean_dec(v_fst_1415_);
lean_dec_ref(v_msg_1392_);
lean_dec_ref(v_tag_1388_);
lean_dec(v_cls_1386_);
v___x_1437_ = lean_st_ref_take(v___y_1397_);
v_traceState_1438_ = lean_ctor_get(v___x_1437_, 4);
v_env_1439_ = lean_ctor_get(v___x_1437_, 0);
v_nextMacroScope_1440_ = lean_ctor_get(v___x_1437_, 1);
v_ngen_1441_ = lean_ctor_get(v___x_1437_, 2);
v_auxDeclNGen_1442_ = lean_ctor_get(v___x_1437_, 3);
v_cache_1443_ = lean_ctor_get(v___x_1437_, 5);
v_messages_1444_ = lean_ctor_get(v___x_1437_, 6);
v_infoState_1445_ = lean_ctor_get(v___x_1437_, 7);
v_snapshotTasks_1446_ = lean_ctor_get(v___x_1437_, 8);
v_isSharedCheck_1465_ = !lean_is_exclusive(v___x_1437_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1448_ = v___x_1437_;
v_isShared_1449_ = v_isSharedCheck_1465_;
goto v_resetjp_1447_;
}
else
{
lean_inc(v_snapshotTasks_1446_);
lean_inc(v_infoState_1445_);
lean_inc(v_messages_1444_);
lean_inc(v_cache_1443_);
lean_inc(v_traceState_1438_);
lean_inc(v_auxDeclNGen_1442_);
lean_inc(v_ngen_1441_);
lean_inc(v_nextMacroScope_1440_);
lean_inc(v_env_1439_);
lean_dec(v___x_1437_);
v___x_1448_ = lean_box(0);
v_isShared_1449_ = v_isSharedCheck_1465_;
goto v_resetjp_1447_;
}
v_resetjp_1447_:
{
uint64_t v_tid_1450_; lean_object* v_traces_1451_; lean_object* v___x_1453_; uint8_t v_isShared_1454_; uint8_t v_isSharedCheck_1464_; 
v_tid_1450_ = lean_ctor_get_uint64(v_traceState_1438_, sizeof(void*)*1);
v_traces_1451_ = lean_ctor_get(v_traceState_1438_, 0);
v_isSharedCheck_1464_ = !lean_is_exclusive(v_traceState_1438_);
if (v_isSharedCheck_1464_ == 0)
{
v___x_1453_ = v_traceState_1438_;
v_isShared_1454_ = v_isSharedCheck_1464_;
goto v_resetjp_1452_;
}
else
{
lean_inc(v_traces_1451_);
lean_dec(v_traceState_1438_);
v___x_1453_ = lean_box(0);
v_isShared_1454_ = v_isSharedCheck_1464_;
goto v_resetjp_1452_;
}
v_resetjp_1452_:
{
lean_object* v___x_1455_; lean_object* v___x_1457_; 
v___x_1455_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1391_, v_traces_1451_);
lean_dec_ref(v_traces_1451_);
if (v_isShared_1454_ == 0)
{
lean_ctor_set(v___x_1453_, 0, v___x_1455_);
v___x_1457_ = v___x_1453_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v___x_1455_);
lean_ctor_set_uint64(v_reuseFailAlloc_1463_, sizeof(void*)*1, v_tid_1450_);
v___x_1457_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
lean_object* v___x_1459_; 
if (v_isShared_1449_ == 0)
{
lean_ctor_set(v___x_1448_, 4, v___x_1457_);
v___x_1459_ = v___x_1448_;
goto v_reusejp_1458_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v_env_1439_);
lean_ctor_set(v_reuseFailAlloc_1462_, 1, v_nextMacroScope_1440_);
lean_ctor_set(v_reuseFailAlloc_1462_, 2, v_ngen_1441_);
lean_ctor_set(v_reuseFailAlloc_1462_, 3, v_auxDeclNGen_1442_);
lean_ctor_set(v_reuseFailAlloc_1462_, 4, v___x_1457_);
lean_ctor_set(v_reuseFailAlloc_1462_, 5, v_cache_1443_);
lean_ctor_set(v_reuseFailAlloc_1462_, 6, v_messages_1444_);
lean_ctor_set(v_reuseFailAlloc_1462_, 7, v_infoState_1445_);
lean_ctor_set(v_reuseFailAlloc_1462_, 8, v_snapshotTasks_1446_);
v___x_1459_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1458_;
}
v_reusejp_1458_:
{
lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1460_ = lean_st_ref_set(v___y_1397_, v___x_1459_);
v___x_1461_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_1399_);
return v___x_1461_;
}
}
}
}
}
else
{
goto v___jp_1430_;
}
}
else
{
goto v___jp_1430_;
}
}
v___jp_1466_:
{
double v___x_1468_; double v___x_1469_; double v___x_1470_; uint8_t v___x_1471_; 
v___x_1468_ = lean_unbox_float(v_snd_1416_);
v___x_1469_ = lean_unbox_float(v_fst_1415_);
v___x_1470_ = lean_float_sub(v___x_1468_, v___x_1469_);
v___x_1471_ = lean_float_decLt(v___y_1467_, v___x_1470_);
v___y_1436_ = v___x_1471_;
goto v___jp_1435_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___boxed(lean_object* v_cls_1482_, lean_object* v_collapsed_1483_, lean_object* v_tag_1484_, lean_object* v_opts_1485_, lean_object* v_clsEnabled_1486_, lean_object* v_oldTraces_1487_, lean_object* v_msg_1488_, lean_object* v_resStartStop_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_){
_start:
{
uint8_t v_collapsed_boxed_1495_; uint8_t v_clsEnabled_boxed_1496_; lean_object* v_res_1497_; 
v_collapsed_boxed_1495_ = lean_unbox(v_collapsed_1483_);
v_clsEnabled_boxed_1496_ = lean_unbox(v_clsEnabled_1486_);
v_res_1497_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(v_cls_1482_, v_collapsed_boxed_1495_, v_tag_1484_, v_opts_1485_, v_clsEnabled_boxed_1496_, v_oldTraces_1487_, v_msg_1488_, v_resStartStop_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_);
lean_dec(v___y_1493_);
lean_dec_ref(v___y_1492_);
lean_dec(v___y_1491_);
lean_dec_ref(v___y_1490_);
lean_dec_ref(v_opts_1485_);
return v_res_1497_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6(void){
_start:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; 
v___x_1507_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2));
v___x_1508_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
v___x_1509_ = l_Lean_Name_append(v___x_1508_, v___x_1507_);
return v___x_1509_;
}
}
static double _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7(void){
_start:
{
lean_object* v___x_1510_; double v___x_1511_; 
v___x_1510_ = lean_unsigned_to_nat(1000000000u);
v___x_1511_ = lean_float_of_nat(v___x_1510_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(lean_object* v_s_1512_, lean_object* v_f_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_){
_start:
{
lean_object* v_options_1519_; uint8_t v_hasTrace_1520_; 
v_options_1519_ = lean_ctor_get(v_a_1516_, 2);
v_hasTrace_1520_ = lean_ctor_get_uint8(v_options_1519_, sizeof(void*)*1);
if (v_hasTrace_1520_ == 0)
{
lean_object* v___x_1521_; 
lean_dec_ref(v_s_1512_);
lean_inc(v_a_1517_);
lean_inc_ref(v_a_1516_);
lean_inc(v_a_1515_);
lean_inc_ref(v_a_1514_);
v___x_1521_ = lean_apply_5(v_f_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, lean_box(0));
return v___x_1521_;
}
else
{
lean_object* v_inheritedTraceOptions_1522_; lean_object* v___f_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; uint8_t v___x_1527_; lean_object* v___y_1529_; lean_object* v___y_1530_; lean_object* v_a_1531_; lean_object* v___y_1544_; lean_object* v___y_1545_; lean_object* v_a_1546_; 
v_inheritedTraceOptions_1522_ = lean_ctor_get(v_a_1516_, 13);
v___f_1523_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1523_, 0, v_s_1512_);
v___x_1524_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2));
v___x_1525_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3));
v___x_1526_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6);
v___x_1527_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1522_, v_options_1519_, v___x_1526_);
if (v___x_1527_ == 0)
{
lean_object* v___x_1596_; uint8_t v___x_1597_; 
v___x_1596_ = l_Lean_trace_profiler;
v___x_1597_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_options_1519_, v___x_1596_);
if (v___x_1597_ == 0)
{
lean_object* v___x_1598_; 
lean_dec_ref(v___f_1523_);
lean_inc(v_a_1517_);
lean_inc_ref(v_a_1516_);
lean_inc(v_a_1515_);
lean_inc_ref(v_a_1514_);
v___x_1598_ = lean_apply_5(v_f_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, lean_box(0));
return v___x_1598_;
}
else
{
goto v___jp_1555_;
}
}
else
{
goto v___jp_1555_;
}
v___jp_1528_:
{
lean_object* v___x_1532_; double v___x_1533_; double v___x_1534_; double v___x_1535_; double v___x_1536_; double v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
v___x_1532_ = lean_io_mono_nanos_now();
v___x_1533_ = lean_float_of_nat(v___y_1529_);
v___x_1534_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7);
v___x_1535_ = lean_float_div(v___x_1533_, v___x_1534_);
v___x_1536_ = lean_float_of_nat(v___x_1532_);
v___x_1537_ = lean_float_div(v___x_1536_, v___x_1534_);
v___x_1538_ = lean_box_float(v___x_1535_);
v___x_1539_ = lean_box_float(v___x_1537_);
v___x_1540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1540_, 0, v___x_1538_);
lean_ctor_set(v___x_1540_, 1, v___x_1539_);
v___x_1541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1541_, 0, v_a_1531_);
lean_ctor_set(v___x_1541_, 1, v___x_1540_);
v___x_1542_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(v___x_1524_, v_hasTrace_1520_, v___x_1525_, v_options_1519_, v___x_1527_, v___y_1530_, v___f_1523_, v___x_1541_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_);
return v___x_1542_;
}
v___jp_1543_:
{
lean_object* v___x_1547_; double v___x_1548_; double v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; 
v___x_1547_ = lean_io_get_num_heartbeats();
v___x_1548_ = lean_float_of_nat(v___y_1544_);
v___x_1549_ = lean_float_of_nat(v___x_1547_);
v___x_1550_ = lean_box_float(v___x_1548_);
v___x_1551_ = lean_box_float(v___x_1549_);
v___x_1552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1550_);
lean_ctor_set(v___x_1552_, 1, v___x_1551_);
v___x_1553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1553_, 0, v_a_1546_);
lean_ctor_set(v___x_1553_, 1, v___x_1552_);
v___x_1554_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(v___x_1524_, v_hasTrace_1520_, v___x_1525_, v_options_1519_, v___x_1527_, v___y_1545_, v___f_1523_, v___x_1553_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_);
return v___x_1554_;
}
v___jp_1555_:
{
lean_object* v___x_1556_; lean_object* v_a_1557_; lean_object* v___x_1558_; uint8_t v___x_1559_; 
v___x_1556_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(v_a_1517_);
v_a_1557_ = lean_ctor_get(v___x_1556_, 0);
lean_inc(v_a_1557_);
lean_dec_ref(v___x_1556_);
v___x_1558_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1559_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_options_1519_, v___x_1558_);
if (v___x_1559_ == 0)
{
lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1560_ = lean_io_mono_nanos_now();
lean_inc(v_a_1517_);
lean_inc_ref(v_a_1516_);
lean_inc(v_a_1515_);
lean_inc_ref(v_a_1514_);
v___x_1561_ = lean_apply_5(v_f_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, lean_box(0));
if (lean_obj_tag(v___x_1561_) == 0)
{
lean_object* v_a_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1569_; 
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
v_isSharedCheck_1569_ = !lean_is_exclusive(v___x_1561_);
if (v_isSharedCheck_1569_ == 0)
{
v___x_1564_ = v___x_1561_;
v_isShared_1565_ = v_isSharedCheck_1569_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_a_1562_);
lean_dec(v___x_1561_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1569_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v___x_1567_; 
if (v_isShared_1565_ == 0)
{
lean_ctor_set_tag(v___x_1564_, 1);
v___x_1567_ = v___x_1564_;
goto v_reusejp_1566_;
}
else
{
lean_object* v_reuseFailAlloc_1568_; 
v_reuseFailAlloc_1568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1568_, 0, v_a_1562_);
v___x_1567_ = v_reuseFailAlloc_1568_;
goto v_reusejp_1566_;
}
v_reusejp_1566_:
{
v___y_1529_ = v___x_1560_;
v___y_1530_ = v_a_1557_;
v_a_1531_ = v___x_1567_;
goto v___jp_1528_;
}
}
}
else
{
lean_object* v_a_1570_; lean_object* v___x_1572_; uint8_t v_isShared_1573_; uint8_t v_isSharedCheck_1577_; 
v_a_1570_ = lean_ctor_get(v___x_1561_, 0);
v_isSharedCheck_1577_ = !lean_is_exclusive(v___x_1561_);
if (v_isSharedCheck_1577_ == 0)
{
v___x_1572_ = v___x_1561_;
v_isShared_1573_ = v_isSharedCheck_1577_;
goto v_resetjp_1571_;
}
else
{
lean_inc(v_a_1570_);
lean_dec(v___x_1561_);
v___x_1572_ = lean_box(0);
v_isShared_1573_ = v_isSharedCheck_1577_;
goto v_resetjp_1571_;
}
v_resetjp_1571_:
{
lean_object* v___x_1575_; 
if (v_isShared_1573_ == 0)
{
lean_ctor_set_tag(v___x_1572_, 0);
v___x_1575_ = v___x_1572_;
goto v_reusejp_1574_;
}
else
{
lean_object* v_reuseFailAlloc_1576_; 
v_reuseFailAlloc_1576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1576_, 0, v_a_1570_);
v___x_1575_ = v_reuseFailAlloc_1576_;
goto v_reusejp_1574_;
}
v_reusejp_1574_:
{
v___y_1529_ = v___x_1560_;
v___y_1530_ = v_a_1557_;
v_a_1531_ = v___x_1575_;
goto v___jp_1528_;
}
}
}
}
else
{
lean_object* v___x_1578_; lean_object* v___x_1579_; 
v___x_1578_ = lean_io_get_num_heartbeats();
lean_inc(v_a_1517_);
lean_inc_ref(v_a_1516_);
lean_inc(v_a_1515_);
lean_inc_ref(v_a_1514_);
v___x_1579_ = lean_apply_5(v_f_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, lean_box(0));
if (lean_obj_tag(v___x_1579_) == 0)
{
lean_object* v_a_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1587_; 
v_a_1580_ = lean_ctor_get(v___x_1579_, 0);
v_isSharedCheck_1587_ = !lean_is_exclusive(v___x_1579_);
if (v_isSharedCheck_1587_ == 0)
{
v___x_1582_ = v___x_1579_;
v_isShared_1583_ = v_isSharedCheck_1587_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_a_1580_);
lean_dec(v___x_1579_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1587_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
lean_object* v___x_1585_; 
if (v_isShared_1583_ == 0)
{
lean_ctor_set_tag(v___x_1582_, 1);
v___x_1585_ = v___x_1582_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1586_; 
v_reuseFailAlloc_1586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1586_, 0, v_a_1580_);
v___x_1585_ = v_reuseFailAlloc_1586_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
v___y_1544_ = v___x_1578_;
v___y_1545_ = v_a_1557_;
v_a_1546_ = v___x_1585_;
goto v___jp_1543_;
}
}
}
else
{
lean_object* v_a_1588_; lean_object* v___x_1590_; uint8_t v_isShared_1591_; uint8_t v_isSharedCheck_1595_; 
v_a_1588_ = lean_ctor_get(v___x_1579_, 0);
v_isSharedCheck_1595_ = !lean_is_exclusive(v___x_1579_);
if (v_isSharedCheck_1595_ == 0)
{
v___x_1590_ = v___x_1579_;
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
else
{
lean_inc(v_a_1588_);
lean_dec(v___x_1579_);
v___x_1590_ = lean_box(0);
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
v_resetjp_1589_:
{
lean_object* v___x_1593_; 
if (v_isShared_1591_ == 0)
{
lean_ctor_set_tag(v___x_1590_, 0);
v___x_1593_ = v___x_1590_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v_a_1588_);
v___x_1593_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
v___y_1544_ = v___x_1578_;
v___y_1545_ = v_a_1557_;
v_a_1546_ = v___x_1593_;
goto v___jp_1543_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___boxed(lean_object* v_s_1599_, lean_object* v_f_1600_, lean_object* v_a_1601_, lean_object* v_a_1602_, lean_object* v_a_1603_, lean_object* v_a_1604_, lean_object* v_a_1605_){
_start:
{
lean_object* v_res_1606_; 
v_res_1606_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v_s_1599_, v_f_1600_, v_a_1601_, v_a_1602_, v_a_1603_, v_a_1604_);
lean_dec(v_a_1604_);
lean_dec_ref(v_a_1603_);
lean_dec(v_a_1602_);
lean_dec_ref(v_a_1601_);
return v_res_1606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace(lean_object* v_00_u03b1_1607_, lean_object* v_s_1608_, lean_object* v_f_1609_, lean_object* v_a_1610_, lean_object* v_a_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_){
_start:
{
lean_object* v___x_1615_; 
v___x_1615_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v_s_1608_, v_f_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_);
return v___x_1615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___boxed(lean_object* v_00_u03b1_1616_, lean_object* v_s_1617_, lean_object* v_f_1618_, lean_object* v_a_1619_, lean_object* v_a_1620_, lean_object* v_a_1621_, lean_object* v_a_1622_, lean_object* v_a_1623_){
_start:
{
lean_object* v_res_1624_; 
v_res_1624_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace(v_00_u03b1_1616_, v_s_1617_, v_f_1618_, v_a_1619_, v_a_1620_, v_a_1621_, v_a_1622_);
lean_dec(v_a_1622_);
lean_dec_ref(v_a_1621_);
lean_dec(v_a_1620_);
lean_dec_ref(v_a_1619_);
return v_res_1624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3(lean_object* v_00_u03b1_1625_, lean_object* v_x_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_){
_start:
{
lean_object* v___x_1632_; 
v___x_1632_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_x_1626_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1633_, lean_object* v_x_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_){
_start:
{
lean_object* v_res_1640_; 
v_res_1640_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3(v_00_u03b1_1633_, v_x_1634_, v___y_1635_, v___y_1636_, v___y_1637_, v___y_1638_);
lean_dec(v___y_1638_);
lean_dec_ref(v___y_1637_);
lean_dec(v___y_1636_);
lean_dec_ref(v___y_1635_);
return v_res_1640_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4(lean_object* v_00_u03b1_1641_, lean_object* v_e_1642_){
_start:
{
uint8_t v___x_1643_; 
v___x_1643_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___redArg(v_e_1642_);
return v___x_1643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4___boxed(lean_object* v_00_u03b1_1644_, lean_object* v_e_1645_){
_start:
{
uint8_t v_res_1646_; lean_object* v_r_1647_; 
v_res_1646_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__4(v_00_u03b1_1644_, v_e_1645_);
lean_dec_ref(v_e_1645_);
v_r_1647_ = lean_box(v_res_1646_);
return v_r_1647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2(lean_object* v_00_u03b1_1648_, lean_object* v_cls_1649_, uint8_t v_collapsed_1650_, lean_object* v_tag_1651_, lean_object* v_opts_1652_, uint8_t v_clsEnabled_1653_, lean_object* v_oldTraces_1654_, lean_object* v_msg_1655_, lean_object* v_resStartStop_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_){
_start:
{
lean_object* v___x_1662_; 
v___x_1662_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg(v_cls_1649_, v_collapsed_1650_, v_tag_1651_, v_opts_1652_, v_clsEnabled_1653_, v_oldTraces_1654_, v_msg_1655_, v_resStartStop_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_);
return v___x_1662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___boxed(lean_object* v_00_u03b1_1663_, lean_object* v_cls_1664_, lean_object* v_collapsed_1665_, lean_object* v_tag_1666_, lean_object* v_opts_1667_, lean_object* v_clsEnabled_1668_, lean_object* v_oldTraces_1669_, lean_object* v_msg_1670_, lean_object* v_resStartStop_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_){
_start:
{
uint8_t v_collapsed_boxed_1677_; uint8_t v_clsEnabled_boxed_1678_; lean_object* v_res_1679_; 
v_collapsed_boxed_1677_ = lean_unbox(v_collapsed_1665_);
v_clsEnabled_boxed_1678_ = lean_unbox(v_clsEnabled_1668_);
v_res_1679_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2(v_00_u03b1_1663_, v_cls_1664_, v_collapsed_boxed_1677_, v_tag_1666_, v_opts_1667_, v_clsEnabled_boxed_1678_, v_oldTraces_1669_, v_msg_1670_, v_resStartStop_1671_, v___y_1672_, v___y_1673_, v___y_1674_, v___y_1675_);
lean_dec(v___y_1675_);
lean_dec_ref(v___y_1674_);
lean_dec(v___y_1673_);
lean_dec_ref(v___y_1672_);
lean_dec_ref(v_opts_1667_);
return v_res_1679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8(lean_object* v_as_1681_){
_start:
{
lean_object* v___f_1682_; lean_object* v___x_1683_; 
v___f_1682_ = ((lean_object*)(lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8___closed__0));
v___x_1683_ = l_List_eraseDupsBy___redArg(v___f_1682_, v_as_1681_);
return v___x_1683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__0(lean_object* v_a_1684_, lean_object* v_a_1685_){
_start:
{
if (lean_obj_tag(v_a_1684_) == 0)
{
lean_object* v___x_1686_; 
v___x_1686_ = l_List_reverse___redArg(v_a_1685_);
return v___x_1686_;
}
else
{
lean_object* v_head_1687_; lean_object* v_tail_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1706_; 
v_head_1687_ = lean_ctor_get(v_a_1684_, 0);
v_tail_1688_ = lean_ctor_get(v_a_1684_, 1);
v_isSharedCheck_1706_ = !lean_is_exclusive(v_a_1684_);
if (v_isSharedCheck_1706_ == 0)
{
v___x_1690_ = v_a_1684_;
v_isShared_1691_ = v_isSharedCheck_1706_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_tail_1688_);
lean_inc(v_head_1687_);
lean_dec(v_a_1684_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1706_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v_fst_1692_; lean_object* v_snd_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1705_; 
v_fst_1692_ = lean_ctor_get(v_head_1687_, 0);
v_snd_1693_ = lean_ctor_get(v_head_1687_, 1);
v_isSharedCheck_1705_ = !lean_is_exclusive(v_head_1687_);
if (v_isSharedCheck_1705_ == 0)
{
v___x_1695_ = v_head_1687_;
v_isShared_1696_ = v_isSharedCheck_1705_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_snd_1693_);
lean_inc(v_fst_1692_);
lean_dec(v_head_1687_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1705_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v___x_1697_; lean_object* v___x_1699_; 
v___x_1697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1697_, 0, v_snd_1693_);
if (v_isShared_1696_ == 0)
{
lean_ctor_set(v___x_1695_, 1, v___x_1697_);
v___x_1699_ = v___x_1695_;
goto v_reusejp_1698_;
}
else
{
lean_object* v_reuseFailAlloc_1704_; 
v_reuseFailAlloc_1704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1704_, 0, v_fst_1692_);
lean_ctor_set(v_reuseFailAlloc_1704_, 1, v___x_1697_);
v___x_1699_ = v_reuseFailAlloc_1704_;
goto v_reusejp_1698_;
}
v_reusejp_1698_:
{
lean_object* v___x_1701_; 
if (v_isShared_1691_ == 0)
{
lean_ctor_set(v___x_1690_, 1, v_a_1685_);
lean_ctor_set(v___x_1690_, 0, v___x_1699_);
v___x_1701_ = v___x_1690_;
goto v_reusejp_1700_;
}
else
{
lean_object* v_reuseFailAlloc_1703_; 
v_reuseFailAlloc_1703_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1703_, 0, v___x_1699_);
lean_ctor_set(v_reuseFailAlloc_1703_, 1, v_a_1685_);
v___x_1701_ = v_reuseFailAlloc_1703_;
goto v_reusejp_1700_;
}
v_reusejp_1700_:
{
v_a_1684_ = v_tail_1688_;
v_a_1685_ = v___x_1701_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0(lean_object* v_head_1707_, lean_object* v_a_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_){
_start:
{
lean_object* v___x_1714_; 
v___x_1714_ = lp_mathlib_Mathlib_Tactic_Linarith_typeOfIneqProof(v_head_1707_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
if (lean_obj_tag(v___x_1714_) == 0)
{
lean_object* v_a_1715_; lean_object* v___x_1716_; 
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___x_1714_, 1);
v___x_1716_ = lp_mathlib_Mathlib_Tactic_Linarith_mkNegOneLtZeroProof(v_a_1715_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
if (lean_obj_tag(v___x_1716_) == 0)
{
lean_object* v_a_1717_; lean_object* v___x_1719_; uint8_t v_isShared_1720_; uint8_t v_isSharedCheck_1730_; 
v_a_1717_ = lean_ctor_get(v___x_1716_, 0);
v_isSharedCheck_1730_ = !lean_is_exclusive(v___x_1716_);
if (v_isSharedCheck_1730_ == 0)
{
v___x_1719_ = v___x_1716_;
v_isShared_1720_ = v_isSharedCheck_1730_;
goto v_resetjp_1718_;
}
else
{
lean_inc(v_a_1717_);
lean_dec(v___x_1716_);
v___x_1719_ = lean_box(0);
v_isShared_1720_ = v_isSharedCheck_1730_;
goto v_resetjp_1718_;
}
v_resetjp_1718_:
{
lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1728_; 
v___x_1721_ = lean_box(0);
v___x_1722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1722_, 0, v_a_1717_);
lean_ctor_set(v___x_1722_, 1, v___x_1721_);
v___x_1723_ = l_List_reverse___redArg(v_a_1708_);
v___x_1724_ = lean_box(0);
v___x_1725_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__0(v___x_1723_, v___x_1724_);
v___x_1726_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1726_, 0, v___x_1722_);
lean_ctor_set(v___x_1726_, 1, v___x_1725_);
if (v_isShared_1720_ == 0)
{
lean_ctor_set(v___x_1719_, 0, v___x_1726_);
v___x_1728_ = v___x_1719_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v___x_1726_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
else
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1738_; 
lean_dec(v_a_1708_);
v_a_1731_ = lean_ctor_get(v___x_1716_, 0);
v_isSharedCheck_1738_ = !lean_is_exclusive(v___x_1716_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1733_ = v___x_1716_;
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1716_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1736_; 
if (v_isShared_1734_ == 0)
{
v___x_1736_ = v___x_1733_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_a_1731_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
else
{
lean_object* v_a_1739_; lean_object* v___x_1741_; uint8_t v_isShared_1742_; uint8_t v_isSharedCheck_1746_; 
lean_dec(v_a_1708_);
v_a_1739_ = lean_ctor_get(v___x_1714_, 0);
v_isSharedCheck_1746_ = !lean_is_exclusive(v___x_1714_);
if (v_isSharedCheck_1746_ == 0)
{
v___x_1741_ = v___x_1714_;
v_isShared_1742_ = v_isSharedCheck_1746_;
goto v_resetjp_1740_;
}
else
{
lean_inc(v_a_1739_);
lean_dec(v___x_1714_);
v___x_1741_ = lean_box(0);
v_isShared_1742_ = v_isSharedCheck_1746_;
goto v_resetjp_1740_;
}
v_resetjp_1740_:
{
lean_object* v___x_1744_; 
if (v_isShared_1742_ == 0)
{
v___x_1744_ = v___x_1741_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v_a_1739_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
return v___x_1744_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0___boxed(lean_object* v_head_1747_, lean_object* v_a_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_){
_start:
{
lean_object* v_res_1754_; 
v_res_1754_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0(v_head_1747_, v_a_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_);
lean_dec(v___y_1752_);
lean_dec_ref(v___y_1751_);
lean_dec(v___y_1750_);
lean_dec_ref(v___y_1749_);
return v_res_1754_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1756_; lean_object* v___x_1757_; 
v___x_1756_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__0));
v___x_1757_ = l_Lean_stringToMessageData(v___x_1756_);
return v___x_1757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1(lean_object* v_x_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_){
_start:
{
lean_object* v___x_1764_; lean_object* v___x_1765_; 
v___x_1764_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___closed__1);
v___x_1765_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1765_, 0, v___x_1764_);
return v___x_1765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1___boxed(lean_object* v_x_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_){
_start:
{
lean_object* v_res_1772_; 
v_res_1772_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__1(v_x_1766_, v___y_1767_, v___y_1768_, v___y_1769_, v___y_1770_);
lean_dec(v___y_1770_);
lean_dec_ref(v___y_1769_);
lean_dec(v___y_1768_);
lean_dec_ref(v___y_1767_);
lean_dec_ref(v_x_1766_);
return v_res_1772_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1774_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__0));
v___x_1775_ = l_Lean_stringToMessageData(v___x_1774_);
return v___x_1775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2(lean_object* v_x_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_){
_start:
{
lean_object* v___x_1782_; lean_object* v___x_1783_; 
v___x_1782_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___closed__1);
v___x_1783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1783_, 0, v___x_1782_);
return v___x_1783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2___boxed(lean_object* v_x_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v_res_1790_; 
v_res_1790_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__2(v_x_1784_, v___y_1785_, v___y_1786_, v___y_1787_, v___y_1788_);
lean_dec(v___y_1788_);
lean_dec_ref(v___y_1787_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec_ref(v_x_1784_);
return v_res_1790_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1(void){
_start:
{
lean_object* v___x_1792_; lean_object* v___x_1793_; 
v___x_1792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__0));
v___x_1793_ = l_Lean_stringToMessageData(v___x_1792_);
return v___x_1793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(lean_object* v_____r_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_){
_start:
{
lean_object* v___x_1800_; lean_object* v___x_1801_; 
v___x_1800_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1);
v___x_1801_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v___x_1800_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_);
return v___x_1801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___boxed(lean_object* v_____r_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_){
_start:
{
lean_object* v_res_1808_; 
v_res_1808_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v_____r_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v___y_1804_);
lean_dec_ref(v___y_1803_);
return v_res_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6(lean_object* v_a_1814_, uint8_t v___x_1815_, lean_object* v_x_1816_, lean_object* v_a_1817_, lean_object* v___x_1818_, lean_object* v___x_1819_, lean_object* v___x_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_){
_start:
{
lean_object* v___x_1826_; 
lean_inc(v___y_1824_);
lean_inc_ref(v___y_1823_);
lean_inc(v___y_1822_);
lean_inc_ref(v___y_1821_);
lean_inc_ref(v_a_1814_);
v___x_1826_ = lean_infer_type(v_a_1814_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_);
if (lean_obj_tag(v___x_1826_) == 0)
{
lean_object* v_a_1827_; uint8_t v___x_1828_; uint8_t v___x_1829_; lean_object* v___x_1830_; uint8_t v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; 
v_a_1827_ = lean_ctor_get(v___x_1826_, 0);
lean_inc(v_a_1827_);
lean_dec_ref_known(v___x_1826_, 1);
v___x_1828_ = 0;
v___x_1829_ = 2;
v___x_1830_ = lean_box(0);
v___x_1831_ = 0;
v___x_1832_ = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(v___x_1832_, 0, v___x_1830_);
lean_ctor_set_uint8(v___x_1832_, sizeof(void*)*1, v___x_1829_);
lean_ctor_set_uint8(v___x_1832_, sizeof(void*)*1 + 1, v___x_1815_);
lean_ctor_set_uint8(v___x_1832_, sizeof(void*)*1 + 2, v___x_1831_);
v___x_1833_ = l_Lean_MVarId_rewrite(v_x_1816_, v_a_1827_, v_a_1817_, v___x_1828_, v___x_1832_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_);
if (lean_obj_tag(v___x_1833_) == 0)
{
lean_object* v_a_1834_; lean_object* v_eqProof_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; 
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
lean_inc(v_a_1834_);
lean_dec_ref_known(v___x_1833_, 1);
v_eqProof_1835_ = lean_ctor_get(v_a_1834_, 1);
lean_inc_ref(v_eqProof_1835_);
lean_dec(v_a_1834_);
v___x_1836_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__1));
v___x_1837_ = lean_unsigned_to_nat(2u);
v___x_1838_ = lean_mk_empty_array_with_capacity(v___x_1837_);
v___x_1839_ = lean_array_push(v___x_1838_, v_eqProof_1835_);
v___x_1840_ = lean_array_push(v___x_1839_, v_a_1814_);
v___x_1841_ = l_Lean_Meta_mkAppM(v___x_1836_, v___x_1840_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_);
if (lean_obj_tag(v___x_1841_) == 0)
{
lean_object* v_a_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; 
v_a_1842_ = lean_ctor_get(v___x_1841_, 0);
lean_inc(v_a_1842_);
lean_dec_ref_known(v___x_1841_, 1);
v___x_1843_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___closed__2));
v___x_1844_ = l_Lean_Name_mkStr4(v___x_1818_, v___x_1819_, v___x_1820_, v___x_1843_);
v___x_1845_ = lean_unsigned_to_nat(1u);
v___x_1846_ = lean_mk_empty_array_with_capacity(v___x_1845_);
v___x_1847_ = lean_array_push(v___x_1846_, v_a_1842_);
v___x_1848_ = l_Lean_Meta_mkAppM(v___x_1844_, v___x_1847_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_);
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec(v___y_1822_);
lean_dec_ref(v___y_1821_);
return v___x_1848_;
}
else
{
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec(v___y_1822_);
lean_dec_ref(v___y_1821_);
lean_dec_ref(v___x_1820_);
lean_dec_ref(v___x_1819_);
lean_dec_ref(v___x_1818_);
return v___x_1841_;
}
}
else
{
lean_object* v_a_1849_; lean_object* v___x_1851_; uint8_t v_isShared_1852_; uint8_t v_isSharedCheck_1856_; 
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec(v___y_1822_);
lean_dec_ref(v___y_1821_);
lean_dec_ref(v___x_1820_);
lean_dec_ref(v___x_1819_);
lean_dec_ref(v___x_1818_);
lean_dec_ref(v_a_1814_);
v_a_1849_ = lean_ctor_get(v___x_1833_, 0);
v_isSharedCheck_1856_ = !lean_is_exclusive(v___x_1833_);
if (v_isSharedCheck_1856_ == 0)
{
v___x_1851_ = v___x_1833_;
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
else
{
lean_inc(v_a_1849_);
lean_dec(v___x_1833_);
v___x_1851_ = lean_box(0);
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
v_resetjp_1850_:
{
lean_object* v___x_1854_; 
if (v_isShared_1852_ == 0)
{
v___x_1854_ = v___x_1851_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v_a_1849_);
v___x_1854_ = v_reuseFailAlloc_1855_;
goto v_reusejp_1853_;
}
v_reusejp_1853_:
{
return v___x_1854_;
}
}
}
}
else
{
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec(v___y_1822_);
lean_dec_ref(v___y_1821_);
lean_dec_ref(v___x_1820_);
lean_dec_ref(v___x_1819_);
lean_dec_ref(v___x_1818_);
lean_dec_ref(v_a_1817_);
lean_dec(v_x_1816_);
lean_dec_ref(v_a_1814_);
return v___x_1826_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___boxed(lean_object* v_a_1857_, lean_object* v___x_1858_, lean_object* v_x_1859_, lean_object* v_a_1860_, lean_object* v___x_1861_, lean_object* v___x_1862_, lean_object* v___x_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
uint8_t v___x_54731__boxed_1869_; lean_object* v_res_1870_; 
v___x_54731__boxed_1869_ = lean_unbox(v___x_1858_);
v_res_1870_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6(v_a_1857_, v___x_54731__boxed_1869_, v_x_1859_, v_a_1860_, v___x_1861_, v___x_1862_, v___x_1863_, v___y_1864_, v___y_1865_, v___y_1866_, v___y_1867_);
return v_res_1870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__7(lean_object* v_x_1871_, lean_object* v_x_1872_){
_start:
{
if (lean_obj_tag(v_x_1872_) == 0)
{
return v_x_1871_;
}
else
{
lean_object* v_head_1873_; lean_object* v_snd_1874_; lean_object* v_snd_1875_; 
v_head_1873_ = lean_ctor_get(v_x_1872_, 0);
v_snd_1874_ = lean_ctor_get(v_head_1873_, 1);
v_snd_1875_ = lean_ctor_get(v_snd_1874_, 1);
if (lean_obj_tag(v_snd_1875_) == 0)
{
lean_object* v_tail_1876_; 
v_tail_1876_ = lean_ctor_get(v_x_1872_, 1);
lean_inc(v_tail_1876_);
lean_dec_ref_known(v_x_1872_, 2);
v_x_1872_ = v_tail_1876_;
goto _start;
}
else
{
lean_object* v_tail_1878_; lean_object* v___x_1880_; uint8_t v_isShared_1881_; uint8_t v_isSharedCheck_1887_; 
lean_inc_ref(v_snd_1875_);
v_tail_1878_ = lean_ctor_get(v_x_1872_, 1);
v_isSharedCheck_1887_ = !lean_is_exclusive(v_x_1872_);
if (v_isSharedCheck_1887_ == 0)
{
lean_object* v_unused_1888_; 
v_unused_1888_ = lean_ctor_get(v_x_1872_, 0);
lean_dec(v_unused_1888_);
v___x_1880_ = v_x_1872_;
v_isShared_1881_ = v_isSharedCheck_1887_;
goto v_resetjp_1879_;
}
else
{
lean_inc(v_tail_1878_);
lean_dec(v_x_1872_);
v___x_1880_ = lean_box(0);
v_isShared_1881_ = v_isSharedCheck_1887_;
goto v_resetjp_1879_;
}
v_resetjp_1879_:
{
lean_object* v_val_1882_; lean_object* v___x_1884_; 
v_val_1882_ = lean_ctor_get(v_snd_1875_, 0);
lean_inc(v_val_1882_);
lean_dec_ref_known(v_snd_1875_, 1);
if (v_isShared_1881_ == 0)
{
lean_ctor_set(v___x_1880_, 1, v_x_1871_);
lean_ctor_set(v___x_1880_, 0, v_val_1882_);
v___x_1884_ = v___x_1880_;
goto v_reusejp_1883_;
}
else
{
lean_object* v_reuseFailAlloc_1886_; 
v_reuseFailAlloc_1886_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1886_, 0, v_val_1882_);
lean_ctor_set(v_reuseFailAlloc_1886_, 1, v_x_1871_);
v___x_1884_ = v_reuseFailAlloc_1886_;
goto v_reusejp_1883_;
}
v_reusejp_1883_:
{
v_x_1871_ = v___x_1884_;
v_x_1872_ = v_tail_1878_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(lean_object* v___x_1889_, lean_object* v___x_1890_, lean_object* v_a_1891_, lean_object* v_____r_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_){
_start:
{
lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; 
v___x_1898_ = lean_box(0);
v___x_1899_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__7(v___x_1898_, v___x_1889_);
v___x_1900_ = lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8(v___x_1899_);
v___x_1901_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1901_, 0, v___x_1890_);
lean_ctor_set(v___x_1901_, 1, v___x_1900_);
v___x_1902_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1902_, 0, v_a_1891_);
lean_ctor_set(v___x_1902_, 1, v___x_1901_);
v___x_1903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1903_, 0, v___x_1902_);
return v___x_1903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4___boxed(lean_object* v___x_1904_, lean_object* v___x_1905_, lean_object* v_a_1906_, lean_object* v_____r_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_){
_start:
{
lean_object* v_res_1913_; 
v_res_1913_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(v___x_1904_, v___x_1905_, v_a_1906_, v_____r_1907_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_);
lean_dec(v___y_1911_);
lean_dec_ref(v___y_1910_);
lean_dec(v___y_1909_);
lean_dec_ref(v___y_1908_);
return v_res_1913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16(lean_object* v_x_1914_, lean_object* v_x_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_){
_start:
{
if (lean_obj_tag(v_x_1914_) == 0)
{
lean_object* v___x_1921_; lean_object* v___x_1922_; 
v___x_1921_ = l_List_reverse___redArg(v_x_1915_);
v___x_1922_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1922_, 0, v___x_1921_);
return v___x_1922_;
}
else
{
lean_object* v_head_1923_; lean_object* v_tail_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1942_; 
v_head_1923_ = lean_ctor_get(v_x_1914_, 0);
v_tail_1924_ = lean_ctor_get(v_x_1914_, 1);
v_isSharedCheck_1942_ = !lean_is_exclusive(v_x_1914_);
if (v_isSharedCheck_1942_ == 0)
{
v___x_1926_ = v_x_1914_;
v_isShared_1927_ = v_isSharedCheck_1942_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_tail_1924_);
lean_inc(v_head_1923_);
lean_dec(v_x_1914_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1942_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1928_; 
lean_inc(v___y_1919_);
lean_inc_ref(v___y_1918_);
lean_inc(v___y_1917_);
lean_inc_ref(v___y_1916_);
v___x_1928_ = lean_infer_type(v_head_1923_, v___y_1916_, v___y_1917_, v___y_1918_, v___y_1919_);
if (lean_obj_tag(v___x_1928_) == 0)
{
lean_object* v_a_1929_; lean_object* v___x_1931_; 
v_a_1929_ = lean_ctor_get(v___x_1928_, 0);
lean_inc(v_a_1929_);
lean_dec_ref_known(v___x_1928_, 1);
if (v_isShared_1927_ == 0)
{
lean_ctor_set(v___x_1926_, 1, v_x_1915_);
lean_ctor_set(v___x_1926_, 0, v_a_1929_);
v___x_1931_ = v___x_1926_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_1933_; 
v_reuseFailAlloc_1933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1933_, 0, v_a_1929_);
lean_ctor_set(v_reuseFailAlloc_1933_, 1, v_x_1915_);
v___x_1931_ = v_reuseFailAlloc_1933_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
v_x_1914_ = v_tail_1924_;
v_x_1915_ = v___x_1931_;
goto _start;
}
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_1941_; 
lean_del_object(v___x_1926_);
lean_dec(v_tail_1924_);
lean_dec(v_x_1915_);
v_a_1934_ = lean_ctor_get(v___x_1928_, 0);
v_isSharedCheck_1941_ = !lean_is_exclusive(v___x_1928_);
if (v_isSharedCheck_1941_ == 0)
{
v___x_1936_ = v___x_1928_;
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1928_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___x_1939_; 
if (v_isShared_1937_ == 0)
{
v___x_1939_ = v___x_1936_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v_a_1934_);
v___x_1939_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
return v___x_1939_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16___boxed(lean_object* v_x_1943_, lean_object* v_x_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v_res_1950_; 
v_res_1950_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16(v_x_1943_, v_x_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec(v___y_1946_);
lean_dec_ref(v___y_1945_);
return v_res_1950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg(lean_object* v_a_1951_, lean_object* v_x_1952_){
_start:
{
if (lean_obj_tag(v_x_1952_) == 0)
{
lean_object* v___x_1953_; 
v___x_1953_ = lean_box(0);
return v___x_1953_;
}
else
{
lean_object* v_key_1954_; lean_object* v_value_1955_; lean_object* v_tail_1956_; uint8_t v___x_1957_; 
v_key_1954_ = lean_ctor_get(v_x_1952_, 0);
v_value_1955_ = lean_ctor_get(v_x_1952_, 1);
v_tail_1956_ = lean_ctor_get(v_x_1952_, 2);
v___x_1957_ = lean_nat_dec_eq(v_key_1954_, v_a_1951_);
if (v___x_1957_ == 0)
{
v_x_1952_ = v_tail_1956_;
goto _start;
}
else
{
lean_object* v___x_1959_; 
lean_inc(v_value_1955_);
v___x_1959_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1959_, 0, v_value_1955_);
return v___x_1959_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg___boxed(lean_object* v_a_1960_, lean_object* v_x_1961_){
_start:
{
lean_object* v_res_1962_; 
v_res_1962_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg(v_a_1960_, v_x_1961_);
lean_dec(v_x_1961_);
lean_dec(v_a_1960_);
return v_res_1962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg(lean_object* v_m_1963_, lean_object* v_a_1964_){
_start:
{
lean_object* v_buckets_1965_; lean_object* v___x_1966_; uint64_t v___x_1967_; uint64_t v___x_1968_; uint64_t v___x_1969_; uint64_t v_fold_1970_; uint64_t v___x_1971_; uint64_t v___x_1972_; uint64_t v___x_1973_; size_t v___x_1974_; size_t v___x_1975_; size_t v___x_1976_; size_t v___x_1977_; size_t v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; 
v_buckets_1965_ = lean_ctor_get(v_m_1963_, 1);
v___x_1966_ = lean_array_get_size(v_buckets_1965_);
v___x_1967_ = lean_uint64_of_nat(v_a_1964_);
v___x_1968_ = 32ULL;
v___x_1969_ = lean_uint64_shift_right(v___x_1967_, v___x_1968_);
v_fold_1970_ = lean_uint64_xor(v___x_1967_, v___x_1969_);
v___x_1971_ = 16ULL;
v___x_1972_ = lean_uint64_shift_right(v_fold_1970_, v___x_1971_);
v___x_1973_ = lean_uint64_xor(v_fold_1970_, v___x_1972_);
v___x_1974_ = lean_uint64_to_usize(v___x_1973_);
v___x_1975_ = lean_usize_of_nat(v___x_1966_);
v___x_1976_ = ((size_t)1ULL);
v___x_1977_ = lean_usize_sub(v___x_1975_, v___x_1976_);
v___x_1978_ = lean_usize_land(v___x_1974_, v___x_1977_);
v___x_1979_ = lean_array_uget_borrowed(v_buckets_1965_, v___x_1978_);
v___x_1980_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg(v_a_1964_, v___x_1979_);
return v___x_1980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg___boxed(lean_object* v_m_1981_, lean_object* v_a_1982_){
_start:
{
lean_object* v_res_1983_; 
v_res_1983_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg(v_m_1981_, v_a_1982_);
lean_dec(v_a_1982_);
lean_dec_ref(v_m_1981_);
return v_res_1983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3(lean_object* v_a_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_){
_start:
{
if (lean_obj_tag(v_a_1985_) == 0)
{
lean_object* v___x_1987_; 
v___x_1987_ = lean_array_to_list(v_a_1986_);
return v___x_1987_;
}
else
{
lean_object* v_head_1988_; lean_object* v_fst_1989_; lean_object* v_tail_1990_; lean_object* v_snd_1991_; lean_object* v___x_1993_; uint8_t v_isShared_1994_; uint8_t v_isSharedCheck_2012_; 
v_head_1988_ = lean_ctor_get(v_a_1985_, 0);
lean_inc(v_head_1988_);
v_fst_1989_ = lean_ctor_get(v_head_1988_, 0);
lean_inc(v_fst_1989_);
v_tail_1990_ = lean_ctor_get(v_a_1985_, 1);
lean_inc(v_tail_1990_);
lean_dec_ref_known(v_a_1985_, 2);
v_snd_1991_ = lean_ctor_get(v_head_1988_, 1);
v_isSharedCheck_2012_ = !lean_is_exclusive(v_head_1988_);
if (v_isSharedCheck_2012_ == 0)
{
lean_object* v_unused_2013_; 
v_unused_2013_ = lean_ctor_get(v_head_1988_, 0);
lean_dec(v_unused_2013_);
v___x_1993_ = v_head_1988_;
v_isShared_1994_ = v_isSharedCheck_2012_;
goto v_resetjp_1992_;
}
else
{
lean_inc(v_snd_1991_);
lean_dec(v_head_1988_);
v___x_1993_ = lean_box(0);
v_isShared_1994_ = v_isSharedCheck_2012_;
goto v_resetjp_1992_;
}
v_resetjp_1992_:
{
lean_object* v_fst_1995_; lean_object* v_snd_1996_; lean_object* v___x_1998_; uint8_t v_isShared_1999_; uint8_t v_isSharedCheck_2011_; 
v_fst_1995_ = lean_ctor_get(v_fst_1989_, 0);
v_snd_1996_ = lean_ctor_get(v_fst_1989_, 1);
v_isSharedCheck_2011_ = !lean_is_exclusive(v_fst_1989_);
if (v_isSharedCheck_2011_ == 0)
{
v___x_1998_ = v_fst_1989_;
v_isShared_1999_ = v_isSharedCheck_2011_;
goto v_resetjp_1997_;
}
else
{
lean_inc(v_snd_1996_);
lean_inc(v_fst_1995_);
lean_dec(v_fst_1989_);
v___x_1998_ = lean_box(0);
v_isShared_1999_ = v_isSharedCheck_2011_;
goto v_resetjp_1997_;
}
v_resetjp_1997_:
{
lean_object* v___x_2000_; 
v___x_2000_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg(v_a_1984_, v_snd_1991_);
lean_dec(v_snd_1991_);
if (lean_obj_tag(v___x_2000_) == 0)
{
lean_del_object(v___x_1998_);
lean_dec(v_snd_1996_);
lean_dec(v_fst_1995_);
lean_del_object(v___x_1993_);
v_a_1985_ = v_tail_1990_;
goto _start;
}
else
{
lean_object* v_val_2002_; lean_object* v___x_2004_; 
v_val_2002_ = lean_ctor_get(v___x_2000_, 0);
lean_inc(v_val_2002_);
lean_dec_ref_known(v___x_2000_, 1);
if (v_isShared_1999_ == 0)
{
lean_ctor_set(v___x_1998_, 0, v_val_2002_);
v___x_2004_ = v___x_1998_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2010_; 
v_reuseFailAlloc_2010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2010_, 0, v_val_2002_);
lean_ctor_set(v_reuseFailAlloc_2010_, 1, v_snd_1996_);
v___x_2004_ = v_reuseFailAlloc_2010_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
lean_object* v___x_2006_; 
if (v_isShared_1994_ == 0)
{
lean_ctor_set(v___x_1993_, 1, v___x_2004_);
lean_ctor_set(v___x_1993_, 0, v_fst_1995_);
v___x_2006_ = v___x_1993_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2009_; 
v_reuseFailAlloc_2009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2009_, 0, v_fst_1995_);
lean_ctor_set(v_reuseFailAlloc_2009_, 1, v___x_2004_);
v___x_2006_ = v_reuseFailAlloc_2009_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
lean_object* v___x_2007_; 
v___x_2007_ = lean_array_push(v_a_1986_, v___x_2006_);
v_a_1985_ = v_tail_1990_;
v_a_1986_ = v___x_2007_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3___boxed(lean_object* v_a_2014_, lean_object* v_a_2015_, lean_object* v_a_2016_){
_start:
{
lean_object* v_res_2017_; 
v_res_2017_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3(v_a_2014_, v_a_2015_, v_a_2016_);
lean_dec_ref(v_a_2014_);
return v_res_2017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__17(lean_object* v_a_2018_, lean_object* v_a_2019_){
_start:
{
if (lean_obj_tag(v_a_2018_) == 0)
{
lean_object* v___x_2020_; 
v___x_2020_ = l_List_reverse___redArg(v_a_2019_);
return v___x_2020_;
}
else
{
lean_object* v_head_2021_; lean_object* v_tail_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2031_; 
v_head_2021_ = lean_ctor_get(v_a_2018_, 0);
v_tail_2022_ = lean_ctor_get(v_a_2018_, 1);
v_isSharedCheck_2031_ = !lean_is_exclusive(v_a_2018_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_2024_ = v_a_2018_;
v_isShared_2025_ = v_isSharedCheck_2031_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_tail_2022_);
lean_inc(v_head_2021_);
lean_dec(v_a_2018_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2031_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v___x_2026_; lean_object* v___x_2028_; 
v___x_2026_ = l_Lean_MessageData_ofExpr(v_head_2021_);
if (v_isShared_2025_ == 0)
{
lean_ctor_set(v___x_2024_, 1, v_a_2019_);
lean_ctor_set(v___x_2024_, 0, v___x_2026_);
v___x_2028_ = v___x_2024_;
goto v_reusejp_2027_;
}
else
{
lean_object* v_reuseFailAlloc_2030_; 
v_reuseFailAlloc_2030_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2030_, 0, v___x_2026_);
lean_ctor_set(v_reuseFailAlloc_2030_, 1, v_a_2019_);
v___x_2028_ = v_reuseFailAlloc_2030_;
goto v_reusejp_2027_;
}
v_reusejp_2027_:
{
v_a_2018_ = v_tail_2022_;
v_a_2019_ = v___x_2028_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15(lean_object* v_e_2032_){
_start:
{
if (lean_obj_tag(v_e_2032_) == 0)
{
uint8_t v___x_2033_; 
v___x_2033_ = 2;
return v___x_2033_;
}
else
{
uint8_t v___x_2034_; 
v___x_2034_ = 0;
return v___x_2034_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15___boxed(lean_object* v_e_2035_){
_start:
{
uint8_t v_res_2036_; lean_object* v_r_2037_; 
v_res_2036_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15(v_e_2035_);
lean_dec_ref(v_e_2035_);
v_r_2037_ = lean_box(v_res_2036_);
return v_r_2037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13(lean_object* v_cls_2038_, uint8_t v_collapsed_2039_, lean_object* v_tag_2040_, lean_object* v_opts_2041_, uint8_t v_clsEnabled_2042_, lean_object* v_oldTraces_2043_, lean_object* v_msg_2044_, lean_object* v_resStartStop_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_){
_start:
{
lean_object* v_fst_2051_; lean_object* v_snd_2052_; lean_object* v___y_2054_; lean_object* v___y_2055_; lean_object* v_data_2056_; lean_object* v_fst_2067_; lean_object* v_snd_2068_; lean_object* v___x_2069_; uint8_t v___x_2070_; lean_object* v___y_2072_; lean_object* v_a_2073_; uint8_t v___y_2088_; double v___y_2119_; 
v_fst_2051_ = lean_ctor_get(v_resStartStop_2045_, 0);
lean_inc(v_fst_2051_);
v_snd_2052_ = lean_ctor_get(v_resStartStop_2045_, 1);
lean_inc(v_snd_2052_);
lean_dec_ref(v_resStartStop_2045_);
v_fst_2067_ = lean_ctor_get(v_snd_2052_, 0);
lean_inc(v_fst_2067_);
v_snd_2068_ = lean_ctor_get(v_snd_2052_, 1);
lean_inc(v_snd_2068_);
lean_dec(v_snd_2052_);
v___x_2069_ = l_Lean_trace_profiler;
v___x_2070_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_2041_, v___x_2069_);
if (v___x_2070_ == 0)
{
v___y_2088_ = v___x_2070_;
goto v___jp_2087_;
}
else
{
lean_object* v___x_2124_; uint8_t v___x_2125_; 
v___x_2124_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2125_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_2041_, v___x_2124_);
if (v___x_2125_ == 0)
{
lean_object* v___x_2126_; lean_object* v___x_2127_; double v___x_2128_; double v___x_2129_; double v___x_2130_; 
v___x_2126_ = l_Lean_trace_profiler_threshold;
v___x_2127_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_2041_, v___x_2126_);
v___x_2128_ = lean_float_of_nat(v___x_2127_);
v___x_2129_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3);
v___x_2130_ = lean_float_div(v___x_2128_, v___x_2129_);
v___y_2119_ = v___x_2130_;
goto v___jp_2118_;
}
else
{
lean_object* v___x_2131_; lean_object* v___x_2132_; double v___x_2133_; 
v___x_2131_ = l_Lean_trace_profiler_threshold;
v___x_2132_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_2041_, v___x_2131_);
v___x_2133_ = lean_float_of_nat(v___x_2132_);
v___y_2119_ = v___x_2133_;
goto v___jp_2118_;
}
}
v___jp_2053_:
{
lean_object* v___x_2057_; 
lean_inc(v___y_2054_);
v___x_2057_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(v_oldTraces_2043_, v_data_2056_, v___y_2054_, v___y_2055_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_);
if (lean_obj_tag(v___x_2057_) == 0)
{
lean_object* v___x_2058_; 
lean_dec_ref_known(v___x_2057_, 1);
v___x_2058_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_2051_);
return v___x_2058_;
}
else
{
lean_object* v_a_2059_; lean_object* v___x_2061_; uint8_t v_isShared_2062_; uint8_t v_isSharedCheck_2066_; 
lean_dec(v_fst_2051_);
v_a_2059_ = lean_ctor_get(v___x_2057_, 0);
v_isSharedCheck_2066_ = !lean_is_exclusive(v___x_2057_);
if (v_isSharedCheck_2066_ == 0)
{
v___x_2061_ = v___x_2057_;
v_isShared_2062_ = v_isSharedCheck_2066_;
goto v_resetjp_2060_;
}
else
{
lean_inc(v_a_2059_);
lean_dec(v___x_2057_);
v___x_2061_ = lean_box(0);
v_isShared_2062_ = v_isSharedCheck_2066_;
goto v_resetjp_2060_;
}
v_resetjp_2060_:
{
lean_object* v___x_2064_; 
if (v_isShared_2062_ == 0)
{
v___x_2064_ = v___x_2061_;
goto v_reusejp_2063_;
}
else
{
lean_object* v_reuseFailAlloc_2065_; 
v_reuseFailAlloc_2065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2065_, 0, v_a_2059_);
v___x_2064_ = v_reuseFailAlloc_2065_;
goto v_reusejp_2063_;
}
v_reusejp_2063_:
{
return v___x_2064_;
}
}
}
}
v___jp_2071_:
{
uint8_t v_result_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; double v___x_2077_; lean_object* v_data_2078_; 
v_result_2074_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13_spec__15(v_fst_2051_);
v___x_2075_ = lean_box(v_result_2074_);
v___x_2076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2076_, 0, v___x_2075_);
v___x_2077_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0);
lean_inc_ref(v_tag_2040_);
lean_inc_ref(v___x_2076_);
lean_inc(v_cls_2038_);
v_data_2078_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2078_, 0, v_cls_2038_);
lean_ctor_set(v_data_2078_, 1, v___x_2076_);
lean_ctor_set(v_data_2078_, 2, v_tag_2040_);
lean_ctor_set_float(v_data_2078_, sizeof(void*)*3, v___x_2077_);
lean_ctor_set_float(v_data_2078_, sizeof(void*)*3 + 8, v___x_2077_);
lean_ctor_set_uint8(v_data_2078_, sizeof(void*)*3 + 16, v_collapsed_2039_);
if (v___x_2070_ == 0)
{
lean_dec_ref_known(v___x_2076_, 1);
lean_dec(v_snd_2068_);
lean_dec(v_fst_2067_);
lean_dec_ref(v_tag_2040_);
lean_dec(v_cls_2038_);
v___y_2054_ = v___y_2072_;
v___y_2055_ = v_a_2073_;
v_data_2056_ = v_data_2078_;
goto v___jp_2053_;
}
else
{
lean_object* v_data_2079_; double v___x_2080_; double v___x_2081_; 
lean_dec_ref_known(v_data_2078_, 3);
v_data_2079_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2079_, 0, v_cls_2038_);
lean_ctor_set(v_data_2079_, 1, v___x_2076_);
lean_ctor_set(v_data_2079_, 2, v_tag_2040_);
v___x_2080_ = lean_unbox_float(v_fst_2067_);
lean_dec(v_fst_2067_);
lean_ctor_set_float(v_data_2079_, sizeof(void*)*3, v___x_2080_);
v___x_2081_ = lean_unbox_float(v_snd_2068_);
lean_dec(v_snd_2068_);
lean_ctor_set_float(v_data_2079_, sizeof(void*)*3 + 8, v___x_2081_);
lean_ctor_set_uint8(v_data_2079_, sizeof(void*)*3 + 16, v_collapsed_2039_);
v___y_2054_ = v___y_2072_;
v___y_2055_ = v_a_2073_;
v_data_2056_ = v_data_2079_;
goto v___jp_2053_;
}
}
v___jp_2082_:
{
lean_object* v_ref_2083_; lean_object* v___x_2084_; 
v_ref_2083_ = lean_ctor_get(v___y_2048_, 5);
lean_inc(v___y_2049_);
lean_inc_ref(v___y_2048_);
lean_inc(v___y_2047_);
lean_inc_ref(v___y_2046_);
lean_inc(v_fst_2051_);
v___x_2084_ = lean_apply_6(v_msg_2044_, v_fst_2051_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_, lean_box(0));
if (lean_obj_tag(v___x_2084_) == 0)
{
lean_object* v_a_2085_; 
v_a_2085_ = lean_ctor_get(v___x_2084_, 0);
lean_inc(v_a_2085_);
lean_dec_ref_known(v___x_2084_, 1);
v___y_2072_ = v_ref_2083_;
v_a_2073_ = v_a_2085_;
goto v___jp_2071_;
}
else
{
lean_object* v___x_2086_; 
lean_dec_ref_known(v___x_2084_, 1);
v___x_2086_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2);
v___y_2072_ = v_ref_2083_;
v_a_2073_ = v___x_2086_;
goto v___jp_2071_;
}
}
v___jp_2087_:
{
if (v_clsEnabled_2042_ == 0)
{
if (v___y_2088_ == 0)
{
lean_object* v___x_2089_; lean_object* v_traceState_2090_; lean_object* v_env_2091_; lean_object* v_nextMacroScope_2092_; lean_object* v_ngen_2093_; lean_object* v_auxDeclNGen_2094_; lean_object* v_cache_2095_; lean_object* v_messages_2096_; lean_object* v_infoState_2097_; lean_object* v_snapshotTasks_2098_; lean_object* v___x_2100_; uint8_t v_isShared_2101_; uint8_t v_isSharedCheck_2117_; 
lean_dec(v_snd_2068_);
lean_dec(v_fst_2067_);
lean_dec_ref(v_msg_2044_);
lean_dec_ref(v_tag_2040_);
lean_dec(v_cls_2038_);
v___x_2089_ = lean_st_ref_take(v___y_2049_);
v_traceState_2090_ = lean_ctor_get(v___x_2089_, 4);
v_env_2091_ = lean_ctor_get(v___x_2089_, 0);
v_nextMacroScope_2092_ = lean_ctor_get(v___x_2089_, 1);
v_ngen_2093_ = lean_ctor_get(v___x_2089_, 2);
v_auxDeclNGen_2094_ = lean_ctor_get(v___x_2089_, 3);
v_cache_2095_ = lean_ctor_get(v___x_2089_, 5);
v_messages_2096_ = lean_ctor_get(v___x_2089_, 6);
v_infoState_2097_ = lean_ctor_get(v___x_2089_, 7);
v_snapshotTasks_2098_ = lean_ctor_get(v___x_2089_, 8);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2100_ = v___x_2089_;
v_isShared_2101_ = v_isSharedCheck_2117_;
goto v_resetjp_2099_;
}
else
{
lean_inc(v_snapshotTasks_2098_);
lean_inc(v_infoState_2097_);
lean_inc(v_messages_2096_);
lean_inc(v_cache_2095_);
lean_inc(v_traceState_2090_);
lean_inc(v_auxDeclNGen_2094_);
lean_inc(v_ngen_2093_);
lean_inc(v_nextMacroScope_2092_);
lean_inc(v_env_2091_);
lean_dec(v___x_2089_);
v___x_2100_ = lean_box(0);
v_isShared_2101_ = v_isSharedCheck_2117_;
goto v_resetjp_2099_;
}
v_resetjp_2099_:
{
uint64_t v_tid_2102_; lean_object* v_traces_2103_; lean_object* v___x_2105_; uint8_t v_isShared_2106_; uint8_t v_isSharedCheck_2116_; 
v_tid_2102_ = lean_ctor_get_uint64(v_traceState_2090_, sizeof(void*)*1);
v_traces_2103_ = lean_ctor_get(v_traceState_2090_, 0);
v_isSharedCheck_2116_ = !lean_is_exclusive(v_traceState_2090_);
if (v_isSharedCheck_2116_ == 0)
{
v___x_2105_ = v_traceState_2090_;
v_isShared_2106_ = v_isSharedCheck_2116_;
goto v_resetjp_2104_;
}
else
{
lean_inc(v_traces_2103_);
lean_dec(v_traceState_2090_);
v___x_2105_ = lean_box(0);
v_isShared_2106_ = v_isSharedCheck_2116_;
goto v_resetjp_2104_;
}
v_resetjp_2104_:
{
lean_object* v___x_2107_; lean_object* v___x_2109_; 
v___x_2107_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2043_, v_traces_2103_);
lean_dec_ref(v_traces_2103_);
if (v_isShared_2106_ == 0)
{
lean_ctor_set(v___x_2105_, 0, v___x_2107_);
v___x_2109_ = v___x_2105_;
goto v_reusejp_2108_;
}
else
{
lean_object* v_reuseFailAlloc_2115_; 
v_reuseFailAlloc_2115_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2115_, 0, v___x_2107_);
lean_ctor_set_uint64(v_reuseFailAlloc_2115_, sizeof(void*)*1, v_tid_2102_);
v___x_2109_ = v_reuseFailAlloc_2115_;
goto v_reusejp_2108_;
}
v_reusejp_2108_:
{
lean_object* v___x_2111_; 
if (v_isShared_2101_ == 0)
{
lean_ctor_set(v___x_2100_, 4, v___x_2109_);
v___x_2111_ = v___x_2100_;
goto v_reusejp_2110_;
}
else
{
lean_object* v_reuseFailAlloc_2114_; 
v_reuseFailAlloc_2114_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2114_, 0, v_env_2091_);
lean_ctor_set(v_reuseFailAlloc_2114_, 1, v_nextMacroScope_2092_);
lean_ctor_set(v_reuseFailAlloc_2114_, 2, v_ngen_2093_);
lean_ctor_set(v_reuseFailAlloc_2114_, 3, v_auxDeclNGen_2094_);
lean_ctor_set(v_reuseFailAlloc_2114_, 4, v___x_2109_);
lean_ctor_set(v_reuseFailAlloc_2114_, 5, v_cache_2095_);
lean_ctor_set(v_reuseFailAlloc_2114_, 6, v_messages_2096_);
lean_ctor_set(v_reuseFailAlloc_2114_, 7, v_infoState_2097_);
lean_ctor_set(v_reuseFailAlloc_2114_, 8, v_snapshotTasks_2098_);
v___x_2111_ = v_reuseFailAlloc_2114_;
goto v_reusejp_2110_;
}
v_reusejp_2110_:
{
lean_object* v___x_2112_; lean_object* v___x_2113_; 
v___x_2112_ = lean_st_ref_set(v___y_2049_, v___x_2111_);
v___x_2113_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_2051_);
return v___x_2113_;
}
}
}
}
}
else
{
goto v___jp_2082_;
}
}
else
{
goto v___jp_2082_;
}
}
v___jp_2118_:
{
double v___x_2120_; double v___x_2121_; double v___x_2122_; uint8_t v___x_2123_; 
v___x_2120_ = lean_unbox_float(v_snd_2068_);
v___x_2121_ = lean_unbox_float(v_fst_2067_);
v___x_2122_ = lean_float_sub(v___x_2120_, v___x_2121_);
v___x_2123_ = lean_float_decLt(v___y_2119_, v___x_2122_);
v___y_2088_ = v___x_2123_;
goto v___jp_2087_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13___boxed(lean_object* v_cls_2134_, lean_object* v_collapsed_2135_, lean_object* v_tag_2136_, lean_object* v_opts_2137_, lean_object* v_clsEnabled_2138_, lean_object* v_oldTraces_2139_, lean_object* v_msg_2140_, lean_object* v_resStartStop_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_){
_start:
{
uint8_t v_collapsed_boxed_2147_; uint8_t v_clsEnabled_boxed_2148_; lean_object* v_res_2149_; 
v_collapsed_boxed_2147_ = lean_unbox(v_collapsed_2135_);
v_clsEnabled_boxed_2148_ = lean_unbox(v_clsEnabled_2138_);
v_res_2149_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13(v_cls_2134_, v_collapsed_boxed_2147_, v_tag_2136_, v_opts_2137_, v_clsEnabled_boxed_2148_, v_oldTraces_2139_, v_msg_2140_, v_resStartStop_2141_, v___y_2142_, v___y_2143_, v___y_2144_, v___y_2145_);
lean_dec(v___y_2145_);
lean_dec_ref(v___y_2144_);
lean_dec(v___y_2143_);
lean_dec_ref(v___y_2142_);
lean_dec_ref(v_opts_2137_);
return v_res_2149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__4(lean_object* v_a_2150_, lean_object* v_a_2151_){
_start:
{
if (lean_obj_tag(v_a_2150_) == 0)
{
lean_object* v___x_2152_; 
v___x_2152_ = l_List_reverse___redArg(v_a_2151_);
return v___x_2152_;
}
else
{
lean_object* v_head_2153_; lean_object* v_snd_2154_; lean_object* v_tail_2155_; lean_object* v___x_2157_; uint8_t v_isShared_2158_; uint8_t v_isSharedCheck_2173_; 
v_head_2153_ = lean_ctor_get(v_a_2150_, 0);
lean_inc(v_head_2153_);
v_snd_2154_ = lean_ctor_get(v_head_2153_, 1);
lean_inc(v_snd_2154_);
v_tail_2155_ = lean_ctor_get(v_a_2150_, 1);
v_isSharedCheck_2173_ = !lean_is_exclusive(v_a_2150_);
if (v_isSharedCheck_2173_ == 0)
{
lean_object* v_unused_2174_; 
v_unused_2174_ = lean_ctor_get(v_a_2150_, 0);
lean_dec(v_unused_2174_);
v___x_2157_ = v_a_2150_;
v_isShared_2158_ = v_isSharedCheck_2173_;
goto v_resetjp_2156_;
}
else
{
lean_inc(v_tail_2155_);
lean_dec(v_a_2150_);
v___x_2157_ = lean_box(0);
v_isShared_2158_ = v_isSharedCheck_2173_;
goto v_resetjp_2156_;
}
v_resetjp_2156_:
{
lean_object* v_fst_2159_; lean_object* v_fst_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2171_; 
v_fst_2159_ = lean_ctor_get(v_head_2153_, 0);
lean_inc(v_fst_2159_);
lean_dec(v_head_2153_);
v_fst_2160_ = lean_ctor_get(v_snd_2154_, 0);
v_isSharedCheck_2171_ = !lean_is_exclusive(v_snd_2154_);
if (v_isSharedCheck_2171_ == 0)
{
lean_object* v_unused_2172_; 
v_unused_2172_ = lean_ctor_get(v_snd_2154_, 1);
lean_dec(v_unused_2172_);
v___x_2162_ = v_snd_2154_;
v_isShared_2163_ = v_isSharedCheck_2171_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_fst_2160_);
lean_dec(v_snd_2154_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2171_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
lean_object* v___x_2165_; 
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 1, v_fst_2160_);
lean_ctor_set(v___x_2162_, 0, v_fst_2159_);
v___x_2165_ = v___x_2162_;
goto v_reusejp_2164_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v_fst_2159_);
lean_ctor_set(v_reuseFailAlloc_2170_, 1, v_fst_2160_);
v___x_2165_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2164_;
}
v_reusejp_2164_:
{
lean_object* v___x_2167_; 
if (v_isShared_2158_ == 0)
{
lean_ctor_set(v___x_2157_, 1, v_a_2151_);
lean_ctor_set(v___x_2157_, 0, v___x_2165_);
v___x_2167_ = v___x_2157_;
goto v_reusejp_2166_;
}
else
{
lean_object* v_reuseFailAlloc_2169_; 
v_reuseFailAlloc_2169_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2169_, 0, v___x_2165_);
lean_ctor_set(v_reuseFailAlloc_2169_, 1, v_a_2151_);
v___x_2167_ = v_reuseFailAlloc_2169_;
goto v_reusejp_2166_;
}
v_reusejp_2166_:
{
v_a_2150_ = v_tail_2155_;
v_a_2151_ = v___x_2167_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__1(lean_object* v_a_2175_, lean_object* v_a_2176_){
_start:
{
if (lean_obj_tag(v_a_2175_) == 0)
{
lean_object* v___x_2177_; 
v___x_2177_ = l_List_reverse___redArg(v_a_2176_);
return v___x_2177_;
}
else
{
lean_object* v_head_2178_; lean_object* v_tail_2179_; lean_object* v___x_2181_; uint8_t v_isShared_2182_; uint8_t v_isSharedCheck_2188_; 
v_head_2178_ = lean_ctor_get(v_a_2175_, 0);
v_tail_2179_ = lean_ctor_get(v_a_2175_, 1);
v_isSharedCheck_2188_ = !lean_is_exclusive(v_a_2175_);
if (v_isSharedCheck_2188_ == 0)
{
v___x_2181_ = v_a_2175_;
v_isShared_2182_ = v_isSharedCheck_2188_;
goto v_resetjp_2180_;
}
else
{
lean_inc(v_tail_2179_);
lean_inc(v_head_2178_);
lean_dec(v_a_2175_);
v___x_2181_ = lean_box(0);
v_isShared_2182_ = v_isSharedCheck_2188_;
goto v_resetjp_2180_;
}
v_resetjp_2180_:
{
lean_object* v_fst_2183_; lean_object* v___x_2185_; 
v_fst_2183_ = lean_ctor_get(v_head_2178_, 0);
lean_inc(v_fst_2183_);
lean_dec(v_head_2178_);
if (v_isShared_2182_ == 0)
{
lean_ctor_set(v___x_2181_, 1, v_a_2176_);
lean_ctor_set(v___x_2181_, 0, v_fst_2183_);
v___x_2185_ = v___x_2181_;
goto v_reusejp_2184_;
}
else
{
lean_object* v_reuseFailAlloc_2187_; 
v_reuseFailAlloc_2187_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2187_, 0, v_fst_2183_);
lean_ctor_set(v_reuseFailAlloc_2187_, 1, v_a_2176_);
v___x_2185_ = v_reuseFailAlloc_2187_;
goto v_reusejp_2184_;
}
v_reusejp_2184_:
{
v_a_2175_ = v_tail_2179_;
v_a_2176_ = v___x_2185_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11(lean_object* v_x_2189_, lean_object* v_x_2190_){
_start:
{
if (lean_obj_tag(v_x_2190_) == 0)
{
lean_inc(v_x_2189_);
return v_x_2189_;
}
else
{
lean_object* v_key_2191_; lean_object* v_value_2192_; lean_object* v_tail_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; 
v_key_2191_ = lean_ctor_get(v_x_2190_, 0);
v_value_2192_ = lean_ctor_get(v_x_2190_, 1);
v_tail_2193_ = lean_ctor_get(v_x_2190_, 2);
v___x_2194_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11(v_x_2189_, v_tail_2193_);
lean_inc(v_value_2192_);
lean_inc(v_key_2191_);
v___x_2195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2195_, 0, v_key_2191_);
lean_ctor_set(v___x_2195_, 1, v_value_2192_);
v___x_2196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2196_, 0, v___x_2195_);
lean_ctor_set(v___x_2196_, 1, v___x_2194_);
return v___x_2196_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11___boxed(lean_object* v_x_2197_, lean_object* v_x_2198_){
_start:
{
lean_object* v_res_2199_; 
v_res_2199_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11(v_x_2197_, v_x_2198_);
lean_dec(v_x_2198_);
lean_dec(v_x_2197_);
return v_res_2199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(lean_object* v_as_2200_, size_t v_i_2201_, size_t v_stop_2202_, lean_object* v_b_2203_){
_start:
{
uint8_t v___x_2204_; 
v___x_2204_ = lean_usize_dec_eq(v_i_2201_, v_stop_2202_);
if (v___x_2204_ == 0)
{
size_t v___x_2205_; size_t v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; 
v___x_2205_ = ((size_t)1ULL);
v___x_2206_ = lean_usize_sub(v_i_2201_, v___x_2205_);
v___x_2207_ = lean_array_uget_borrowed(v_as_2200_, v___x_2206_);
v___x_2208_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__11(v_b_2203_, v___x_2207_);
lean_dec(v_b_2203_);
v_i_2201_ = v___x_2206_;
v_b_2203_ = v___x_2208_;
goto _start;
}
else
{
return v_b_2203_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12___boxed(lean_object* v_as_2210_, lean_object* v_i_2211_, lean_object* v_stop_2212_, lean_object* v_b_2213_){
_start:
{
size_t v_i_boxed_2214_; size_t v_stop_boxed_2215_; lean_object* v_res_2216_; 
v_i_boxed_2214_ = lean_unbox_usize(v_i_2211_);
lean_dec(v_i_2211_);
v_stop_boxed_2215_ = lean_unbox_usize(v_stop_2212_);
lean_dec(v_stop_2212_);
v_res_2216_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(v_as_2210_, v_i_boxed_2214_, v_stop_boxed_2215_, v_b_2213_);
lean_dec_ref(v_as_2210_);
return v_res_2216_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6(lean_object* v_e_2217_){
_start:
{
if (lean_obj_tag(v_e_2217_) == 0)
{
uint8_t v___x_2218_; 
v___x_2218_ = 2;
return v___x_2218_;
}
else
{
uint8_t v___x_2219_; 
v___x_2219_ = 0;
return v___x_2219_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6___boxed(lean_object* v_e_2220_){
_start:
{
uint8_t v_res_2221_; lean_object* v_r_2222_; 
v_res_2221_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6(v_e_2220_);
lean_dec_ref(v_e_2220_);
v_r_2222_ = lean_box(v_res_2221_);
return v_r_2222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5(lean_object* v_cls_2223_, uint8_t v_collapsed_2224_, lean_object* v_tag_2225_, lean_object* v_opts_2226_, uint8_t v_clsEnabled_2227_, lean_object* v_oldTraces_2228_, lean_object* v_msg_2229_, lean_object* v_resStartStop_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_){
_start:
{
lean_object* v_fst_2236_; lean_object* v_snd_2237_; lean_object* v___y_2239_; lean_object* v___y_2240_; lean_object* v_data_2241_; lean_object* v_fst_2252_; lean_object* v_snd_2253_; lean_object* v___x_2254_; uint8_t v___x_2255_; lean_object* v___y_2257_; lean_object* v_a_2258_; uint8_t v___y_2273_; double v___y_2304_; 
v_fst_2236_ = lean_ctor_get(v_resStartStop_2230_, 0);
lean_inc(v_fst_2236_);
v_snd_2237_ = lean_ctor_get(v_resStartStop_2230_, 1);
lean_inc(v_snd_2237_);
lean_dec_ref(v_resStartStop_2230_);
v_fst_2252_ = lean_ctor_get(v_snd_2237_, 0);
lean_inc(v_fst_2252_);
v_snd_2253_ = lean_ctor_get(v_snd_2237_, 1);
lean_inc(v_snd_2253_);
lean_dec(v_snd_2237_);
v___x_2254_ = l_Lean_trace_profiler;
v___x_2255_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_2226_, v___x_2254_);
if (v___x_2255_ == 0)
{
v___y_2273_ = v___x_2255_;
goto v___jp_2272_;
}
else
{
lean_object* v___x_2309_; uint8_t v___x_2310_; 
v___x_2309_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2310_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_opts_2226_, v___x_2309_);
if (v___x_2310_ == 0)
{
lean_object* v___x_2311_; lean_object* v___x_2312_; double v___x_2313_; double v___x_2314_; double v___x_2315_; 
v___x_2311_ = l_Lean_trace_profiler_threshold;
v___x_2312_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_2226_, v___x_2311_);
v___x_2313_ = lean_float_of_nat(v___x_2312_);
v___x_2314_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__3);
v___x_2315_ = lean_float_div(v___x_2313_, v___x_2314_);
v___y_2304_ = v___x_2315_;
goto v___jp_2303_;
}
else
{
lean_object* v___x_2316_; lean_object* v___x_2317_; double v___x_2318_; 
v___x_2316_ = l_Lean_trace_profiler_threshold;
v___x_2317_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__5(v_opts_2226_, v___x_2316_);
v___x_2318_ = lean_float_of_nat(v___x_2317_);
v___y_2304_ = v___x_2318_;
goto v___jp_2303_;
}
}
v___jp_2238_:
{
lean_object* v___x_2242_; 
lean_inc(v___y_2240_);
v___x_2242_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__2(v_oldTraces_2228_, v_data_2241_, v___y_2240_, v___y_2239_, v___y_2231_, v___y_2232_, v___y_2233_, v___y_2234_);
if (lean_obj_tag(v___x_2242_) == 0)
{
lean_object* v___x_2243_; 
lean_dec_ref_known(v___x_2242_, 1);
v___x_2243_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_2236_);
return v___x_2243_;
}
else
{
lean_object* v_a_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2251_; 
lean_dec(v_fst_2236_);
v_a_2244_ = lean_ctor_get(v___x_2242_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v___x_2242_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2246_ = v___x_2242_;
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_a_2244_);
lean_dec(v___x_2242_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2249_; 
if (v_isShared_2247_ == 0)
{
v___x_2249_ = v___x_2246_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v_a_2244_);
v___x_2249_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
return v___x_2249_;
}
}
}
}
v___jp_2256_:
{
uint8_t v_result_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; double v___x_2262_; lean_object* v_data_2263_; 
v_result_2259_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5_spec__6(v_fst_2236_);
v___x_2260_ = lean_box(v_result_2259_);
v___x_2261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2261_, 0, v___x_2260_);
v___x_2262_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0);
lean_inc_ref(v_tag_2225_);
lean_inc_ref(v___x_2261_);
lean_inc(v_cls_2223_);
v_data_2263_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2263_, 0, v_cls_2223_);
lean_ctor_set(v_data_2263_, 1, v___x_2261_);
lean_ctor_set(v_data_2263_, 2, v_tag_2225_);
lean_ctor_set_float(v_data_2263_, sizeof(void*)*3, v___x_2262_);
lean_ctor_set_float(v_data_2263_, sizeof(void*)*3 + 8, v___x_2262_);
lean_ctor_set_uint8(v_data_2263_, sizeof(void*)*3 + 16, v_collapsed_2224_);
if (v___x_2255_ == 0)
{
lean_dec_ref_known(v___x_2261_, 1);
lean_dec(v_snd_2253_);
lean_dec(v_fst_2252_);
lean_dec_ref(v_tag_2225_);
lean_dec(v_cls_2223_);
v___y_2239_ = v_a_2258_;
v___y_2240_ = v___y_2257_;
v_data_2241_ = v_data_2263_;
goto v___jp_2238_;
}
else
{
lean_object* v_data_2264_; double v___x_2265_; double v___x_2266_; 
lean_dec_ref_known(v_data_2263_, 3);
v_data_2264_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2264_, 0, v_cls_2223_);
lean_ctor_set(v_data_2264_, 1, v___x_2261_);
lean_ctor_set(v_data_2264_, 2, v_tag_2225_);
v___x_2265_ = lean_unbox_float(v_fst_2252_);
lean_dec(v_fst_2252_);
lean_ctor_set_float(v_data_2264_, sizeof(void*)*3, v___x_2265_);
v___x_2266_ = lean_unbox_float(v_snd_2253_);
lean_dec(v_snd_2253_);
lean_ctor_set_float(v_data_2264_, sizeof(void*)*3 + 8, v___x_2266_);
lean_ctor_set_uint8(v_data_2264_, sizeof(void*)*3 + 16, v_collapsed_2224_);
v___y_2239_ = v_a_2258_;
v___y_2240_ = v___y_2257_;
v_data_2241_ = v_data_2264_;
goto v___jp_2238_;
}
}
v___jp_2267_:
{
lean_object* v_ref_2268_; lean_object* v___x_2269_; 
v_ref_2268_ = lean_ctor_get(v___y_2233_, 5);
lean_inc(v___y_2234_);
lean_inc_ref(v___y_2233_);
lean_inc(v___y_2232_);
lean_inc_ref(v___y_2231_);
lean_inc(v_fst_2236_);
v___x_2269_ = lean_apply_6(v_msg_2229_, v_fst_2236_, v___y_2231_, v___y_2232_, v___y_2233_, v___y_2234_, lean_box(0));
if (lean_obj_tag(v___x_2269_) == 0)
{
lean_object* v_a_2270_; 
v_a_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc(v_a_2270_);
lean_dec_ref_known(v___x_2269_, 1);
v___y_2257_ = v_ref_2268_;
v_a_2258_ = v_a_2270_;
goto v___jp_2256_;
}
else
{
lean_object* v___x_2271_; 
lean_dec_ref_known(v___x_2269_, 1);
v___x_2271_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__2);
v___y_2257_ = v_ref_2268_;
v_a_2258_ = v___x_2271_;
goto v___jp_2256_;
}
}
v___jp_2272_:
{
if (v_clsEnabled_2227_ == 0)
{
if (v___y_2273_ == 0)
{
lean_object* v___x_2274_; lean_object* v_traceState_2275_; lean_object* v_env_2276_; lean_object* v_nextMacroScope_2277_; lean_object* v_ngen_2278_; lean_object* v_auxDeclNGen_2279_; lean_object* v_cache_2280_; lean_object* v_messages_2281_; lean_object* v_infoState_2282_; lean_object* v_snapshotTasks_2283_; lean_object* v___x_2285_; uint8_t v_isShared_2286_; uint8_t v_isSharedCheck_2302_; 
lean_dec(v_snd_2253_);
lean_dec(v_fst_2252_);
lean_dec_ref(v_msg_2229_);
lean_dec_ref(v_tag_2225_);
lean_dec(v_cls_2223_);
v___x_2274_ = lean_st_ref_take(v___y_2234_);
v_traceState_2275_ = lean_ctor_get(v___x_2274_, 4);
v_env_2276_ = lean_ctor_get(v___x_2274_, 0);
v_nextMacroScope_2277_ = lean_ctor_get(v___x_2274_, 1);
v_ngen_2278_ = lean_ctor_get(v___x_2274_, 2);
v_auxDeclNGen_2279_ = lean_ctor_get(v___x_2274_, 3);
v_cache_2280_ = lean_ctor_get(v___x_2274_, 5);
v_messages_2281_ = lean_ctor_get(v___x_2274_, 6);
v_infoState_2282_ = lean_ctor_get(v___x_2274_, 7);
v_snapshotTasks_2283_ = lean_ctor_get(v___x_2274_, 8);
v_isSharedCheck_2302_ = !lean_is_exclusive(v___x_2274_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2285_ = v___x_2274_;
v_isShared_2286_ = v_isSharedCheck_2302_;
goto v_resetjp_2284_;
}
else
{
lean_inc(v_snapshotTasks_2283_);
lean_inc(v_infoState_2282_);
lean_inc(v_messages_2281_);
lean_inc(v_cache_2280_);
lean_inc(v_traceState_2275_);
lean_inc(v_auxDeclNGen_2279_);
lean_inc(v_ngen_2278_);
lean_inc(v_nextMacroScope_2277_);
lean_inc(v_env_2276_);
lean_dec(v___x_2274_);
v___x_2285_ = lean_box(0);
v_isShared_2286_ = v_isSharedCheck_2302_;
goto v_resetjp_2284_;
}
v_resetjp_2284_:
{
uint64_t v_tid_2287_; lean_object* v_traces_2288_; lean_object* v___x_2290_; uint8_t v_isShared_2291_; uint8_t v_isSharedCheck_2301_; 
v_tid_2287_ = lean_ctor_get_uint64(v_traceState_2275_, sizeof(void*)*1);
v_traces_2288_ = lean_ctor_get(v_traceState_2275_, 0);
v_isSharedCheck_2301_ = !lean_is_exclusive(v_traceState_2275_);
if (v_isSharedCheck_2301_ == 0)
{
v___x_2290_ = v_traceState_2275_;
v_isShared_2291_ = v_isSharedCheck_2301_;
goto v_resetjp_2289_;
}
else
{
lean_inc(v_traces_2288_);
lean_dec(v_traceState_2275_);
v___x_2290_ = lean_box(0);
v_isShared_2291_ = v_isSharedCheck_2301_;
goto v_resetjp_2289_;
}
v_resetjp_2289_:
{
lean_object* v___x_2292_; lean_object* v___x_2294_; 
v___x_2292_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2228_, v_traces_2288_);
lean_dec_ref(v_traces_2288_);
if (v_isShared_2291_ == 0)
{
lean_ctor_set(v___x_2290_, 0, v___x_2292_);
v___x_2294_ = v___x_2290_;
goto v_reusejp_2293_;
}
else
{
lean_object* v_reuseFailAlloc_2300_; 
v_reuseFailAlloc_2300_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2300_, 0, v___x_2292_);
lean_ctor_set_uint64(v_reuseFailAlloc_2300_, sizeof(void*)*1, v_tid_2287_);
v___x_2294_ = v_reuseFailAlloc_2300_;
goto v_reusejp_2293_;
}
v_reusejp_2293_:
{
lean_object* v___x_2296_; 
if (v_isShared_2286_ == 0)
{
lean_ctor_set(v___x_2285_, 4, v___x_2294_);
v___x_2296_ = v___x_2285_;
goto v_reusejp_2295_;
}
else
{
lean_object* v_reuseFailAlloc_2299_; 
v_reuseFailAlloc_2299_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2299_, 0, v_env_2276_);
lean_ctor_set(v_reuseFailAlloc_2299_, 1, v_nextMacroScope_2277_);
lean_ctor_set(v_reuseFailAlloc_2299_, 2, v_ngen_2278_);
lean_ctor_set(v_reuseFailAlloc_2299_, 3, v_auxDeclNGen_2279_);
lean_ctor_set(v_reuseFailAlloc_2299_, 4, v___x_2294_);
lean_ctor_set(v_reuseFailAlloc_2299_, 5, v_cache_2280_);
lean_ctor_set(v_reuseFailAlloc_2299_, 6, v_messages_2281_);
lean_ctor_set(v_reuseFailAlloc_2299_, 7, v_infoState_2282_);
lean_ctor_set(v_reuseFailAlloc_2299_, 8, v_snapshotTasks_2283_);
v___x_2296_ = v_reuseFailAlloc_2299_;
goto v_reusejp_2295_;
}
v_reusejp_2295_:
{
lean_object* v___x_2297_; lean_object* v___x_2298_; 
v___x_2297_ = lean_st_ref_set(v___y_2234_, v___x_2296_);
v___x_2298_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2_spec__3___redArg(v_fst_2236_);
return v___x_2298_;
}
}
}
}
}
else
{
goto v___jp_2267_;
}
}
else
{
goto v___jp_2267_;
}
}
v___jp_2303_:
{
double v___x_2305_; double v___x_2306_; double v___x_2307_; uint8_t v___x_2308_; 
v___x_2305_ = lean_unbox_float(v_snd_2253_);
v___x_2306_ = lean_unbox_float(v_fst_2252_);
v___x_2307_ = lean_float_sub(v___x_2305_, v___x_2306_);
v___x_2308_ = lean_float_decLt(v___y_2304_, v___x_2307_);
v___y_2273_ = v___x_2308_;
goto v___jp_2272_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5___boxed(lean_object* v_cls_2319_, lean_object* v_collapsed_2320_, lean_object* v_tag_2321_, lean_object* v_opts_2322_, lean_object* v_clsEnabled_2323_, lean_object* v_oldTraces_2324_, lean_object* v_msg_2325_, lean_object* v_resStartStop_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_){
_start:
{
uint8_t v_collapsed_boxed_2332_; uint8_t v_clsEnabled_boxed_2333_; lean_object* v_res_2334_; 
v_collapsed_boxed_2332_ = lean_unbox(v_collapsed_2320_);
v_clsEnabled_boxed_2333_ = lean_unbox(v_clsEnabled_2323_);
v_res_2334_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5(v_cls_2319_, v_collapsed_boxed_2332_, v_tag_2321_, v_opts_2322_, v_clsEnabled_boxed_2333_, v_oldTraces_2324_, v_msg_2325_, v_resStartStop_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_);
lean_dec(v___y_2330_);
lean_dec_ref(v___y_2329_);
lean_dec(v___y_2328_);
lean_dec_ref(v___y_2327_);
lean_dec_ref(v_opts_2322_);
return v_res_2334_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2(void){
_start:
{
lean_object* v___x_2338_; lean_object* v___x_2339_; 
v___x_2338_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1));
v___x_2339_ = l_Lean_MessageData_ofFormat(v___x_2338_);
return v___x_2339_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3(void){
_start:
{
lean_object* v___x_2340_; lean_object* v___x_2341_; 
v___x_2340_ = lean_box(1);
v___x_2341_ = l_Lean_MessageData_ofFormat(v___x_2340_);
return v___x_2341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10(lean_object* v_a_2342_, lean_object* v_a_2343_){
_start:
{
if (lean_obj_tag(v_a_2342_) == 0)
{
lean_object* v___x_2344_; 
v___x_2344_ = l_List_reverse___redArg(v_a_2343_);
return v___x_2344_;
}
else
{
lean_object* v_head_2345_; lean_object* v_tail_2346_; lean_object* v___x_2348_; uint8_t v_isShared_2349_; uint8_t v_isSharedCheck_2374_; 
v_head_2345_ = lean_ctor_get(v_a_2342_, 0);
v_tail_2346_ = lean_ctor_get(v_a_2342_, 1);
v_isSharedCheck_2374_ = !lean_is_exclusive(v_a_2342_);
if (v_isSharedCheck_2374_ == 0)
{
v___x_2348_ = v_a_2342_;
v_isShared_2349_ = v_isSharedCheck_2374_;
goto v_resetjp_2347_;
}
else
{
lean_inc(v_tail_2346_);
lean_inc(v_head_2345_);
lean_dec(v_a_2342_);
v___x_2348_ = lean_box(0);
v_isShared_2349_ = v_isSharedCheck_2374_;
goto v_resetjp_2347_;
}
v_resetjp_2347_:
{
lean_object* v_fst_2350_; lean_object* v_snd_2351_; lean_object* v___x_2353_; uint8_t v_isShared_2354_; uint8_t v_isSharedCheck_2373_; 
v_fst_2350_ = lean_ctor_get(v_head_2345_, 0);
v_snd_2351_ = lean_ctor_get(v_head_2345_, 1);
v_isSharedCheck_2373_ = !lean_is_exclusive(v_head_2345_);
if (v_isSharedCheck_2373_ == 0)
{
v___x_2353_ = v_head_2345_;
v_isShared_2354_ = v_isSharedCheck_2373_;
goto v_resetjp_2352_;
}
else
{
lean_inc(v_snd_2351_);
lean_inc(v_fst_2350_);
lean_dec(v_head_2345_);
v___x_2353_ = lean_box(0);
v_isShared_2354_ = v_isSharedCheck_2373_;
goto v_resetjp_2352_;
}
v_resetjp_2352_:
{
lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2360_; 
v___x_2355_ = l_Nat_reprFast(v_fst_2350_);
v___x_2356_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2356_, 0, v___x_2355_);
v___x_2357_ = l_Lean_MessageData_ofFormat(v___x_2356_);
v___x_2358_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__2);
if (v_isShared_2354_ == 0)
{
lean_ctor_set_tag(v___x_2353_, 7);
lean_ctor_set(v___x_2353_, 1, v___x_2358_);
lean_ctor_set(v___x_2353_, 0, v___x_2357_);
v___x_2360_ = v___x_2353_;
goto v_reusejp_2359_;
}
else
{
lean_object* v_reuseFailAlloc_2372_; 
v_reuseFailAlloc_2372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2372_, 0, v___x_2357_);
lean_ctor_set(v_reuseFailAlloc_2372_, 1, v___x_2358_);
v___x_2360_ = v_reuseFailAlloc_2372_;
goto v_reusejp_2359_;
}
v_reusejp_2359_:
{
lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2369_; 
v___x_2361_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__3);
v___x_2362_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2362_, 0, v___x_2360_);
lean_ctor_set(v___x_2362_, 1, v___x_2361_);
v___x_2363_ = l_Nat_reprFast(v_snd_2351_);
v___x_2364_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2364_, 0, v___x_2363_);
v___x_2365_ = l_Lean_MessageData_ofFormat(v___x_2364_);
v___x_2366_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2366_, 0, v___x_2362_);
lean_ctor_set(v___x_2366_, 1, v___x_2365_);
v___x_2367_ = l_Lean_MessageData_paren(v___x_2366_);
if (v_isShared_2349_ == 0)
{
lean_ctor_set(v___x_2348_, 1, v_a_2343_);
lean_ctor_set(v___x_2348_, 0, v___x_2367_);
v___x_2369_ = v___x_2348_;
goto v_reusejp_2368_;
}
else
{
lean_object* v_reuseFailAlloc_2371_; 
v_reuseFailAlloc_2371_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2371_, 0, v___x_2367_);
lean_ctor_set(v_reuseFailAlloc_2371_, 1, v_a_2343_);
v___x_2369_ = v_reuseFailAlloc_2371_;
goto v_reusejp_2368_;
}
v_reusejp_2368_:
{
v_a_2342_ = v_tail_2346_;
v_a_2343_ = v___x_2369_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(lean_object* v_x_2375_, lean_object* v_x_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_){
_start:
{
if (lean_obj_tag(v_x_2375_) == 0)
{
lean_object* v___x_2382_; lean_object* v___x_2383_; 
v___x_2382_ = l_List_reverse___redArg(v_x_2376_);
v___x_2383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2383_, 0, v___x_2382_);
return v___x_2383_;
}
else
{
lean_object* v_head_2384_; lean_object* v_tail_2385_; lean_object* v___x_2387_; uint8_t v_isShared_2388_; uint8_t v_isSharedCheck_2410_; 
v_head_2384_ = lean_ctor_get(v_x_2375_, 0);
v_tail_2385_ = lean_ctor_get(v_x_2375_, 1);
v_isSharedCheck_2410_ = !lean_is_exclusive(v_x_2375_);
if (v_isSharedCheck_2410_ == 0)
{
v___x_2387_ = v_x_2375_;
v_isShared_2388_ = v_isSharedCheck_2410_;
goto v_resetjp_2386_;
}
else
{
lean_inc(v_tail_2385_);
lean_inc(v_head_2384_);
lean_dec(v_x_2375_);
v___x_2387_ = lean_box(0);
v_isShared_2388_ = v_isSharedCheck_2410_;
goto v_resetjp_2386_;
}
v_resetjp_2386_:
{
lean_object* v___y_2390_; lean_object* v_snd_2404_; lean_object* v_fst_2405_; lean_object* v_fst_2406_; lean_object* v___x_2407_; 
v_snd_2404_ = lean_ctor_get(v_head_2384_, 1);
lean_inc(v_snd_2404_);
v_fst_2405_ = lean_ctor_get(v_head_2384_, 0);
lean_inc(v_fst_2405_);
lean_dec(v_head_2384_);
v_fst_2406_ = lean_ctor_get(v_snd_2404_, 0);
lean_inc(v_fst_2406_);
lean_dec(v_snd_2404_);
v___x_2407_ = lp_mathlib_Mathlib_Tactic_Linarith_leftOfIneqProof(v_fst_2405_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_);
if (lean_obj_tag(v___x_2407_) == 0)
{
lean_object* v_a_2408_; lean_object* v___x_2409_; 
v_a_2408_ = lean_ctor_get(v___x_2407_, 0);
lean_inc(v_a_2408_);
lean_dec_ref_known(v___x_2407_, 1);
v___x_2409_ = lp_mathlib_Mathlib_Tactic_Linarith_mulExpr(v_fst_2406_, v_a_2408_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_);
v___y_2390_ = v___x_2409_;
goto v___jp_2389_;
}
else
{
lean_dec(v_fst_2406_);
v___y_2390_ = v___x_2407_;
goto v___jp_2389_;
}
v___jp_2389_:
{
if (lean_obj_tag(v___y_2390_) == 0)
{
lean_object* v_a_2391_; lean_object* v___x_2393_; 
v_a_2391_ = lean_ctor_get(v___y_2390_, 0);
lean_inc(v_a_2391_);
lean_dec_ref_known(v___y_2390_, 1);
if (v_isShared_2388_ == 0)
{
lean_ctor_set(v___x_2387_, 1, v_x_2376_);
lean_ctor_set(v___x_2387_, 0, v_a_2391_);
v___x_2393_ = v___x_2387_;
goto v_reusejp_2392_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_a_2391_);
lean_ctor_set(v_reuseFailAlloc_2395_, 1, v_x_2376_);
v___x_2393_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2392_;
}
v_reusejp_2392_:
{
v_x_2375_ = v_tail_2385_;
v_x_2376_ = v___x_2393_;
goto _start;
}
}
else
{
lean_object* v_a_2396_; lean_object* v___x_2398_; uint8_t v_isShared_2399_; uint8_t v_isSharedCheck_2403_; 
lean_del_object(v___x_2387_);
lean_dec(v_tail_2385_);
lean_dec(v_x_2376_);
v_a_2396_ = lean_ctor_get(v___y_2390_, 0);
v_isSharedCheck_2403_ = !lean_is_exclusive(v___y_2390_);
if (v_isSharedCheck_2403_ == 0)
{
v___x_2398_ = v___y_2390_;
v_isShared_2399_ = v_isSharedCheck_2403_;
goto v_resetjp_2397_;
}
else
{
lean_inc(v_a_2396_);
lean_dec(v___y_2390_);
v___x_2398_ = lean_box(0);
v_isShared_2399_ = v_isSharedCheck_2403_;
goto v_resetjp_2397_;
}
v_resetjp_2397_:
{
lean_object* v___x_2401_; 
if (v_isShared_2399_ == 0)
{
v___x_2401_ = v___x_2398_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v_a_2396_);
v___x_2401_ = v_reuseFailAlloc_2402_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
return v___x_2401_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6___boxed(lean_object* v_x_2411_, lean_object* v_x_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_){
_start:
{
lean_object* v_res_2418_; 
v_res_2418_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(v_x_2411_, v_x_2412_, v___y_2413_, v___y_2414_, v___y_2415_, v___y_2416_);
lean_dec(v___y_2416_);
lean_dec_ref(v___y_2415_);
lean_dec(v___y_2414_);
lean_dec_ref(v___y_2413_);
return v_res_2418_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2(void){
_start:
{
lean_object* v___x_2421_; lean_object* v___x_2422_; 
v___x_2421_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__0));
v___x_2422_ = lean_string_length(v___x_2421_);
return v___x_2422_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3(void){
_start:
{
lean_object* v___x_2423_; lean_object* v___x_2424_; 
v___x_2423_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2, &lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2_once, _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__2);
v___x_2424_ = lean_nat_to_int(v___x_2423_);
return v___x_2424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18(lean_object* v_x_2429_, lean_object* v_x_2430_, lean_object* v_x_2431_){
_start:
{
if (lean_obj_tag(v_x_2431_) == 0)
{
lean_dec(v_x_2429_);
return v_x_2430_;
}
else
{
lean_object* v_head_2432_; lean_object* v_tail_2433_; lean_object* v___x_2435_; uint8_t v_isShared_2436_; uint8_t v_isSharedCheck_2467_; 
v_head_2432_ = lean_ctor_get(v_x_2431_, 0);
v_tail_2433_ = lean_ctor_get(v_x_2431_, 1);
v_isSharedCheck_2467_ = !lean_is_exclusive(v_x_2431_);
if (v_isSharedCheck_2467_ == 0)
{
v___x_2435_ = v_x_2431_;
v_isShared_2436_ = v_isSharedCheck_2467_;
goto v_resetjp_2434_;
}
else
{
lean_inc(v_tail_2433_);
lean_inc(v_head_2432_);
lean_dec(v_x_2431_);
v___x_2435_ = lean_box(0);
v_isShared_2436_ = v_isSharedCheck_2467_;
goto v_resetjp_2434_;
}
v_resetjp_2434_:
{
lean_object* v_fst_2437_; lean_object* v_snd_2438_; lean_object* v___x_2440_; uint8_t v_isShared_2441_; uint8_t v_isSharedCheck_2466_; 
v_fst_2437_ = lean_ctor_get(v_head_2432_, 0);
v_snd_2438_ = lean_ctor_get(v_head_2432_, 1);
v_isSharedCheck_2466_ = !lean_is_exclusive(v_head_2432_);
if (v_isSharedCheck_2466_ == 0)
{
v___x_2440_ = v_head_2432_;
v_isShared_2441_ = v_isSharedCheck_2466_;
goto v_resetjp_2439_;
}
else
{
lean_inc(v_snd_2438_);
lean_inc(v_fst_2437_);
lean_dec(v_head_2432_);
v___x_2440_ = lean_box(0);
v_isShared_2441_ = v_isSharedCheck_2466_;
goto v_resetjp_2439_;
}
v_resetjp_2439_:
{
lean_object* v___x_2443_; 
lean_inc(v_x_2429_);
if (v_isShared_2441_ == 0)
{
lean_ctor_set_tag(v___x_2440_, 5);
lean_ctor_set(v___x_2440_, 1, v_x_2429_);
lean_ctor_set(v___x_2440_, 0, v_x_2430_);
v___x_2443_ = v___x_2440_;
goto v_reusejp_2442_;
}
else
{
lean_object* v_reuseFailAlloc_2465_; 
v_reuseFailAlloc_2465_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2465_, 0, v_x_2430_);
lean_ctor_set(v_reuseFailAlloc_2465_, 1, v_x_2429_);
v___x_2443_ = v_reuseFailAlloc_2465_;
goto v_reusejp_2442_;
}
v_reusejp_2442_:
{
lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2448_; 
v___x_2444_ = l_Nat_reprFast(v_fst_2437_);
v___x_2445_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2445_, 0, v___x_2444_);
v___x_2446_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1));
if (v_isShared_2436_ == 0)
{
lean_ctor_set_tag(v___x_2435_, 5);
lean_ctor_set(v___x_2435_, 1, v___x_2446_);
lean_ctor_set(v___x_2435_, 0, v___x_2445_);
v___x_2448_ = v___x_2435_;
goto v_reusejp_2447_;
}
else
{
lean_object* v_reuseFailAlloc_2464_; 
v_reuseFailAlloc_2464_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2464_, 0, v___x_2445_);
lean_ctor_set(v_reuseFailAlloc_2464_, 1, v___x_2446_);
v___x_2448_ = v_reuseFailAlloc_2464_;
goto v_reusejp_2447_;
}
v_reusejp_2447_:
{
lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; uint8_t v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; 
v___x_2449_ = lean_box(1);
v___x_2450_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2450_, 0, v___x_2448_);
lean_ctor_set(v___x_2450_, 1, v___x_2449_);
v___x_2451_ = l_Int_repr(v_snd_2438_);
lean_dec(v_snd_2438_);
v___x_2452_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2452_, 0, v___x_2451_);
v___x_2453_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2453_, 0, v___x_2450_);
lean_ctor_set(v___x_2453_, 1, v___x_2452_);
v___x_2454_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3, &lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3_once, _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3);
v___x_2455_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4));
v___x_2456_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2456_, 0, v___x_2455_);
lean_ctor_set(v___x_2456_, 1, v___x_2453_);
v___x_2457_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5));
v___x_2458_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2458_, 0, v___x_2456_);
lean_ctor_set(v___x_2458_, 1, v___x_2457_);
v___x_2459_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2459_, 0, v___x_2454_);
lean_ctor_set(v___x_2459_, 1, v___x_2458_);
v___x_2460_ = 0;
v___x_2461_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2461_, 0, v___x_2459_);
lean_ctor_set_uint8(v___x_2461_, sizeof(void*)*1, v___x_2460_);
v___x_2462_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2462_, 0, v___x_2443_);
lean_ctor_set(v___x_2462_, 1, v___x_2461_);
v_x_2430_ = v___x_2462_;
v_x_2431_ = v_tail_2433_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17(lean_object* v_x_2468_, lean_object* v_x_2469_){
_start:
{
if (lean_obj_tag(v_x_2468_) == 0)
{
lean_object* v___x_2470_; 
lean_dec(v_x_2469_);
v___x_2470_ = lean_box(0);
return v___x_2470_;
}
else
{
lean_object* v_tail_2471_; 
v_tail_2471_ = lean_ctor_get(v_x_2468_, 1);
if (lean_obj_tag(v_tail_2471_) == 0)
{
lean_object* v_head_2472_; lean_object* v___x_2474_; uint8_t v_isShared_2475_; uint8_t v_isSharedCheck_2503_; 
lean_dec(v_x_2469_);
v_head_2472_ = lean_ctor_get(v_x_2468_, 0);
v_isSharedCheck_2503_ = !lean_is_exclusive(v_x_2468_);
if (v_isSharedCheck_2503_ == 0)
{
lean_object* v_unused_2504_; 
v_unused_2504_ = lean_ctor_get(v_x_2468_, 1);
lean_dec(v_unused_2504_);
v___x_2474_ = v_x_2468_;
v_isShared_2475_ = v_isSharedCheck_2503_;
goto v_resetjp_2473_;
}
else
{
lean_inc(v_head_2472_);
lean_dec(v_x_2468_);
v___x_2474_ = lean_box(0);
v_isShared_2475_ = v_isSharedCheck_2503_;
goto v_resetjp_2473_;
}
v_resetjp_2473_:
{
lean_object* v_fst_2476_; lean_object* v_snd_2477_; lean_object* v___x_2479_; uint8_t v_isShared_2480_; uint8_t v_isSharedCheck_2502_; 
v_fst_2476_ = lean_ctor_get(v_head_2472_, 0);
v_snd_2477_ = lean_ctor_get(v_head_2472_, 1);
v_isSharedCheck_2502_ = !lean_is_exclusive(v_head_2472_);
if (v_isSharedCheck_2502_ == 0)
{
v___x_2479_ = v_head_2472_;
v_isShared_2480_ = v_isSharedCheck_2502_;
goto v_resetjp_2478_;
}
else
{
lean_inc(v_snd_2477_);
lean_inc(v_fst_2476_);
lean_dec(v_head_2472_);
v___x_2479_ = lean_box(0);
v_isShared_2480_ = v_isSharedCheck_2502_;
goto v_resetjp_2478_;
}
v_resetjp_2478_:
{
lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2485_; 
v___x_2481_ = l_Nat_reprFast(v_fst_2476_);
v___x_2482_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2482_, 0, v___x_2481_);
v___x_2483_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1));
if (v_isShared_2480_ == 0)
{
lean_ctor_set_tag(v___x_2479_, 5);
lean_ctor_set(v___x_2479_, 1, v___x_2483_);
lean_ctor_set(v___x_2479_, 0, v___x_2482_);
v___x_2485_ = v___x_2479_;
goto v_reusejp_2484_;
}
else
{
lean_object* v_reuseFailAlloc_2501_; 
v_reuseFailAlloc_2501_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2501_, 0, v___x_2482_);
lean_ctor_set(v_reuseFailAlloc_2501_, 1, v___x_2483_);
v___x_2485_ = v_reuseFailAlloc_2501_;
goto v_reusejp_2484_;
}
v_reusejp_2484_:
{
lean_object* v___x_2486_; lean_object* v___x_2488_; 
v___x_2486_ = lean_box(1);
if (v_isShared_2475_ == 0)
{
lean_ctor_set_tag(v___x_2474_, 5);
lean_ctor_set(v___x_2474_, 1, v___x_2486_);
lean_ctor_set(v___x_2474_, 0, v___x_2485_);
v___x_2488_ = v___x_2474_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2500_; 
v_reuseFailAlloc_2500_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2500_, 0, v___x_2485_);
lean_ctor_set(v_reuseFailAlloc_2500_, 1, v___x_2486_);
v___x_2488_ = v_reuseFailAlloc_2500_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; uint8_t v___x_2498_; lean_object* v___x_2499_; 
v___x_2489_ = l_Int_repr(v_snd_2477_);
lean_dec(v_snd_2477_);
v___x_2490_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2490_, 0, v___x_2489_);
v___x_2491_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2488_);
lean_ctor_set(v___x_2491_, 1, v___x_2490_);
v___x_2492_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3, &lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3_once, _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3);
v___x_2493_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4));
v___x_2494_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2494_, 0, v___x_2493_);
lean_ctor_set(v___x_2494_, 1, v___x_2491_);
v___x_2495_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5));
v___x_2496_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2494_);
lean_ctor_set(v___x_2496_, 1, v___x_2495_);
v___x_2497_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2497_, 0, v___x_2492_);
lean_ctor_set(v___x_2497_, 1, v___x_2496_);
v___x_2498_ = 0;
v___x_2499_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2499_, 0, v___x_2497_);
lean_ctor_set_uint8(v___x_2499_, sizeof(void*)*1, v___x_2498_);
return v___x_2499_;
}
}
}
}
}
else
{
lean_object* v_head_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2537_; 
lean_inc(v_tail_2471_);
v_head_2505_ = lean_ctor_get(v_x_2468_, 0);
v_isSharedCheck_2537_ = !lean_is_exclusive(v_x_2468_);
if (v_isSharedCheck_2537_ == 0)
{
lean_object* v_unused_2538_; 
v_unused_2538_ = lean_ctor_get(v_x_2468_, 1);
lean_dec(v_unused_2538_);
v___x_2507_ = v_x_2468_;
v_isShared_2508_ = v_isSharedCheck_2537_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_head_2505_);
lean_dec(v_x_2468_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2537_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v_fst_2509_; lean_object* v_snd_2510_; lean_object* v___x_2512_; uint8_t v_isShared_2513_; uint8_t v_isSharedCheck_2536_; 
v_fst_2509_ = lean_ctor_get(v_head_2505_, 0);
v_snd_2510_ = lean_ctor_get(v_head_2505_, 1);
v_isSharedCheck_2536_ = !lean_is_exclusive(v_head_2505_);
if (v_isSharedCheck_2536_ == 0)
{
v___x_2512_ = v_head_2505_;
v_isShared_2513_ = v_isSharedCheck_2536_;
goto v_resetjp_2511_;
}
else
{
lean_inc(v_snd_2510_);
lean_inc(v_fst_2509_);
lean_dec(v_head_2505_);
v___x_2512_ = lean_box(0);
v_isShared_2513_ = v_isSharedCheck_2536_;
goto v_resetjp_2511_;
}
v_resetjp_2511_:
{
lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2518_; 
v___x_2514_ = l_Nat_reprFast(v_fst_2509_);
v___x_2515_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2515_, 0, v___x_2514_);
v___x_2516_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10___closed__1));
if (v_isShared_2513_ == 0)
{
lean_ctor_set_tag(v___x_2512_, 5);
lean_ctor_set(v___x_2512_, 1, v___x_2516_);
lean_ctor_set(v___x_2512_, 0, v___x_2515_);
v___x_2518_ = v___x_2512_;
goto v_reusejp_2517_;
}
else
{
lean_object* v_reuseFailAlloc_2535_; 
v_reuseFailAlloc_2535_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2535_, 0, v___x_2515_);
lean_ctor_set(v_reuseFailAlloc_2535_, 1, v___x_2516_);
v___x_2518_ = v_reuseFailAlloc_2535_;
goto v_reusejp_2517_;
}
v_reusejp_2517_:
{
lean_object* v___x_2519_; lean_object* v___x_2521_; 
v___x_2519_ = lean_box(1);
if (v_isShared_2508_ == 0)
{
lean_ctor_set_tag(v___x_2507_, 5);
lean_ctor_set(v___x_2507_, 1, v___x_2519_);
lean_ctor_set(v___x_2507_, 0, v___x_2518_);
v___x_2521_ = v___x_2507_;
goto v_reusejp_2520_;
}
else
{
lean_object* v_reuseFailAlloc_2534_; 
v_reuseFailAlloc_2534_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2534_, 0, v___x_2518_);
lean_ctor_set(v_reuseFailAlloc_2534_, 1, v___x_2519_);
v___x_2521_ = v_reuseFailAlloc_2534_;
goto v_reusejp_2520_;
}
v_reusejp_2520_:
{
lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; uint8_t v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; 
v___x_2522_ = l_Int_repr(v_snd_2510_);
lean_dec(v_snd_2510_);
v___x_2523_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2523_, 0, v___x_2522_);
v___x_2524_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2524_, 0, v___x_2521_);
lean_ctor_set(v___x_2524_, 1, v___x_2523_);
v___x_2525_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3, &lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3_once, _init_lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__3);
v___x_2526_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__4));
v___x_2527_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2527_, 0, v___x_2526_);
lean_ctor_set(v___x_2527_, 1, v___x_2524_);
v___x_2528_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18___closed__5));
v___x_2529_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2529_, 0, v___x_2527_);
lean_ctor_set(v___x_2529_, 1, v___x_2528_);
v___x_2530_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2530_, 0, v___x_2525_);
lean_ctor_set(v___x_2530_, 1, v___x_2529_);
v___x_2531_ = 0;
v___x_2532_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2532_, 0, v___x_2530_);
lean_ctor_set_uint8(v___x_2532_, sizeof(void*)*1, v___x_2531_);
v___x_2533_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17_spec__18(v_x_2469_, v___x_2532_, v_tail_2471_);
return v___x_2533_;
}
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5(void){
_start:
{
lean_object* v___x_2547_; lean_object* v___x_2548_; 
v___x_2547_ = ((lean_object*)(lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__3));
v___x_2548_ = lean_string_length(v___x_2547_);
return v___x_2548_;
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6(void){
_start:
{
lean_object* v___x_2549_; lean_object* v___x_2550_; 
v___x_2549_ = lean_obj_once(&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5, &lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5_once, _init_lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__5);
v___x_2550_ = lean_nat_to_int(v___x_2549_);
return v___x_2550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14(lean_object* v_x_2555_){
_start:
{
if (lean_obj_tag(v_x_2555_) == 0)
{
lean_object* v___x_2556_; 
v___x_2556_ = ((lean_object*)(lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__1));
return v___x_2556_;
}
else
{
lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; uint8_t v___x_2565_; lean_object* v___x_2566_; 
v___x_2557_ = ((lean_object*)(lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__2));
v___x_2558_ = lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__17(v_x_2555_, v___x_2557_);
v___x_2559_ = lean_obj_once(&lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6, &lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6_once, _init_lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__6);
v___x_2560_ = ((lean_object*)(lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__7));
v___x_2561_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2561_, 0, v___x_2560_);
lean_ctor_set(v___x_2561_, 1, v___x_2558_);
v___x_2562_ = ((lean_object*)(lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14___closed__8));
v___x_2563_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2563_, 0, v___x_2561_);
lean_ctor_set(v___x_2563_, 1, v___x_2562_);
v___x_2564_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2564_, 0, v___x_2559_);
lean_ctor_set(v___x_2564_, 1, v___x_2563_);
v___x_2565_ = 0;
v___x_2566_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2566_, 0, v___x_2564_);
lean_ctor_set_uint8(v___x_2566_, sizeof(void*)*1, v___x_2565_);
return v___x_2566_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15(lean_object* v_a_2570_, lean_object* v_a_2571_){
_start:
{
if (lean_obj_tag(v_a_2570_) == 0)
{
lean_object* v___x_2572_; 
v___x_2572_ = l_List_reverse___redArg(v_a_2571_);
return v___x_2572_;
}
else
{
lean_object* v_head_2573_; lean_object* v_tail_2574_; lean_object* v___x_2576_; uint8_t v_isShared_2577_; uint8_t v_isSharedCheck_2591_; 
v_head_2573_ = lean_ctor_get(v_a_2570_, 0);
v_tail_2574_ = lean_ctor_get(v_a_2570_, 1);
v_isSharedCheck_2591_ = !lean_is_exclusive(v_a_2570_);
if (v_isSharedCheck_2591_ == 0)
{
v___x_2576_ = v_a_2570_;
v_isShared_2577_ = v_isSharedCheck_2591_;
goto v_resetjp_2575_;
}
else
{
lean_inc(v_tail_2574_);
lean_inc(v_head_2573_);
lean_dec(v_a_2570_);
v___x_2576_ = lean_box(0);
v_isShared_2577_ = v_isSharedCheck_2591_;
goto v_resetjp_2575_;
}
v_resetjp_2575_:
{
uint8_t v_str_2578_; lean_object* v_coeffs_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2588_; 
v_str_2578_ = lean_ctor_get_uint8(v_head_2573_, sizeof(void*)*1);
v_coeffs_2579_ = lean_ctor_get(v_head_2573_, 0);
lean_inc(v_coeffs_2579_);
lean_dec(v_head_2573_);
v___x_2580_ = lp_mathlib_List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14(v_coeffs_2579_);
v___x_2581_ = lp_mathlib_Mathlib_Ineq_toString(v_str_2578_);
v___x_2582_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2582_, 0, v___x_2581_);
v___x_2583_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2583_, 0, v___x_2580_);
lean_ctor_set(v___x_2583_, 1, v___x_2582_);
v___x_2584_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15___closed__1));
v___x_2585_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2585_, 0, v___x_2583_);
lean_ctor_set(v___x_2585_, 1, v___x_2584_);
v___x_2586_ = l_Lean_MessageData_ofFormat(v___x_2585_);
if (v_isShared_2577_ == 0)
{
lean_ctor_set(v___x_2576_, 1, v_a_2571_);
lean_ctor_set(v___x_2576_, 0, v___x_2586_);
v___x_2588_ = v___x_2576_;
goto v_reusejp_2587_;
}
else
{
lean_object* v_reuseFailAlloc_2590_; 
v_reuseFailAlloc_2590_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2590_, 0, v___x_2586_);
lean_ctor_set(v_reuseFailAlloc_2590_, 1, v_a_2571_);
v___x_2588_ = v_reuseFailAlloc_2590_;
goto v_reusejp_2587_;
}
v_reusejp_2587_:
{
v_a_2570_ = v_tail_2574_;
v_a_2571_ = v___x_2588_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(lean_object* v_cls_2594_, lean_object* v_msg_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_, lean_object* v___y_2598_, lean_object* v___y_2599_){
_start:
{
lean_object* v_ref_2601_; lean_object* v___x_2602_; lean_object* v_a_2603_; lean_object* v___x_2605_; uint8_t v_isShared_2606_; uint8_t v_isSharedCheck_2647_; 
v_ref_2601_ = lean_ctor_get(v___y_2598_, 5);
v___x_2602_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0_spec__0(v_msg_2595_, v___y_2596_, v___y_2597_, v___y_2598_, v___y_2599_);
v_a_2603_ = lean_ctor_get(v___x_2602_, 0);
v_isSharedCheck_2647_ = !lean_is_exclusive(v___x_2602_);
if (v_isSharedCheck_2647_ == 0)
{
v___x_2605_ = v___x_2602_;
v_isShared_2606_ = v_isSharedCheck_2647_;
goto v_resetjp_2604_;
}
else
{
lean_inc(v_a_2603_);
lean_dec(v___x_2602_);
v___x_2605_ = lean_box(0);
v_isShared_2606_ = v_isSharedCheck_2647_;
goto v_resetjp_2604_;
}
v_resetjp_2604_:
{
lean_object* v___x_2607_; lean_object* v_traceState_2608_; lean_object* v_env_2609_; lean_object* v_nextMacroScope_2610_; lean_object* v_ngen_2611_; lean_object* v_auxDeclNGen_2612_; lean_object* v_cache_2613_; lean_object* v_messages_2614_; lean_object* v_infoState_2615_; lean_object* v_snapshotTasks_2616_; lean_object* v___x_2618_; uint8_t v_isShared_2619_; uint8_t v_isSharedCheck_2646_; 
v___x_2607_ = lean_st_ref_take(v___y_2599_);
v_traceState_2608_ = lean_ctor_get(v___x_2607_, 4);
v_env_2609_ = lean_ctor_get(v___x_2607_, 0);
v_nextMacroScope_2610_ = lean_ctor_get(v___x_2607_, 1);
v_ngen_2611_ = lean_ctor_get(v___x_2607_, 2);
v_auxDeclNGen_2612_ = lean_ctor_get(v___x_2607_, 3);
v_cache_2613_ = lean_ctor_get(v___x_2607_, 5);
v_messages_2614_ = lean_ctor_get(v___x_2607_, 6);
v_infoState_2615_ = lean_ctor_get(v___x_2607_, 7);
v_snapshotTasks_2616_ = lean_ctor_get(v___x_2607_, 8);
v_isSharedCheck_2646_ = !lean_is_exclusive(v___x_2607_);
if (v_isSharedCheck_2646_ == 0)
{
v___x_2618_ = v___x_2607_;
v_isShared_2619_ = v_isSharedCheck_2646_;
goto v_resetjp_2617_;
}
else
{
lean_inc(v_snapshotTasks_2616_);
lean_inc(v_infoState_2615_);
lean_inc(v_messages_2614_);
lean_inc(v_cache_2613_);
lean_inc(v_traceState_2608_);
lean_inc(v_auxDeclNGen_2612_);
lean_inc(v_ngen_2611_);
lean_inc(v_nextMacroScope_2610_);
lean_inc(v_env_2609_);
lean_dec(v___x_2607_);
v___x_2618_ = lean_box(0);
v_isShared_2619_ = v_isSharedCheck_2646_;
goto v_resetjp_2617_;
}
v_resetjp_2617_:
{
uint64_t v_tid_2620_; lean_object* v_traces_2621_; lean_object* v___x_2623_; uint8_t v_isShared_2624_; uint8_t v_isSharedCheck_2645_; 
v_tid_2620_ = lean_ctor_get_uint64(v_traceState_2608_, sizeof(void*)*1);
v_traces_2621_ = lean_ctor_get(v_traceState_2608_, 0);
v_isSharedCheck_2645_ = !lean_is_exclusive(v_traceState_2608_);
if (v_isSharedCheck_2645_ == 0)
{
v___x_2623_ = v_traceState_2608_;
v_isShared_2624_ = v_isSharedCheck_2645_;
goto v_resetjp_2622_;
}
else
{
lean_inc(v_traces_2621_);
lean_dec(v_traceState_2608_);
v___x_2623_ = lean_box(0);
v_isShared_2624_ = v_isSharedCheck_2645_;
goto v_resetjp_2622_;
}
v_resetjp_2622_:
{
lean_object* v___x_2625_; double v___x_2626_; uint8_t v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2635_; 
v___x_2625_ = lean_box(0);
v___x_2626_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__2___redArg___closed__0);
v___x_2627_ = 0;
v___x_2628_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3));
v___x_2629_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2629_, 0, v_cls_2594_);
lean_ctor_set(v___x_2629_, 1, v___x_2625_);
lean_ctor_set(v___x_2629_, 2, v___x_2628_);
lean_ctor_set_float(v___x_2629_, sizeof(void*)*3, v___x_2626_);
lean_ctor_set_float(v___x_2629_, sizeof(void*)*3 + 8, v___x_2626_);
lean_ctor_set_uint8(v___x_2629_, sizeof(void*)*3 + 16, v___x_2627_);
v___x_2630_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___closed__0));
v___x_2631_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2629_);
lean_ctor_set(v___x_2631_, 1, v_a_2603_);
lean_ctor_set(v___x_2631_, 2, v___x_2630_);
lean_inc(v_ref_2601_);
v___x_2632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2632_, 0, v_ref_2601_);
lean_ctor_set(v___x_2632_, 1, v___x_2631_);
v___x_2633_ = l_Lean_PersistentArray_push___redArg(v_traces_2621_, v___x_2632_);
if (v_isShared_2624_ == 0)
{
lean_ctor_set(v___x_2623_, 0, v___x_2633_);
v___x_2635_ = v___x_2623_;
goto v_reusejp_2634_;
}
else
{
lean_object* v_reuseFailAlloc_2644_; 
v_reuseFailAlloc_2644_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2644_, 0, v___x_2633_);
lean_ctor_set_uint64(v_reuseFailAlloc_2644_, sizeof(void*)*1, v_tid_2620_);
v___x_2635_ = v_reuseFailAlloc_2644_;
goto v_reusejp_2634_;
}
v_reusejp_2634_:
{
lean_object* v___x_2637_; 
if (v_isShared_2619_ == 0)
{
lean_ctor_set(v___x_2618_, 4, v___x_2635_);
v___x_2637_ = v___x_2618_;
goto v_reusejp_2636_;
}
else
{
lean_object* v_reuseFailAlloc_2643_; 
v_reuseFailAlloc_2643_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2643_, 0, v_env_2609_);
lean_ctor_set(v_reuseFailAlloc_2643_, 1, v_nextMacroScope_2610_);
lean_ctor_set(v_reuseFailAlloc_2643_, 2, v_ngen_2611_);
lean_ctor_set(v_reuseFailAlloc_2643_, 3, v_auxDeclNGen_2612_);
lean_ctor_set(v_reuseFailAlloc_2643_, 4, v___x_2635_);
lean_ctor_set(v_reuseFailAlloc_2643_, 5, v_cache_2613_);
lean_ctor_set(v_reuseFailAlloc_2643_, 6, v_messages_2614_);
lean_ctor_set(v_reuseFailAlloc_2643_, 7, v_infoState_2615_);
lean_ctor_set(v_reuseFailAlloc_2643_, 8, v_snapshotTasks_2616_);
v___x_2637_ = v_reuseFailAlloc_2643_;
goto v_reusejp_2636_;
}
v_reusejp_2636_:
{
lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2641_; 
v___x_2638_ = lean_st_ref_set(v___y_2599_, v___x_2637_);
v___x_2639_ = lean_box(0);
if (v_isShared_2606_ == 0)
{
lean_ctor_set(v___x_2605_, 0, v___x_2639_);
v___x_2641_ = v___x_2605_;
goto v_reusejp_2640_;
}
else
{
lean_object* v_reuseFailAlloc_2642_; 
v_reuseFailAlloc_2642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2642_, 0, v___x_2639_);
v___x_2641_ = v_reuseFailAlloc_2642_;
goto v_reusejp_2640_;
}
v_reusejp_2640_:
{
return v___x_2641_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9___boxed(lean_object* v_cls_2648_, lean_object* v_msg_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_){
_start:
{
lean_object* v_res_2655_; 
v_res_2655_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v_cls_2648_, v_msg_2649_, v___y_2650_, v___y_2651_, v___y_2652_, v___y_2653_);
lean_dec(v___y_2653_);
lean_dec_ref(v___y_2652_);
lean_dec(v___y_2651_);
lean_dec_ref(v___y_2650_);
return v_res_2655_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1(void){
_start:
{
lean_object* v___x_2657_; lean_object* v___x_2658_; 
v___x_2657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__0));
v___x_2658_ = l_Lean_stringToMessageData(v___x_2657_);
return v___x_2658_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7(void){
_start:
{
uint8_t v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; 
v___x_2668_ = 1;
v___x_2669_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__3));
v___x_2670_ = l_Lean_Name_toString(v___x_2669_, v___x_2668_);
return v___x_2670_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13(void){
_start:
{
lean_object* v___x_2676_; lean_object* v___x_2677_; 
v___x_2676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__12));
v___x_2677_ = l_Lean_stringToMessageData(v___x_2676_);
return v___x_2677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16(void){
_start:
{
lean_object* v___x_2681_; lean_object* v___x_2682_; 
v___x_2681_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__15));
v___x_2682_ = l_Lean_stringToMessageData(v___x_2681_);
return v___x_2682_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18(void){
_start:
{
lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; 
v___x_2685_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__17));
v___x_2686_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
v___x_2687_ = l_Lean_Name_append(v___x_2686_, v___x_2685_);
return v___x_2687_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21(void){
_start:
{
lean_object* v___x_2690_; lean_object* v___x_2691_; 
v___x_2690_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__20));
v___x_2691_ = l_Lean_stringToMessageData(v___x_2690_);
return v___x_2691_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23(void){
_start:
{
lean_object* v___x_2693_; lean_object* v___x_2694_; 
v___x_2693_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__22));
v___x_2694_ = l_Lean_stringToMessageData(v___x_2693_);
return v___x_2694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith(uint8_t v_transparency_2695_, lean_object* v_oracle_2696_, lean_object* v_discharger_2697_, lean_object* v_x_2698_, lean_object* v_x_2699_, lean_object* v_a_2700_, lean_object* v_a_2701_, lean_object* v_a_2702_, lean_object* v_a_2703_){
_start:
{
if (lean_obj_tag(v_x_2699_) == 0)
{
lean_object* v___x_2705_; lean_object* v___x_2706_; 
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v___x_2705_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__1);
v___x_2706_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v___x_2705_, v_a_2700_, v_a_2701_, v_a_2702_, v_a_2703_);
return v___x_2706_;
}
else
{
lean_object* v_head_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; uint8_t v___x_2711_; lean_object* v___y_2713_; lean_object* v___y_2714_; lean_object* v___y_2715_; lean_object* v___y_2716_; lean_object* v_fst_2717_; lean_object* v_fst_2718_; lean_object* v_snd_2719_; lean_object* v___y_2766_; lean_object* v___y_2767_; lean_object* v___y_2768_; lean_object* v___y_2769_; lean_object* v___y_2770_; lean_object* v___x_2784_; lean_object* v___x_2785_; 
v_head_2707_ = lean_ctor_get(v_x_2699_, 0);
lean_inc(v_head_2707_);
v___x_2708_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__0));
v___x_2709_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__1));
v___x_2710_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_addIneq___closed__2));
v___x_2711_ = 1;
v___x_2784_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__7);
v___x_2785_ = l_Lean_Core_checkSystem(v___x_2784_, v_a_2702_, v_a_2703_);
if (lean_obj_tag(v___x_2785_) == 0)
{
lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; 
lean_dec_ref_known(v___x_2785_, 1);
v___x_2786_ = lean_unsigned_to_nat(0u);
v___x_2787_ = l_List_zipIdxTR___redArg(v_x_2699_, v___x_2786_);
v___x_2788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__8));
v___x_2789_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_addNegEqProofsIdx___boxed), 6, 1);
lean_closure_set(v___x_2789_, 0, v___x_2787_);
v___x_2790_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_2788_, v___x_2789_, v_a_2700_, v_a_2701_, v_a_2702_, v_a_2703_);
if (lean_obj_tag(v___x_2790_) == 0)
{
lean_object* v_a_2791_; lean_object* v___x_2793_; uint8_t v_isShared_2794_; uint8_t v_isSharedCheck_3658_; 
v_a_2791_ = lean_ctor_get(v___x_2790_, 0);
v_isSharedCheck_3658_ = !lean_is_exclusive(v___x_2790_);
if (v_isSharedCheck_3658_ == 0)
{
v___x_2793_ = v___x_2790_;
v_isShared_2794_ = v_isSharedCheck_3658_;
goto v_resetjp_2792_;
}
else
{
lean_inc(v_a_2791_);
lean_dec(v___x_2790_);
v___x_2793_ = lean_box(0);
v_isShared_2794_ = v_isSharedCheck_3658_;
goto v_resetjp_2792_;
}
v_resetjp_2792_:
{
lean_object* v___f_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; 
v___f_2795_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2795_, 0, v_head_2707_);
lean_closure_set(v___f_2795_, 1, v_a_2791_);
v___x_2796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__9));
v___x_2797_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_2796_, v___f_2795_, v_a_2700_, v_a_2701_, v_a_2702_, v_a_2703_);
if (lean_obj_tag(v___x_2797_) == 0)
{
lean_object* v_options_2798_; lean_object* v_a_2799_; lean_object* v___x_2801_; uint8_t v_isShared_2802_; uint8_t v_isSharedCheck_3649_; 
v_options_2798_ = lean_ctor_get(v_a_2702_, 2);
v_a_2799_ = lean_ctor_get(v___x_2797_, 0);
v_isSharedCheck_3649_ = !lean_is_exclusive(v___x_2797_);
if (v_isSharedCheck_3649_ == 0)
{
v___x_2801_ = v___x_2797_;
v_isShared_2802_ = v_isSharedCheck_3649_;
goto v_resetjp_2800_;
}
else
{
lean_inc(v_a_2799_);
lean_dec(v___x_2797_);
v___x_2801_ = lean_box(0);
v_isShared_2802_ = v_isSharedCheck_3649_;
goto v_resetjp_2800_;
}
v_resetjp_2800_:
{
lean_object* v_inheritedTraceOptions_2803_; uint8_t v_hasTrace_2804_; lean_object* v___f_2805_; lean_object* v___f_2806_; uint8_t v___y_2808_; lean_object* v___y_2809_; lean_object* v___y_2810_; lean_object* v___y_2811_; lean_object* v___y_2812_; lean_object* v___y_2813_; lean_object* v___y_2814_; lean_object* v___y_2815_; lean_object* v___y_2816_; lean_object* v___y_2817_; lean_object* v_a_2818_; uint8_t v___y_2828_; lean_object* v___y_2829_; lean_object* v___y_2830_; lean_object* v___y_2831_; lean_object* v___y_2832_; lean_object* v___y_2833_; lean_object* v___y_2834_; lean_object* v___y_2835_; lean_object* v___y_2836_; lean_object* v___y_2837_; lean_object* v_a_2838_; uint8_t v___y_2841_; lean_object* v___y_2842_; lean_object* v___y_2843_; lean_object* v___y_2844_; lean_object* v___y_2845_; lean_object* v___y_2846_; lean_object* v___y_2847_; lean_object* v___y_2848_; lean_object* v___y_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; uint8_t v___y_2862_; lean_object* v___y_2863_; lean_object* v___y_2864_; lean_object* v___y_2865_; lean_object* v___y_2866_; lean_object* v___y_2867_; lean_object* v___y_2868_; lean_object* v___y_2869_; lean_object* v___y_2870_; lean_object* v___y_2871_; lean_object* v___y_2872_; uint8_t v___y_2876_; lean_object* v___y_2877_; lean_object* v___y_2878_; lean_object* v___y_2879_; lean_object* v___y_2880_; lean_object* v___y_2881_; lean_object* v___y_2882_; lean_object* v___y_2883_; lean_object* v___y_2884_; lean_object* v___y_2885_; lean_object* v_a_2886_; uint8_t v___y_2899_; lean_object* v___y_2900_; lean_object* v___y_2901_; lean_object* v___y_2902_; lean_object* v___y_2903_; lean_object* v___y_2904_; lean_object* v___y_2905_; lean_object* v___y_2906_; lean_object* v___y_2907_; lean_object* v___y_2908_; lean_object* v_a_2909_; uint8_t v___y_2912_; lean_object* v___y_2913_; lean_object* v___y_2914_; lean_object* v___y_2915_; lean_object* v___y_2916_; lean_object* v___y_2917_; lean_object* v___y_2918_; lean_object* v___y_2919_; lean_object* v___y_2920_; lean_object* v___y_2921_; lean_object* v___y_2922_; uint8_t v___y_2933_; lean_object* v___y_2934_; lean_object* v___y_2935_; lean_object* v___y_2936_; lean_object* v___y_2937_; lean_object* v___y_2938_; lean_object* v___y_2939_; lean_object* v___y_2940_; lean_object* v___y_2941_; lean_object* v___y_2942_; lean_object* v___y_2943_; lean_object* v___x_2946_; lean_object* v___y_2948_; lean_object* v___y_2949_; uint8_t v___y_2950_; lean_object* v___y_2951_; lean_object* v___y_2952_; uint8_t v___y_2953_; lean_object* v___y_2954_; lean_object* v___y_2955_; lean_object* v___y_2956_; lean_object* v___y_2957_; lean_object* v___y_2958_; lean_object* v___y_2959_; lean_object* v___y_2960_; lean_object* v___y_3004_; lean_object* v___y_3005_; lean_object* v___y_3006_; lean_object* v___y_3007_; lean_object* v___y_3008_; lean_object* v_options_3009_; uint8_t v_hasTrace_3010_; lean_object* v_inheritedTraceOptions_3011_; lean_object* v___y_3012_; lean_object* v_a_3013_; lean_object* v___y_3084_; lean_object* v___y_3085_; lean_object* v___y_3086_; lean_object* v___y_3087_; lean_object* v___y_3088_; lean_object* v___y_3089_; lean_object* v_a_3090_; lean_object* v___y_3095_; lean_object* v___y_3096_; lean_object* v___y_3097_; lean_object* v___y_3098_; lean_object* v___y_3099_; lean_object* v___y_3100_; lean_object* v___y_3101_; lean_object* v___y_3102_; lean_object* v___y_3103_; lean_object* v___y_3117_; lean_object* v___y_3118_; lean_object* v___y_3119_; lean_object* v___y_3120_; lean_object* v___y_3121_; lean_object* v___y_3122_; lean_object* v___y_3123_; lean_object* v___y_3134_; uint8_t v___y_3135_; lean_object* v___y_3136_; lean_object* v___y_3137_; lean_object* v___y_3138_; lean_object* v___y_3139_; lean_object* v___y_3140_; lean_object* v___y_3141_; lean_object* v___y_3142_; lean_object* v___y_3143_; lean_object* v_a_3144_; lean_object* v___y_3154_; uint8_t v___y_3155_; lean_object* v___y_3156_; lean_object* v___y_3157_; lean_object* v___y_3158_; lean_object* v___y_3159_; lean_object* v___y_3160_; lean_object* v___y_3161_; lean_object* v___y_3162_; lean_object* v___y_3163_; lean_object* v_a_3164_; lean_object* v___y_3167_; uint8_t v___y_3168_; lean_object* v___y_3169_; lean_object* v___y_3170_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; lean_object* v___y_3174_; lean_object* v___y_3175_; lean_object* v___y_3176_; lean_object* v_a_3177_; lean_object* v___y_3180_; lean_object* v___y_3181_; lean_object* v___y_3182_; lean_object* v___y_3183_; lean_object* v___y_3184_; lean_object* v___y_3185_; lean_object* v___y_3186_; uint8_t v___y_3187_; lean_object* v___y_3188_; lean_object* v___y_3189_; lean_object* v___y_3190_; lean_object* v___y_3191_; lean_object* v___y_3192_; lean_object* v___y_3199_; uint8_t v___y_3200_; lean_object* v___y_3201_; uint8_t v___y_3202_; lean_object* v___y_3203_; lean_object* v___y_3204_; lean_object* v___y_3205_; lean_object* v___y_3206_; lean_object* v___y_3207_; lean_object* v___y_3208_; lean_object* v___y_3209_; lean_object* v___y_3210_; lean_object* v_a_3211_; lean_object* v___y_3223_; uint8_t v___y_3224_; lean_object* v___y_3225_; uint8_t v___y_3226_; lean_object* v___y_3227_; lean_object* v___y_3228_; lean_object* v___y_3229_; lean_object* v___y_3230_; lean_object* v___y_3231_; lean_object* v___y_3232_; lean_object* v___y_3233_; lean_object* v___y_3234_; lean_object* v___y_3235_; lean_object* v___y_3238_; uint8_t v___y_3239_; lean_object* v___y_3240_; uint8_t v___y_3241_; lean_object* v___y_3242_; lean_object* v___y_3243_; lean_object* v___y_3244_; lean_object* v___y_3245_; lean_object* v___y_3246_; lean_object* v___y_3247_; lean_object* v___y_3248_; lean_object* v___y_3249_; lean_object* v___y_3253_; lean_object* v___y_3254_; uint8_t v___y_3255_; lean_object* v___y_3256_; lean_object* v___y_3257_; lean_object* v___y_3258_; lean_object* v___y_3259_; lean_object* v___y_3260_; uint8_t v___y_3261_; lean_object* v___y_3262_; lean_object* v___y_3263_; lean_object* v___y_3264_; lean_object* v___y_3265_; uint8_t v___y_3266_; lean_object* v___y_3276_; lean_object* v___y_3277_; uint8_t v___y_3278_; lean_object* v___y_3279_; lean_object* v___y_3280_; lean_object* v___y_3281_; lean_object* v___y_3282_; lean_object* v___y_3283_; lean_object* v___y_3284_; lean_object* v___y_3285_; lean_object* v_a_3286_; lean_object* v___y_3299_; lean_object* v___y_3300_; uint8_t v___y_3301_; lean_object* v___y_3302_; lean_object* v___y_3303_; lean_object* v___y_3304_; lean_object* v___y_3305_; lean_object* v___y_3306_; lean_object* v___y_3307_; lean_object* v___y_3308_; lean_object* v_a_3309_; lean_object* v___y_3312_; lean_object* v___y_3313_; uint8_t v___y_3314_; lean_object* v___y_3315_; lean_object* v___y_3316_; lean_object* v___y_3317_; lean_object* v___y_3318_; lean_object* v___y_3319_; lean_object* v___y_3320_; lean_object* v___y_3321_; lean_object* v_a_3322_; lean_object* v___y_3325_; lean_object* v___y_3326_; lean_object* v___y_3327_; lean_object* v___y_3328_; lean_object* v___y_3329_; lean_object* v___y_3330_; lean_object* v___y_3331_; uint8_t v___y_3332_; lean_object* v___y_3333_; lean_object* v___y_3334_; lean_object* v___y_3335_; lean_object* v___y_3336_; lean_object* v___y_3337_; lean_object* v___y_3344_; uint8_t v___y_3345_; lean_object* v___y_3346_; lean_object* v___y_3347_; uint8_t v___y_3348_; lean_object* v___y_3349_; lean_object* v___y_3350_; lean_object* v___y_3351_; lean_object* v___y_3352_; lean_object* v___y_3353_; lean_object* v___y_3354_; lean_object* v___y_3355_; lean_object* v_a_3356_; lean_object* v___y_3368_; uint8_t v___y_3369_; lean_object* v___y_3370_; lean_object* v___y_3371_; uint8_t v___y_3372_; lean_object* v___y_3373_; lean_object* v___y_3374_; lean_object* v___y_3375_; lean_object* v___y_3376_; lean_object* v___y_3377_; lean_object* v___y_3378_; lean_object* v___y_3379_; lean_object* v___y_3380_; lean_object* v___y_3383_; uint8_t v___y_3384_; lean_object* v___y_3385_; lean_object* v___y_3386_; uint8_t v___y_3387_; lean_object* v___y_3388_; lean_object* v___y_3389_; lean_object* v___y_3390_; lean_object* v___y_3391_; lean_object* v___y_3392_; lean_object* v___y_3393_; lean_object* v___y_3394_; lean_object* v___y_3398_; uint8_t v___y_3399_; lean_object* v___y_3400_; lean_object* v___y_3401_; lean_object* v___y_3402_; lean_object* v___y_3403_; lean_object* v___y_3404_; uint8_t v___y_3405_; lean_object* v___y_3406_; lean_object* v___y_3407_; lean_object* v___y_3408_; lean_object* v___y_3409_; lean_object* v___y_3410_; uint8_t v___y_3411_; lean_object* v___y_3421_; uint8_t v___y_3422_; lean_object* v___y_3423_; lean_object* v___y_3424_; lean_object* v___y_3425_; lean_object* v___y_3426_; lean_object* v___y_3427_; lean_object* v___y_3428_; uint8_t v___y_3429_; lean_object* v___y_3430_; lean_object* v___y_3431_; lean_object* v___y_3432_; lean_object* v___y_3450_; lean_object* v___y_3451_; lean_object* v___y_3452_; lean_object* v___y_3453_; lean_object* v___y_3454_; lean_object* v_options_3455_; uint8_t v_hasTrace_3456_; lean_object* v_inheritedTraceOptions_3457_; lean_object* v___y_3458_; lean_object* v_a_3459_; lean_object* v___y_3471_; lean_object* v___y_3472_; lean_object* v___y_3473_; lean_object* v___y_3474_; lean_object* v___y_3475_; lean_object* v___y_3476_; lean_object* v___y_3477_; lean_object* v___y_3487_; lean_object* v___y_3488_; lean_object* v___y_3489_; lean_object* v___y_3490_; lean_object* v___y_3491_; lean_object* v___y_3492_; lean_object* v___y_3496_; uint8_t v___y_3497_; lean_object* v___y_3498_; lean_object* v___y_3499_; lean_object* v___y_3500_; lean_object* v___y_3501_; lean_object* v___y_3502_; lean_object* v___y_3503_; lean_object* v___y_3504_; lean_object* v___y_3505_; uint8_t v___y_3506_; lean_object* v___y_3526_; lean_object* v___y_3527_; lean_object* v___y_3528_; lean_object* v___y_3529_; lean_object* v___y_3530_; lean_object* v___y_3531_; lean_object* v___y_3532_; uint8_t v___y_3533_; lean_object* v___x_3547_; lean_object* v___y_3549_; lean_object* v___y_3550_; lean_object* v___y_3551_; lean_object* v___y_3552_; lean_object* v___y_3553_; lean_object* v_options_3554_; uint8_t v_hasTrace_3555_; lean_object* v_inheritedTraceOptions_3556_; lean_object* v___y_3557_; lean_object* v___x_3574_; lean_object* v___y_3576_; lean_object* v___y_3577_; lean_object* v___y_3578_; lean_object* v___y_3579_; 
v_inheritedTraceOptions_2803_ = lean_ctor_get(v_a_2702_, 13);
v_hasTrace_2804_ = lean_ctor_get_uint8(v_options_2798_, sizeof(void*)*1);
v___f_2805_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__10));
v___f_2806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__11));
v___x_2946_ = lean_box(0);
lean_inc(v_a_2799_);
v___x_3547_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__1(v_a_2799_, v___x_2946_);
v___x_3574_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__2));
if (v_hasTrace_2804_ == 0)
{
v___y_3576_ = v_a_2700_;
v___y_3577_ = v_a_2701_;
v___y_3578_ = v_a_2702_;
v___y_3579_ = v_a_2703_;
goto v___jp_3575_;
}
else
{
lean_object* v___x_3623_; uint8_t v___x_3624_; 
v___x_3623_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6);
v___x_3624_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2803_, v_options_2798_, v___x_3623_);
if (v___x_3624_ == 0)
{
v___y_3576_ = v_a_2700_;
v___y_3577_ = v_a_2701_;
v___y_3578_ = v_a_2702_;
v___y_3579_ = v_a_2703_;
goto v___jp_3575_;
}
else
{
lean_object* v___x_3625_; 
lean_inc(v___x_3547_);
v___x_3625_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__16(v___x_3547_, v___x_2946_, v_a_2700_, v_a_2701_, v_a_2702_, v_a_2703_);
if (lean_obj_tag(v___x_3625_) == 0)
{
lean_object* v_a_3626_; lean_object* v___x_3627_; lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3631_; lean_object* v___x_3632_; 
v_a_3626_ = lean_ctor_get(v___x_3625_, 0);
lean_inc(v_a_3626_);
lean_dec_ref_known(v___x_3625_, 1);
v___x_3627_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__23);
v___x_3628_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__17(v_a_3626_, v___x_2946_);
v___x_3629_ = l_Lean_MessageData_ofList(v___x_3628_);
v___x_3630_ = l_Lean_indentD(v___x_3629_);
v___x_3631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3631_, 0, v___x_3627_);
lean_ctor_set(v___x_3631_, 1, v___x_3630_);
v___x_3632_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___x_3574_, v___x_3631_, v_a_2700_, v_a_2701_, v_a_2702_, v_a_2703_);
if (lean_obj_tag(v___x_3632_) == 0)
{
lean_dec_ref_known(v___x_3632_, 1);
v___y_3576_ = v_a_2700_;
v___y_3577_ = v_a_2701_;
v___y_3578_ = v_a_2702_;
v___y_3579_ = v_a_2703_;
goto v___jp_3575_;
}
else
{
lean_object* v_a_3633_; lean_object* v___x_3635_; uint8_t v_isShared_3636_; uint8_t v_isSharedCheck_3640_; 
lean_dec(v___x_3547_);
lean_del_object(v___x_2801_);
lean_dec(v_a_2799_);
lean_del_object(v___x_2793_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3633_ = lean_ctor_get(v___x_3632_, 0);
v_isSharedCheck_3640_ = !lean_is_exclusive(v___x_3632_);
if (v_isSharedCheck_3640_ == 0)
{
v___x_3635_ = v___x_3632_;
v_isShared_3636_ = v_isSharedCheck_3640_;
goto v_resetjp_3634_;
}
else
{
lean_inc(v_a_3633_);
lean_dec(v___x_3632_);
v___x_3635_ = lean_box(0);
v_isShared_3636_ = v_isSharedCheck_3640_;
goto v_resetjp_3634_;
}
v_resetjp_3634_:
{
lean_object* v___x_3638_; 
if (v_isShared_3636_ == 0)
{
v___x_3638_ = v___x_3635_;
goto v_reusejp_3637_;
}
else
{
lean_object* v_reuseFailAlloc_3639_; 
v_reuseFailAlloc_3639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3639_, 0, v_a_3633_);
v___x_3638_ = v_reuseFailAlloc_3639_;
goto v_reusejp_3637_;
}
v_reusejp_3637_:
{
return v___x_3638_;
}
}
}
}
else
{
lean_object* v_a_3641_; lean_object* v___x_3643_; uint8_t v_isShared_3644_; uint8_t v_isSharedCheck_3648_; 
lean_dec(v___x_3547_);
lean_del_object(v___x_2801_);
lean_dec(v_a_2799_);
lean_del_object(v___x_2793_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3641_ = lean_ctor_get(v___x_3625_, 0);
v_isSharedCheck_3648_ = !lean_is_exclusive(v___x_3625_);
if (v_isSharedCheck_3648_ == 0)
{
v___x_3643_ = v___x_3625_;
v_isShared_3644_ = v_isSharedCheck_3648_;
goto v_resetjp_3642_;
}
else
{
lean_inc(v_a_3641_);
lean_dec(v___x_3625_);
v___x_3643_ = lean_box(0);
v_isShared_3644_ = v_isSharedCheck_3648_;
goto v_resetjp_3642_;
}
v_resetjp_3642_:
{
lean_object* v___x_3646_; 
if (v_isShared_3644_ == 0)
{
v___x_3646_ = v___x_3643_;
goto v_reusejp_3645_;
}
else
{
lean_object* v_reuseFailAlloc_3647_; 
v_reuseFailAlloc_3647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3647_, 0, v_a_3641_);
v___x_3646_ = v_reuseFailAlloc_3647_;
goto v_reusejp_3645_;
}
v_reusejp_3645_:
{
return v___x_3646_;
}
}
}
}
}
v___jp_2807_:
{
lean_object* v___x_2819_; double v___x_2820_; double v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; 
v___x_2819_ = lean_io_get_num_heartbeats();
v___x_2820_ = lean_float_of_nat(v___y_2810_);
v___x_2821_ = lean_float_of_nat(v___x_2819_);
v___x_2822_ = lean_box_float(v___x_2820_);
v___x_2823_ = lean_box_float(v___x_2821_);
v___x_2824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2824_, 0, v___x_2822_);
lean_ctor_set(v___x_2824_, 1, v___x_2823_);
v___x_2825_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2825_, 0, v_a_2818_);
lean_ctor_set(v___x_2825_, 1, v___x_2824_);
lean_inc_ref(v___y_2814_);
lean_inc(v___y_2815_);
v___x_2826_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5(v___y_2815_, v___x_2711_, v___y_2814_, v___y_2809_, v___y_2808_, v___y_2812_, v___f_2806_, v___x_2825_, v___y_2813_, v___y_2811_, v___y_2816_, v___y_2817_);
v___y_2766_ = v___y_2811_;
v___y_2767_ = v___y_2813_;
v___y_2768_ = v___y_2816_;
v___y_2769_ = v___y_2817_;
v___y_2770_ = v___x_2826_;
goto v___jp_2765_;
}
v___jp_2827_:
{
lean_object* v___x_2839_; 
v___x_2839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2839_, 0, v_a_2838_);
v___y_2808_ = v___y_2828_;
v___y_2809_ = v___y_2830_;
v___y_2810_ = v___y_2829_;
v___y_2811_ = v___y_2831_;
v___y_2812_ = v___y_2832_;
v___y_2813_ = v___y_2833_;
v___y_2814_ = v___y_2834_;
v___y_2815_ = v___y_2835_;
v___y_2816_ = v___y_2836_;
v___y_2817_ = v___y_2837_;
v_a_2818_ = v___x_2839_;
goto v___jp_2807_;
}
v___jp_2840_:
{
if (lean_obj_tag(v___y_2851_) == 0)
{
lean_object* v_a_2852_; lean_object* v___x_2854_; uint8_t v_isShared_2855_; uint8_t v_isSharedCheck_2859_; 
v_a_2852_ = lean_ctor_get(v___y_2851_, 0);
v_isSharedCheck_2859_ = !lean_is_exclusive(v___y_2851_);
if (v_isSharedCheck_2859_ == 0)
{
v___x_2854_ = v___y_2851_;
v_isShared_2855_ = v_isSharedCheck_2859_;
goto v_resetjp_2853_;
}
else
{
lean_inc(v_a_2852_);
lean_dec(v___y_2851_);
v___x_2854_ = lean_box(0);
v_isShared_2855_ = v_isSharedCheck_2859_;
goto v_resetjp_2853_;
}
v_resetjp_2853_:
{
lean_object* v___x_2857_; 
if (v_isShared_2855_ == 0)
{
lean_ctor_set_tag(v___x_2854_, 1);
v___x_2857_ = v___x_2854_;
goto v_reusejp_2856_;
}
else
{
lean_object* v_reuseFailAlloc_2858_; 
v_reuseFailAlloc_2858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2858_, 0, v_a_2852_);
v___x_2857_ = v_reuseFailAlloc_2858_;
goto v_reusejp_2856_;
}
v_reusejp_2856_:
{
v___y_2808_ = v___y_2841_;
v___y_2809_ = v___y_2843_;
v___y_2810_ = v___y_2842_;
v___y_2811_ = v___y_2844_;
v___y_2812_ = v___y_2845_;
v___y_2813_ = v___y_2846_;
v___y_2814_ = v___y_2847_;
v___y_2815_ = v___y_2848_;
v___y_2816_ = v___y_2849_;
v___y_2817_ = v___y_2850_;
v_a_2818_ = v___x_2857_;
goto v___jp_2807_;
}
}
}
else
{
lean_object* v_a_2860_; 
v_a_2860_ = lean_ctor_get(v___y_2851_, 0);
lean_inc(v_a_2860_);
lean_dec_ref_known(v___y_2851_, 1);
v___y_2828_ = v___y_2841_;
v___y_2829_ = v___y_2842_;
v___y_2830_ = v___y_2843_;
v___y_2831_ = v___y_2844_;
v___y_2832_ = v___y_2845_;
v___y_2833_ = v___y_2846_;
v___y_2834_ = v___y_2847_;
v___y_2835_ = v___y_2848_;
v___y_2836_ = v___y_2849_;
v___y_2837_ = v___y_2850_;
v_a_2838_ = v_a_2860_;
goto v___jp_2827_;
}
}
v___jp_2861_:
{
lean_object* v___x_2873_; lean_object* v___x_2874_; 
v___x_2873_ = lean_box(0);
lean_inc(v___y_2872_);
lean_inc_ref(v___y_2871_);
lean_inc(v___y_2866_);
lean_inc_ref(v___y_2868_);
v___x_2874_ = lean_apply_6(v___y_2863_, v___x_2873_, v___y_2868_, v___y_2866_, v___y_2871_, v___y_2872_, lean_box(0));
v___y_2841_ = v___y_2862_;
v___y_2842_ = v___y_2865_;
v___y_2843_ = v___y_2864_;
v___y_2844_ = v___y_2866_;
v___y_2845_ = v___y_2867_;
v___y_2846_ = v___y_2868_;
v___y_2847_ = v___y_2869_;
v___y_2848_ = v___y_2870_;
v___y_2849_ = v___y_2871_;
v___y_2850_ = v___y_2872_;
v___y_2851_ = v___x_2874_;
goto v___jp_2840_;
}
v___jp_2875_:
{
lean_object* v___x_2887_; double v___x_2888_; double v___x_2889_; double v___x_2890_; double v___x_2891_; double v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; 
v___x_2887_ = lean_io_mono_nanos_now();
v___x_2888_ = lean_float_of_nat(v___y_2877_);
v___x_2889_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7);
v___x_2890_ = lean_float_div(v___x_2888_, v___x_2889_);
v___x_2891_ = lean_float_of_nat(v___x_2887_);
v___x_2892_ = lean_float_div(v___x_2891_, v___x_2889_);
v___x_2893_ = lean_box_float(v___x_2890_);
v___x_2894_ = lean_box_float(v___x_2892_);
v___x_2895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2895_, 0, v___x_2893_);
lean_ctor_set(v___x_2895_, 1, v___x_2894_);
v___x_2896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2896_, 0, v_a_2886_);
lean_ctor_set(v___x_2896_, 1, v___x_2895_);
lean_inc_ref(v___y_2882_);
lean_inc(v___y_2883_);
v___x_2897_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__5(v___y_2883_, v___x_2711_, v___y_2882_, v___y_2878_, v___y_2876_, v___y_2880_, v___f_2806_, v___x_2896_, v___y_2881_, v___y_2879_, v___y_2884_, v___y_2885_);
v___y_2766_ = v___y_2879_;
v___y_2767_ = v___y_2881_;
v___y_2768_ = v___y_2884_;
v___y_2769_ = v___y_2885_;
v___y_2770_ = v___x_2897_;
goto v___jp_2765_;
}
v___jp_2898_:
{
lean_object* v___x_2910_; 
v___x_2910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2910_, 0, v_a_2909_);
v___y_2876_ = v___y_2899_;
v___y_2877_ = v___y_2900_;
v___y_2878_ = v___y_2901_;
v___y_2879_ = v___y_2902_;
v___y_2880_ = v___y_2903_;
v___y_2881_ = v___y_2904_;
v___y_2882_ = v___y_2905_;
v___y_2883_ = v___y_2906_;
v___y_2884_ = v___y_2907_;
v___y_2885_ = v___y_2908_;
v_a_2886_ = v___x_2910_;
goto v___jp_2875_;
}
v___jp_2911_:
{
if (lean_obj_tag(v___y_2922_) == 0)
{
lean_object* v_a_2923_; lean_object* v___x_2925_; uint8_t v_isShared_2926_; uint8_t v_isSharedCheck_2930_; 
v_a_2923_ = lean_ctor_get(v___y_2922_, 0);
v_isSharedCheck_2930_ = !lean_is_exclusive(v___y_2922_);
if (v_isSharedCheck_2930_ == 0)
{
v___x_2925_ = v___y_2922_;
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
else
{
lean_inc(v_a_2923_);
lean_dec(v___y_2922_);
v___x_2925_ = lean_box(0);
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
v_resetjp_2924_:
{
lean_object* v___x_2928_; 
if (v_isShared_2926_ == 0)
{
lean_ctor_set_tag(v___x_2925_, 1);
v___x_2928_ = v___x_2925_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v_a_2923_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
v___y_2876_ = v___y_2912_;
v___y_2877_ = v___y_2913_;
v___y_2878_ = v___y_2914_;
v___y_2879_ = v___y_2915_;
v___y_2880_ = v___y_2916_;
v___y_2881_ = v___y_2917_;
v___y_2882_ = v___y_2918_;
v___y_2883_ = v___y_2919_;
v___y_2884_ = v___y_2920_;
v___y_2885_ = v___y_2921_;
v_a_2886_ = v___x_2928_;
goto v___jp_2875_;
}
}
}
else
{
lean_object* v_a_2931_; 
v_a_2931_ = lean_ctor_get(v___y_2922_, 0);
lean_inc(v_a_2931_);
lean_dec_ref_known(v___y_2922_, 1);
v___y_2899_ = v___y_2912_;
v___y_2900_ = v___y_2913_;
v___y_2901_ = v___y_2914_;
v___y_2902_ = v___y_2915_;
v___y_2903_ = v___y_2916_;
v___y_2904_ = v___y_2917_;
v___y_2905_ = v___y_2918_;
v___y_2906_ = v___y_2919_;
v___y_2907_ = v___y_2920_;
v___y_2908_ = v___y_2921_;
v_a_2909_ = v_a_2931_;
goto v___jp_2898_;
}
}
v___jp_2932_:
{
lean_object* v___x_2944_; lean_object* v___x_2945_; 
v___x_2944_ = lean_box(0);
lean_inc(v___y_2943_);
lean_inc_ref(v___y_2942_);
lean_inc(v___y_2936_);
lean_inc_ref(v___y_2939_);
v___x_2945_ = lean_apply_6(v___y_2937_, v___x_2944_, v___y_2939_, v___y_2936_, v___y_2942_, v___y_2943_, lean_box(0));
v___y_2912_ = v___y_2933_;
v___y_2913_ = v___y_2934_;
v___y_2914_ = v___y_2935_;
v___y_2915_ = v___y_2936_;
v___y_2916_ = v___y_2938_;
v___y_2917_ = v___y_2939_;
v___y_2918_ = v___y_2940_;
v___y_2919_ = v___y_2941_;
v___y_2920_ = v___y_2942_;
v___y_2921_ = v___y_2943_;
v___y_2922_ = v___x_2945_;
goto v___jp_2911_;
}
v___jp_2947_:
{
lean_object* v___x_2961_; lean_object* v_a_2962_; lean_object* v___x_2963_; uint8_t v___x_2964_; 
v___x_2961_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(v___y_2958_);
v_a_2962_ = lean_ctor_get(v___x_2961_, 0);
lean_inc(v_a_2962_);
lean_dec_ref(v___x_2961_);
v___x_2963_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2964_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v___y_2951_, v___x_2963_);
if (v___x_2964_ == 0)
{
lean_object* v___x_2965_; lean_object* v___x_2966_; 
v___x_2965_ = lean_io_mono_nanos_now();
v___x_2966_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(v___y_2959_, v___x_2946_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2966_) == 0)
{
lean_object* v_a_2967_; lean_object* v___x_2968_; 
v_a_2967_ = lean_ctor_get(v___x_2966_, 0);
lean_inc(v_a_2967_);
lean_dec_ref_known(v___x_2966_, 1);
v___x_2968_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs(v_a_2967_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2968_) == 0)
{
lean_object* v_a_2969_; lean_object* v___f_2970_; 
v_a_2969_ = lean_ctor_get(v___x_2968_, 0);
lean_inc_n(v_a_2969_, 2);
lean_dec_ref_known(v___x_2968_, 1);
lean_inc(v___y_2948_);
lean_inc(v___y_2949_);
v___f_2970_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4___boxed), 9, 3);
lean_closure_set(v___f_2970_, 0, v___y_2949_);
lean_closure_set(v___f_2970_, 1, v___y_2948_);
lean_closure_set(v___f_2970_, 2, v_a_2969_);
if (v___y_2953_ == 0)
{
lean_dec(v_a_2969_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v___y_2933_ = v___y_2950_;
v___y_2934_ = v___x_2965_;
v___y_2935_ = v___y_2951_;
v___y_2936_ = v___y_2952_;
v___y_2937_ = v___f_2970_;
v___y_2938_ = v_a_2962_;
v___y_2939_ = v___y_2954_;
v___y_2940_ = v___y_2955_;
v___y_2941_ = v___y_2956_;
v___y_2942_ = v___y_2957_;
v___y_2943_ = v___y_2958_;
goto v___jp_2932_;
}
else
{
lean_object* v___x_2971_; lean_object* v___x_2972_; uint8_t v___x_2973_; 
v___x_2971_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_2956_);
v___x_2972_ = l_Lean_Name_append(v___x_2971_, v___y_2956_);
v___x_2973_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_2960_, v___y_2951_, v___x_2972_);
lean_dec(v___x_2972_);
if (v___x_2973_ == 0)
{
lean_dec(v_a_2969_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v___y_2933_ = v___y_2950_;
v___y_2934_ = v___x_2965_;
v___y_2935_ = v___y_2951_;
v___y_2936_ = v___y_2952_;
v___y_2937_ = v___f_2970_;
v___y_2938_ = v_a_2962_;
v___y_2939_ = v___y_2954_;
v___y_2940_ = v___y_2955_;
v___y_2941_ = v___y_2956_;
v___y_2942_ = v___y_2957_;
v___y_2943_ = v___y_2958_;
goto v___jp_2932_;
}
else
{
lean_object* v___x_2974_; lean_object* v___x_2975_; lean_object* v___x_2976_; lean_object* v___x_2977_; lean_object* v___x_2978_; 
lean_dec_ref(v___f_2970_);
lean_inc(v_a_2969_);
v___x_2974_ = l_Lean_MessageData_ofExpr(v_a_2969_);
v___x_2975_ = l_Lean_indentD(v___x_2974_);
v___x_2976_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13);
v___x_2977_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2977_, 0, v___x_2975_);
lean_ctor_set(v___x_2977_, 1, v___x_2976_);
lean_inc(v___y_2956_);
v___x_2978_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_2956_, v___x_2977_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2978_) == 0)
{
lean_object* v_a_2979_; lean_object* v___x_2980_; 
v_a_2979_ = lean_ctor_get(v___x_2978_, 0);
lean_inc(v_a_2979_);
lean_dec_ref_known(v___x_2978_, 1);
v___x_2980_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(v___y_2949_, v___y_2948_, v_a_2969_, v_a_2979_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
v___y_2912_ = v___y_2950_;
v___y_2913_ = v___x_2965_;
v___y_2914_ = v___y_2951_;
v___y_2915_ = v___y_2952_;
v___y_2916_ = v_a_2962_;
v___y_2917_ = v___y_2954_;
v___y_2918_ = v___y_2955_;
v___y_2919_ = v___y_2956_;
v___y_2920_ = v___y_2957_;
v___y_2921_ = v___y_2958_;
v___y_2922_ = v___x_2980_;
goto v___jp_2911_;
}
else
{
lean_object* v_a_2981_; 
lean_dec(v_a_2969_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_2981_ = lean_ctor_get(v___x_2978_, 0);
lean_inc(v_a_2981_);
lean_dec_ref_known(v___x_2978_, 1);
v___y_2899_ = v___y_2950_;
v___y_2900_ = v___x_2965_;
v___y_2901_ = v___y_2951_;
v___y_2902_ = v___y_2952_;
v___y_2903_ = v_a_2962_;
v___y_2904_ = v___y_2954_;
v___y_2905_ = v___y_2955_;
v___y_2906_ = v___y_2956_;
v___y_2907_ = v___y_2957_;
v___y_2908_ = v___y_2958_;
v_a_2909_ = v_a_2981_;
goto v___jp_2898_;
}
}
}
}
else
{
lean_object* v_a_2982_; 
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_2982_ = lean_ctor_get(v___x_2968_, 0);
lean_inc(v_a_2982_);
lean_dec_ref_known(v___x_2968_, 1);
v___y_2899_ = v___y_2950_;
v___y_2900_ = v___x_2965_;
v___y_2901_ = v___y_2951_;
v___y_2902_ = v___y_2952_;
v___y_2903_ = v_a_2962_;
v___y_2904_ = v___y_2954_;
v___y_2905_ = v___y_2955_;
v___y_2906_ = v___y_2956_;
v___y_2907_ = v___y_2957_;
v___y_2908_ = v___y_2958_;
v_a_2909_ = v_a_2982_;
goto v___jp_2898_;
}
}
else
{
lean_object* v_a_2983_; 
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_2983_ = lean_ctor_get(v___x_2966_, 0);
lean_inc(v_a_2983_);
lean_dec_ref_known(v___x_2966_, 1);
v___y_2899_ = v___y_2950_;
v___y_2900_ = v___x_2965_;
v___y_2901_ = v___y_2951_;
v___y_2902_ = v___y_2952_;
v___y_2903_ = v_a_2962_;
v___y_2904_ = v___y_2954_;
v___y_2905_ = v___y_2955_;
v___y_2906_ = v___y_2956_;
v___y_2907_ = v___y_2957_;
v___y_2908_ = v___y_2958_;
v_a_2909_ = v_a_2983_;
goto v___jp_2898_;
}
}
else
{
lean_object* v___x_2984_; lean_object* v___x_2985_; 
v___x_2984_ = lean_io_get_num_heartbeats();
v___x_2985_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(v___y_2959_, v___x_2946_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2985_) == 0)
{
lean_object* v_a_2986_; lean_object* v___x_2987_; 
v_a_2986_ = lean_ctor_get(v___x_2985_, 0);
lean_inc(v_a_2986_);
lean_dec_ref_known(v___x_2985_, 1);
v___x_2987_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs(v_a_2986_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2987_) == 0)
{
lean_object* v_a_2988_; lean_object* v___f_2989_; 
v_a_2988_ = lean_ctor_get(v___x_2987_, 0);
lean_inc_n(v_a_2988_, 2);
lean_dec_ref_known(v___x_2987_, 1);
lean_inc(v___y_2948_);
lean_inc(v___y_2949_);
v___f_2989_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4___boxed), 9, 3);
lean_closure_set(v___f_2989_, 0, v___y_2949_);
lean_closure_set(v___f_2989_, 1, v___y_2948_);
lean_closure_set(v___f_2989_, 2, v_a_2988_);
if (v___y_2953_ == 0)
{
lean_dec(v_a_2988_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v___y_2862_ = v___y_2950_;
v___y_2863_ = v___f_2989_;
v___y_2864_ = v___y_2951_;
v___y_2865_ = v___x_2984_;
v___y_2866_ = v___y_2952_;
v___y_2867_ = v_a_2962_;
v___y_2868_ = v___y_2954_;
v___y_2869_ = v___y_2955_;
v___y_2870_ = v___y_2956_;
v___y_2871_ = v___y_2957_;
v___y_2872_ = v___y_2958_;
goto v___jp_2861_;
}
else
{
lean_object* v___x_2990_; lean_object* v___x_2991_; uint8_t v___x_2992_; 
v___x_2990_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_2956_);
v___x_2991_ = l_Lean_Name_append(v___x_2990_, v___y_2956_);
v___x_2992_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_2960_, v___y_2951_, v___x_2991_);
lean_dec(v___x_2991_);
if (v___x_2992_ == 0)
{
lean_dec(v_a_2988_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v___y_2862_ = v___y_2950_;
v___y_2863_ = v___f_2989_;
v___y_2864_ = v___y_2951_;
v___y_2865_ = v___x_2984_;
v___y_2866_ = v___y_2952_;
v___y_2867_ = v_a_2962_;
v___y_2868_ = v___y_2954_;
v___y_2869_ = v___y_2955_;
v___y_2870_ = v___y_2956_;
v___y_2871_ = v___y_2957_;
v___y_2872_ = v___y_2958_;
goto v___jp_2861_;
}
else
{
lean_object* v___x_2993_; lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; 
lean_dec_ref(v___f_2989_);
lean_inc(v_a_2988_);
v___x_2993_ = l_Lean_MessageData_ofExpr(v_a_2988_);
v___x_2994_ = l_Lean_indentD(v___x_2993_);
v___x_2995_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13);
v___x_2996_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2996_, 0, v___x_2994_);
lean_ctor_set(v___x_2996_, 1, v___x_2995_);
lean_inc(v___y_2956_);
v___x_2997_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_2956_, v___x_2996_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
if (lean_obj_tag(v___x_2997_) == 0)
{
lean_object* v_a_2998_; lean_object* v___x_2999_; 
v_a_2998_ = lean_ctor_get(v___x_2997_, 0);
lean_inc(v_a_2998_);
lean_dec_ref_known(v___x_2997_, 1);
v___x_2999_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(v___y_2949_, v___y_2948_, v_a_2988_, v_a_2998_, v___y_2954_, v___y_2952_, v___y_2957_, v___y_2958_);
v___y_2841_ = v___y_2950_;
v___y_2842_ = v___x_2984_;
v___y_2843_ = v___y_2951_;
v___y_2844_ = v___y_2952_;
v___y_2845_ = v_a_2962_;
v___y_2846_ = v___y_2954_;
v___y_2847_ = v___y_2955_;
v___y_2848_ = v___y_2956_;
v___y_2849_ = v___y_2957_;
v___y_2850_ = v___y_2958_;
v___y_2851_ = v___x_2999_;
goto v___jp_2840_;
}
else
{
lean_object* v_a_3000_; 
lean_dec(v_a_2988_);
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_3000_ = lean_ctor_get(v___x_2997_, 0);
lean_inc(v_a_3000_);
lean_dec_ref_known(v___x_2997_, 1);
v___y_2828_ = v___y_2950_;
v___y_2829_ = v___x_2984_;
v___y_2830_ = v___y_2951_;
v___y_2831_ = v___y_2952_;
v___y_2832_ = v_a_2962_;
v___y_2833_ = v___y_2954_;
v___y_2834_ = v___y_2955_;
v___y_2835_ = v___y_2956_;
v___y_2836_ = v___y_2957_;
v___y_2837_ = v___y_2958_;
v_a_2838_ = v_a_3000_;
goto v___jp_2827_;
}
}
}
}
else
{
lean_object* v_a_3001_; 
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_3001_ = lean_ctor_get(v___x_2987_, 0);
lean_inc(v_a_3001_);
lean_dec_ref_known(v___x_2987_, 1);
v___y_2828_ = v___y_2950_;
v___y_2829_ = v___x_2984_;
v___y_2830_ = v___y_2951_;
v___y_2831_ = v___y_2952_;
v___y_2832_ = v_a_2962_;
v___y_2833_ = v___y_2954_;
v___y_2834_ = v___y_2955_;
v___y_2835_ = v___y_2956_;
v___y_2836_ = v___y_2957_;
v___y_2837_ = v___y_2958_;
v_a_2838_ = v_a_3001_;
goto v___jp_2827_;
}
}
else
{
lean_object* v_a_3002_; 
lean_dec(v___y_2949_);
lean_dec(v___y_2948_);
v_a_3002_ = lean_ctor_get(v___x_2985_, 0);
lean_inc(v_a_3002_);
lean_dec_ref_known(v___x_2985_, 1);
v___y_2828_ = v___y_2950_;
v___y_2829_ = v___x_2984_;
v___y_2830_ = v___y_2951_;
v___y_2831_ = v___y_2952_;
v___y_2832_ = v_a_2962_;
v___y_2833_ = v___y_2954_;
v___y_2834_ = v___y_2955_;
v___y_2835_ = v___y_2956_;
v___y_2836_ = v___y_2957_;
v___y_2837_ = v___y_2958_;
v_a_2838_ = v_a_3002_;
goto v___jp_2827_;
}
}
}
v___jp_3003_:
{
lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3017_; 
v___x_3014_ = l_List_zipIdxTR___redArg(v_a_2799_, v___x_2786_);
v___x_3015_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__14));
v___x_3016_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__3(v_a_3013_, v___x_3014_, v___x_3015_);
lean_dec_ref(v_a_3013_);
lean_inc(v___x_3016_);
v___x_3017_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__4(v___x_3016_, v___x_2946_);
if (v_hasTrace_3010_ == 0)
{
lean_object* v___x_3018_; 
lean_inc(v___x_3016_);
v___x_3018_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(v___x_3016_, v___x_2946_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
if (lean_obj_tag(v___x_3018_) == 0)
{
lean_object* v_a_3019_; lean_object* v___x_3020_; 
v_a_3019_ = lean_ctor_get(v___x_3018_, 0);
lean_inc(v_a_3019_);
lean_dec_ref_known(v___x_3018_, 1);
v___x_3020_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs(v_a_3019_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
if (lean_obj_tag(v___x_3020_) == 0)
{
lean_object* v_a_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; 
v_a_3021_ = lean_ctor_get(v___x_3020_, 0);
lean_inc(v_a_3021_);
lean_dec_ref_known(v___x_3020_, 1);
v___x_3022_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__7(v___x_2946_, v___x_3016_);
v___x_3023_ = lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__8(v___x_3022_);
v___y_2713_ = v___y_3004_;
v___y_2714_ = v___y_3005_;
v___y_2715_ = v___y_3008_;
v___y_2716_ = v___y_3012_;
v_fst_2717_ = v_a_3021_;
v_fst_2718_ = v___x_3017_;
v_snd_2719_ = v___x_3023_;
goto v___jp_2712_;
}
else
{
lean_object* v_a_3024_; lean_object* v___x_3026_; uint8_t v_isShared_3027_; uint8_t v_isSharedCheck_3031_; 
lean_dec(v___x_3017_);
lean_dec(v___x_3016_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3024_ = lean_ctor_get(v___x_3020_, 0);
v_isSharedCheck_3031_ = !lean_is_exclusive(v___x_3020_);
if (v_isSharedCheck_3031_ == 0)
{
v___x_3026_ = v___x_3020_;
v_isShared_3027_ = v_isSharedCheck_3031_;
goto v_resetjp_3025_;
}
else
{
lean_inc(v_a_3024_);
lean_dec(v___x_3020_);
v___x_3026_ = lean_box(0);
v_isShared_3027_ = v_isSharedCheck_3031_;
goto v_resetjp_3025_;
}
v_resetjp_3025_:
{
lean_object* v___x_3029_; 
if (v_isShared_3027_ == 0)
{
v___x_3029_ = v___x_3026_;
goto v_reusejp_3028_;
}
else
{
lean_object* v_reuseFailAlloc_3030_; 
v_reuseFailAlloc_3030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3030_, 0, v_a_3024_);
v___x_3029_ = v_reuseFailAlloc_3030_;
goto v_reusejp_3028_;
}
v_reusejp_3028_:
{
return v___x_3029_;
}
}
}
}
else
{
lean_object* v_a_3032_; lean_object* v___x_3034_; uint8_t v_isShared_3035_; uint8_t v_isSharedCheck_3039_; 
lean_dec(v___x_3017_);
lean_dec(v___x_3016_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3032_ = lean_ctor_get(v___x_3018_, 0);
v_isSharedCheck_3039_ = !lean_is_exclusive(v___x_3018_);
if (v_isSharedCheck_3039_ == 0)
{
v___x_3034_ = v___x_3018_;
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
else
{
lean_inc(v_a_3032_);
lean_dec(v___x_3018_);
v___x_3034_ = lean_box(0);
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
v_resetjp_3033_:
{
lean_object* v___x_3037_; 
if (v_isShared_3035_ == 0)
{
v___x_3037_ = v___x_3034_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v_a_3032_);
v___x_3037_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
return v___x_3037_;
}
}
}
}
else
{
lean_object* v___x_3040_; lean_object* v___x_3041_; uint8_t v___x_3042_; 
v___x_3040_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3007_);
v___x_3041_ = l_Lean_Name_append(v___x_3040_, v___y_3007_);
v___x_3042_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3011_, v_options_3009_, v___x_3041_);
lean_dec(v___x_3041_);
if (v___x_3042_ == 0)
{
lean_object* v___x_3043_; uint8_t v___x_3044_; 
v___x_3043_ = l_Lean_trace_profiler;
v___x_3044_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_options_3009_, v___x_3043_);
if (v___x_3044_ == 0)
{
lean_object* v___x_3045_; 
lean_inc(v___x_3016_);
v___x_3045_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__6(v___x_3016_, v___x_2946_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
if (lean_obj_tag(v___x_3045_) == 0)
{
lean_object* v_a_3046_; lean_object* v___x_3047_; 
v_a_3046_ = lean_ctor_get(v___x_3045_, 0);
lean_inc(v_a_3046_);
lean_dec_ref_known(v___x_3045_, 1);
v___x_3047_ = lp_mathlib_Mathlib_Tactic_Linarith_addExprs(v_a_3046_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
if (lean_obj_tag(v___x_3047_) == 0)
{
if (v___x_3042_ == 0)
{
lean_object* v_a_3048_; lean_object* v___x_3049_; lean_object* v___x_3050_; 
v_a_3048_ = lean_ctor_get(v___x_3047_, 0);
lean_inc(v_a_3048_);
lean_dec_ref_known(v___x_3047_, 1);
v___x_3049_ = lean_box(0);
v___x_3050_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(v___x_3016_, v___x_3017_, v_a_3048_, v___x_3049_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
v___y_2766_ = v___y_3004_;
v___y_2767_ = v___y_3005_;
v___y_2768_ = v___y_3008_;
v___y_2769_ = v___y_3012_;
v___y_2770_ = v___x_3050_;
goto v___jp_2765_;
}
else
{
lean_object* v_a_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3055_; lean_object* v___x_3056_; 
v_a_3051_ = lean_ctor_get(v___x_3047_, 0);
lean_inc_n(v_a_3051_, 2);
lean_dec_ref_known(v___x_3047_, 1);
v___x_3052_ = l_Lean_MessageData_ofExpr(v_a_3051_);
v___x_3053_ = l_Lean_indentD(v___x_3052_);
v___x_3054_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__13);
v___x_3055_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3055_, 0, v___x_3053_);
lean_ctor_set(v___x_3055_, 1, v___x_3054_);
lean_inc(v___y_3007_);
v___x_3056_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3007_, v___x_3055_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
if (lean_obj_tag(v___x_3056_) == 0)
{
lean_object* v_a_3057_; lean_object* v___x_3058_; 
v_a_3057_ = lean_ctor_get(v___x_3056_, 0);
lean_inc(v_a_3057_);
lean_dec_ref_known(v___x_3056_, 1);
v___x_3058_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__4(v___x_3016_, v___x_3017_, v_a_3051_, v_a_3057_, v___y_3005_, v___y_3004_, v___y_3008_, v___y_3012_);
v___y_2766_ = v___y_3004_;
v___y_2767_ = v___y_3005_;
v___y_2768_ = v___y_3008_;
v___y_2769_ = v___y_3012_;
v___y_2770_ = v___x_3058_;
goto v___jp_2765_;
}
else
{
lean_object* v_a_3059_; lean_object* v___x_3061_; uint8_t v_isShared_3062_; uint8_t v_isSharedCheck_3066_; 
lean_dec(v_a_3051_);
lean_dec(v___x_3017_);
lean_dec(v___x_3016_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3059_ = lean_ctor_get(v___x_3056_, 0);
v_isSharedCheck_3066_ = !lean_is_exclusive(v___x_3056_);
if (v_isSharedCheck_3066_ == 0)
{
v___x_3061_ = v___x_3056_;
v_isShared_3062_ = v_isSharedCheck_3066_;
goto v_resetjp_3060_;
}
else
{
lean_inc(v_a_3059_);
lean_dec(v___x_3056_);
v___x_3061_ = lean_box(0);
v_isShared_3062_ = v_isSharedCheck_3066_;
goto v_resetjp_3060_;
}
v_resetjp_3060_:
{
lean_object* v___x_3064_; 
if (v_isShared_3062_ == 0)
{
v___x_3064_ = v___x_3061_;
goto v_reusejp_3063_;
}
else
{
lean_object* v_reuseFailAlloc_3065_; 
v_reuseFailAlloc_3065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3065_, 0, v_a_3059_);
v___x_3064_ = v_reuseFailAlloc_3065_;
goto v_reusejp_3063_;
}
v_reusejp_3063_:
{
return v___x_3064_;
}
}
}
}
}
else
{
lean_object* v_a_3067_; lean_object* v___x_3069_; uint8_t v_isShared_3070_; uint8_t v_isSharedCheck_3074_; 
lean_dec(v___x_3017_);
lean_dec(v___x_3016_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3067_ = lean_ctor_get(v___x_3047_, 0);
v_isSharedCheck_3074_ = !lean_is_exclusive(v___x_3047_);
if (v_isSharedCheck_3074_ == 0)
{
v___x_3069_ = v___x_3047_;
v_isShared_3070_ = v_isSharedCheck_3074_;
goto v_resetjp_3068_;
}
else
{
lean_inc(v_a_3067_);
lean_dec(v___x_3047_);
v___x_3069_ = lean_box(0);
v_isShared_3070_ = v_isSharedCheck_3074_;
goto v_resetjp_3068_;
}
v_resetjp_3068_:
{
lean_object* v___x_3072_; 
if (v_isShared_3070_ == 0)
{
v___x_3072_ = v___x_3069_;
goto v_reusejp_3071_;
}
else
{
lean_object* v_reuseFailAlloc_3073_; 
v_reuseFailAlloc_3073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3073_, 0, v_a_3067_);
v___x_3072_ = v_reuseFailAlloc_3073_;
goto v_reusejp_3071_;
}
v_reusejp_3071_:
{
return v___x_3072_;
}
}
}
}
else
{
lean_object* v_a_3075_; lean_object* v___x_3077_; uint8_t v_isShared_3078_; uint8_t v_isSharedCheck_3082_; 
lean_dec(v___x_3017_);
lean_dec(v___x_3016_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3075_ = lean_ctor_get(v___x_3045_, 0);
v_isSharedCheck_3082_ = !lean_is_exclusive(v___x_3045_);
if (v_isSharedCheck_3082_ == 0)
{
v___x_3077_ = v___x_3045_;
v_isShared_3078_ = v_isSharedCheck_3082_;
goto v_resetjp_3076_;
}
else
{
lean_inc(v_a_3075_);
lean_dec(v___x_3045_);
v___x_3077_ = lean_box(0);
v_isShared_3078_ = v_isSharedCheck_3082_;
goto v_resetjp_3076_;
}
v_resetjp_3076_:
{
lean_object* v___x_3080_; 
if (v_isShared_3078_ == 0)
{
v___x_3080_ = v___x_3077_;
goto v_reusejp_3079_;
}
else
{
lean_object* v_reuseFailAlloc_3081_; 
v_reuseFailAlloc_3081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3081_, 0, v_a_3075_);
v___x_3080_ = v_reuseFailAlloc_3081_;
goto v_reusejp_3079_;
}
v_reusejp_3079_:
{
return v___x_3080_;
}
}
}
}
else
{
lean_inc(v___x_3016_);
v___y_2948_ = v___x_3017_;
v___y_2949_ = v___x_3016_;
v___y_2950_ = v___x_3042_;
v___y_2951_ = v_options_3009_;
v___y_2952_ = v___y_3004_;
v___y_2953_ = v_hasTrace_3010_;
v___y_2954_ = v___y_3005_;
v___y_2955_ = v___y_3006_;
v___y_2956_ = v___y_3007_;
v___y_2957_ = v___y_3008_;
v___y_2958_ = v___y_3012_;
v___y_2959_ = v___x_3016_;
v___y_2960_ = v_inheritedTraceOptions_3011_;
goto v___jp_2947_;
}
}
else
{
lean_inc(v___x_3016_);
v___y_2948_ = v___x_3017_;
v___y_2949_ = v___x_3016_;
v___y_2950_ = v___x_3042_;
v___y_2951_ = v_options_3009_;
v___y_2952_ = v___y_3004_;
v___y_2953_ = v_hasTrace_3010_;
v___y_2954_ = v___y_3005_;
v___y_2955_ = v___y_3006_;
v___y_2956_ = v___y_3007_;
v___y_2957_ = v___y_3008_;
v___y_2958_ = v___y_3012_;
v___y_2959_ = v___x_3016_;
v___y_2960_ = v_inheritedTraceOptions_3011_;
goto v___jp_2947_;
}
}
}
v___jp_3083_:
{
lean_object* v_options_3091_; lean_object* v_inheritedTraceOptions_3092_; uint8_t v_hasTrace_3093_; 
v_options_3091_ = lean_ctor_get(v___y_3088_, 2);
v_inheritedTraceOptions_3092_ = lean_ctor_get(v___y_3088_, 13);
v_hasTrace_3093_ = lean_ctor_get_uint8(v_options_3091_, sizeof(void*)*1);
v___y_3004_ = v___y_3084_;
v___y_3005_ = v___y_3085_;
v___y_3006_ = v___y_3086_;
v___y_3007_ = v___y_3087_;
v___y_3008_ = v___y_3088_;
v_options_3009_ = v_options_3091_;
v_hasTrace_3010_ = v_hasTrace_3093_;
v_inheritedTraceOptions_3011_ = v_inheritedTraceOptions_3092_;
v___y_3012_ = v___y_3089_;
v_a_3013_ = v_a_3090_;
goto v___jp_3003_;
}
v___jp_3094_:
{
lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; 
v___x_3104_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10(v___y_3103_, v___x_2946_);
v___x_3105_ = l_Lean_MessageData_ofList(v___x_3104_);
lean_inc_ref(v___y_3098_);
v___x_3106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3106_, 0, v___y_3098_);
lean_ctor_set(v___x_3106_, 1, v___x_3105_);
lean_inc(v___y_3100_);
v___x_3107_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3100_, v___x_3106_, v___y_3097_, v___y_3096_, v___y_3101_, v___y_3102_);
if (lean_obj_tag(v___x_3107_) == 0)
{
lean_dec_ref_known(v___x_3107_, 1);
v___y_3084_ = v___y_3096_;
v___y_3085_ = v___y_3097_;
v___y_3086_ = v___y_3099_;
v___y_3087_ = v___y_3100_;
v___y_3088_ = v___y_3101_;
v___y_3089_ = v___y_3102_;
v_a_3090_ = v___y_3095_;
goto v___jp_3083_;
}
else
{
lean_object* v_a_3108_; lean_object* v___x_3110_; uint8_t v_isShared_3111_; uint8_t v_isSharedCheck_3115_; 
lean_dec_ref(v___y_3095_);
lean_dec(v_a_2799_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3108_ = lean_ctor_get(v___x_3107_, 0);
v_isSharedCheck_3115_ = !lean_is_exclusive(v___x_3107_);
if (v_isSharedCheck_3115_ == 0)
{
v___x_3110_ = v___x_3107_;
v_isShared_3111_ = v_isSharedCheck_3115_;
goto v_resetjp_3109_;
}
else
{
lean_inc(v_a_3108_);
lean_dec(v___x_3107_);
v___x_3110_ = lean_box(0);
v_isShared_3111_ = v_isSharedCheck_3115_;
goto v_resetjp_3109_;
}
v_resetjp_3109_:
{
lean_object* v___x_3113_; 
if (v_isShared_3111_ == 0)
{
v___x_3113_ = v___x_3110_;
goto v_reusejp_3112_;
}
else
{
lean_object* v_reuseFailAlloc_3114_; 
v_reuseFailAlloc_3114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3114_, 0, v_a_3108_);
v___x_3113_ = v_reuseFailAlloc_3114_;
goto v_reusejp_3112_;
}
v_reusejp_3112_:
{
return v___x_3113_;
}
}
}
}
v___jp_3116_:
{
if (lean_obj_tag(v___y_3123_) == 0)
{
lean_object* v_a_3124_; 
v_a_3124_ = lean_ctor_get(v___y_3123_, 0);
lean_inc(v_a_3124_);
lean_dec_ref_known(v___y_3123_, 1);
v___y_3084_ = v___y_3117_;
v___y_3085_ = v___y_3118_;
v___y_3086_ = v___y_3119_;
v___y_3087_ = v___y_3120_;
v___y_3088_ = v___y_3121_;
v___y_3089_ = v___y_3122_;
v_a_3090_ = v_a_3124_;
goto v___jp_3083_;
}
else
{
lean_object* v_a_3125_; lean_object* v___x_3127_; uint8_t v_isShared_3128_; uint8_t v_isSharedCheck_3132_; 
lean_dec(v_a_2799_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3125_ = lean_ctor_get(v___y_3123_, 0);
v_isSharedCheck_3132_ = !lean_is_exclusive(v___y_3123_);
if (v_isSharedCheck_3132_ == 0)
{
v___x_3127_ = v___y_3123_;
v_isShared_3128_ = v_isSharedCheck_3132_;
goto v_resetjp_3126_;
}
else
{
lean_inc(v_a_3125_);
lean_dec(v___y_3123_);
v___x_3127_ = lean_box(0);
v_isShared_3128_ = v_isSharedCheck_3132_;
goto v_resetjp_3126_;
}
v_resetjp_3126_:
{
lean_object* v___x_3130_; 
if (v_isShared_3128_ == 0)
{
v___x_3130_ = v___x_3127_;
goto v_reusejp_3129_;
}
else
{
lean_object* v_reuseFailAlloc_3131_; 
v_reuseFailAlloc_3131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3131_, 0, v_a_3125_);
v___x_3130_ = v_reuseFailAlloc_3131_;
goto v_reusejp_3129_;
}
v_reusejp_3129_:
{
return v___x_3130_;
}
}
}
}
v___jp_3133_:
{
lean_object* v___x_3145_; double v___x_3146_; double v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v___x_3151_; lean_object* v___x_3152_; 
v___x_3145_ = lean_io_get_num_heartbeats();
v___x_3146_ = lean_float_of_nat(v___y_3136_);
v___x_3147_ = lean_float_of_nat(v___x_3145_);
v___x_3148_ = lean_box_float(v___x_3146_);
v___x_3149_ = lean_box_float(v___x_3147_);
v___x_3150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3150_, 0, v___x_3148_);
lean_ctor_set(v___x_3150_, 1, v___x_3149_);
v___x_3151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3151_, 0, v_a_3144_);
lean_ctor_set(v___x_3151_, 1, v___x_3150_);
lean_inc_ref(v___y_3138_);
lean_inc(v___y_3139_);
v___x_3152_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13(v___y_3139_, v___x_2711_, v___y_3138_, v___y_3141_, v___y_3135_, v___y_3143_, v___f_2805_, v___x_3151_, v___y_3137_, v___y_3134_, v___y_3140_, v___y_3142_);
v___y_3117_ = v___y_3134_;
v___y_3118_ = v___y_3137_;
v___y_3119_ = v___y_3138_;
v___y_3120_ = v___y_3139_;
v___y_3121_ = v___y_3140_;
v___y_3122_ = v___y_3142_;
v___y_3123_ = v___x_3152_;
goto v___jp_3116_;
}
v___jp_3153_:
{
lean_object* v___x_3165_; 
v___x_3165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3165_, 0, v_a_3164_);
v___y_3134_ = v___y_3154_;
v___y_3135_ = v___y_3155_;
v___y_3136_ = v___y_3156_;
v___y_3137_ = v___y_3157_;
v___y_3138_ = v___y_3158_;
v___y_3139_ = v___y_3159_;
v___y_3140_ = v___y_3161_;
v___y_3141_ = v___y_3160_;
v___y_3142_ = v___y_3162_;
v___y_3143_ = v___y_3163_;
v_a_3144_ = v___x_3165_;
goto v___jp_3133_;
}
v___jp_3166_:
{
lean_object* v___x_3178_; 
v___x_3178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3178_, 0, v_a_3177_);
v___y_3134_ = v___y_3167_;
v___y_3135_ = v___y_3168_;
v___y_3136_ = v___y_3169_;
v___y_3137_ = v___y_3170_;
v___y_3138_ = v___y_3171_;
v___y_3139_ = v___y_3172_;
v___y_3140_ = v___y_3174_;
v___y_3141_ = v___y_3173_;
v___y_3142_ = v___y_3175_;
v___y_3143_ = v___y_3176_;
v_a_3144_ = v___x_3178_;
goto v___jp_3133_;
}
v___jp_3179_:
{
lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; 
v___x_3193_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10(v___y_3192_, v___x_2946_);
v___x_3194_ = l_Lean_MessageData_ofList(v___x_3193_);
lean_inc_ref(v___y_3180_);
v___x_3195_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3195_, 0, v___y_3180_);
lean_ctor_set(v___x_3195_, 1, v___x_3194_);
lean_inc(v___y_3188_);
v___x_3196_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3188_, v___x_3195_, v___y_3184_, v___y_3181_, v___y_3186_, v___y_3190_);
if (lean_obj_tag(v___x_3196_) == 0)
{
lean_dec_ref_known(v___x_3196_, 1);
v___y_3167_ = v___y_3181_;
v___y_3168_ = v___y_3187_;
v___y_3169_ = v___y_3183_;
v___y_3170_ = v___y_3184_;
v___y_3171_ = v___y_3185_;
v___y_3172_ = v___y_3188_;
v___y_3173_ = v___y_3189_;
v___y_3174_ = v___y_3186_;
v___y_3175_ = v___y_3190_;
v___y_3176_ = v___y_3191_;
v_a_3177_ = v___y_3182_;
goto v___jp_3166_;
}
else
{
lean_object* v_a_3197_; 
lean_dec_ref(v___y_3182_);
v_a_3197_ = lean_ctor_get(v___x_3196_, 0);
lean_inc(v_a_3197_);
lean_dec_ref_known(v___x_3196_, 1);
v___y_3154_ = v___y_3181_;
v___y_3155_ = v___y_3187_;
v___y_3156_ = v___y_3183_;
v___y_3157_ = v___y_3184_;
v___y_3158_ = v___y_3185_;
v___y_3159_ = v___y_3188_;
v___y_3160_ = v___y_3189_;
v___y_3161_ = v___y_3186_;
v___y_3162_ = v___y_3190_;
v___y_3163_ = v___y_3191_;
v_a_3164_ = v_a_3197_;
goto v___jp_3153_;
}
}
v___jp_3198_:
{
if (v___y_3200_ == 0)
{
v___y_3167_ = v___y_3201_;
v___y_3168_ = v___y_3202_;
v___y_3169_ = v___y_3203_;
v___y_3170_ = v___y_3204_;
v___y_3171_ = v___y_3205_;
v___y_3172_ = v___y_3206_;
v___y_3173_ = v___y_3208_;
v___y_3174_ = v___y_3207_;
v___y_3175_ = v___y_3209_;
v___y_3176_ = v___y_3210_;
v_a_3177_ = v_a_3211_;
goto v___jp_3166_;
}
else
{
lean_object* v___x_3212_; lean_object* v___x_3213_; uint8_t v___x_3214_; 
v___x_3212_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3206_);
v___x_3213_ = l_Lean_Name_append(v___x_3212_, v___y_3206_);
v___x_3214_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_3199_, v___y_3208_, v___x_3213_);
lean_dec(v___x_3213_);
if (v___x_3214_ == 0)
{
v___y_3167_ = v___y_3201_;
v___y_3168_ = v___y_3202_;
v___y_3169_ = v___y_3203_;
v___y_3170_ = v___y_3204_;
v___y_3171_ = v___y_3205_;
v___y_3172_ = v___y_3206_;
v___y_3173_ = v___y_3208_;
v___y_3174_ = v___y_3207_;
v___y_3175_ = v___y_3209_;
v___y_3176_ = v___y_3210_;
v_a_3177_ = v_a_3211_;
goto v___jp_3166_;
}
else
{
lean_object* v_buckets_3215_; lean_object* v___x_3216_; lean_object* v___x_3217_; uint8_t v___x_3218_; 
v_buckets_3215_ = lean_ctor_get(v_a_3211_, 1);
v___x_3216_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16);
v___x_3217_ = lean_array_get_size(v_buckets_3215_);
v___x_3218_ = lean_nat_dec_lt(v___x_2786_, v___x_3217_);
if (v___x_3218_ == 0)
{
v___y_3180_ = v___x_3216_;
v___y_3181_ = v___y_3201_;
v___y_3182_ = v_a_3211_;
v___y_3183_ = v___y_3203_;
v___y_3184_ = v___y_3204_;
v___y_3185_ = v___y_3205_;
v___y_3186_ = v___y_3207_;
v___y_3187_ = v___y_3202_;
v___y_3188_ = v___y_3206_;
v___y_3189_ = v___y_3208_;
v___y_3190_ = v___y_3209_;
v___y_3191_ = v___y_3210_;
v___y_3192_ = v___x_2946_;
goto v___jp_3179_;
}
else
{
size_t v___x_3219_; size_t v___x_3220_; lean_object* v___x_3221_; 
v___x_3219_ = lean_usize_of_nat(v___x_3217_);
v___x_3220_ = ((size_t)0ULL);
v___x_3221_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(v_buckets_3215_, v___x_3219_, v___x_3220_, v___x_2946_);
v___y_3180_ = v___x_3216_;
v___y_3181_ = v___y_3201_;
v___y_3182_ = v_a_3211_;
v___y_3183_ = v___y_3203_;
v___y_3184_ = v___y_3204_;
v___y_3185_ = v___y_3205_;
v___y_3186_ = v___y_3207_;
v___y_3187_ = v___y_3202_;
v___y_3188_ = v___y_3206_;
v___y_3189_ = v___y_3208_;
v___y_3190_ = v___y_3209_;
v___y_3191_ = v___y_3210_;
v___y_3192_ = v___x_3221_;
goto v___jp_3179_;
}
}
}
}
v___jp_3222_:
{
lean_object* v_a_3236_; 
v_a_3236_ = lean_ctor_get(v___y_3235_, 0);
lean_inc(v_a_3236_);
lean_dec_ref(v___y_3235_);
v___y_3154_ = v___y_3225_;
v___y_3155_ = v___y_3226_;
v___y_3156_ = v___y_3227_;
v___y_3157_ = v___y_3228_;
v___y_3158_ = v___y_3229_;
v___y_3159_ = v___y_3230_;
v___y_3160_ = v___y_3231_;
v___y_3161_ = v___y_3232_;
v___y_3162_ = v___y_3233_;
v___y_3163_ = v___y_3234_;
v_a_3164_ = v_a_3236_;
goto v___jp_3153_;
}
v___jp_3237_:
{
lean_object* v___x_3250_; lean_object* v___x_3251_; 
v___x_3250_ = lean_box(0);
v___x_3251_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v___x_3250_, v___y_3243_, v___y_3240_, v___y_3247_, v___y_3248_);
v___y_3223_ = v___y_3238_;
v___y_3224_ = v___y_3239_;
v___y_3225_ = v___y_3240_;
v___y_3226_ = v___y_3241_;
v___y_3227_ = v___y_3242_;
v___y_3228_ = v___y_3243_;
v___y_3229_ = v___y_3244_;
v___y_3230_ = v___y_3245_;
v___y_3231_ = v___y_3246_;
v___y_3232_ = v___y_3247_;
v___y_3233_ = v___y_3248_;
v___y_3234_ = v___y_3249_;
v___y_3235_ = v___x_3251_;
goto v___jp_3222_;
}
v___jp_3252_:
{
if (v___y_3266_ == 0)
{
if (v___y_3255_ == 0)
{
lean_dec_ref(v___y_3254_);
v___y_3238_ = v___y_3253_;
v___y_3239_ = v___y_3255_;
v___y_3240_ = v___y_3256_;
v___y_3241_ = v___y_3261_;
v___y_3242_ = v___y_3257_;
v___y_3243_ = v___y_3258_;
v___y_3244_ = v___y_3259_;
v___y_3245_ = v___y_3262_;
v___y_3246_ = v___y_3263_;
v___y_3247_ = v___y_3260_;
v___y_3248_ = v___y_3264_;
v___y_3249_ = v___y_3265_;
goto v___jp_3237_;
}
else
{
lean_object* v___x_3267_; lean_object* v___x_3268_; uint8_t v___x_3269_; 
v___x_3267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3262_);
v___x_3268_ = l_Lean_Name_append(v___x_3267_, v___y_3262_);
v___x_3269_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_3253_, v___y_3263_, v___x_3268_);
lean_dec(v___x_3268_);
if (v___x_3269_ == 0)
{
lean_dec_ref(v___y_3254_);
v___y_3238_ = v___y_3253_;
v___y_3239_ = v___y_3255_;
v___y_3240_ = v___y_3256_;
v___y_3241_ = v___y_3261_;
v___y_3242_ = v___y_3257_;
v___y_3243_ = v___y_3258_;
v___y_3244_ = v___y_3259_;
v___y_3245_ = v___y_3262_;
v___y_3246_ = v___y_3263_;
v___y_3247_ = v___y_3260_;
v___y_3248_ = v___y_3264_;
v___y_3249_ = v___y_3265_;
goto v___jp_3237_;
}
else
{
lean_object* v___x_3270_; lean_object* v___x_3271_; 
v___x_3270_ = l_Lean_Exception_toMessageData(v___y_3254_);
lean_inc(v___y_3262_);
v___x_3271_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3262_, v___x_3270_, v___y_3258_, v___y_3256_, v___y_3260_, v___y_3264_);
if (lean_obj_tag(v___x_3271_) == 0)
{
lean_object* v_a_3272_; lean_object* v___x_3273_; 
v_a_3272_ = lean_ctor_get(v___x_3271_, 0);
lean_inc(v_a_3272_);
lean_dec_ref_known(v___x_3271_, 1);
v___x_3273_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v_a_3272_, v___y_3258_, v___y_3256_, v___y_3260_, v___y_3264_);
v___y_3223_ = v___y_3253_;
v___y_3224_ = v___y_3255_;
v___y_3225_ = v___y_3256_;
v___y_3226_ = v___y_3261_;
v___y_3227_ = v___y_3257_;
v___y_3228_ = v___y_3258_;
v___y_3229_ = v___y_3259_;
v___y_3230_ = v___y_3262_;
v___y_3231_ = v___y_3263_;
v___y_3232_ = v___y_3260_;
v___y_3233_ = v___y_3264_;
v___y_3234_ = v___y_3265_;
v___y_3235_ = v___x_3273_;
goto v___jp_3222_;
}
else
{
lean_object* v_a_3274_; 
v_a_3274_ = lean_ctor_get(v___x_3271_, 0);
lean_inc(v_a_3274_);
lean_dec_ref_known(v___x_3271_, 1);
v___y_3154_ = v___y_3256_;
v___y_3155_ = v___y_3261_;
v___y_3156_ = v___y_3257_;
v___y_3157_ = v___y_3258_;
v___y_3158_ = v___y_3259_;
v___y_3159_ = v___y_3262_;
v___y_3160_ = v___y_3263_;
v___y_3161_ = v___y_3260_;
v___y_3162_ = v___y_3264_;
v___y_3163_ = v___y_3265_;
v_a_3164_ = v_a_3274_;
goto v___jp_3153_;
}
}
}
}
else
{
v___y_3154_ = v___y_3256_;
v___y_3155_ = v___y_3261_;
v___y_3156_ = v___y_3257_;
v___y_3157_ = v___y_3258_;
v___y_3158_ = v___y_3259_;
v___y_3159_ = v___y_3262_;
v___y_3160_ = v___y_3263_;
v___y_3161_ = v___y_3260_;
v___y_3162_ = v___y_3264_;
v___y_3163_ = v___y_3265_;
v_a_3164_ = v___y_3254_;
goto v___jp_3153_;
}
}
v___jp_3275_:
{
lean_object* v___x_3287_; double v___x_3288_; double v___x_3289_; double v___x_3290_; double v___x_3291_; double v___x_3292_; lean_object* v___x_3293_; lean_object* v___x_3294_; lean_object* v___x_3295_; lean_object* v___x_3296_; lean_object* v___x_3297_; 
v___x_3287_ = lean_io_mono_nanos_now();
v___x_3288_ = lean_float_of_nat(v___y_3276_);
v___x_3289_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__7);
v___x_3290_ = lean_float_div(v___x_3288_, v___x_3289_);
v___x_3291_ = lean_float_of_nat(v___x_3287_);
v___x_3292_ = lean_float_div(v___x_3291_, v___x_3289_);
v___x_3293_ = lean_box_float(v___x_3290_);
v___x_3294_ = lean_box_float(v___x_3292_);
v___x_3295_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3295_, 0, v___x_3293_);
lean_ctor_set(v___x_3295_, 1, v___x_3294_);
v___x_3296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3296_, 0, v_a_3286_);
lean_ctor_set(v___x_3296_, 1, v___x_3295_);
lean_inc_ref(v___y_3280_);
lean_inc(v___y_3281_);
v___x_3297_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__13(v___y_3281_, v___x_2711_, v___y_3280_, v___y_3283_, v___y_3278_, v___y_3285_, v___f_2805_, v___x_3296_, v___y_3279_, v___y_3277_, v___y_3282_, v___y_3284_);
v___y_3117_ = v___y_3277_;
v___y_3118_ = v___y_3279_;
v___y_3119_ = v___y_3280_;
v___y_3120_ = v___y_3281_;
v___y_3121_ = v___y_3282_;
v___y_3122_ = v___y_3284_;
v___y_3123_ = v___x_3297_;
goto v___jp_3116_;
}
v___jp_3298_:
{
lean_object* v___x_3310_; 
v___x_3310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3310_, 0, v_a_3309_);
v___y_3276_ = v___y_3299_;
v___y_3277_ = v___y_3300_;
v___y_3278_ = v___y_3301_;
v___y_3279_ = v___y_3302_;
v___y_3280_ = v___y_3303_;
v___y_3281_ = v___y_3304_;
v___y_3282_ = v___y_3306_;
v___y_3283_ = v___y_3305_;
v___y_3284_ = v___y_3307_;
v___y_3285_ = v___y_3308_;
v_a_3286_ = v___x_3310_;
goto v___jp_3275_;
}
v___jp_3311_:
{
lean_object* v___x_3323_; 
v___x_3323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3323_, 0, v_a_3322_);
v___y_3276_ = v___y_3312_;
v___y_3277_ = v___y_3313_;
v___y_3278_ = v___y_3314_;
v___y_3279_ = v___y_3315_;
v___y_3280_ = v___y_3316_;
v___y_3281_ = v___y_3317_;
v___y_3282_ = v___y_3319_;
v___y_3283_ = v___y_3318_;
v___y_3284_ = v___y_3320_;
v___y_3285_ = v___y_3321_;
v_a_3286_ = v___x_3323_;
goto v___jp_3275_;
}
v___jp_3324_:
{
lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3340_; lean_object* v___x_3341_; 
v___x_3338_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__10(v___y_3337_, v___x_2946_);
v___x_3339_ = l_Lean_MessageData_ofList(v___x_3338_);
lean_inc_ref(v___y_3328_);
v___x_3340_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3340_, 0, v___y_3328_);
lean_ctor_set(v___x_3340_, 1, v___x_3339_);
lean_inc(v___y_3333_);
v___x_3341_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3333_, v___x_3340_, v___y_3329_, v___y_3326_, v___y_3331_, v___y_3335_);
if (lean_obj_tag(v___x_3341_) == 0)
{
lean_dec_ref_known(v___x_3341_, 1);
v___y_3312_ = v___y_3325_;
v___y_3313_ = v___y_3326_;
v___y_3314_ = v___y_3332_;
v___y_3315_ = v___y_3329_;
v___y_3316_ = v___y_3330_;
v___y_3317_ = v___y_3333_;
v___y_3318_ = v___y_3334_;
v___y_3319_ = v___y_3331_;
v___y_3320_ = v___y_3335_;
v___y_3321_ = v___y_3336_;
v_a_3322_ = v___y_3327_;
goto v___jp_3311_;
}
else
{
lean_object* v_a_3342_; 
lean_dec_ref(v___y_3327_);
v_a_3342_ = lean_ctor_get(v___x_3341_, 0);
lean_inc(v_a_3342_);
lean_dec_ref_known(v___x_3341_, 1);
v___y_3299_ = v___y_3325_;
v___y_3300_ = v___y_3326_;
v___y_3301_ = v___y_3332_;
v___y_3302_ = v___y_3329_;
v___y_3303_ = v___y_3330_;
v___y_3304_ = v___y_3333_;
v___y_3305_ = v___y_3334_;
v___y_3306_ = v___y_3331_;
v___y_3307_ = v___y_3335_;
v___y_3308_ = v___y_3336_;
v_a_3309_ = v_a_3342_;
goto v___jp_3298_;
}
}
v___jp_3343_:
{
if (v___y_3345_ == 0)
{
v___y_3312_ = v___y_3346_;
v___y_3313_ = v___y_3347_;
v___y_3314_ = v___y_3348_;
v___y_3315_ = v___y_3349_;
v___y_3316_ = v___y_3350_;
v___y_3317_ = v___y_3351_;
v___y_3318_ = v___y_3353_;
v___y_3319_ = v___y_3352_;
v___y_3320_ = v___y_3354_;
v___y_3321_ = v___y_3355_;
v_a_3322_ = v_a_3356_;
goto v___jp_3311_;
}
else
{
lean_object* v___x_3357_; lean_object* v___x_3358_; uint8_t v___x_3359_; 
v___x_3357_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3351_);
v___x_3358_ = l_Lean_Name_append(v___x_3357_, v___y_3351_);
v___x_3359_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_3344_, v___y_3353_, v___x_3358_);
lean_dec(v___x_3358_);
if (v___x_3359_ == 0)
{
v___y_3312_ = v___y_3346_;
v___y_3313_ = v___y_3347_;
v___y_3314_ = v___y_3348_;
v___y_3315_ = v___y_3349_;
v___y_3316_ = v___y_3350_;
v___y_3317_ = v___y_3351_;
v___y_3318_ = v___y_3353_;
v___y_3319_ = v___y_3352_;
v___y_3320_ = v___y_3354_;
v___y_3321_ = v___y_3355_;
v_a_3322_ = v_a_3356_;
goto v___jp_3311_;
}
else
{
lean_object* v_buckets_3360_; lean_object* v___x_3361_; lean_object* v___x_3362_; uint8_t v___x_3363_; 
v_buckets_3360_ = lean_ctor_get(v_a_3356_, 1);
v___x_3361_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16);
v___x_3362_ = lean_array_get_size(v_buckets_3360_);
v___x_3363_ = lean_nat_dec_lt(v___x_2786_, v___x_3362_);
if (v___x_3363_ == 0)
{
v___y_3325_ = v___y_3346_;
v___y_3326_ = v___y_3347_;
v___y_3327_ = v_a_3356_;
v___y_3328_ = v___x_3361_;
v___y_3329_ = v___y_3349_;
v___y_3330_ = v___y_3350_;
v___y_3331_ = v___y_3352_;
v___y_3332_ = v___y_3348_;
v___y_3333_ = v___y_3351_;
v___y_3334_ = v___y_3353_;
v___y_3335_ = v___y_3354_;
v___y_3336_ = v___y_3355_;
v___y_3337_ = v___x_2946_;
goto v___jp_3324_;
}
else
{
size_t v___x_3364_; size_t v___x_3365_; lean_object* v___x_3366_; 
v___x_3364_ = lean_usize_of_nat(v___x_3362_);
v___x_3365_ = ((size_t)0ULL);
v___x_3366_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(v_buckets_3360_, v___x_3364_, v___x_3365_, v___x_2946_);
v___y_3325_ = v___y_3346_;
v___y_3326_ = v___y_3347_;
v___y_3327_ = v_a_3356_;
v___y_3328_ = v___x_3361_;
v___y_3329_ = v___y_3349_;
v___y_3330_ = v___y_3350_;
v___y_3331_ = v___y_3352_;
v___y_3332_ = v___y_3348_;
v___y_3333_ = v___y_3351_;
v___y_3334_ = v___y_3353_;
v___y_3335_ = v___y_3354_;
v___y_3336_ = v___y_3355_;
v___y_3337_ = v___x_3366_;
goto v___jp_3324_;
}
}
}
}
v___jp_3367_:
{
lean_object* v_a_3381_; 
v_a_3381_ = lean_ctor_get(v___y_3380_, 0);
lean_inc(v_a_3381_);
lean_dec_ref(v___y_3380_);
v___y_3299_ = v___y_3370_;
v___y_3300_ = v___y_3371_;
v___y_3301_ = v___y_3372_;
v___y_3302_ = v___y_3373_;
v___y_3303_ = v___y_3374_;
v___y_3304_ = v___y_3375_;
v___y_3305_ = v___y_3376_;
v___y_3306_ = v___y_3377_;
v___y_3307_ = v___y_3378_;
v___y_3308_ = v___y_3379_;
v_a_3309_ = v_a_3381_;
goto v___jp_3298_;
}
v___jp_3382_:
{
lean_object* v___x_3395_; lean_object* v___x_3396_; 
v___x_3395_ = lean_box(0);
v___x_3396_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v___x_3395_, v___y_3388_, v___y_3386_, v___y_3392_, v___y_3393_);
v___y_3368_ = v___y_3383_;
v___y_3369_ = v___y_3384_;
v___y_3370_ = v___y_3385_;
v___y_3371_ = v___y_3386_;
v___y_3372_ = v___y_3387_;
v___y_3373_ = v___y_3388_;
v___y_3374_ = v___y_3389_;
v___y_3375_ = v___y_3390_;
v___y_3376_ = v___y_3391_;
v___y_3377_ = v___y_3392_;
v___y_3378_ = v___y_3393_;
v___y_3379_ = v___y_3394_;
v___y_3380_ = v___x_3396_;
goto v___jp_3367_;
}
v___jp_3397_:
{
if (v___y_3411_ == 0)
{
if (v___y_3399_ == 0)
{
lean_dec_ref(v___y_3406_);
v___y_3383_ = v___y_3398_;
v___y_3384_ = v___y_3399_;
v___y_3385_ = v___y_3400_;
v___y_3386_ = v___y_3401_;
v___y_3387_ = v___y_3405_;
v___y_3388_ = v___y_3402_;
v___y_3389_ = v___y_3403_;
v___y_3390_ = v___y_3407_;
v___y_3391_ = v___y_3408_;
v___y_3392_ = v___y_3404_;
v___y_3393_ = v___y_3409_;
v___y_3394_ = v___y_3410_;
goto v___jp_3382_;
}
else
{
lean_object* v___x_3412_; lean_object* v___x_3413_; uint8_t v___x_3414_; 
v___x_3412_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3407_);
v___x_3413_ = l_Lean_Name_append(v___x_3412_, v___y_3407_);
v___x_3414_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_3398_, v___y_3408_, v___x_3413_);
lean_dec(v___x_3413_);
if (v___x_3414_ == 0)
{
lean_dec_ref(v___y_3406_);
v___y_3383_ = v___y_3398_;
v___y_3384_ = v___y_3399_;
v___y_3385_ = v___y_3400_;
v___y_3386_ = v___y_3401_;
v___y_3387_ = v___y_3405_;
v___y_3388_ = v___y_3402_;
v___y_3389_ = v___y_3403_;
v___y_3390_ = v___y_3407_;
v___y_3391_ = v___y_3408_;
v___y_3392_ = v___y_3404_;
v___y_3393_ = v___y_3409_;
v___y_3394_ = v___y_3410_;
goto v___jp_3382_;
}
else
{
lean_object* v___x_3415_; lean_object* v___x_3416_; 
v___x_3415_ = l_Lean_Exception_toMessageData(v___y_3406_);
lean_inc(v___y_3407_);
v___x_3416_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3407_, v___x_3415_, v___y_3402_, v___y_3401_, v___y_3404_, v___y_3409_);
if (lean_obj_tag(v___x_3416_) == 0)
{
lean_object* v_a_3417_; lean_object* v___x_3418_; 
v_a_3417_ = lean_ctor_get(v___x_3416_, 0);
lean_inc(v_a_3417_);
lean_dec_ref_known(v___x_3416_, 1);
v___x_3418_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v_a_3417_, v___y_3402_, v___y_3401_, v___y_3404_, v___y_3409_);
v___y_3368_ = v___y_3398_;
v___y_3369_ = v___y_3399_;
v___y_3370_ = v___y_3400_;
v___y_3371_ = v___y_3401_;
v___y_3372_ = v___y_3405_;
v___y_3373_ = v___y_3402_;
v___y_3374_ = v___y_3403_;
v___y_3375_ = v___y_3407_;
v___y_3376_ = v___y_3408_;
v___y_3377_ = v___y_3404_;
v___y_3378_ = v___y_3409_;
v___y_3379_ = v___y_3410_;
v___y_3380_ = v___x_3418_;
goto v___jp_3367_;
}
else
{
lean_object* v_a_3419_; 
v_a_3419_ = lean_ctor_get(v___x_3416_, 0);
lean_inc(v_a_3419_);
lean_dec_ref_known(v___x_3416_, 1);
v___y_3299_ = v___y_3400_;
v___y_3300_ = v___y_3401_;
v___y_3301_ = v___y_3405_;
v___y_3302_ = v___y_3402_;
v___y_3303_ = v___y_3403_;
v___y_3304_ = v___y_3407_;
v___y_3305_ = v___y_3408_;
v___y_3306_ = v___y_3404_;
v___y_3307_ = v___y_3409_;
v___y_3308_ = v___y_3410_;
v_a_3309_ = v_a_3419_;
goto v___jp_3298_;
}
}
}
}
else
{
v___y_3299_ = v___y_3400_;
v___y_3300_ = v___y_3401_;
v___y_3301_ = v___y_3405_;
v___y_3302_ = v___y_3402_;
v___y_3303_ = v___y_3403_;
v___y_3304_ = v___y_3407_;
v___y_3305_ = v___y_3408_;
v___y_3306_ = v___y_3404_;
v___y_3307_ = v___y_3409_;
v___y_3308_ = v___y_3410_;
v_a_3309_ = v___y_3406_;
goto v___jp_3298_;
}
}
v___jp_3420_:
{
lean_object* v___x_3433_; lean_object* v_a_3434_; lean_object* v___x_3435_; uint8_t v___x_3436_; 
v___x_3433_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__0___redArg(v___y_3432_);
v_a_3434_ = lean_ctor_get(v___x_3433_, 0);
lean_inc(v_a_3434_);
lean_dec_ref(v___x_3433_);
v___x_3435_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3436_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v___y_3431_, v___x_3435_);
if (v___x_3436_ == 0)
{
lean_object* v___x_3437_; lean_object* v___x_3438_; 
v___x_3437_ = lean_io_mono_nanos_now();
lean_inc(v___y_3432_);
lean_inc_ref(v___y_3428_);
lean_inc(v___y_3423_);
lean_inc_ref(v___y_3426_);
v___x_3438_ = lean_apply_7(v_oracle_2696_, v___y_3425_, v___y_3424_, v___y_3426_, v___y_3423_, v___y_3428_, v___y_3432_, lean_box(0));
if (lean_obj_tag(v___x_3438_) == 0)
{
lean_object* v_a_3439_; 
v_a_3439_ = lean_ctor_get(v___x_3438_, 0);
lean_inc(v_a_3439_);
lean_dec_ref_known(v___x_3438_, 1);
v___y_3344_ = v___y_3421_;
v___y_3345_ = v___y_3422_;
v___y_3346_ = v___x_3437_;
v___y_3347_ = v___y_3423_;
v___y_3348_ = v___y_3429_;
v___y_3349_ = v___y_3426_;
v___y_3350_ = v___y_3427_;
v___y_3351_ = v___y_3430_;
v___y_3352_ = v___y_3428_;
v___y_3353_ = v___y_3431_;
v___y_3354_ = v___y_3432_;
v___y_3355_ = v_a_3434_;
v_a_3356_ = v_a_3439_;
goto v___jp_3343_;
}
else
{
lean_object* v_a_3440_; uint8_t v___x_3441_; 
v_a_3440_ = lean_ctor_get(v___x_3438_, 0);
lean_inc(v_a_3440_);
lean_dec_ref_known(v___x_3438_, 1);
v___x_3441_ = l_Lean_Exception_isInterrupt(v_a_3440_);
if (v___x_3441_ == 0)
{
uint8_t v___x_3442_; 
lean_inc(v_a_3440_);
v___x_3442_ = l_Lean_Exception_isRuntime(v_a_3440_);
v___y_3398_ = v___y_3421_;
v___y_3399_ = v___y_3422_;
v___y_3400_ = v___x_3437_;
v___y_3401_ = v___y_3423_;
v___y_3402_ = v___y_3426_;
v___y_3403_ = v___y_3427_;
v___y_3404_ = v___y_3428_;
v___y_3405_ = v___y_3429_;
v___y_3406_ = v_a_3440_;
v___y_3407_ = v___y_3430_;
v___y_3408_ = v___y_3431_;
v___y_3409_ = v___y_3432_;
v___y_3410_ = v_a_3434_;
v___y_3411_ = v___x_3442_;
goto v___jp_3397_;
}
else
{
v___y_3398_ = v___y_3421_;
v___y_3399_ = v___y_3422_;
v___y_3400_ = v___x_3437_;
v___y_3401_ = v___y_3423_;
v___y_3402_ = v___y_3426_;
v___y_3403_ = v___y_3427_;
v___y_3404_ = v___y_3428_;
v___y_3405_ = v___y_3429_;
v___y_3406_ = v_a_3440_;
v___y_3407_ = v___y_3430_;
v___y_3408_ = v___y_3431_;
v___y_3409_ = v___y_3432_;
v___y_3410_ = v_a_3434_;
v___y_3411_ = v___x_3441_;
goto v___jp_3397_;
}
}
}
else
{
lean_object* v___x_3443_; lean_object* v___x_3444_; 
v___x_3443_ = lean_io_get_num_heartbeats();
lean_inc(v___y_3432_);
lean_inc_ref(v___y_3428_);
lean_inc(v___y_3423_);
lean_inc_ref(v___y_3426_);
v___x_3444_ = lean_apply_7(v_oracle_2696_, v___y_3425_, v___y_3424_, v___y_3426_, v___y_3423_, v___y_3428_, v___y_3432_, lean_box(0));
if (lean_obj_tag(v___x_3444_) == 0)
{
lean_object* v_a_3445_; 
v_a_3445_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3445_);
lean_dec_ref_known(v___x_3444_, 1);
v___y_3199_ = v___y_3421_;
v___y_3200_ = v___y_3422_;
v___y_3201_ = v___y_3423_;
v___y_3202_ = v___y_3429_;
v___y_3203_ = v___x_3443_;
v___y_3204_ = v___y_3426_;
v___y_3205_ = v___y_3427_;
v___y_3206_ = v___y_3430_;
v___y_3207_ = v___y_3428_;
v___y_3208_ = v___y_3431_;
v___y_3209_ = v___y_3432_;
v___y_3210_ = v_a_3434_;
v_a_3211_ = v_a_3445_;
goto v___jp_3198_;
}
else
{
lean_object* v_a_3446_; uint8_t v___x_3447_; 
v_a_3446_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3446_);
lean_dec_ref_known(v___x_3444_, 1);
v___x_3447_ = l_Lean_Exception_isInterrupt(v_a_3446_);
if (v___x_3447_ == 0)
{
uint8_t v___x_3448_; 
lean_inc(v_a_3446_);
v___x_3448_ = l_Lean_Exception_isRuntime(v_a_3446_);
v___y_3253_ = v___y_3421_;
v___y_3254_ = v_a_3446_;
v___y_3255_ = v___y_3422_;
v___y_3256_ = v___y_3423_;
v___y_3257_ = v___x_3443_;
v___y_3258_ = v___y_3426_;
v___y_3259_ = v___y_3427_;
v___y_3260_ = v___y_3428_;
v___y_3261_ = v___y_3429_;
v___y_3262_ = v___y_3430_;
v___y_3263_ = v___y_3431_;
v___y_3264_ = v___y_3432_;
v___y_3265_ = v_a_3434_;
v___y_3266_ = v___x_3448_;
goto v___jp_3252_;
}
else
{
v___y_3253_ = v___y_3421_;
v___y_3254_ = v_a_3446_;
v___y_3255_ = v___y_3422_;
v___y_3256_ = v___y_3423_;
v___y_3257_ = v___x_3443_;
v___y_3258_ = v___y_3426_;
v___y_3259_ = v___y_3427_;
v___y_3260_ = v___y_3428_;
v___y_3261_ = v___y_3429_;
v___y_3262_ = v___y_3430_;
v___y_3263_ = v___y_3431_;
v___y_3264_ = v___y_3432_;
v___y_3265_ = v_a_3434_;
v___y_3266_ = v___x_3447_;
goto v___jp_3252_;
}
}
}
}
v___jp_3449_:
{
lean_object* v___x_3460_; lean_object* v___x_3461_; uint8_t v___x_3462_; 
v___x_3460_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3453_);
v___x_3461_ = l_Lean_Name_append(v___x_3460_, v___y_3453_);
v___x_3462_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3457_, v_options_3455_, v___x_3461_);
lean_dec(v___x_3461_);
if (v___x_3462_ == 0)
{
v___y_3004_ = v___y_3450_;
v___y_3005_ = v___y_3451_;
v___y_3006_ = v___y_3452_;
v___y_3007_ = v___y_3453_;
v___y_3008_ = v___y_3454_;
v_options_3009_ = v_options_3455_;
v_hasTrace_3010_ = v_hasTrace_3456_;
v_inheritedTraceOptions_3011_ = v_inheritedTraceOptions_3457_;
v___y_3012_ = v___y_3458_;
v_a_3013_ = v_a_3459_;
goto v___jp_3003_;
}
else
{
lean_object* v_buckets_3463_; lean_object* v___x_3464_; lean_object* v___x_3465_; uint8_t v___x_3466_; 
v_buckets_3463_ = lean_ctor_get(v_a_3459_, 1);
v___x_3464_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__16);
v___x_3465_ = lean_array_get_size(v_buckets_3463_);
v___x_3466_ = lean_nat_dec_lt(v___x_2786_, v___x_3465_);
if (v___x_3466_ == 0)
{
v___y_3095_ = v_a_3459_;
v___y_3096_ = v___y_3450_;
v___y_3097_ = v___y_3451_;
v___y_3098_ = v___x_3464_;
v___y_3099_ = v___y_3452_;
v___y_3100_ = v___y_3453_;
v___y_3101_ = v___y_3454_;
v___y_3102_ = v___y_3458_;
v___y_3103_ = v___x_2946_;
goto v___jp_3094_;
}
else
{
size_t v___x_3467_; size_t v___x_3468_; lean_object* v___x_3469_; 
v___x_3467_ = lean_usize_of_nat(v___x_3465_);
v___x_3468_ = ((size_t)0ULL);
v___x_3469_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__12(v_buckets_3463_, v___x_3467_, v___x_3468_, v___x_2946_);
v___y_3095_ = v_a_3459_;
v___y_3096_ = v___y_3450_;
v___y_3097_ = v___y_3451_;
v___y_3098_ = v___x_3464_;
v___y_3099_ = v___y_3452_;
v___y_3100_ = v___y_3453_;
v___y_3101_ = v___y_3454_;
v___y_3102_ = v___y_3458_;
v___y_3103_ = v___x_3469_;
goto v___jp_3094_;
}
}
}
v___jp_3470_:
{
lean_object* v_a_3478_; lean_object* v___x_3480_; uint8_t v_isShared_3481_; uint8_t v_isSharedCheck_3485_; 
v_a_3478_ = lean_ctor_get(v___y_3477_, 0);
v_isSharedCheck_3485_ = !lean_is_exclusive(v___y_3477_);
if (v_isSharedCheck_3485_ == 0)
{
v___x_3480_ = v___y_3477_;
v_isShared_3481_ = v_isSharedCheck_3485_;
goto v_resetjp_3479_;
}
else
{
lean_inc(v_a_3478_);
lean_dec(v___y_3477_);
v___x_3480_ = lean_box(0);
v_isShared_3481_ = v_isSharedCheck_3485_;
goto v_resetjp_3479_;
}
v_resetjp_3479_:
{
lean_object* v___x_3483_; 
if (v_isShared_3481_ == 0)
{
v___x_3483_ = v___x_3480_;
goto v_reusejp_3482_;
}
else
{
lean_object* v_reuseFailAlloc_3484_; 
v_reuseFailAlloc_3484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3484_, 0, v_a_3478_);
v___x_3483_ = v_reuseFailAlloc_3484_;
goto v_reusejp_3482_;
}
v_reusejp_3482_:
{
return v___x_3483_;
}
}
}
v___jp_3486_:
{
lean_object* v___x_3493_; lean_object* v___x_3494_; 
v___x_3493_ = lean_box(0);
v___x_3494_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v___x_3493_, v___y_3488_, v___y_3487_, v___y_3491_, v___y_3492_);
v___y_3471_ = v___y_3487_;
v___y_3472_ = v___y_3488_;
v___y_3473_ = v___y_3489_;
v___y_3474_ = v___y_3490_;
v___y_3475_ = v___y_3491_;
v___y_3476_ = v___y_3492_;
v___y_3477_ = v___x_3494_;
goto v___jp_3470_;
}
v___jp_3495_:
{
if (v___y_3506_ == 0)
{
lean_del_object(v___x_2801_);
if (v___y_3497_ == 0)
{
lean_dec_ref(v___y_3505_);
v___y_3487_ = v___y_3498_;
v___y_3488_ = v___y_3499_;
v___y_3489_ = v___y_3500_;
v___y_3490_ = v___y_3501_;
v___y_3491_ = v___y_3503_;
v___y_3492_ = v___y_3504_;
goto v___jp_3486_;
}
else
{
lean_object* v___x_3507_; lean_object* v___x_3508_; uint8_t v___x_3509_; 
v___x_3507_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__5));
lean_inc(v___y_3501_);
v___x_3508_ = l_Lean_Name_append(v___x_3507_, v___y_3501_);
v___x_3509_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_3496_, v___y_3502_, v___x_3508_);
lean_dec(v___x_3508_);
if (v___x_3509_ == 0)
{
lean_dec_ref(v___y_3505_);
v___y_3487_ = v___y_3498_;
v___y_3488_ = v___y_3499_;
v___y_3489_ = v___y_3500_;
v___y_3490_ = v___y_3501_;
v___y_3491_ = v___y_3503_;
v___y_3492_ = v___y_3504_;
goto v___jp_3486_;
}
else
{
lean_object* v___x_3510_; lean_object* v___x_3511_; 
v___x_3510_ = l_Lean_Exception_toMessageData(v___y_3505_);
lean_inc(v___y_3501_);
v___x_3511_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___y_3501_, v___x_3510_, v___y_3499_, v___y_3498_, v___y_3503_, v___y_3504_);
if (lean_obj_tag(v___x_3511_) == 0)
{
lean_object* v_a_3512_; lean_object* v___x_3513_; 
v_a_3512_ = lean_ctor_get(v___x_3511_, 0);
lean_inc(v_a_3512_);
lean_dec_ref_known(v___x_3511_, 1);
v___x_3513_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3(v_a_3512_, v___y_3499_, v___y_3498_, v___y_3503_, v___y_3504_);
v___y_3471_ = v___y_3498_;
v___y_3472_ = v___y_3499_;
v___y_3473_ = v___y_3500_;
v___y_3474_ = v___y_3501_;
v___y_3475_ = v___y_3503_;
v___y_3476_ = v___y_3504_;
v___y_3477_ = v___x_3513_;
goto v___jp_3470_;
}
else
{
lean_object* v_a_3514_; lean_object* v___x_3516_; uint8_t v_isShared_3517_; uint8_t v_isSharedCheck_3521_; 
v_a_3514_ = lean_ctor_get(v___x_3511_, 0);
v_isSharedCheck_3521_ = !lean_is_exclusive(v___x_3511_);
if (v_isSharedCheck_3521_ == 0)
{
v___x_3516_ = v___x_3511_;
v_isShared_3517_ = v_isSharedCheck_3521_;
goto v_resetjp_3515_;
}
else
{
lean_inc(v_a_3514_);
lean_dec(v___x_3511_);
v___x_3516_ = lean_box(0);
v_isShared_3517_ = v_isSharedCheck_3521_;
goto v_resetjp_3515_;
}
v_resetjp_3515_:
{
lean_object* v___x_3519_; 
if (v_isShared_3517_ == 0)
{
v___x_3519_ = v___x_3516_;
goto v_reusejp_3518_;
}
else
{
lean_object* v_reuseFailAlloc_3520_; 
v_reuseFailAlloc_3520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3520_, 0, v_a_3514_);
v___x_3519_ = v_reuseFailAlloc_3520_;
goto v_reusejp_3518_;
}
v_reusejp_3518_:
{
return v___x_3519_;
}
}
}
}
}
}
else
{
lean_object* v___x_3523_; 
if (v_isShared_2802_ == 0)
{
lean_ctor_set_tag(v___x_2801_, 1);
lean_ctor_set(v___x_2801_, 0, v___y_3505_);
v___x_3523_ = v___x_2801_;
goto v_reusejp_3522_;
}
else
{
lean_object* v_reuseFailAlloc_3524_; 
v_reuseFailAlloc_3524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3524_, 0, v___y_3505_);
v___x_3523_ = v_reuseFailAlloc_3524_;
goto v_reusejp_3522_;
}
v_reusejp_3522_:
{
return v___x_3523_;
}
}
}
v___jp_3525_:
{
if (v___y_3533_ == 0)
{
lean_object* v___x_3534_; lean_object* v___x_3535_; lean_object* v_a_3536_; lean_object* v___x_3538_; uint8_t v_isShared_3539_; uint8_t v_isSharedCheck_3543_; 
lean_dec_ref(v___y_3529_);
lean_del_object(v___x_2793_);
v___x_3534_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__3___closed__1);
v___x_3535_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Linarith_mkLTZeroProof_spec__0___redArg(v___x_3534_, v___y_3527_, v___y_3526_, v___y_3531_, v___y_3532_);
v_a_3536_ = lean_ctor_get(v___x_3535_, 0);
v_isSharedCheck_3543_ = !lean_is_exclusive(v___x_3535_);
if (v_isSharedCheck_3543_ == 0)
{
v___x_3538_ = v___x_3535_;
v_isShared_3539_ = v_isSharedCheck_3543_;
goto v_resetjp_3537_;
}
else
{
lean_inc(v_a_3536_);
lean_dec(v___x_3535_);
v___x_3538_ = lean_box(0);
v_isShared_3539_ = v_isSharedCheck_3543_;
goto v_resetjp_3537_;
}
v_resetjp_3537_:
{
lean_object* v___x_3541_; 
if (v_isShared_3539_ == 0)
{
v___x_3541_ = v___x_3538_;
goto v_reusejp_3540_;
}
else
{
lean_object* v_reuseFailAlloc_3542_; 
v_reuseFailAlloc_3542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3542_, 0, v_a_3536_);
v___x_3541_ = v_reuseFailAlloc_3542_;
goto v_reusejp_3540_;
}
v_reusejp_3540_:
{
return v___x_3541_;
}
}
}
else
{
lean_object* v___x_3545_; 
if (v_isShared_2794_ == 0)
{
lean_ctor_set_tag(v___x_2793_, 1);
lean_ctor_set(v___x_2793_, 0, v___y_3529_);
v___x_3545_ = v___x_2793_;
goto v_reusejp_3544_;
}
else
{
lean_object* v_reuseFailAlloc_3546_; 
v_reuseFailAlloc_3546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3546_, 0, v___y_3529_);
v___x_3545_ = v_reuseFailAlloc_3546_;
goto v_reusejp_3544_;
}
v_reusejp_3544_:
{
return v___x_3545_;
}
}
}
v___jp_3548_:
{
lean_object* v___x_3558_; lean_object* v___x_3559_; 
v___x_3558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__17));
v___x_3559_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__3));
if (v_hasTrace_3555_ == 0)
{
lean_object* v___x_3560_; 
lean_del_object(v___x_2801_);
lean_inc(v___y_3557_);
lean_inc_ref(v___y_3553_);
lean_inc(v___y_3552_);
lean_inc_ref(v___y_3551_);
v___x_3560_ = lean_apply_7(v_oracle_2696_, v___y_3550_, v___y_3549_, v___y_3551_, v___y_3552_, v___y_3553_, v___y_3557_, lean_box(0));
if (lean_obj_tag(v___x_3560_) == 0)
{
lean_object* v_a_3561_; 
lean_del_object(v___x_2793_);
v_a_3561_ = lean_ctor_get(v___x_3560_, 0);
lean_inc(v_a_3561_);
lean_dec_ref_known(v___x_3560_, 1);
v___y_3004_ = v___y_3552_;
v___y_3005_ = v___y_3551_;
v___y_3006_ = v___x_3559_;
v___y_3007_ = v___x_3558_;
v___y_3008_ = v___y_3553_;
v_options_3009_ = v_options_3554_;
v_hasTrace_3010_ = v_hasTrace_3555_;
v_inheritedTraceOptions_3011_ = v_inheritedTraceOptions_3556_;
v___y_3012_ = v___y_3557_;
v_a_3013_ = v_a_3561_;
goto v___jp_3003_;
}
else
{
lean_object* v_a_3562_; uint8_t v___x_3563_; 
lean_dec(v_a_2799_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3562_ = lean_ctor_get(v___x_3560_, 0);
lean_inc(v_a_3562_);
lean_dec_ref_known(v___x_3560_, 1);
v___x_3563_ = l_Lean_Exception_isInterrupt(v_a_3562_);
if (v___x_3563_ == 0)
{
uint8_t v___x_3564_; 
lean_inc(v_a_3562_);
v___x_3564_ = l_Lean_Exception_isRuntime(v_a_3562_);
v___y_3526_ = v___y_3552_;
v___y_3527_ = v___y_3551_;
v___y_3528_ = v___x_3559_;
v___y_3529_ = v_a_3562_;
v___y_3530_ = v___x_3558_;
v___y_3531_ = v___y_3553_;
v___y_3532_ = v___y_3557_;
v___y_3533_ = v___x_3564_;
goto v___jp_3525_;
}
else
{
v___y_3526_ = v___y_3552_;
v___y_3527_ = v___y_3551_;
v___y_3528_ = v___x_3559_;
v___y_3529_ = v_a_3562_;
v___y_3530_ = v___x_3558_;
v___y_3531_ = v___y_3553_;
v___y_3532_ = v___y_3557_;
v___y_3533_ = v___x_3563_;
goto v___jp_3525_;
}
}
}
else
{
lean_object* v___x_3565_; uint8_t v___x_3566_; 
lean_del_object(v___x_2793_);
v___x_3565_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__18);
v___x_3566_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3556_, v_options_3554_, v___x_3565_);
if (v___x_3566_ == 0)
{
lean_object* v___x_3567_; uint8_t v___x_3568_; 
v___x_3567_ = l_Lean_trace_profiler;
v___x_3568_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace_spec__1(v_options_3554_, v___x_3567_);
if (v___x_3568_ == 0)
{
lean_object* v___x_3569_; 
lean_inc(v___y_3557_);
lean_inc_ref(v___y_3553_);
lean_inc(v___y_3552_);
lean_inc_ref(v___y_3551_);
v___x_3569_ = lean_apply_7(v_oracle_2696_, v___y_3550_, v___y_3549_, v___y_3551_, v___y_3552_, v___y_3553_, v___y_3557_, lean_box(0));
if (lean_obj_tag(v___x_3569_) == 0)
{
lean_object* v_a_3570_; 
lean_del_object(v___x_2801_);
v_a_3570_ = lean_ctor_get(v___x_3569_, 0);
lean_inc(v_a_3570_);
lean_dec_ref_known(v___x_3569_, 1);
v___y_3450_ = v___y_3552_;
v___y_3451_ = v___y_3551_;
v___y_3452_ = v___x_3559_;
v___y_3453_ = v___x_3558_;
v___y_3454_ = v___y_3553_;
v_options_3455_ = v_options_3554_;
v_hasTrace_3456_ = v_hasTrace_3555_;
v_inheritedTraceOptions_3457_ = v_inheritedTraceOptions_3556_;
v___y_3458_ = v___y_3557_;
v_a_3459_ = v_a_3570_;
goto v___jp_3449_;
}
else
{
lean_object* v_a_3571_; uint8_t v___x_3572_; 
lean_dec(v_a_2799_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_3571_ = lean_ctor_get(v___x_3569_, 0);
lean_inc(v_a_3571_);
lean_dec_ref_known(v___x_3569_, 1);
v___x_3572_ = l_Lean_Exception_isInterrupt(v_a_3571_);
if (v___x_3572_ == 0)
{
uint8_t v___x_3573_; 
lean_inc(v_a_3571_);
v___x_3573_ = l_Lean_Exception_isRuntime(v_a_3571_);
v___y_3496_ = v_inheritedTraceOptions_3556_;
v___y_3497_ = v_hasTrace_3555_;
v___y_3498_ = v___y_3552_;
v___y_3499_ = v___y_3551_;
v___y_3500_ = v___x_3559_;
v___y_3501_ = v___x_3558_;
v___y_3502_ = v_options_3554_;
v___y_3503_ = v___y_3553_;
v___y_3504_ = v___y_3557_;
v___y_3505_ = v_a_3571_;
v___y_3506_ = v___x_3573_;
goto v___jp_3495_;
}
else
{
v___y_3496_ = v_inheritedTraceOptions_3556_;
v___y_3497_ = v_hasTrace_3555_;
v___y_3498_ = v___y_3552_;
v___y_3499_ = v___y_3551_;
v___y_3500_ = v___x_3559_;
v___y_3501_ = v___x_3558_;
v___y_3502_ = v_options_3554_;
v___y_3503_ = v___y_3553_;
v___y_3504_ = v___y_3557_;
v___y_3505_ = v_a_3571_;
v___y_3506_ = v___x_3572_;
goto v___jp_3495_;
}
}
}
else
{
lean_del_object(v___x_2801_);
v___y_3421_ = v_inheritedTraceOptions_3556_;
v___y_3422_ = v_hasTrace_3555_;
v___y_3423_ = v___y_3552_;
v___y_3424_ = v___y_3549_;
v___y_3425_ = v___y_3550_;
v___y_3426_ = v___y_3551_;
v___y_3427_ = v___x_3559_;
v___y_3428_ = v___y_3553_;
v___y_3429_ = v___x_3566_;
v___y_3430_ = v___x_3558_;
v___y_3431_ = v_options_3554_;
v___y_3432_ = v___y_3557_;
goto v___jp_3420_;
}
}
else
{
lean_del_object(v___x_2801_);
v___y_3421_ = v_inheritedTraceOptions_3556_;
v___y_3422_ = v_hasTrace_3555_;
v___y_3423_ = v___y_3552_;
v___y_3424_ = v___y_3549_;
v___y_3425_ = v___y_3550_;
v___y_3426_ = v___y_3551_;
v___y_3427_ = v___x_3559_;
v___y_3428_ = v___y_3553_;
v___y_3429_ = v___x_3566_;
v___y_3430_ = v___x_3558_;
v___y_3431_ = v_options_3554_;
v___y_3432_ = v___y_3557_;
goto v___jp_3420_;
}
}
}
v___jp_3575_:
{
lean_object* v___x_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; 
v___x_3580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__19));
v___x_3581_ = lean_box(v_transparency_2695_);
v___x_3582_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___boxed), 7, 2);
lean_closure_set(v___x_3582_, 0, v___x_3581_);
lean_closure_set(v___x_3582_, 1, v___x_3547_);
v___x_3583_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_3580_, v___x_3582_, v___y_3576_, v___y_3577_, v___y_3578_, v___y_3579_);
if (lean_obj_tag(v___x_3583_) == 0)
{
lean_object* v_a_3584_; lean_object* v_options_3585_; uint8_t v_hasTrace_3586_; 
v_a_3584_ = lean_ctor_get(v___x_3583_, 0);
lean_inc(v_a_3584_);
lean_dec_ref_known(v___x_3583_, 1);
v_options_3585_ = lean_ctor_get(v___y_3578_, 2);
v_hasTrace_3586_ = lean_ctor_get_uint8(v_options_3585_, sizeof(void*)*1);
if (v_hasTrace_3586_ == 0)
{
lean_object* v_fst_3587_; lean_object* v_snd_3588_; lean_object* v_inheritedTraceOptions_3589_; 
v_fst_3587_ = lean_ctor_get(v_a_3584_, 0);
lean_inc(v_fst_3587_);
v_snd_3588_ = lean_ctor_get(v_a_3584_, 1);
lean_inc(v_snd_3588_);
lean_dec(v_a_3584_);
v_inheritedTraceOptions_3589_ = lean_ctor_get(v___y_3578_, 13);
v___y_3549_ = v_snd_3588_;
v___y_3550_ = v_fst_3587_;
v___y_3551_ = v___y_3576_;
v___y_3552_ = v___y_3577_;
v___y_3553_ = v___y_3578_;
v_options_3554_ = v_options_3585_;
v_hasTrace_3555_ = v_hasTrace_3586_;
v_inheritedTraceOptions_3556_ = v_inheritedTraceOptions_3589_;
v___y_3557_ = v___y_3579_;
goto v___jp_3548_;
}
else
{
lean_object* v_fst_3590_; lean_object* v_snd_3591_; lean_object* v___x_3593_; uint8_t v_isShared_3594_; uint8_t v_isSharedCheck_3614_; 
v_fst_3590_ = lean_ctor_get(v_a_3584_, 0);
v_snd_3591_ = lean_ctor_get(v_a_3584_, 1);
v_isSharedCheck_3614_ = !lean_is_exclusive(v_a_3584_);
if (v_isSharedCheck_3614_ == 0)
{
v___x_3593_ = v_a_3584_;
v_isShared_3594_ = v_isSharedCheck_3614_;
goto v_resetjp_3592_;
}
else
{
lean_inc(v_snd_3591_);
lean_inc(v_fst_3590_);
lean_dec(v_a_3584_);
v___x_3593_ = lean_box(0);
v_isShared_3594_ = v_isSharedCheck_3614_;
goto v_resetjp_3592_;
}
v_resetjp_3592_:
{
lean_object* v_inheritedTraceOptions_3595_; lean_object* v___x_3596_; uint8_t v___x_3597_; 
v_inheritedTraceOptions_3595_ = lean_ctor_get(v___y_3578_, 13);
v___x_3596_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg___closed__6);
v___x_3597_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3595_, v_options_3585_, v___x_3596_);
if (v___x_3597_ == 0)
{
lean_del_object(v___x_3593_);
v___y_3549_ = v_snd_3591_;
v___y_3550_ = v_fst_3590_;
v___y_3551_ = v___y_3576_;
v___y_3552_ = v___y_3577_;
v___y_3553_ = v___y_3578_;
v_options_3554_ = v_options_3585_;
v_hasTrace_3555_ = v_hasTrace_3586_;
v_inheritedTraceOptions_3556_ = v_inheritedTraceOptions_3595_;
v___y_3557_ = v___y_3579_;
goto v___jp_3548_;
}
else
{
lean_object* v___x_3598_; lean_object* v___x_3599_; lean_object* v___x_3600_; lean_object* v___x_3601_; lean_object* v___x_3603_; 
v___x_3598_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21, &lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__21);
lean_inc(v_fst_3590_);
v___x_3599_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__15(v_fst_3590_, v___x_2946_);
v___x_3600_ = l_Lean_MessageData_ofList(v___x_3599_);
v___x_3601_ = l_Lean_indentD(v___x_3600_);
if (v_isShared_3594_ == 0)
{
lean_ctor_set_tag(v___x_3593_, 7);
lean_ctor_set(v___x_3593_, 1, v___x_3601_);
lean_ctor_set(v___x_3593_, 0, v___x_3598_);
v___x_3603_ = v___x_3593_;
goto v_reusejp_3602_;
}
else
{
lean_object* v_reuseFailAlloc_3613_; 
v_reuseFailAlloc_3613_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3613_, 0, v___x_3598_);
lean_ctor_set(v_reuseFailAlloc_3613_, 1, v___x_3601_);
v___x_3603_ = v_reuseFailAlloc_3613_;
goto v_reusejp_3602_;
}
v_reusejp_3602_:
{
lean_object* v___x_3604_; 
v___x_3604_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__9(v___x_3574_, v___x_3603_, v___y_3576_, v___y_3577_, v___y_3578_, v___y_3579_);
if (lean_obj_tag(v___x_3604_) == 0)
{
lean_dec_ref_known(v___x_3604_, 1);
v___y_3549_ = v_snd_3591_;
v___y_3550_ = v_fst_3590_;
v___y_3551_ = v___y_3576_;
v___y_3552_ = v___y_3577_;
v___y_3553_ = v___y_3578_;
v_options_3554_ = v_options_3585_;
v_hasTrace_3555_ = v_hasTrace_3586_;
v_inheritedTraceOptions_3556_ = v_inheritedTraceOptions_3595_;
v___y_3557_ = v___y_3579_;
goto v___jp_3548_;
}
else
{
lean_object* v_a_3605_; lean_object* v___x_3607_; uint8_t v_isShared_3608_; uint8_t v_isSharedCheck_3612_; 
lean_dec(v_snd_3591_);
lean_dec(v_fst_3590_);
lean_del_object(v___x_2801_);
lean_dec(v_a_2799_);
lean_del_object(v___x_2793_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3605_ = lean_ctor_get(v___x_3604_, 0);
v_isSharedCheck_3612_ = !lean_is_exclusive(v___x_3604_);
if (v_isSharedCheck_3612_ == 0)
{
v___x_3607_ = v___x_3604_;
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
else
{
lean_inc(v_a_3605_);
lean_dec(v___x_3604_);
v___x_3607_ = lean_box(0);
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
v_resetjp_3606_:
{
lean_object* v___x_3610_; 
if (v_isShared_3608_ == 0)
{
v___x_3610_ = v___x_3607_;
goto v_reusejp_3609_;
}
else
{
lean_object* v_reuseFailAlloc_3611_; 
v_reuseFailAlloc_3611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3611_, 0, v_a_3605_);
v___x_3610_ = v_reuseFailAlloc_3611_;
goto v_reusejp_3609_;
}
v_reusejp_3609_:
{
return v___x_3610_;
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
lean_object* v_a_3615_; lean_object* v___x_3617_; uint8_t v_isShared_3618_; uint8_t v_isSharedCheck_3622_; 
lean_del_object(v___x_2801_);
lean_dec(v_a_2799_);
lean_del_object(v___x_2793_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3615_ = lean_ctor_get(v___x_3583_, 0);
v_isSharedCheck_3622_ = !lean_is_exclusive(v___x_3583_);
if (v_isSharedCheck_3622_ == 0)
{
v___x_3617_ = v___x_3583_;
v_isShared_3618_ = v_isSharedCheck_3622_;
goto v_resetjp_3616_;
}
else
{
lean_inc(v_a_3615_);
lean_dec(v___x_3583_);
v___x_3617_ = lean_box(0);
v_isShared_3618_ = v_isSharedCheck_3622_;
goto v_resetjp_3616_;
}
v_resetjp_3616_:
{
lean_object* v___x_3620_; 
if (v_isShared_3618_ == 0)
{
v___x_3620_ = v___x_3617_;
goto v_reusejp_3619_;
}
else
{
lean_object* v_reuseFailAlloc_3621_; 
v_reuseFailAlloc_3621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3621_, 0, v_a_3615_);
v___x_3620_ = v_reuseFailAlloc_3621_;
goto v_reusejp_3619_;
}
v_reusejp_3619_:
{
return v___x_3620_;
}
}
}
}
}
}
else
{
lean_object* v_a_3650_; lean_object* v___x_3652_; uint8_t v_isShared_3653_; uint8_t v_isSharedCheck_3657_; 
lean_del_object(v___x_2793_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3650_ = lean_ctor_get(v___x_2797_, 0);
v_isSharedCheck_3657_ = !lean_is_exclusive(v___x_2797_);
if (v_isSharedCheck_3657_ == 0)
{
v___x_3652_ = v___x_2797_;
v_isShared_3653_ = v_isSharedCheck_3657_;
goto v_resetjp_3651_;
}
else
{
lean_inc(v_a_3650_);
lean_dec(v___x_2797_);
v___x_3652_ = lean_box(0);
v_isShared_3653_ = v_isSharedCheck_3657_;
goto v_resetjp_3651_;
}
v_resetjp_3651_:
{
lean_object* v___x_3655_; 
if (v_isShared_3653_ == 0)
{
v___x_3655_ = v___x_3652_;
goto v_reusejp_3654_;
}
else
{
lean_object* v_reuseFailAlloc_3656_; 
v_reuseFailAlloc_3656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3656_, 0, v_a_3650_);
v___x_3655_ = v_reuseFailAlloc_3656_;
goto v_reusejp_3654_;
}
v_reusejp_3654_:
{
return v___x_3655_;
}
}
}
}
}
else
{
lean_object* v_a_3659_; lean_object* v___x_3661_; uint8_t v_isShared_3662_; uint8_t v_isSharedCheck_3666_; 
lean_dec(v_head_2707_);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3659_ = lean_ctor_get(v___x_2790_, 0);
v_isSharedCheck_3666_ = !lean_is_exclusive(v___x_2790_);
if (v_isSharedCheck_3666_ == 0)
{
v___x_3661_ = v___x_2790_;
v_isShared_3662_ = v_isSharedCheck_3666_;
goto v_resetjp_3660_;
}
else
{
lean_inc(v_a_3659_);
lean_dec(v___x_2790_);
v___x_3661_ = lean_box(0);
v_isShared_3662_ = v_isSharedCheck_3666_;
goto v_resetjp_3660_;
}
v_resetjp_3660_:
{
lean_object* v___x_3664_; 
if (v_isShared_3662_ == 0)
{
v___x_3664_ = v___x_3661_;
goto v_reusejp_3663_;
}
else
{
lean_object* v_reuseFailAlloc_3665_; 
v_reuseFailAlloc_3665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3665_, 0, v_a_3659_);
v___x_3664_ = v_reuseFailAlloc_3665_;
goto v_reusejp_3663_;
}
v_reusejp_3663_:
{
return v___x_3664_;
}
}
}
}
else
{
lean_object* v_a_3667_; lean_object* v___x_3669_; uint8_t v_isShared_3670_; uint8_t v_isSharedCheck_3674_; 
lean_dec(v_head_2707_);
lean_dec_ref_known(v_x_2699_, 2);
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
lean_dec_ref(v_oracle_2696_);
v_a_3667_ = lean_ctor_get(v___x_2785_, 0);
v_isSharedCheck_3674_ = !lean_is_exclusive(v___x_2785_);
if (v_isSharedCheck_3674_ == 0)
{
v___x_3669_ = v___x_2785_;
v_isShared_3670_ = v_isSharedCheck_3674_;
goto v_resetjp_3668_;
}
else
{
lean_inc(v_a_3667_);
lean_dec(v___x_2785_);
v___x_3669_ = lean_box(0);
v_isShared_3670_ = v_isSharedCheck_3674_;
goto v_resetjp_3668_;
}
v_resetjp_3668_:
{
lean_object* v___x_3672_; 
if (v_isShared_3670_ == 0)
{
v___x_3672_ = v___x_3669_;
goto v_reusejp_3671_;
}
else
{
lean_object* v_reuseFailAlloc_3673_; 
v_reuseFailAlloc_3673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3673_, 0, v_a_3667_);
v___x_3672_ = v_reuseFailAlloc_3673_;
goto v_reusejp_3671_;
}
v_reusejp_3671_:
{
return v___x_3672_;
}
}
}
v___jp_2712_:
{
lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; 
v___x_2720_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__4));
v___x_2721_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_proveEqZeroUsing___boxed), 7, 2);
lean_closure_set(v___x_2721_, 0, v_discharger_2697_);
lean_closure_set(v___x_2721_, 1, v_fst_2717_);
v___x_2722_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_2720_, v___x_2721_, v___y_2714_, v___y_2713_, v___y_2715_, v___y_2716_);
if (lean_obj_tag(v___x_2722_) == 0)
{
lean_object* v_a_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; 
v_a_2723_ = lean_ctor_get(v___x_2722_, 0);
lean_inc(v_a_2723_);
lean_dec_ref_known(v___x_2722_, 1);
v___x_2724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__5));
v___x_2725_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_mkLTZeroProof___boxed), 6, 1);
lean_closure_set(v___x_2725_, 0, v_fst_2718_);
v___x_2726_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_2724_, v___x_2725_, v___y_2714_, v___y_2713_, v___y_2715_, v___y_2716_);
if (lean_obj_tag(v___x_2726_) == 0)
{
lean_object* v_a_2727_; lean_object* v___x_2728_; lean_object* v___f_2729_; lean_object* v___x_2730_; lean_object* v___x_2731_; 
v_a_2727_ = lean_ctor_get(v___x_2726_, 0);
lean_inc(v_a_2727_);
lean_dec_ref_known(v___x_2726_, 1);
v___x_2728_ = lean_box(v___x_2711_);
v___f_2729_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___lam__6___boxed), 12, 7);
lean_closure_set(v___f_2729_, 0, v_a_2727_);
lean_closure_set(v___f_2729_, 1, v___x_2728_);
lean_closure_set(v___f_2729_, 2, v_x_2698_);
lean_closure_set(v___f_2729_, 3, v_a_2723_);
lean_closure_set(v___f_2729_, 4, v___x_2708_);
lean_closure_set(v___f_2729_, 5, v___x_2709_);
lean_closure_set(v___f_2729_, 6, v___x_2710_);
v___x_2730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___closed__6));
v___x_2731_ = lp_mathlib___private_Mathlib_Tactic_Linarith_Verification_0__Mathlib_Tactic_Linarith_proveFalseByLinarith_detailTrace___redArg(v___x_2730_, v___f_2729_, v___y_2714_, v___y_2713_, v___y_2715_, v___y_2716_);
if (lean_obj_tag(v___x_2731_) == 0)
{
lean_object* v_a_2732_; lean_object* v___x_2734_; uint8_t v_isShared_2735_; uint8_t v_isSharedCheck_2740_; 
v_a_2732_ = lean_ctor_get(v___x_2731_, 0);
v_isSharedCheck_2740_ = !lean_is_exclusive(v___x_2731_);
if (v_isSharedCheck_2740_ == 0)
{
v___x_2734_ = v___x_2731_;
v_isShared_2735_ = v_isSharedCheck_2740_;
goto v_resetjp_2733_;
}
else
{
lean_inc(v_a_2732_);
lean_dec(v___x_2731_);
v___x_2734_ = lean_box(0);
v_isShared_2735_ = v_isSharedCheck_2740_;
goto v_resetjp_2733_;
}
v_resetjp_2733_:
{
lean_object* v___x_2736_; lean_object* v___x_2738_; 
v___x_2736_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2736_, 0, v_a_2732_);
lean_ctor_set(v___x_2736_, 1, v_snd_2719_);
if (v_isShared_2735_ == 0)
{
lean_ctor_set(v___x_2734_, 0, v___x_2736_);
v___x_2738_ = v___x_2734_;
goto v_reusejp_2737_;
}
else
{
lean_object* v_reuseFailAlloc_2739_; 
v_reuseFailAlloc_2739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2739_, 0, v___x_2736_);
v___x_2738_ = v_reuseFailAlloc_2739_;
goto v_reusejp_2737_;
}
v_reusejp_2737_:
{
return v___x_2738_;
}
}
}
else
{
lean_object* v_a_2741_; lean_object* v___x_2743_; uint8_t v_isShared_2744_; uint8_t v_isSharedCheck_2748_; 
lean_dec(v_snd_2719_);
v_a_2741_ = lean_ctor_get(v___x_2731_, 0);
v_isSharedCheck_2748_ = !lean_is_exclusive(v___x_2731_);
if (v_isSharedCheck_2748_ == 0)
{
v___x_2743_ = v___x_2731_;
v_isShared_2744_ = v_isSharedCheck_2748_;
goto v_resetjp_2742_;
}
else
{
lean_inc(v_a_2741_);
lean_dec(v___x_2731_);
v___x_2743_ = lean_box(0);
v_isShared_2744_ = v_isSharedCheck_2748_;
goto v_resetjp_2742_;
}
v_resetjp_2742_:
{
lean_object* v___x_2746_; 
if (v_isShared_2744_ == 0)
{
v___x_2746_ = v___x_2743_;
goto v_reusejp_2745_;
}
else
{
lean_object* v_reuseFailAlloc_2747_; 
v_reuseFailAlloc_2747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2747_, 0, v_a_2741_);
v___x_2746_ = v_reuseFailAlloc_2747_;
goto v_reusejp_2745_;
}
v_reusejp_2745_:
{
return v___x_2746_;
}
}
}
}
else
{
lean_object* v_a_2749_; lean_object* v___x_2751_; uint8_t v_isShared_2752_; uint8_t v_isSharedCheck_2756_; 
lean_dec(v_a_2723_);
lean_dec(v_snd_2719_);
lean_dec(v_x_2698_);
v_a_2749_ = lean_ctor_get(v___x_2726_, 0);
v_isSharedCheck_2756_ = !lean_is_exclusive(v___x_2726_);
if (v_isSharedCheck_2756_ == 0)
{
v___x_2751_ = v___x_2726_;
v_isShared_2752_ = v_isSharedCheck_2756_;
goto v_resetjp_2750_;
}
else
{
lean_inc(v_a_2749_);
lean_dec(v___x_2726_);
v___x_2751_ = lean_box(0);
v_isShared_2752_ = v_isSharedCheck_2756_;
goto v_resetjp_2750_;
}
v_resetjp_2750_:
{
lean_object* v___x_2754_; 
if (v_isShared_2752_ == 0)
{
v___x_2754_ = v___x_2751_;
goto v_reusejp_2753_;
}
else
{
lean_object* v_reuseFailAlloc_2755_; 
v_reuseFailAlloc_2755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2755_, 0, v_a_2749_);
v___x_2754_ = v_reuseFailAlloc_2755_;
goto v_reusejp_2753_;
}
v_reusejp_2753_:
{
return v___x_2754_;
}
}
}
}
else
{
lean_object* v_a_2757_; lean_object* v___x_2759_; uint8_t v_isShared_2760_; uint8_t v_isSharedCheck_2764_; 
lean_dec(v_snd_2719_);
lean_dec(v_fst_2718_);
lean_dec(v_x_2698_);
v_a_2757_ = lean_ctor_get(v___x_2722_, 0);
v_isSharedCheck_2764_ = !lean_is_exclusive(v___x_2722_);
if (v_isSharedCheck_2764_ == 0)
{
v___x_2759_ = v___x_2722_;
v_isShared_2760_ = v_isSharedCheck_2764_;
goto v_resetjp_2758_;
}
else
{
lean_inc(v_a_2757_);
lean_dec(v___x_2722_);
v___x_2759_ = lean_box(0);
v_isShared_2760_ = v_isSharedCheck_2764_;
goto v_resetjp_2758_;
}
v_resetjp_2758_:
{
lean_object* v___x_2762_; 
if (v_isShared_2760_ == 0)
{
v___x_2762_ = v___x_2759_;
goto v_reusejp_2761_;
}
else
{
lean_object* v_reuseFailAlloc_2763_; 
v_reuseFailAlloc_2763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2763_, 0, v_a_2757_);
v___x_2762_ = v_reuseFailAlloc_2763_;
goto v_reusejp_2761_;
}
v_reusejp_2761_:
{
return v___x_2762_;
}
}
}
}
v___jp_2765_:
{
if (lean_obj_tag(v___y_2770_) == 0)
{
lean_object* v_a_2771_; lean_object* v_snd_2772_; lean_object* v_fst_2773_; lean_object* v_fst_2774_; lean_object* v_snd_2775_; 
v_a_2771_ = lean_ctor_get(v___y_2770_, 0);
lean_inc(v_a_2771_);
lean_dec_ref_known(v___y_2770_, 1);
v_snd_2772_ = lean_ctor_get(v_a_2771_, 1);
lean_inc(v_snd_2772_);
v_fst_2773_ = lean_ctor_get(v_a_2771_, 0);
lean_inc(v_fst_2773_);
lean_dec(v_a_2771_);
v_fst_2774_ = lean_ctor_get(v_snd_2772_, 0);
lean_inc(v_fst_2774_);
v_snd_2775_ = lean_ctor_get(v_snd_2772_, 1);
lean_inc(v_snd_2775_);
lean_dec(v_snd_2772_);
v___y_2713_ = v___y_2766_;
v___y_2714_ = v___y_2767_;
v___y_2715_ = v___y_2768_;
v___y_2716_ = v___y_2769_;
v_fst_2717_ = v_fst_2773_;
v_fst_2718_ = v_fst_2774_;
v_snd_2719_ = v_snd_2775_;
goto v___jp_2712_;
}
else
{
lean_object* v_a_2776_; lean_object* v___x_2778_; uint8_t v_isShared_2779_; uint8_t v_isSharedCheck_2783_; 
lean_dec(v_x_2698_);
lean_dec_ref(v_discharger_2697_);
v_a_2776_ = lean_ctor_get(v___y_2770_, 0);
v_isSharedCheck_2783_ = !lean_is_exclusive(v___y_2770_);
if (v_isSharedCheck_2783_ == 0)
{
v___x_2778_ = v___y_2770_;
v_isShared_2779_ = v_isSharedCheck_2783_;
goto v_resetjp_2777_;
}
else
{
lean_inc(v_a_2776_);
lean_dec(v___y_2770_);
v___x_2778_ = lean_box(0);
v_isShared_2779_ = v_isSharedCheck_2783_;
goto v_resetjp_2777_;
}
v_resetjp_2777_:
{
lean_object* v___x_2781_; 
if (v_isShared_2779_ == 0)
{
v___x_2781_ = v___x_2778_;
goto v_reusejp_2780_;
}
else
{
lean_object* v_reuseFailAlloc_2782_; 
v_reuseFailAlloc_2782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2782_, 0, v_a_2776_);
v___x_2781_ = v_reuseFailAlloc_2782_;
goto v_reusejp_2780_;
}
v_reusejp_2780_:
{
return v___x_2781_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith___boxed(lean_object* v_transparency_3675_, lean_object* v_oracle_3676_, lean_object* v_discharger_3677_, lean_object* v_x_3678_, lean_object* v_x_3679_, lean_object* v_a_3680_, lean_object* v_a_3681_, lean_object* v_a_3682_, lean_object* v_a_3683_, lean_object* v_a_3684_){
_start:
{
uint8_t v_transparency_boxed_3685_; lean_object* v_res_3686_; 
v_transparency_boxed_3685_ = lean_unbox(v_transparency_3675_);
v_res_3686_ = lp_mathlib_Mathlib_Tactic_Linarith_proveFalseByLinarith(v_transparency_boxed_3685_, v_oracle_3676_, v_discharger_3677_, v_x_3678_, v_x_3679_, v_a_3680_, v_a_3681_, v_a_3682_, v_a_3683_);
lean_dec(v_a_3683_);
lean_dec_ref(v_a_3682_);
lean_dec(v_a_3681_);
lean_dec_ref(v_a_3680_);
return v_res_3686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2(lean_object* v_00_u03b2_3687_, lean_object* v_m_3688_, lean_object* v_a_3689_){
_start:
{
lean_object* v___x_3690_; 
v___x_3690_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___redArg(v_m_3688_, v_a_3689_);
return v___x_3690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2___boxed(lean_object* v_00_u03b2_3691_, lean_object* v_m_3692_, lean_object* v_a_3693_){
_start:
{
lean_object* v_res_3694_; 
v_res_3694_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2(v_00_u03b2_3691_, v_m_3692_, v_a_3693_);
lean_dec(v_a_3693_);
lean_dec_ref(v_m_3692_);
return v_res_3694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00List_format___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__14_spec__18(lean_object* v_a_3695_){
_start:
{
lean_object* v___x_3696_; 
v___x_3696_ = lean_nat_to_int(v_a_3695_);
return v___x_3696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2(lean_object* v_00_u03b2_3697_, lean_object* v_a_3698_, lean_object* v_x_3699_){
_start:
{
lean_object* v___x_3700_; 
v___x_3700_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___redArg(v_a_3698_, v_x_3699_);
return v___x_3700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2___boxed(lean_object* v_00_u03b2_3701_, lean_object* v_a_3702_, lean_object* v_x_3703_){
_start:
{
lean_object* v_res_3704_; 
v_res_3704_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_Linarith_proveFalseByLinarith_spec__2_spec__2(v_00_u03b2_3701_, v_a_3702_, v_x_3703_);
lean_dec(v_x_3703_);
lean_dec(v_a_3702_);
return v_res_3704_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Verification(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Verification(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Verification(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Verification(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Verification(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Verification(builtin);
}
#ifdef __cplusplus
}
#endif
