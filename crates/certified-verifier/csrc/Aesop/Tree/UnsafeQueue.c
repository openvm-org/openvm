// Lean compiler output
// Module: Aesop.Tree.UnsafeQueue
// Imports: public import Init public meta import Init public import Aesop.Rule import Aesop.Constants
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
extern double lp_aesop_Aesop_postponedSafeRuleSuccessProbability;
uint8_t lean_float_decLt(double, double);
uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
double lean_float_sub(double, double);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Subarray_empty(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
extern lean_object* lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
lean_object* lp_aesop_Aesop_instInhabitedRule_default___redArg(lean_object*);
extern double lp_aesop_Aesop_instInhabitedPercent_default;
lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleName_compare___boxed(lean_object*, lean_object*);
lean_object* l_compareOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_compareLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedRuleTacOutput_default;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPostponedSafeRule_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPostponedSafeRule;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_unsafeRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_unsafeRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_postponedSafeRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_postponedSafeRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnsafeQueueEntry;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12_value;
static const lean_string_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instToString___closed__0_value;
LEAN_EXPORT double lp_aesop_Aesop_UnsafeQueueEntry_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_successProbability___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_name___boxed(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_compare___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnsafeQueueEntry_name___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__2 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_compareOn___boxed, .m_arity = 6, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__1_value),((lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__2_value)} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__3 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_compareLex___boxed, .m_arity = 6, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__0_value),((lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__3_value)} };
static const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__4 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__4_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd = (const lean_object*)&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_instEmptyCollection;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_instInhabited___aux__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_instInhabited;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnsafeQueue_initial___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnsafeQueue_initial___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnsafeQueue_initial___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueue_initial___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0___redArg(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_UnsafeQueue_entriesToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnsafeQueue_entriesToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnsafeQueue_entriesToMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_entriesToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
v___x_2_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lp_aesop_Aesop_instInhabitedRuleTacOutput_default;
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0, &lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__0);
v___x_5_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5_, 0, v___x_4_);
lean_ctor_set(v___x_5_, 1, v___x_3_);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1, &lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default___closed__1);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_aesop_Aesop_instInhabitedPostponedSafeRule_default;
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1(void){
_start:
{
double v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lp_aesop_Aesop_postponedSafeRuleSuccessProbability;
v___x_9_ = lean_box_float(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule(lean_object* v_r_10_){
_start:
{
lean_object* v_rule_11_; lean_object* v_name_12_; lean_object* v_indexingMode_13_; lean_object* v_pattern_x3f_14_; lean_object* v_tac_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_23_; 
v_rule_11_ = lean_ctor_get(v_r_10_, 0);
lean_inc_ref(v_rule_11_);
lean_dec_ref(v_r_10_);
v_name_12_ = lean_ctor_get(v_rule_11_, 0);
v_indexingMode_13_ = lean_ctor_get(v_rule_11_, 1);
v_pattern_x3f_14_ = lean_ctor_get(v_rule_11_, 2);
v_tac_15_ = lean_ctor_get(v_rule_11_, 4);
v_isSharedCheck_23_ = !lean_is_exclusive(v_rule_11_);
if (v_isSharedCheck_23_ == 0)
{
lean_object* v_unused_24_; 
v_unused_24_ = lean_ctor_get(v_rule_11_, 3);
lean_dec(v_unused_24_);
v___x_17_ = v_rule_11_;
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_tac_15_);
lean_inc(v_pattern_x3f_14_);
lean_inc(v_indexingMode_13_);
lean_inc(v_name_12_);
lean_dec(v_rule_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_19_ = lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1;
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 3, v___x_19_);
v___x_21_ = v___x_17_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v_name_12_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v_indexingMode_13_);
lean_ctor_set(v_reuseFailAlloc_22_, 2, v_pattern_x3f_14_);
lean_ctor_set(v_reuseFailAlloc_22_, 3, v___x_19_);
lean_ctor_set(v_reuseFailAlloc_22_, 4, v_tac_15_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorIdx(lean_object* v_x_25_){
_start:
{
if (lean_obj_tag(v_x_25_) == 0)
{
lean_object* v___x_26_; 
v___x_26_ = lean_unsigned_to_nat(0u);
return v___x_26_;
}
else
{
lean_object* v___x_27_; 
v___x_27_ = lean_unsigned_to_nat(1u);
return v___x_27_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorIdx___boxed(lean_object* v_x_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorIdx(v_x_28_);
lean_dec_ref(v_x_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(lean_object* v_t_30_, lean_object* v_k_31_){
_start:
{
lean_object* v_r_32_; lean_object* v___x_33_; 
v_r_32_ = lean_ctor_get(v_t_30_, 0);
lean_inc_ref(v_r_32_);
lean_dec_ref(v_t_30_);
v___x_33_ = lean_apply_1(v_k_31_, v_r_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim(lean_object* v_motive_34_, lean_object* v_ctorIdx_35_, lean_object* v_t_36_, lean_object* v_h_37_, lean_object* v_k_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(v_t_36_, v_k_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___boxed(lean_object* v_motive_40_, lean_object* v_ctorIdx_41_, lean_object* v_t_42_, lean_object* v_h_43_, lean_object* v_k_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim(v_motive_40_, v_ctorIdx_41_, v_t_42_, v_h_43_, v_k_44_);
lean_dec(v_ctorIdx_41_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_unsafeRule_elim___redArg(lean_object* v_t_46_, lean_object* v_unsafeRule_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(v_t_46_, v_unsafeRule_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_unsafeRule_elim(lean_object* v_motive_49_, lean_object* v_t_50_, lean_object* v_h_51_, lean_object* v_unsafeRule_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(v_t_50_, v_unsafeRule_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_postponedSafeRule_elim___redArg(lean_object* v_t_54_, lean_object* v_postponedSafeRule_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(v_t_54_, v_postponedSafeRule_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_postponedSafeRule_elim(lean_object* v_motive_57_, lean_object* v_t_58_, lean_object* v_h_59_, lean_object* v_postponedSafeRule_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_aesop_Aesop_UnsafeQueueEntry_ctorElim___redArg(v_t_58_, v_postponedSafeRule_60_);
return v___x_61_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1(void){
_start:
{
double v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lp_aesop_Aesop_instInhabitedPercent_default;
v___x_63_ = lean_box_float(v___x_62_);
return v___x_63_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1;
v___x_65_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v___x_64_);
return v___x_65_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0, &lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0);
v___x_67_ = lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(v___x_66_);
return v___x_67_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1, &lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__1);
v___x_69_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
return v___x_69_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default(void){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2, &lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__2);
return v___x_70_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry(void){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default;
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0(lean_object* v_x_86_){
_start:
{
if (lean_obj_tag(v_x_86_) == 0)
{
lean_object* v_r_87_; lean_object* v_rule_88_; lean_object* v_name_89_; lean_object* v_name_90_; uint8_t v_builder_91_; uint8_t v_phase_92_; uint8_t v_scope_93_; lean_object* v___y_95_; lean_object* v___y_96_; lean_object* v___y_97_; lean_object* v___y_104_; lean_object* v___y_105_; lean_object* v___y_106_; lean_object* v___y_112_; 
v_r_87_ = lean_ctor_get(v_x_86_, 0);
lean_inc_ref(v_r_87_);
lean_dec_ref_known(v_x_86_, 1);
v_rule_88_ = lean_ctor_get(v_r_87_, 0);
lean_inc(v_rule_88_);
lean_dec_ref(v_r_87_);
v_name_89_ = lean_ctor_get(v_rule_88_, 0);
lean_inc_ref(v_name_89_);
lean_dec(v_rule_88_);
v_name_90_ = lean_ctor_get(v_name_89_, 0);
lean_inc(v_name_90_);
v_builder_91_ = lean_ctor_get_uint8(v_name_89_, sizeof(void*)*1 + 8);
v_phase_92_ = lean_ctor_get_uint8(v_name_89_, sizeof(void*)*1 + 9);
v_scope_93_ = lean_ctor_get_uint8(v_name_89_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_89_);
switch(v_phase_92_)
{
case 0:
{
lean_object* v___x_123_; 
v___x_123_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11));
v___y_112_ = v___x_123_;
goto v___jp_111_;
}
case 1:
{
lean_object* v___x_124_; 
v___x_124_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12));
v___y_112_ = v___x_124_;
goto v___jp_111_;
}
default: 
{
lean_object* v___x_125_; 
v___x_125_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13));
v___y_112_ = v___x_125_;
goto v___jp_111_;
}
}
v___jp_94_:
{
lean_object* v___x_98_; lean_object* v___x_99_; uint8_t v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_98_ = lean_string_append(v___y_96_, v___y_97_);
v___x_99_ = lean_string_append(v___x_98_, v___y_95_);
v___x_100_ = 1;
v___x_101_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_90_, v___x_100_);
v___x_102_ = lean_string_append(v___x_99_, v___x_101_);
lean_dec_ref(v___x_101_);
return v___x_102_;
}
v___jp_103_:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = lean_string_append(v___y_104_, v___y_106_);
v___x_108_ = lean_string_append(v___x_107_, v___y_105_);
if (v_scope_93_ == 0)
{
lean_object* v___x_109_; 
v___x_109_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0));
v___y_95_ = v___y_105_;
v___y_96_ = v___x_108_;
v___y_97_ = v___x_109_;
goto v___jp_94_;
}
else
{
lean_object* v___x_110_; 
v___x_110_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1));
v___y_95_ = v___y_105_;
v___y_96_ = v___x_108_;
v___y_97_ = v___x_110_;
goto v___jp_94_;
}
}
v___jp_111_:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2));
lean_inc_ref(v___y_112_);
v___x_114_ = lean_string_append(v___y_112_, v___x_113_);
switch(v_builder_91_)
{
case 0:
{
lean_object* v___x_115_; 
v___x_115_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_115_;
goto v___jp_103_;
}
case 1:
{
lean_object* v___x_116_; 
v___x_116_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_116_;
goto v___jp_103_;
}
case 2:
{
lean_object* v___x_117_; 
v___x_117_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_117_;
goto v___jp_103_;
}
case 3:
{
lean_object* v___x_118_; 
v___x_118_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_118_;
goto v___jp_103_;
}
case 4:
{
lean_object* v___x_119_; 
v___x_119_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_119_;
goto v___jp_103_;
}
case 5:
{
lean_object* v___x_120_; 
v___x_120_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_120_;
goto v___jp_103_;
}
case 6:
{
lean_object* v___x_121_; 
v___x_121_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_121_;
goto v___jp_103_;
}
default: 
{
lean_object* v___x_122_; 
v___x_122_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10));
v___y_104_ = v___x_114_;
v___y_105_ = v___x_113_;
v___y_106_ = v___x_122_;
goto v___jp_103_;
}
}
}
}
else
{
lean_object* v_r_126_; lean_object* v_rule_127_; lean_object* v_name_128_; lean_object* v_name_129_; uint8_t v_builder_130_; uint8_t v_phase_131_; uint8_t v_scope_132_; lean_object* v___y_134_; lean_object* v___y_135_; lean_object* v___y_136_; lean_object* v___y_143_; lean_object* v___y_144_; lean_object* v___y_145_; lean_object* v___y_151_; 
v_r_126_ = lean_ctor_get(v_x_86_, 0);
lean_inc_ref(v_r_126_);
lean_dec_ref_known(v_x_86_, 1);
v_rule_127_ = lean_ctor_get(v_r_126_, 0);
lean_inc_ref(v_rule_127_);
lean_dec_ref(v_r_126_);
v_name_128_ = lean_ctor_get(v_rule_127_, 0);
lean_inc_ref(v_name_128_);
lean_dec_ref(v_rule_127_);
v_name_129_ = lean_ctor_get(v_name_128_, 0);
lean_inc(v_name_129_);
v_builder_130_ = lean_ctor_get_uint8(v_name_128_, sizeof(void*)*1 + 8);
v_phase_131_ = lean_ctor_get_uint8(v_name_128_, sizeof(void*)*1 + 9);
v_scope_132_ = lean_ctor_get_uint8(v_name_128_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_128_);
switch(v_phase_131_)
{
case 0:
{
lean_object* v___x_162_; 
v___x_162_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11));
v___y_151_ = v___x_162_;
goto v___jp_150_;
}
case 1:
{
lean_object* v___x_163_; 
v___x_163_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12));
v___y_151_ = v___x_163_;
goto v___jp_150_;
}
default: 
{
lean_object* v___x_164_; 
v___x_164_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13));
v___y_151_ = v___x_164_;
goto v___jp_150_;
}
}
v___jp_133_:
{
lean_object* v___x_137_; lean_object* v___x_138_; uint8_t v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_137_ = lean_string_append(v___y_135_, v___y_136_);
v___x_138_ = lean_string_append(v___x_137_, v___y_134_);
v___x_139_ = 1;
v___x_140_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_129_, v___x_139_);
v___x_141_ = lean_string_append(v___x_138_, v___x_140_);
lean_dec_ref(v___x_140_);
return v___x_141_;
}
v___jp_142_:
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = lean_string_append(v___y_144_, v___y_145_);
v___x_147_ = lean_string_append(v___x_146_, v___y_143_);
if (v_scope_132_ == 0)
{
lean_object* v___x_148_; 
v___x_148_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0));
v___y_134_ = v___y_143_;
v___y_135_ = v___x_147_;
v___y_136_ = v___x_148_;
goto v___jp_133_;
}
else
{
lean_object* v___x_149_; 
v___x_149_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1));
v___y_134_ = v___y_143_;
v___y_135_ = v___x_147_;
v___y_136_ = v___x_149_;
goto v___jp_133_;
}
}
v___jp_150_:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2));
lean_inc_ref(v___y_151_);
v___x_153_ = lean_string_append(v___y_151_, v___x_152_);
switch(v_builder_130_)
{
case 0:
{
lean_object* v___x_154_; 
v___x_154_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_154_;
goto v___jp_142_;
}
case 1:
{
lean_object* v___x_155_; 
v___x_155_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_155_;
goto v___jp_142_;
}
case 2:
{
lean_object* v___x_156_; 
v___x_156_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_156_;
goto v___jp_142_;
}
case 3:
{
lean_object* v___x_157_; 
v___x_157_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_157_;
goto v___jp_142_;
}
case 4:
{
lean_object* v___x_158_; 
v___x_158_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_158_;
goto v___jp_142_;
}
case 5:
{
lean_object* v___x_159_; 
v___x_159_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_159_;
goto v___jp_142_;
}
case 6:
{
lean_object* v___x_160_; 
v___x_160_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_160_;
goto v___jp_142_;
}
default: 
{
lean_object* v___x_161_; 
v___x_161_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10));
v___y_143_ = v___x_152_;
v___y_144_ = v___x_153_;
v___y_145_ = v___x_161_;
goto v___jp_142_;
}
}
}
}
}
}
LEAN_EXPORT double lp_aesop_Aesop_UnsafeQueueEntry_successProbability(lean_object* v_x_167_){
_start:
{
if (lean_obj_tag(v_x_167_) == 0)
{
lean_object* v_r_168_; lean_object* v_rule_169_; lean_object* v_extra_170_; double v___x_171_; 
v_r_168_ = lean_ctor_get(v_x_167_, 0);
v_rule_169_ = lean_ctor_get(v_r_168_, 0);
v_extra_170_ = lean_ctor_get(v_rule_169_, 3);
v___x_171_ = lean_unbox_float(v_extra_170_);
return v___x_171_;
}
else
{
double v___x_172_; 
v___x_172_ = lp_aesop_Aesop_postponedSafeRuleSuccessProbability;
return v___x_172_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_successProbability___boxed(lean_object* v_x_173_){
_start:
{
double v_res_174_; lean_object* v_r_175_; 
v_res_174_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_x_173_);
lean_dec_ref(v_x_173_);
v_r_175_ = lean_box_float(v_res_174_);
return v_r_175_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_name(lean_object* v_x_176_){
_start:
{
if (lean_obj_tag(v_x_176_) == 0)
{
lean_object* v_r_177_; lean_object* v_rule_178_; lean_object* v_name_179_; 
v_r_177_ = lean_ctor_get(v_x_176_, 0);
v_rule_178_ = lean_ctor_get(v_r_177_, 0);
v_name_179_ = lean_ctor_get(v_rule_178_, 0);
lean_inc_ref(v_name_179_);
return v_name_179_;
}
else
{
lean_object* v_r_180_; lean_object* v_rule_181_; lean_object* v_name_182_; 
v_r_180_ = lean_ctor_get(v_x_176_, 0);
v_rule_181_ = lean_ctor_get(v_r_180_, 0);
v_name_182_ = lean_ctor_get(v_rule_181_, 0);
lean_inc_ref(v_name_182_);
return v_name_182_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_name___boxed(lean_object* v_x_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_x_183_);
lean_dec_ref(v_x_183_);
return v_res_184_;
}
}
static double _init_lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0(void){
_start:
{
lean_object* v___x_185_; uint8_t v___x_186_; lean_object* v___x_187_; double v___x_188_; 
v___x_185_ = lean_unsigned_to_nat(5u);
v___x_186_ = 1;
v___x_187_ = lean_unsigned_to_nat(1u);
v___x_188_ = l_Float_ofScientific(v___x_187_, v___x_186_, v___x_185_);
return v___x_188_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0(lean_object* v_x_189_, lean_object* v_y_190_){
_start:
{
double v_toFloat_191_; double v_toFloat_192_; uint8_t v___y_194_; uint8_t v___x_199_; 
v_toFloat_191_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_x_189_);
v_toFloat_192_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_y_190_);
v___x_199_ = lean_float_decLt(v_toFloat_192_, v_toFloat_191_);
if (v___x_199_ == 0)
{
double v___x_200_; double v___x_201_; uint8_t v___x_202_; 
v___x_200_ = lean_float_sub(v_toFloat_192_, v_toFloat_191_);
v___x_201_ = lean_float_once(&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0, &lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0);
v___x_202_ = lean_float_decLt(v___x_200_, v___x_201_);
v___y_194_ = v___x_202_;
goto v___jp_193_;
}
else
{
double v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; double v___x_206_; uint8_t v___x_207_; 
v___x_203_ = lean_float_sub(v_toFloat_191_, v_toFloat_192_);
v___x_204_ = lean_unsigned_to_nat(1u);
v___x_205_ = lean_unsigned_to_nat(5u);
v___x_206_ = l_Float_ofScientific(v___x_204_, v___x_199_, v___x_205_);
v___x_207_ = lean_float_decLt(v___x_203_, v___x_206_);
v___y_194_ = v___x_207_;
goto v___jp_193_;
}
v___jp_193_:
{
if (v___y_194_ == 0)
{
uint8_t v___x_195_; 
v___x_195_ = lean_float_decLt(v_toFloat_191_, v_toFloat_192_);
if (v___x_195_ == 0)
{
uint8_t v___x_196_; 
v___x_196_ = 0;
return v___x_196_;
}
else
{
uint8_t v___x_197_; 
v___x_197_ = 2;
return v___x_197_;
}
}
else
{
uint8_t v___x_198_; 
v___x_198_ = 1;
return v___x_198_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___boxed(lean_object* v_x_208_, lean_object* v_y_209_){
_start:
{
uint8_t v_res_210_; lean_object* v_r_211_; 
v_res_210_ = lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0(v_x_208_, v_y_209_);
lean_dec_ref(v_y_209_);
lean_dec_ref(v_x_208_);
v_r_211_ = lean_box(v_res_210_);
return v_r_211_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = l_Subarray_empty(lean_box(0));
return v___x_222_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1(void){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_obj_once(&lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0, &lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0);
return v___x_223_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection(void){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0, &lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0);
return v___x_224_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnsafeQueue_instInhabited___aux__1(void){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lean_obj_once(&lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0, &lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0);
return v___x_225_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnsafeQueue_instInhabited(void){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lean_obj_once(&lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0, &lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1___closed__0);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial___lam__0(lean_object* v_x_227_, lean_object* v_x_228_){
_start:
{
lean_inc_ref(v_x_227_);
return v_x_227_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial___lam__0___boxed(lean_object* v_x_229_, lean_object* v_x_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_aesop_Aesop_UnsafeQueue_initial___lam__0(v_x_229_, v_x_230_);
lean_dec_ref(v_x_230_);
lean_dec_ref(v_x_229_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg(lean_object* v_hi_232_, lean_object* v_pivot_233_, lean_object* v_as_234_, lean_object* v_i_235_, lean_object* v_k_236_){
_start:
{
uint8_t v___x_247_; 
v___x_247_ = lean_nat_dec_lt(v_k_236_, v_hi_232_);
if (v___x_247_ == 0)
{
lean_object* v___x_248_; lean_object* v___x_249_; 
lean_dec(v_k_236_);
v___x_248_ = lean_array_fswap(v_as_234_, v_i_235_, v_hi_232_);
v___x_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_249_, 0, v_i_235_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
return v___x_249_;
}
else
{
lean_object* v___x_250_; double v_toFloat_251_; double v_toFloat_252_; uint8_t v___y_254_; uint8_t v___x_259_; 
v___x_250_ = lean_array_fget_borrowed(v_as_234_, v_k_236_);
v_toFloat_251_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v___x_250_);
v_toFloat_252_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_pivot_233_);
v___x_259_ = lean_float_decLt(v_toFloat_252_, v_toFloat_251_);
if (v___x_259_ == 0)
{
double v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; double v___x_263_; uint8_t v___x_264_; 
v___x_260_ = lean_float_sub(v_toFloat_252_, v_toFloat_251_);
v___x_261_ = lean_unsigned_to_nat(1u);
v___x_262_ = lean_unsigned_to_nat(5u);
v___x_263_ = l_Float_ofScientific(v___x_261_, v___x_247_, v___x_262_);
v___x_264_ = lean_float_decLt(v___x_260_, v___x_263_);
v___y_254_ = v___x_264_;
goto v___jp_253_;
}
else
{
double v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; double v___x_268_; uint8_t v___x_269_; 
v___x_265_ = lean_float_sub(v_toFloat_251_, v_toFloat_252_);
v___x_266_ = lean_unsigned_to_nat(1u);
v___x_267_ = lean_unsigned_to_nat(5u);
v___x_268_ = l_Float_ofScientific(v___x_266_, v___x_259_, v___x_267_);
v___x_269_ = lean_float_decLt(v___x_265_, v___x_268_);
v___y_254_ = v___x_269_;
goto v___jp_253_;
}
v___jp_253_:
{
if (v___y_254_ == 0)
{
uint8_t v___x_255_; 
v___x_255_ = lean_float_decLt(v_toFloat_251_, v_toFloat_252_);
if (v___x_255_ == 0)
{
goto v___jp_241_;
}
else
{
goto v___jp_237_;
}
}
else
{
lean_object* v___x_256_; lean_object* v___x_257_; uint8_t v___x_258_; 
v___x_256_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v___x_250_);
v___x_257_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_pivot_233_);
v___x_258_ = lp_aesop_Aesop_RuleName_compare(v___x_256_, v___x_257_);
lean_dec_ref(v___x_257_);
lean_dec_ref(v___x_256_);
if (v___x_258_ == 0)
{
goto v___jp_241_;
}
else
{
goto v___jp_237_;
}
}
}
}
v___jp_237_:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_unsigned_to_nat(1u);
v___x_239_ = lean_nat_add(v_k_236_, v___x_238_);
lean_dec(v_k_236_);
v_k_236_ = v___x_239_;
goto _start;
}
v___jp_241_:
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_242_ = lean_array_fswap(v_as_234_, v_i_235_, v_k_236_);
v___x_243_ = lean_unsigned_to_nat(1u);
v___x_244_ = lean_nat_add(v_i_235_, v___x_243_);
lean_dec(v_i_235_);
v___x_245_ = lean_nat_add(v_k_236_, v___x_243_);
lean_dec(v_k_236_);
v_as_234_ = v___x_242_;
v_i_235_ = v___x_244_;
v_k_236_ = v___x_245_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg___boxed(lean_object* v_hi_270_, lean_object* v_pivot_271_, lean_object* v_as_272_, lean_object* v_i_273_, lean_object* v_k_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg(v_hi_270_, v_pivot_271_, v_as_272_, v_i_273_, v_k_274_);
lean_dec_ref(v_pivot_271_);
lean_dec(v_hi_270_);
return v_res_275_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(uint8_t v___x_276_, lean_object* v_x_277_, lean_object* v_y_278_){
_start:
{
double v_toFloat_279_; double v_toFloat_280_; uint8_t v___y_282_; uint8_t v___x_288_; 
v_toFloat_279_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_x_277_);
v_toFloat_280_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_y_278_);
v___x_288_ = lean_float_decLt(v_toFloat_280_, v_toFloat_279_);
if (v___x_288_ == 0)
{
double v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; double v___x_292_; uint8_t v___x_293_; 
v___x_289_ = lean_float_sub(v_toFloat_280_, v_toFloat_279_);
v___x_290_ = lean_unsigned_to_nat(1u);
v___x_291_ = lean_unsigned_to_nat(5u);
v___x_292_ = l_Float_ofScientific(v___x_290_, v___x_276_, v___x_291_);
v___x_293_ = lean_float_decLt(v___x_289_, v___x_292_);
v___y_282_ = v___x_293_;
goto v___jp_281_;
}
else
{
double v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; double v___x_297_; uint8_t v___x_298_; 
v___x_294_ = lean_float_sub(v_toFloat_279_, v_toFloat_280_);
v___x_295_ = lean_unsigned_to_nat(1u);
v___x_296_ = lean_unsigned_to_nat(5u);
v___x_297_ = l_Float_ofScientific(v___x_295_, v___x_288_, v___x_296_);
v___x_298_ = lean_float_decLt(v___x_294_, v___x_297_);
v___y_282_ = v___x_298_;
goto v___jp_281_;
}
v___jp_281_:
{
if (v___y_282_ == 0)
{
uint8_t v___x_283_; 
v___x_283_ = lean_float_decLt(v_toFloat_279_, v_toFloat_280_);
if (v___x_283_ == 0)
{
return v___x_276_;
}
else
{
return v___y_282_;
}
}
else
{
lean_object* v___x_284_; lean_object* v___x_285_; uint8_t v___x_286_; 
v___x_284_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_x_277_);
v___x_285_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_y_278_);
v___x_286_ = lp_aesop_Aesop_RuleName_compare(v___x_284_, v___x_285_);
lean_dec_ref(v___x_285_);
lean_dec_ref(v___x_284_);
if (v___x_286_ == 0)
{
return v___y_282_;
}
else
{
uint8_t v___x_287_; 
v___x_287_ = 0;
return v___x_287_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0___boxed(lean_object* v___x_299_, lean_object* v_x_300_, lean_object* v_y_301_){
_start:
{
uint8_t v___x_917__boxed_302_; uint8_t v_res_303_; lean_object* v_r_304_; 
v___x_917__boxed_302_ = lean_unbox(v___x_299_);
v_res_303_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(v___x_917__boxed_302_, v_x_300_, v_y_301_);
lean_dec_ref(v_y_301_);
lean_dec_ref(v_x_300_);
v_r_304_ = lean_box(v_res_303_);
return v_r_304_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(lean_object* v_n_305_, lean_object* v_as_306_, lean_object* v_lo_307_, lean_object* v_hi_308_){
_start:
{
lean_object* v___y_310_; uint8_t v___x_320_; 
v___x_320_ = lean_nat_dec_lt(v_lo_307_, v_hi_308_);
if (v___x_320_ == 0)
{
lean_dec(v_lo_307_);
return v_as_306_;
}
else
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v_mid_323_; lean_object* v___y_325_; lean_object* v___y_331_; lean_object* v___x_336_; lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_321_ = lean_nat_add(v_lo_307_, v_hi_308_);
v___x_322_ = lean_unsigned_to_nat(1u);
v_mid_323_ = lean_nat_shiftr(v___x_321_, v___x_322_);
lean_dec(v___x_321_);
v___x_336_ = lean_array_fget_borrowed(v_as_306_, v_mid_323_);
v___x_337_ = lean_array_fget_borrowed(v_as_306_, v_lo_307_);
v___x_338_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(v___x_320_, v___x_336_, v___x_337_);
if (v___x_338_ == 0)
{
v___y_331_ = v_as_306_;
goto v___jp_330_;
}
else
{
lean_object* v___x_339_; 
v___x_339_ = lean_array_fswap(v_as_306_, v_lo_307_, v_mid_323_);
v___y_331_ = v___x_339_;
goto v___jp_330_;
}
v___jp_324_:
{
lean_object* v___x_326_; lean_object* v___x_327_; uint8_t v___x_328_; 
v___x_326_ = lean_array_fget_borrowed(v___y_325_, v_mid_323_);
v___x_327_ = lean_array_fget_borrowed(v___y_325_, v_hi_308_);
v___x_328_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(v___x_320_, v___x_326_, v___x_327_);
if (v___x_328_ == 0)
{
lean_dec(v_mid_323_);
v___y_310_ = v___y_325_;
goto v___jp_309_;
}
else
{
lean_object* v___x_329_; 
v___x_329_ = lean_array_fswap(v___y_325_, v_mid_323_, v_hi_308_);
lean_dec(v_mid_323_);
v___y_310_ = v___x_329_;
goto v___jp_309_;
}
}
v___jp_330_:
{
lean_object* v___x_332_; lean_object* v___x_333_; uint8_t v___x_334_; 
v___x_332_ = lean_array_fget_borrowed(v___y_331_, v_hi_308_);
v___x_333_ = lean_array_fget_borrowed(v___y_331_, v_lo_307_);
v___x_334_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___lam__0(v___x_320_, v___x_332_, v___x_333_);
if (v___x_334_ == 0)
{
v___y_325_ = v___y_331_;
goto v___jp_324_;
}
else
{
lean_object* v___x_335_; 
v___x_335_ = lean_array_fswap(v___y_331_, v_lo_307_, v_hi_308_);
v___y_325_ = v___x_335_;
goto v___jp_324_;
}
}
}
v___jp_309_:
{
lean_object* v_pivot_311_; lean_object* v___x_312_; lean_object* v_fst_313_; lean_object* v_snd_314_; uint8_t v___x_315_; 
v_pivot_311_ = lean_array_fget(v___y_310_, v_hi_308_);
lean_inc_n(v_lo_307_, 2);
v___x_312_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg(v_hi_308_, v_pivot_311_, v___y_310_, v_lo_307_, v_lo_307_);
lean_dec(v_pivot_311_);
v_fst_313_ = lean_ctor_get(v___x_312_, 0);
lean_inc(v_fst_313_);
v_snd_314_ = lean_ctor_get(v___x_312_, 1);
lean_inc(v_snd_314_);
lean_dec_ref(v___x_312_);
v___x_315_ = lean_nat_dec_le(v_hi_308_, v_fst_313_);
if (v___x_315_ == 0)
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_316_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(v_n_305_, v_snd_314_, v_lo_307_, v_fst_313_);
v___x_317_ = lean_unsigned_to_nat(1u);
v___x_318_ = lean_nat_add(v_fst_313_, v___x_317_);
lean_dec(v_fst_313_);
v_as_306_ = v___x_316_;
v_lo_307_ = v___x_318_;
goto _start;
}
else
{
lean_dec(v_fst_313_);
lean_dec(v_lo_307_);
return v_snd_314_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg___boxed(lean_object* v_n_340_, lean_object* v_as_341_, lean_object* v_lo_342_, lean_object* v_hi_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(v_n_340_, v_as_341_, v_lo_342_, v_hi_343_);
lean_dec(v_hi_343_);
lean_dec(v_n_340_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2_spec__2(lean_object* v_xs_345_, lean_object* v_ys_346_, lean_object* v_merge_347_, lean_object* v_acc_348_, lean_object* v_i_349_, lean_object* v_j_350_){
_start:
{
lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_351_ = lean_array_get_size(v_xs_345_);
v___x_352_ = lean_nat_dec_le(v___x_351_, v_i_349_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; uint8_t v___x_354_; 
v___x_353_ = lean_array_get_size(v_ys_346_);
v___x_354_ = lean_nat_dec_le(v___x_353_, v_j_350_);
if (v___x_354_ == 0)
{
lean_object* v_x_355_; lean_object* v_y_361_; double v_toFloat_367_; double v_toFloat_368_; uint8_t v___y_370_; uint8_t v___x_381_; 
v_x_355_ = lean_array_fget_borrowed(v_xs_345_, v_i_349_);
v_y_361_ = lean_array_fget_borrowed(v_ys_346_, v_j_350_);
v_toFloat_367_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_x_355_);
v_toFloat_368_ = lp_aesop_Aesop_UnsafeQueueEntry_successProbability(v_y_361_);
v___x_381_ = lean_float_decLt(v_toFloat_368_, v_toFloat_367_);
if (v___x_381_ == 0)
{
double v___x_382_; double v___x_383_; uint8_t v___x_384_; 
v___x_382_ = lean_float_sub(v_toFloat_368_, v_toFloat_367_);
v___x_383_ = lean_float_once(&lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0, &lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0_once, _init_lp_aesop_Aesop_UnsafeQueueEntry_instOrd___lam__0___closed__0);
v___x_384_ = lean_float_decLt(v___x_382_, v___x_383_);
v___y_370_ = v___x_384_;
goto v___jp_369_;
}
else
{
double v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; double v___x_388_; uint8_t v___x_389_; 
v___x_385_ = lean_float_sub(v_toFloat_367_, v_toFloat_368_);
v___x_386_ = lean_unsigned_to_nat(1u);
v___x_387_ = lean_unsigned_to_nat(5u);
v___x_388_ = l_Float_ofScientific(v___x_386_, v___x_381_, v___x_387_);
v___x_389_ = lean_float_decLt(v___x_385_, v___x_388_);
v___y_370_ = v___x_389_;
goto v___jp_369_;
}
v___jp_356_:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
lean_inc(v_x_355_);
v___x_357_ = lean_array_push(v_acc_348_, v_x_355_);
v___x_358_ = lean_unsigned_to_nat(1u);
v___x_359_ = lean_nat_add(v_i_349_, v___x_358_);
lean_dec(v_i_349_);
v_acc_348_ = v___x_357_;
v_i_349_ = v___x_359_;
goto _start;
}
v___jp_362_:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
lean_inc(v_y_361_);
v___x_363_ = lean_array_push(v_acc_348_, v_y_361_);
v___x_364_ = lean_unsigned_to_nat(1u);
v___x_365_ = lean_nat_add(v_j_350_, v___x_364_);
lean_dec(v_j_350_);
v_acc_348_ = v___x_363_;
v_j_350_ = v___x_365_;
goto _start;
}
v___jp_369_:
{
if (v___y_370_ == 0)
{
uint8_t v___x_371_; 
v___x_371_ = lean_float_decLt(v_toFloat_367_, v_toFloat_368_);
if (v___x_371_ == 0)
{
goto v___jp_356_;
}
else
{
goto v___jp_362_;
}
}
else
{
lean_object* v___x_372_; lean_object* v___x_373_; uint8_t v___x_374_; 
v___x_372_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_x_355_);
v___x_373_ = lp_aesop_Aesop_UnsafeQueueEntry_name(v_y_361_);
v___x_374_ = lp_aesop_Aesop_RuleName_compare(v___x_372_, v___x_373_);
lean_dec_ref(v___x_373_);
lean_dec_ref(v___x_372_);
switch(v___x_374_)
{
case 0:
{
goto v___jp_356_;
}
case 1:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
lean_inc_ref(v_merge_347_);
lean_inc(v_y_361_);
lean_inc(v_x_355_);
v___x_375_ = lean_apply_2(v_merge_347_, v_x_355_, v_y_361_);
v___x_376_ = lean_array_push(v_acc_348_, v___x_375_);
v___x_377_ = lean_unsigned_to_nat(1u);
v___x_378_ = lean_nat_add(v_i_349_, v___x_377_);
lean_dec(v_i_349_);
v___x_379_ = lean_nat_add(v_j_350_, v___x_377_);
lean_dec(v_j_350_);
v_acc_348_ = v___x_376_;
v_i_349_ = v___x_378_;
v_j_350_ = v___x_379_;
goto _start;
}
default: 
{
goto v___jp_362_;
}
}
}
}
}
else
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec(v_j_350_);
lean_dec_ref(v_merge_347_);
lean_dec_ref(v_ys_346_);
v___x_390_ = l_Array_toSubarray___redArg(v_xs_345_, v_i_349_, v___x_351_);
v___x_391_ = l_Subarray_copy___redArg(v___x_390_);
v___x_392_ = l_Array_append___redArg(v_acc_348_, v___x_391_);
lean_dec_ref(v___x_391_);
return v___x_392_;
}
}
else
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
lean_dec(v_i_349_);
lean_dec_ref(v_merge_347_);
lean_dec_ref(v_xs_345_);
v___x_393_ = lean_array_get_size(v_ys_346_);
v___x_394_ = l_Array_toSubarray___redArg(v_ys_346_, v_j_350_, v___x_393_);
v___x_395_ = l_Subarray_copy___redArg(v___x_394_);
v___x_396_ = l_Array_append___redArg(v_acc_348_, v___x_395_);
lean_dec_ref(v___x_395_);
return v___x_396_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2(lean_object* v_xs_397_, lean_object* v_ys_398_, lean_object* v_merge_399_){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; 
v___x_400_ = lean_array_get_size(v_xs_397_);
v___x_401_ = lean_array_get_size(v_ys_398_);
v___x_402_ = lean_nat_add(v___x_400_, v___x_401_);
v___x_403_ = lean_mk_empty_array_with_capacity(v___x_402_);
lean_dec(v___x_402_);
v___x_404_ = lean_unsigned_to_nat(0u);
v___x_405_ = lp_aesop_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2_spec__2(v_xs_397_, v_ys_398_, v_merge_399_, v___x_403_, v___x_404_, v___x_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1(size_t v_sz_406_, size_t v_i_407_, lean_object* v_bs_408_){
_start:
{
uint8_t v___x_409_; 
v___x_409_ = lean_usize_dec_lt(v_i_407_, v_sz_406_);
if (v___x_409_ == 0)
{
return v_bs_408_;
}
else
{
lean_object* v_v_410_; lean_object* v___x_411_; lean_object* v_bs_x27_412_; lean_object* v___x_413_; size_t v___x_414_; size_t v___x_415_; lean_object* v___x_416_; 
v_v_410_ = lean_array_uget(v_bs_408_, v_i_407_);
v___x_411_ = lean_unsigned_to_nat(0u);
v_bs_x27_412_ = lean_array_uset(v_bs_408_, v_i_407_, v___x_411_);
v___x_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_413_, 0, v_v_410_);
v___x_414_ = ((size_t)1ULL);
v___x_415_ = lean_usize_add(v_i_407_, v___x_414_);
v___x_416_ = lean_array_uset(v_bs_x27_412_, v_i_407_, v___x_413_);
v_i_407_ = v___x_415_;
v_bs_408_ = v___x_416_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1___boxed(lean_object* v_sz_418_, lean_object* v_i_419_, lean_object* v_bs_420_){
_start:
{
size_t v_sz_boxed_421_; size_t v_i_boxed_422_; lean_object* v_res_423_; 
v_sz_boxed_421_ = lean_unbox_usize(v_sz_418_);
lean_dec(v_sz_418_);
v_i_boxed_422_ = lean_unbox_usize(v_i_419_);
lean_dec(v_i_419_);
v_res_423_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1(v_sz_boxed_421_, v_i_boxed_422_, v_bs_420_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0(size_t v_sz_424_, size_t v_i_425_, lean_object* v_bs_426_){
_start:
{
uint8_t v___x_427_; 
v___x_427_ = lean_usize_dec_lt(v_i_425_, v_sz_424_);
if (v___x_427_ == 0)
{
return v_bs_426_;
}
else
{
lean_object* v_v_428_; lean_object* v___x_429_; lean_object* v_bs_x27_430_; lean_object* v___x_431_; size_t v___x_432_; size_t v___x_433_; lean_object* v___x_434_; 
v_v_428_ = lean_array_uget(v_bs_426_, v_i_425_);
v___x_429_ = lean_unsigned_to_nat(0u);
v_bs_x27_430_ = lean_array_uset(v_bs_426_, v_i_425_, v___x_429_);
v___x_431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_431_, 0, v_v_428_);
v___x_432_ = ((size_t)1ULL);
v___x_433_ = lean_usize_add(v_i_425_, v___x_432_);
v___x_434_ = lean_array_uset(v_bs_x27_430_, v_i_425_, v___x_431_);
v_i_425_ = v___x_433_;
v_bs_426_ = v___x_434_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0___boxed(lean_object* v_sz_436_, lean_object* v_i_437_, lean_object* v_bs_438_){
_start:
{
size_t v_sz_boxed_439_; size_t v_i_boxed_440_; lean_object* v_res_441_; 
v_sz_boxed_439_ = lean_unbox_usize(v_sz_436_);
lean_dec(v_sz_436_);
v_i_boxed_440_ = lean_unbox_usize(v_i_437_);
lean_dec(v_i_437_);
v_res_441_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0(v_sz_boxed_439_, v_i_boxed_440_, v_bs_438_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_initial(lean_object* v_postponedSafeRules_443_, lean_object* v_unsafeRules_444_){
_start:
{
lean_object* v___f_445_; size_t v_sz_446_; size_t v___x_447_; lean_object* v_unsafeRules_448_; size_t v_sz_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___y_453_; lean_object* v___x_457_; lean_object* v___y_459_; lean_object* v___y_460_; uint8_t v___x_462_; 
v___f_445_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueue_initial___closed__0));
v_sz_446_ = lean_array_size(v_unsafeRules_444_);
v___x_447_ = ((size_t)0ULL);
v_unsafeRules_448_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__0(v_sz_446_, v___x_447_, v_unsafeRules_444_);
v_sz_449_ = lean_array_size(v_postponedSafeRules_443_);
v___x_450_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_initial_spec__1(v_sz_449_, v___x_447_, v_postponedSafeRules_443_);
v___x_451_ = lean_unsigned_to_nat(0u);
v___x_457_ = lean_array_get_size(v___x_450_);
v___x_462_ = lean_nat_dec_eq(v___x_457_, v___x_451_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___y_466_; uint8_t v___x_468_; 
v___x_463_ = lean_unsigned_to_nat(1u);
v___x_464_ = lean_nat_sub(v___x_457_, v___x_463_);
v___x_468_ = lean_nat_dec_le(v___x_451_, v___x_464_);
if (v___x_468_ == 0)
{
lean_inc(v___x_464_);
v___y_466_ = v___x_464_;
goto v___jp_465_;
}
else
{
v___y_466_ = v___x_451_;
goto v___jp_465_;
}
v___jp_465_:
{
uint8_t v___x_467_; 
v___x_467_ = lean_nat_dec_le(v___y_466_, v___x_464_);
if (v___x_467_ == 0)
{
lean_dec(v___x_464_);
lean_inc(v___y_466_);
v___y_459_ = v___y_466_;
v___y_460_ = v___y_466_;
goto v___jp_458_;
}
else
{
v___y_459_ = v___y_466_;
v___y_460_ = v___x_464_;
goto v___jp_458_;
}
}
}
else
{
v___y_453_ = v___x_450_;
goto v___jp_452_;
}
v___jp_452_:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_454_ = lp_aesop_Array_mergeDedupWith___at___00Aesop_UnsafeQueue_initial_spec__2(v___y_453_, v_unsafeRules_448_, v___f_445_);
v___x_455_ = lean_array_get_size(v___x_454_);
v___x_456_ = l_Array_toSubarray___redArg(v___x_454_, v___x_451_, v___x_455_);
return v___x_456_;
}
v___jp_458_:
{
lean_object* v___x_461_; 
v___x_461_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(v___x_457_, v___x_450_, v___y_459_, v___y_460_);
lean_dec(v___y_460_);
v___y_453_ = v___x_461_;
goto v___jp_452_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3(lean_object* v_n_469_, lean_object* v_as_470_, lean_object* v_lo_471_, lean_object* v_hi_472_, lean_object* v_w_473_, lean_object* v_hlo_474_, lean_object* v_hhi_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___redArg(v_n_469_, v_as_470_, v_lo_471_, v_hi_472_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3___boxed(lean_object* v_n_477_, lean_object* v_as_478_, lean_object* v_lo_479_, lean_object* v_hi_480_, lean_object* v_w_481_, lean_object* v_hlo_482_, lean_object* v_hhi_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3(v_n_477_, v_as_478_, v_lo_479_, v_hi_480_, v_w_481_, v_hlo_482_, v_hhi_483_);
lean_dec(v_hi_480_);
lean_dec(v_n_477_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4(lean_object* v_n_485_, lean_object* v_lo_486_, lean_object* v_hi_487_, lean_object* v_hhi_488_, lean_object* v_pivot_489_, lean_object* v_as_490_, lean_object* v_i_491_, lean_object* v_k_492_, lean_object* v_ilo_493_, lean_object* v_ik_494_, lean_object* v_w_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___redArg(v_hi_487_, v_pivot_489_, v_as_490_, v_i_491_, v_k_492_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4___boxed(lean_object* v_n_497_, lean_object* v_lo_498_, lean_object* v_hi_499_, lean_object* v_hhi_500_, lean_object* v_pivot_501_, lean_object* v_as_502_, lean_object* v_i_503_, lean_object* v_k_504_, lean_object* v_ilo_505_, lean_object* v_ik_506_, lean_object* v_w_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_UnsafeQueue_initial_spec__3_spec__4(v_n_497_, v_lo_498_, v_hi_499_, v_hhi_500_, v_pivot_501_, v_as_502_, v_i_503_, v_k_504_, v_ilo_505_, v_ik_506_, v_w_507_);
lean_dec_ref(v_pivot_501_);
lean_dec(v_hi_499_);
lean_dec(v_lo_498_);
lean_dec(v_n_497_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1(size_t v_sz_509_, size_t v_i_510_, lean_object* v_bs_511_){
_start:
{
uint8_t v___x_512_; 
v___x_512_ = lean_usize_dec_lt(v_i_510_, v_sz_509_);
if (v___x_512_ == 0)
{
return v_bs_511_;
}
else
{
lean_object* v_v_513_; lean_object* v___x_514_; lean_object* v_bs_x27_515_; lean_object* v___y_517_; 
v_v_513_ = lean_array_uget(v_bs_511_, v_i_510_);
v___x_514_ = lean_unsigned_to_nat(0u);
v_bs_x27_515_ = lean_array_uset(v_bs_511_, v_i_510_, v___x_514_);
if (lean_obj_tag(v_v_513_) == 0)
{
lean_object* v_r_524_; lean_object* v_rule_525_; lean_object* v_name_526_; lean_object* v_name_527_; uint8_t v_builder_528_; uint8_t v_phase_529_; uint8_t v_scope_530_; lean_object* v___y_532_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_540_; lean_object* v___y_541_; lean_object* v___y_542_; lean_object* v___y_548_; 
v_r_524_ = lean_ctor_get(v_v_513_, 0);
lean_inc_ref(v_r_524_);
lean_dec_ref_known(v_v_513_, 1);
v_rule_525_ = lean_ctor_get(v_r_524_, 0);
lean_inc(v_rule_525_);
lean_dec_ref(v_r_524_);
v_name_526_ = lean_ctor_get(v_rule_525_, 0);
lean_inc_ref(v_name_526_);
lean_dec(v_rule_525_);
v_name_527_ = lean_ctor_get(v_name_526_, 0);
lean_inc(v_name_527_);
v_builder_528_ = lean_ctor_get_uint8(v_name_526_, sizeof(void*)*1 + 8);
v_phase_529_ = lean_ctor_get_uint8(v_name_526_, sizeof(void*)*1 + 9);
v_scope_530_ = lean_ctor_get_uint8(v_name_526_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_526_);
switch(v_phase_529_)
{
case 0:
{
lean_object* v___x_559_; 
v___x_559_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11));
v___y_548_ = v___x_559_;
goto v___jp_547_;
}
case 1:
{
lean_object* v___x_560_; 
v___x_560_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12));
v___y_548_ = v___x_560_;
goto v___jp_547_;
}
default: 
{
lean_object* v___x_561_; 
v___x_561_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13));
v___y_548_ = v___x_561_;
goto v___jp_547_;
}
}
v___jp_531_:
{
lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
v___x_535_ = lean_string_append(v___y_532_, v___y_534_);
v___x_536_ = lean_string_append(v___x_535_, v___y_533_);
v___x_537_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_527_, v___x_512_);
v___x_538_ = lean_string_append(v___x_536_, v___x_537_);
lean_dec_ref(v___x_537_);
v___y_517_ = v___x_538_;
goto v___jp_516_;
}
v___jp_539_:
{
lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_543_ = lean_string_append(v___y_540_, v___y_542_);
v___x_544_ = lean_string_append(v___x_543_, v___y_541_);
if (v_scope_530_ == 0)
{
lean_object* v___x_545_; 
v___x_545_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0));
v___y_532_ = v___x_544_;
v___y_533_ = v___y_541_;
v___y_534_ = v___x_545_;
goto v___jp_531_;
}
else
{
lean_object* v___x_546_; 
v___x_546_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1));
v___y_532_ = v___x_544_;
v___y_533_ = v___y_541_;
v___y_534_ = v___x_546_;
goto v___jp_531_;
}
}
v___jp_547_:
{
lean_object* v___x_549_; lean_object* v___x_550_; 
v___x_549_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2));
lean_inc_ref(v___y_548_);
v___x_550_ = lean_string_append(v___y_548_, v___x_549_);
switch(v_builder_528_)
{
case 0:
{
lean_object* v___x_551_; 
v___x_551_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_551_;
goto v___jp_539_;
}
case 1:
{
lean_object* v___x_552_; 
v___x_552_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_552_;
goto v___jp_539_;
}
case 2:
{
lean_object* v___x_553_; 
v___x_553_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_553_;
goto v___jp_539_;
}
case 3:
{
lean_object* v___x_554_; 
v___x_554_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_554_;
goto v___jp_539_;
}
case 4:
{
lean_object* v___x_555_; 
v___x_555_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_555_;
goto v___jp_539_;
}
case 5:
{
lean_object* v___x_556_; 
v___x_556_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_556_;
goto v___jp_539_;
}
case 6:
{
lean_object* v___x_557_; 
v___x_557_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_557_;
goto v___jp_539_;
}
default: 
{
lean_object* v___x_558_; 
v___x_558_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10));
v___y_540_ = v___x_550_;
v___y_541_ = v___x_549_;
v___y_542_ = v___x_558_;
goto v___jp_539_;
}
}
}
}
else
{
lean_object* v_r_562_; lean_object* v_rule_563_; lean_object* v_name_564_; lean_object* v_name_565_; uint8_t v_builder_566_; uint8_t v_phase_567_; uint8_t v_scope_568_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_578_; lean_object* v___y_579_; lean_object* v___y_580_; lean_object* v___y_586_; 
v_r_562_ = lean_ctor_get(v_v_513_, 0);
lean_inc_ref(v_r_562_);
lean_dec_ref_known(v_v_513_, 1);
v_rule_563_ = lean_ctor_get(v_r_562_, 0);
lean_inc_ref(v_rule_563_);
lean_dec_ref(v_r_562_);
v_name_564_ = lean_ctor_get(v_rule_563_, 0);
lean_inc_ref(v_name_564_);
lean_dec_ref(v_rule_563_);
v_name_565_ = lean_ctor_get(v_name_564_, 0);
lean_inc(v_name_565_);
v_builder_566_ = lean_ctor_get_uint8(v_name_564_, sizeof(void*)*1 + 8);
v_phase_567_ = lean_ctor_get_uint8(v_name_564_, sizeof(void*)*1 + 9);
v_scope_568_ = lean_ctor_get_uint8(v_name_564_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_564_);
switch(v_phase_567_)
{
case 0:
{
lean_object* v___x_597_; 
v___x_597_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__11));
v___y_586_ = v___x_597_;
goto v___jp_585_;
}
case 1:
{
lean_object* v___x_598_; 
v___x_598_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__12));
v___y_586_ = v___x_598_;
goto v___jp_585_;
}
default: 
{
lean_object* v___x_599_; 
v___x_599_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__13));
v___y_586_ = v___x_599_;
goto v___jp_585_;
}
}
v___jp_569_:
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_573_ = lean_string_append(v___y_570_, v___y_572_);
v___x_574_ = lean_string_append(v___x_573_, v___y_571_);
v___x_575_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_565_, v___x_512_);
v___x_576_ = lean_string_append(v___x_574_, v___x_575_);
lean_dec_ref(v___x_575_);
v___y_517_ = v___x_576_;
goto v___jp_516_;
}
v___jp_577_:
{
lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_581_ = lean_string_append(v___y_579_, v___y_580_);
v___x_582_ = lean_string_append(v___x_581_, v___y_578_);
if (v_scope_568_ == 0)
{
lean_object* v___x_583_; 
v___x_583_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__0));
v___y_570_ = v___x_582_;
v___y_571_ = v___y_578_;
v___y_572_ = v___x_583_;
goto v___jp_569_;
}
else
{
lean_object* v___x_584_; 
v___x_584_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__1));
v___y_570_ = v___x_582_;
v___y_571_ = v___y_578_;
v___y_572_ = v___x_584_;
goto v___jp_569_;
}
}
v___jp_585_:
{
lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_587_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__2));
lean_inc_ref(v___y_586_);
v___x_588_ = lean_string_append(v___y_586_, v___x_587_);
switch(v_builder_566_)
{
case 0:
{
lean_object* v___x_589_; 
v___x_589_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__3));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_589_;
goto v___jp_577_;
}
case 1:
{
lean_object* v___x_590_; 
v___x_590_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__4));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_590_;
goto v___jp_577_;
}
case 2:
{
lean_object* v___x_591_; 
v___x_591_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__5));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_591_;
goto v___jp_577_;
}
case 3:
{
lean_object* v___x_592_; 
v___x_592_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__6));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_592_;
goto v___jp_577_;
}
case 4:
{
lean_object* v___x_593_; 
v___x_593_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__7));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_593_;
goto v___jp_577_;
}
case 5:
{
lean_object* v___x_594_; 
v___x_594_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__8));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_594_;
goto v___jp_577_;
}
case 6:
{
lean_object* v___x_595_; 
v___x_595_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__9));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_595_;
goto v___jp_577_;
}
default: 
{
lean_object* v___x_596_; 
v___x_596_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueueEntry_instToString___lam__0___closed__10));
v___y_578_ = v___x_587_;
v___y_579_ = v___x_588_;
v___y_580_ = v___x_596_;
goto v___jp_577_;
}
}
}
}
v___jp_516_:
{
lean_object* v___x_518_; lean_object* v___x_519_; size_t v___x_520_; size_t v___x_521_; lean_object* v___x_522_; 
v___x_518_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_518_, 0, v___y_517_);
v___x_519_ = l_Lean_MessageData_ofFormat(v___x_518_);
v___x_520_ = ((size_t)1ULL);
v___x_521_ = lean_usize_add(v_i_510_, v___x_520_);
v___x_522_ = lean_array_uset(v_bs_x27_515_, v_i_510_, v___x_519_);
v_i_510_ = v___x_521_;
v_bs_511_ = v___x_522_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1___boxed(lean_object* v_sz_600_, lean_object* v_i_601_, lean_object* v_bs_602_){
_start:
{
size_t v_sz_boxed_603_; size_t v_i_boxed_604_; lean_object* v_res_605_; 
v_sz_boxed_603_ = lean_unbox_usize(v_sz_600_);
lean_dec(v_sz_600_);
v_i_boxed_604_ = lean_unbox_usize(v_i_601_);
lean_dec(v_i_601_);
v_res_605_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1(v_sz_boxed_603_, v_i_boxed_604_, v_bs_602_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0___redArg(lean_object* v_a_606_, lean_object* v_b_607_){
_start:
{
lean_object* v_array_608_; lean_object* v_start_609_; lean_object* v_stop_610_; lean_object* v___x_612_; uint8_t v_isShared_613_; uint8_t v_isSharedCheck_623_; 
v_array_608_ = lean_ctor_get(v_a_606_, 0);
v_start_609_ = lean_ctor_get(v_a_606_, 1);
v_stop_610_ = lean_ctor_get(v_a_606_, 2);
v_isSharedCheck_623_ = !lean_is_exclusive(v_a_606_);
if (v_isSharedCheck_623_ == 0)
{
v___x_612_ = v_a_606_;
v_isShared_613_ = v_isSharedCheck_623_;
goto v_resetjp_611_;
}
else
{
lean_inc(v_stop_610_);
lean_inc(v_start_609_);
lean_inc(v_array_608_);
lean_dec(v_a_606_);
v___x_612_ = lean_box(0);
v_isShared_613_ = v_isSharedCheck_623_;
goto v_resetjp_611_;
}
v_resetjp_611_:
{
uint8_t v___x_614_; 
v___x_614_ = lean_nat_dec_lt(v_start_609_, v_stop_610_);
if (v___x_614_ == 0)
{
lean_del_object(v___x_612_);
lean_dec(v_stop_610_);
lean_dec(v_start_609_);
lean_dec_ref(v_array_608_);
return v_b_607_;
}
else
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_618_; 
v___x_615_ = lean_unsigned_to_nat(1u);
v___x_616_ = lean_nat_add(v_start_609_, v___x_615_);
lean_inc_ref(v_array_608_);
if (v_isShared_613_ == 0)
{
lean_ctor_set(v___x_612_, 1, v___x_616_);
v___x_618_ = v___x_612_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v_array_608_);
lean_ctor_set(v_reuseFailAlloc_622_, 1, v___x_616_);
lean_ctor_set(v_reuseFailAlloc_622_, 2, v_stop_610_);
v___x_618_ = v_reuseFailAlloc_622_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_619_ = lean_array_fget(v_array_608_, v_start_609_);
lean_dec(v_start_609_);
lean_dec_ref(v_array_608_);
v___x_620_ = lean_array_push(v_b_607_, v___x_619_);
v_a_606_ = v___x_618_;
v_b_607_ = v___x_620_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnsafeQueue_entriesToMessageData(lean_object* v_q_626_){
_start:
{
lean_object* v___x_627_; lean_object* v___x_628_; size_t v_sz_629_; size_t v___x_630_; lean_object* v___x_631_; 
v___x_627_ = ((lean_object*)(lp_aesop_Aesop_UnsafeQueue_entriesToMessageData___closed__0));
v___x_628_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0___redArg(v_q_626_, v___x_627_);
v_sz_629_ = lean_array_size(v___x_628_);
v___x_630_ = ((size_t)0ULL);
v___x_631_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__1(v_sz_629_, v___x_630_, v___x_628_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0(lean_object* v_inst_632_, lean_object* v_R_633_, lean_object* v_a_634_, lean_object* v_b_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_UnsafeQueue_entriesToMessageData_spec__0___redArg(v_a_634_, v_b_635_);
return v___x_636_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Constants(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPostponedSafeRule_default = _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPostponedSafeRule_default);
lp_aesop_Aesop_instInhabitedPostponedSafeRule = _init_lp_aesop_Aesop_instInhabitedPostponedSafeRule();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPostponedSafeRule);
lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1 = _init_lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1();
lean_mark_persistent(lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule___boxed__const__1);
lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1 = _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default___closed__0___boxed__const__1);
lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default = _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedUnsafeQueueEntry_default);
lp_aesop_Aesop_instInhabitedUnsafeQueueEntry = _init_lp_aesop_Aesop_instInhabitedUnsafeQueueEntry();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedUnsafeQueueEntry);
lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1 = _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1();
lean_mark_persistent(lp_aesop_Aesop_UnsafeQueue_instEmptyCollection___aux__1);
lp_aesop_Aesop_UnsafeQueue_instEmptyCollection = _init_lp_aesop_Aesop_UnsafeQueue_instEmptyCollection();
lean_mark_persistent(lp_aesop_Aesop_UnsafeQueue_instEmptyCollection);
lp_aesop_Aesop_UnsafeQueue_instInhabited___aux__1 = _init_lp_aesop_Aesop_UnsafeQueue_instInhabited___aux__1();
lean_mark_persistent(lp_aesop_Aesop_UnsafeQueue_instInhabited___aux__1);
lp_aesop_Aesop_UnsafeQueue_instInhabited = _init_lp_aesop_Aesop_UnsafeQueue_instInhabited();
lean_mark_persistent(lp_aesop_Aesop_UnsafeQueue_instInhabited);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Constants(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
}
#ifdef __cplusplus
}
#endif
