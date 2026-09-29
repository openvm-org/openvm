// Lean compiler output
// Module: Aesop.Index.Forward
// Imports: public import Init public meta import Init public import Aesop.Forward.Match.Types import Batteries.Lean.Meta.DiscrTree
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_ForwardRulePriority_compare(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint64_t l_Lean_Meta_DiscrTree_Key_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_Meta_DiscrTree_instBEqKey_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Meta_DiscrTree_Key_lt(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
lean_object* lp_aesop_Aesop_ForwardRule_instHashable___lam__0___boxed(lean_object*);
lean_object* lp_aesop_Aesop_ForwardRule_instBEq___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t lp_aesop_Aesop_ForwardRuleInfo_isConstant(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardIndex_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardIndex;
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2;
static lean_once_cell_t lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_instEmptyCollection;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___closed__0 = (const lean_object*)&lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4(lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__0_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__1_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_trace___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_trace___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_trace___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_trace___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_trace___lam__1___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_trace___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__1_value;
static const lean_array_object lp_aesop_Aesop_ForwardIndex_trace___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ForwardIndex_trace___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_merge___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_merge___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_merge___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_merge___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_merge___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_merge___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_merge___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_merge___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_merge___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_merge___lam__2, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_merge___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_merge___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__2_value),((lean_object*)&lp_aesop_Aesop_ForwardIndex_trace___closed__2_value)}};
static const lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.Meta.DiscrTree.Basic"};
static const lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__0 = (const lean_object*)&lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__0_value;
static const lean_string_object lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Lean.Meta.DiscrTree.insertKeyValue"};
static const lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__1 = (const lean_object*)&lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__1_value;
static const lean_string_object lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "invalid key sequence"};
static const lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__2 = (const lean_object*)&lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_insert(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getRuleWithName_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getRuleWithName_x3f___boxed(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__0_value;
static const lean_array_object lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___boxed(lean_object*);
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__0);
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0(lean_object* v_00_u03b2_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0___closed__1);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0, &lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__0);
v___x_8_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedForwardIndex_default_spec__0(lean_box(0));
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_10_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2, &lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__2);
v___x_11_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1);
v___x_12_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
lean_ctor_set(v___x_12_, 1, v___x_11_);
lean_ctor_set(v___x_12_, 2, v___x_10_);
return v___x_12_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex_default(void){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3, &lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__3);
return v___x_13_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardIndex(void){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_aesop_Aesop_instInhabitedForwardIndex_default;
return v___x_14_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2(void){
_start:
{
lean_object* v___f_17_; lean_object* v___f_18_; lean_object* v___x_19_; 
v___f_17_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__1));
v___f_18_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__0));
v___x_19_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___f_18_, v___f_17_);
return v___x_19_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3(void){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_20_ = lean_obj_once(&lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2, &lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2_once, _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__2);
v___x_21_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardIndex_default___closed__1);
v___x_22_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
lean_ctor_set(v___x_22_, 2, v___x_20_);
return v___x_22_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection(void){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_obj_once(&lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3, &lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3_once, _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection___closed__3);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__0(lean_object* v_x1_24_, lean_object* v_x2_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lean_array_push(v_x1_24_, v_x2_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(lean_object* v_f_27_, lean_object* v_as_28_, size_t v_i_29_, size_t v_stop_30_, lean_object* v_b_31_){
_start:
{
uint8_t v___x_32_; 
v___x_32_ = lean_usize_dec_eq(v_i_29_, v_stop_30_);
if (v___x_32_ == 0)
{
lean_object* v___x_33_; lean_object* v___x_34_; size_t v___x_35_; size_t v___x_36_; 
v___x_33_ = lean_array_uget_borrowed(v_as_28_, v_i_29_);
lean_inc(v_f_27_);
lean_inc(v___x_33_);
v___x_34_ = lean_apply_2(v_f_27_, v_b_31_, v___x_33_);
v___x_35_ = ((size_t)1ULL);
v___x_36_ = lean_usize_add(v_i_29_, v___x_35_);
v_i_29_ = v___x_36_;
v_b_31_ = v___x_34_;
goto _start;
}
else
{
lean_dec(v_f_27_);
return v_b_31_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg___boxed(lean_object* v_f_38_, lean_object* v_as_39_, lean_object* v_i_40_, lean_object* v_stop_41_, lean_object* v_b_42_){
_start:
{
size_t v_i_boxed_43_; size_t v_stop_boxed_44_; lean_object* v_res_45_; 
v_i_boxed_43_ = lean_unbox_usize(v_i_40_);
lean_dec(v_i_40_);
v_stop_boxed_44_ = lean_unbox_usize(v_stop_41_);
lean_dec(v_stop_41_);
v_res_45_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(v_f_38_, v_as_39_, v_i_boxed_43_, v_stop_boxed_44_, v_b_42_);
lean_dec_ref(v_as_39_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(lean_object* v_f_46_, lean_object* v_x_47_, lean_object* v_x_48_){
_start:
{
lean_object* v_vs_49_; lean_object* v_children_50_; lean_object* v___x_51_; lean_object* v_s_53_; lean_object* v___x_63_; uint8_t v___x_64_; 
v_vs_49_ = lean_ctor_get(v_x_48_, 0);
v_children_50_ = lean_ctor_get(v_x_48_, 1);
v___x_51_ = lean_unsigned_to_nat(0u);
v___x_63_ = lean_array_get_size(v_vs_49_);
v___x_64_ = lean_nat_dec_lt(v___x_51_, v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_array_get_size(v_children_50_);
v___x_66_ = lean_nat_dec_lt(v___x_51_, v___x_65_);
if (v___x_66_ == 0)
{
lean_dec(v_f_46_);
return v_x_47_;
}
else
{
uint8_t v___x_67_; 
v___x_67_ = lean_nat_dec_le(v___x_65_, v___x_65_);
if (v___x_67_ == 0)
{
if (v___x_66_ == 0)
{
lean_dec(v_f_46_);
return v_x_47_;
}
else
{
size_t v___x_68_; size_t v___x_69_; lean_object* v___x_70_; 
v___x_68_ = ((size_t)0ULL);
v___x_69_ = lean_usize_of_nat(v___x_65_);
v___x_70_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_46_, v_children_50_, v___x_68_, v___x_69_, v_x_47_);
return v___x_70_;
}
}
else
{
size_t v___x_71_; size_t v___x_72_; lean_object* v___x_73_; 
v___x_71_ = ((size_t)0ULL);
v___x_72_ = lean_usize_of_nat(v___x_65_);
v___x_73_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_46_, v_children_50_, v___x_71_, v___x_72_, v_x_47_);
return v___x_73_;
}
}
}
else
{
uint8_t v___x_74_; 
v___x_74_ = lean_nat_dec_le(v___x_63_, v___x_63_);
if (v___x_74_ == 0)
{
if (v___x_64_ == 0)
{
v_s_53_ = v_x_47_;
goto v___jp_52_;
}
else
{
size_t v___x_75_; size_t v___x_76_; lean_object* v___x_77_; 
v___x_75_ = ((size_t)0ULL);
v___x_76_ = lean_usize_of_nat(v___x_63_);
lean_inc(v_f_46_);
v___x_77_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(v_f_46_, v_vs_49_, v___x_75_, v___x_76_, v_x_47_);
v_s_53_ = v___x_77_;
goto v___jp_52_;
}
}
else
{
size_t v___x_78_; size_t v___x_79_; lean_object* v___x_80_; 
v___x_78_ = ((size_t)0ULL);
v___x_79_ = lean_usize_of_nat(v___x_63_);
lean_inc(v_f_46_);
v___x_80_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(v_f_46_, v_vs_49_, v___x_78_, v___x_79_, v_x_47_);
v_s_53_ = v___x_80_;
goto v___jp_52_;
}
}
v___jp_52_:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = lean_array_get_size(v_children_50_);
v___x_55_ = lean_nat_dec_lt(v___x_51_, v___x_54_);
if (v___x_55_ == 0)
{
lean_dec(v_f_46_);
return v_s_53_;
}
else
{
uint8_t v___x_56_; 
v___x_56_ = lean_nat_dec_le(v___x_54_, v___x_54_);
if (v___x_56_ == 0)
{
if (v___x_55_ == 0)
{
lean_dec(v_f_46_);
return v_s_53_;
}
else
{
size_t v___x_57_; size_t v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((size_t)0ULL);
v___x_58_ = lean_usize_of_nat(v___x_54_);
v___x_59_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_46_, v_children_50_, v___x_57_, v___x_58_, v_s_53_);
return v___x_59_;
}
}
else
{
size_t v___x_60_; size_t v___x_61_; lean_object* v___x_62_; 
v___x_60_ = ((size_t)0ULL);
v___x_61_ = lean_usize_of_nat(v___x_54_);
v___x_62_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_46_, v_children_50_, v___x_60_, v___x_61_, v_s_53_);
return v___x_62_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(lean_object* v_f_81_, lean_object* v_as_82_, size_t v_i_83_, size_t v_stop_84_, lean_object* v_b_85_){
_start:
{
uint8_t v___x_86_; 
v___x_86_ = lean_usize_dec_eq(v_i_83_, v_stop_84_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; lean_object* v_snd_88_; lean_object* v___x_89_; size_t v___x_90_; size_t v___x_91_; 
v___x_87_ = lean_array_uget_borrowed(v_as_82_, v_i_83_);
v_snd_88_ = lean_ctor_get(v___x_87_, 1);
lean_inc(v_f_81_);
v___x_89_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(v_f_81_, v_b_85_, v_snd_88_);
v___x_90_ = ((size_t)1ULL);
v___x_91_ = lean_usize_add(v_i_83_, v___x_90_);
v_i_83_ = v___x_91_;
v_b_85_ = v___x_89_;
goto _start;
}
else
{
lean_dec(v_f_81_);
return v_b_85_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg___boxed(lean_object* v_f_93_, lean_object* v_as_94_, lean_object* v_i_95_, lean_object* v_stop_96_, lean_object* v_b_97_){
_start:
{
size_t v_i_boxed_98_; size_t v_stop_boxed_99_; lean_object* v_res_100_; 
v_i_boxed_98_ = lean_unbox_usize(v_i_95_);
lean_dec(v_i_95_);
v_stop_boxed_99_ = lean_unbox_usize(v_stop_96_);
lean_dec(v_stop_96_);
v_res_100_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_93_, v_as_94_, v_i_boxed_98_, v_stop_boxed_99_, v_b_97_);
lean_dec_ref(v_as_94_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg___boxed(lean_object* v_f_101_, lean_object* v_x_102_, lean_object* v_x_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(v_f_101_, v_x_102_, v_x_103_);
lean_dec_ref(v_x_103_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__1(lean_object* v___f_105_, lean_object* v_s_106_, lean_object* v_x_107_, lean_object* v_t_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(v___f_105_, v_s_106_, v_t_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___lam__1___boxed(lean_object* v___f_110_, lean_object* v_s_111_, lean_object* v_x_112_, lean_object* v_t_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_aesop_Aesop_ForwardIndex_trace___lam__1(v___f_110_, v_s_111_, v_x_112_, v_t_113_);
lean_dec_ref(v_t_113_);
lean_dec(v_x_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15(lean_object* v_f_115_, lean_object* v_xs_116_, lean_object* v_acc_117_, lean_object* v_i_118_, lean_object* v_hd_119_){
_start:
{
lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_120_ = lean_array_get_size(v_xs_116_);
v___x_121_ = lean_nat_dec_lt(v_i_118_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
lean_dec(v_i_118_);
lean_dec_ref(v_f_115_);
v___x_122_ = lean_array_push(v_acc_117_, v_hd_119_);
return v___x_122_;
}
else
{
lean_object* v_x_123_; uint8_t v___y_130_; lean_object* v_fst_135_; lean_object* v_fst_136_; lean_object* v_name_137_; lean_object* v_name_138_; lean_object* v_name_139_; uint8_t v_builder_140_; uint8_t v_phase_141_; uint8_t v_scope_142_; uint64_t v_hash_143_; lean_object* v_name_144_; uint8_t v_builder_145_; uint8_t v_phase_146_; uint8_t v_scope_147_; uint64_t v_hash_148_; uint8_t v___y_150_; uint8_t v___x_154_; 
v_x_123_ = lean_array_fget_borrowed(v_xs_116_, v_i_118_);
v_fst_135_ = lean_ctor_get(v_x_123_, 0);
v_fst_136_ = lean_ctor_get(v_hd_119_, 0);
v_name_137_ = lean_ctor_get(v_fst_135_, 1);
v_name_138_ = lean_ctor_get(v_fst_136_, 1);
v_name_139_ = lean_ctor_get(v_name_137_, 0);
v_builder_140_ = lean_ctor_get_uint8(v_name_137_, sizeof(void*)*1 + 8);
v_phase_141_ = lean_ctor_get_uint8(v_name_137_, sizeof(void*)*1 + 9);
v_scope_142_ = lean_ctor_get_uint8(v_name_137_, sizeof(void*)*1 + 10);
v_hash_143_ = lean_ctor_get_uint64(v_name_137_, sizeof(void*)*1);
v_name_144_ = lean_ctor_get(v_name_138_, 0);
v_builder_145_ = lean_ctor_get_uint8(v_name_138_, sizeof(void*)*1 + 8);
v_phase_146_ = lean_ctor_get_uint8(v_name_138_, sizeof(void*)*1 + 9);
v_scope_147_ = lean_ctor_get_uint8(v_name_138_, sizeof(void*)*1 + 10);
v_hash_148_ = lean_ctor_get_uint64(v_name_138_, sizeof(void*)*1);
v___x_154_ = lean_uint64_dec_eq(v_hash_143_, v_hash_148_);
if (v___x_154_ == 0)
{
v___y_150_ = v___x_154_;
goto v___jp_149_;
}
else
{
uint8_t v___x_155_; 
v___x_155_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_140_, v_builder_145_);
v___y_150_ = v___x_155_;
goto v___jp_149_;
}
v___jp_124_:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = lean_array_push(v_acc_117_, v_hd_119_);
v___x_126_ = lean_unsigned_to_nat(1u);
v___x_127_ = lean_nat_add(v_i_118_, v___x_126_);
lean_dec(v_i_118_);
lean_inc(v_x_123_);
v_acc_117_ = v___x_125_;
v_i_118_ = v___x_127_;
v_hd_119_ = v_x_123_;
goto _start;
}
v___jp_129_:
{
if (v___y_130_ == 0)
{
goto v___jp_124_;
}
else
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = lean_unsigned_to_nat(1u);
v___x_132_ = lean_nat_add(v_i_118_, v___x_131_);
lean_dec(v_i_118_);
lean_inc_ref(v_f_115_);
lean_inc(v_x_123_);
v___x_133_ = lean_apply_2(v_f_115_, v_hd_119_, v_x_123_);
v_i_118_ = v___x_132_;
v_hd_119_ = v___x_133_;
goto _start;
}
}
v___jp_149_:
{
if (v___y_150_ == 0)
{
goto v___jp_124_;
}
else
{
uint8_t v___x_151_; 
v___x_151_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_141_, v_phase_146_);
if (v___x_151_ == 0)
{
v___y_130_ = v___x_151_;
goto v___jp_129_;
}
else
{
uint8_t v___x_152_; 
v___x_152_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_142_, v_scope_147_);
if (v___x_152_ == 0)
{
v___y_130_ = v___x_152_;
goto v___jp_129_;
}
else
{
uint8_t v___x_153_; 
v___x_153_ = lean_name_eq(v_name_139_, v_name_144_);
v___y_130_ = v___x_153_;
goto v___jp_129_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15___boxed(lean_object* v_f_156_, lean_object* v_xs_157_, lean_object* v_acc_158_, lean_object* v_i_159_, lean_object* v_hd_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15(v_f_156_, v_xs_157_, v_acc_158_, v_i_159_, v_hd_160_);
lean_dec_ref(v_xs_157_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11(lean_object* v_f_162_, lean_object* v_xs_163_){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_164_ = lean_unsigned_to_nat(0u);
v___x_165_ = lean_array_get_size(v_xs_163_);
v___x_166_ = lean_nat_dec_lt(v___x_164_, v___x_165_);
if (v___x_166_ == 0)
{
lean_dec_ref(v_f_162_);
lean_inc_ref(v_xs_163_);
return v_xs_163_;
}
else
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_167_ = lean_mk_empty_array_with_capacity(v___x_165_);
v___x_168_ = lean_unsigned_to_nat(1u);
v___x_169_ = lean_array_fget_borrowed(v_xs_163_, v___x_164_);
lean_inc(v___x_169_);
v___x_170_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11_spec__15(v_f_162_, v_xs_163_, v___x_167_, v___x_168_, v___x_169_);
return v___x_170_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11___boxed(lean_object* v_f_171_, lean_object* v_xs_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11(v_f_171_, v_xs_172_);
lean_dec_ref(v_xs_172_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0(lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
lean_inc_ref(v_x_174_);
return v_x_174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0___boxed(lean_object* v_x_176_, lean_object* v_x_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___lam__0(v_x_176_, v_x_177_);
lean_dec_ref(v_x_177_);
lean_dec_ref(v_x_176_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5(lean_object* v_xs_180_){
_start:
{
lean_object* v___f_181_; lean_object* v___x_182_; 
v___f_181_ = ((lean_object*)(lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___closed__0));
v___x_182_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5_spec__11(v___f_181_, v_xs_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5___boxed(lean_object* v_xs_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5(v_xs_183_);
lean_dec_ref(v_xs_183_);
return v_res_184_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0(lean_object* v_opts_185_, lean_object* v_opt_186_){
_start:
{
lean_object* v_name_187_; lean_object* v_defValue_188_; lean_object* v_map_189_; lean_object* v___x_190_; 
v_name_187_ = lean_ctor_get(v_opt_186_, 0);
v_defValue_188_ = lean_ctor_get(v_opt_186_, 1);
v_map_189_ = lean_ctor_get(v_opts_185_, 0);
v___x_190_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_189_, v_name_187_);
if (lean_obj_tag(v___x_190_) == 0)
{
uint8_t v___x_191_; 
v___x_191_ = lean_unbox(v_defValue_188_);
return v___x_191_;
}
else
{
lean_object* v_val_192_; 
v_val_192_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_val_192_);
lean_dec_ref_known(v___x_190_, 1);
if (lean_obj_tag(v_val_192_) == 1)
{
uint8_t v_v_193_; 
v_v_193_ = lean_ctor_get_uint8(v_val_192_, 0);
lean_dec_ref_known(v_val_192_, 0);
return v_v_193_;
}
else
{
uint8_t v___x_194_; 
lean_dec(v_val_192_);
v___x_194_ = lean_unbox(v_defValue_188_);
return v___x_194_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0___boxed(lean_object* v_opts_195_, lean_object* v_opt_196_){
_start:
{
uint8_t v_res_197_; lean_object* v_r_198_; 
v_res_197_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0(v_opts_195_, v_opt_196_);
lean_dec_ref(v_opt_196_);
lean_dec_ref(v_opts_195_);
v_r_198_ = lean_box(v_res_197_);
return v_r_198_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg(lean_object* v_opt_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_options_202_; lean_object* v_option_203_; uint8_t v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v_options_202_ = lean_ctor_get(v___y_200_, 2);
v_option_203_ = lean_ctor_get(v_opt_199_, 1);
v___x_204_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0_spec__0(v_options_202_, v_option_203_);
v___x_205_ = lean_box(v___x_204_);
v___x_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg___boxed(lean_object* v_opt_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg(v_opt_207_, v___y_208_);
lean_dec_ref(v___y_208_);
lean_dec_ref(v_opt_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg(lean_object* v_f_211_, lean_object* v_keys_212_, lean_object* v_vals_213_, lean_object* v_i_214_, lean_object* v_acc_215_){
_start:
{
lean_object* v___x_216_; uint8_t v___x_217_; 
v___x_216_ = lean_array_get_size(v_keys_212_);
v___x_217_ = lean_nat_dec_lt(v_i_214_, v___x_216_);
if (v___x_217_ == 0)
{
lean_dec(v_i_214_);
lean_dec(v_f_211_);
return v_acc_215_;
}
else
{
lean_object* v_k_218_; lean_object* v_v_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v_k_218_ = lean_array_fget_borrowed(v_keys_212_, v_i_214_);
v_v_219_ = lean_array_fget_borrowed(v_vals_213_, v_i_214_);
lean_inc(v_f_211_);
lean_inc(v_v_219_);
lean_inc(v_k_218_);
v___x_220_ = lean_apply_3(v_f_211_, v_acc_215_, v_k_218_, v_v_219_);
v___x_221_ = lean_unsigned_to_nat(1u);
v___x_222_ = lean_nat_add(v_i_214_, v___x_221_);
lean_dec(v_i_214_);
v_i_214_ = v___x_222_;
v_acc_215_ = v___x_220_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg___boxed(lean_object* v_f_224_, lean_object* v_keys_225_, lean_object* v_vals_226_, lean_object* v_i_227_, lean_object* v_acc_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg(v_f_224_, v_keys_225_, v_vals_226_, v_i_227_, v_acc_228_);
lean_dec_ref(v_vals_226_);
lean_dec_ref(v_keys_225_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(lean_object* v_f_230_, lean_object* v_x_231_, lean_object* v_x_232_){
_start:
{
if (lean_obj_tag(v_x_231_) == 0)
{
lean_object* v_es_233_; lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; 
v_es_233_ = lean_ctor_get(v_x_231_, 0);
v___x_234_ = lean_unsigned_to_nat(0u);
v___x_235_ = lean_array_get_size(v_es_233_);
v___x_236_ = lean_nat_dec_lt(v___x_234_, v___x_235_);
if (v___x_236_ == 0)
{
lean_dec(v_f_230_);
return v_x_232_;
}
else
{
uint8_t v___x_237_; 
v___x_237_ = lean_nat_dec_le(v___x_235_, v___x_235_);
if (v___x_237_ == 0)
{
if (v___x_236_ == 0)
{
lean_dec(v_f_230_);
return v_x_232_;
}
else
{
size_t v___x_238_; size_t v___x_239_; lean_object* v___x_240_; 
v___x_238_ = ((size_t)0ULL);
v___x_239_ = lean_usize_of_nat(v___x_235_);
v___x_240_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(v_f_230_, v_es_233_, v___x_238_, v___x_239_, v_x_232_);
return v___x_240_;
}
}
else
{
size_t v___x_241_; size_t v___x_242_; lean_object* v___x_243_; 
v___x_241_ = ((size_t)0ULL);
v___x_242_ = lean_usize_of_nat(v___x_235_);
v___x_243_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(v_f_230_, v_es_233_, v___x_241_, v___x_242_, v_x_232_);
return v___x_243_;
}
}
}
else
{
lean_object* v_ks_244_; lean_object* v_vs_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v_ks_244_ = lean_ctor_get(v_x_231_, 0);
v_vs_245_ = lean_ctor_get(v_x_231_, 1);
v___x_246_ = lean_unsigned_to_nat(0u);
v___x_247_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg(v_f_230_, v_ks_244_, v_vs_245_, v___x_246_, v_x_232_);
return v___x_247_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(lean_object* v_f_248_, lean_object* v_as_249_, size_t v_i_250_, size_t v_stop_251_, lean_object* v_b_252_){
_start:
{
lean_object* v___y_254_; uint8_t v___x_258_; 
v___x_258_ = lean_usize_dec_eq(v_i_250_, v_stop_251_);
if (v___x_258_ == 0)
{
lean_object* v___x_259_; 
v___x_259_ = lean_array_uget_borrowed(v_as_249_, v_i_250_);
switch(lean_obj_tag(v___x_259_))
{
case 0:
{
lean_object* v_key_260_; lean_object* v_val_261_; lean_object* v___x_262_; 
v_key_260_ = lean_ctor_get(v___x_259_, 0);
v_val_261_ = lean_ctor_get(v___x_259_, 1);
lean_inc(v_f_248_);
lean_inc(v_val_261_);
lean_inc(v_key_260_);
v___x_262_ = lean_apply_3(v_f_248_, v_b_252_, v_key_260_, v_val_261_);
v___y_254_ = v___x_262_;
goto v___jp_253_;
}
case 1:
{
lean_object* v_node_263_; lean_object* v___x_264_; 
v_node_263_ = lean_ctor_get(v___x_259_, 0);
lean_inc(v_f_248_);
v___x_264_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_248_, v_node_263_, v_b_252_);
v___y_254_ = v___x_264_;
goto v___jp_253_;
}
default: 
{
v___y_254_ = v_b_252_;
goto v___jp_253_;
}
}
}
else
{
lean_dec(v_f_248_);
return v_b_252_;
}
v___jp_253_:
{
size_t v___x_255_; size_t v___x_256_; 
v___x_255_ = ((size_t)1ULL);
v___x_256_ = lean_usize_add(v_i_250_, v___x_255_);
v_i_250_ = v___x_256_;
v_b_252_ = v___y_254_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg___boxed(lean_object* v_f_265_, lean_object* v_as_266_, lean_object* v_i_267_, lean_object* v_stop_268_, lean_object* v_b_269_){
_start:
{
size_t v_i_boxed_270_; size_t v_stop_boxed_271_; lean_object* v_res_272_; 
v_i_boxed_270_ = lean_unbox_usize(v_i_267_);
lean_dec(v_i_267_);
v_stop_boxed_271_ = lean_unbox_usize(v_stop_268_);
lean_dec(v_stop_268_);
v_res_272_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(v_f_265_, v_as_266_, v_i_boxed_270_, v_stop_boxed_271_, v_b_269_);
lean_dec_ref(v_as_266_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg___boxed(lean_object* v_f_273_, lean_object* v_x_274_, lean_object* v_x_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_273_, v_x_274_, v_x_275_);
lean_dec_ref(v_x_274_);
return v_res_276_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(uint8_t v___x_277_, lean_object* v_x_278_, lean_object* v_y_279_){
_start:
{
uint8_t v___y_281_; lean_object* v_fst_283_; lean_object* v_fst_284_; lean_object* v_name_285_; lean_object* v_prio_286_; lean_object* v_name_287_; lean_object* v_prio_288_; uint8_t v___x_289_; 
v_fst_283_ = lean_ctor_get(v_x_278_, 0);
v_fst_284_ = lean_ctor_get(v_y_279_, 0);
v_name_285_ = lean_ctor_get(v_fst_283_, 1);
v_prio_286_ = lean_ctor_get(v_fst_283_, 3);
v_name_287_ = lean_ctor_get(v_fst_284_, 1);
v_prio_288_ = lean_ctor_get(v_fst_284_, 3);
v___x_289_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_prio_286_, v_prio_288_);
if (v___x_289_ == 1)
{
uint8_t v___x_290_; 
v___x_290_ = lp_aesop_Aesop_RuleName_compare(v_name_285_, v_name_287_);
v___y_281_ = v___x_290_;
goto v___jp_280_;
}
else
{
v___y_281_ = v___x_289_;
goto v___jp_280_;
}
v___jp_280_:
{
if (v___y_281_ == 0)
{
return v___x_277_;
}
else
{
uint8_t v___x_282_; 
v___x_282_ = 0;
return v___x_282_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0___boxed(lean_object* v___x_291_, lean_object* v_x_292_, lean_object* v_y_293_){
_start:
{
uint8_t v___x_4449__boxed_294_; uint8_t v_res_295_; lean_object* v_r_296_; 
v___x_4449__boxed_294_ = lean_unbox(v___x_291_);
v_res_295_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(v___x_4449__boxed_294_, v_x_292_, v_y_293_);
lean_dec_ref(v_y_293_);
lean_dec_ref(v_x_292_);
v_r_296_ = lean_box(v_res_295_);
return v_r_296_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg(lean_object* v_hi_297_, lean_object* v_pivot_298_, lean_object* v_as_299_, lean_object* v_i_300_, lean_object* v_k_301_){
_start:
{
uint8_t v___y_303_; uint8_t v___x_312_; 
v___x_312_ = lean_nat_dec_lt(v_k_301_, v_hi_297_);
if (v___x_312_ == 0)
{
lean_object* v___x_313_; lean_object* v___x_314_; 
lean_dec(v_k_301_);
v___x_313_ = lean_array_fswap(v_as_299_, v_i_300_, v_hi_297_);
v___x_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_314_, 0, v_i_300_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
return v___x_314_;
}
else
{
lean_object* v___x_315_; lean_object* v_fst_316_; lean_object* v_fst_317_; lean_object* v_name_318_; lean_object* v_prio_319_; lean_object* v_name_320_; lean_object* v_prio_321_; uint8_t v___x_322_; 
v___x_315_ = lean_array_fget_borrowed(v_as_299_, v_k_301_);
v_fst_316_ = lean_ctor_get(v___x_315_, 0);
v_fst_317_ = lean_ctor_get(v_pivot_298_, 0);
v_name_318_ = lean_ctor_get(v_fst_316_, 1);
v_prio_319_ = lean_ctor_get(v_fst_316_, 3);
v_name_320_ = lean_ctor_get(v_fst_317_, 1);
v_prio_321_ = lean_ctor_get(v_fst_317_, 3);
v___x_322_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_prio_319_, v_prio_321_);
if (v___x_322_ == 1)
{
uint8_t v___x_323_; 
v___x_323_ = lp_aesop_Aesop_RuleName_compare(v_name_318_, v_name_320_);
v___y_303_ = v___x_323_;
goto v___jp_302_;
}
else
{
v___y_303_ = v___x_322_;
goto v___jp_302_;
}
}
v___jp_302_:
{
if (v___y_303_ == 0)
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_304_ = lean_array_fswap(v_as_299_, v_i_300_, v_k_301_);
v___x_305_ = lean_unsigned_to_nat(1u);
v___x_306_ = lean_nat_add(v_i_300_, v___x_305_);
lean_dec(v_i_300_);
v___x_307_ = lean_nat_add(v_k_301_, v___x_305_);
lean_dec(v_k_301_);
v_as_299_ = v___x_304_;
v_i_300_ = v___x_306_;
v_k_301_ = v___x_307_;
goto _start;
}
else
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = lean_unsigned_to_nat(1u);
v___x_310_ = lean_nat_add(v_k_301_, v___x_309_);
lean_dec(v_k_301_);
v_k_301_ = v___x_310_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg___boxed(lean_object* v_hi_324_, lean_object* v_pivot_325_, lean_object* v_as_326_, lean_object* v_i_327_, lean_object* v_k_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg(v_hi_324_, v_pivot_325_, v_as_326_, v_i_327_, v_k_328_);
lean_dec_ref(v_pivot_325_);
lean_dec(v_hi_324_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(lean_object* v_n_330_, lean_object* v_as_331_, lean_object* v_lo_332_, lean_object* v_hi_333_){
_start:
{
lean_object* v___y_335_; uint8_t v___x_345_; 
v___x_345_ = lean_nat_dec_lt(v_lo_332_, v_hi_333_);
if (v___x_345_ == 0)
{
lean_dec(v_lo_332_);
return v_as_331_;
}
else
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v_mid_348_; lean_object* v___y_350_; lean_object* v___y_356_; lean_object* v___x_361_; lean_object* v___x_362_; uint8_t v___x_363_; 
v___x_346_ = lean_nat_add(v_lo_332_, v_hi_333_);
v___x_347_ = lean_unsigned_to_nat(1u);
v_mid_348_ = lean_nat_shiftr(v___x_346_, v___x_347_);
lean_dec(v___x_346_);
v___x_361_ = lean_array_fget_borrowed(v_as_331_, v_mid_348_);
v___x_362_ = lean_array_fget_borrowed(v_as_331_, v_lo_332_);
v___x_363_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(v___x_345_, v___x_361_, v___x_362_);
if (v___x_363_ == 0)
{
v___y_356_ = v_as_331_;
goto v___jp_355_;
}
else
{
lean_object* v___x_364_; 
v___x_364_ = lean_array_fswap(v_as_331_, v_lo_332_, v_mid_348_);
v___y_356_ = v___x_364_;
goto v___jp_355_;
}
v___jp_349_:
{
lean_object* v___x_351_; lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_351_ = lean_array_fget_borrowed(v___y_350_, v_mid_348_);
v___x_352_ = lean_array_fget_borrowed(v___y_350_, v_hi_333_);
v___x_353_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(v___x_345_, v___x_351_, v___x_352_);
if (v___x_353_ == 0)
{
lean_dec(v_mid_348_);
v___y_335_ = v___y_350_;
goto v___jp_334_;
}
else
{
lean_object* v___x_354_; 
v___x_354_ = lean_array_fswap(v___y_350_, v_mid_348_, v_hi_333_);
lean_dec(v_mid_348_);
v___y_335_ = v___x_354_;
goto v___jp_334_;
}
}
v___jp_355_:
{
lean_object* v___x_357_; lean_object* v___x_358_; uint8_t v___x_359_; 
v___x_357_ = lean_array_fget_borrowed(v___y_356_, v_hi_333_);
v___x_358_ = lean_array_fget_borrowed(v___y_356_, v_lo_332_);
v___x_359_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___lam__0(v___x_345_, v___x_357_, v___x_358_);
if (v___x_359_ == 0)
{
v___y_350_ = v___y_356_;
goto v___jp_349_;
}
else
{
lean_object* v___x_360_; 
v___x_360_ = lean_array_fswap(v___y_356_, v_lo_332_, v_hi_333_);
v___y_350_ = v___x_360_;
goto v___jp_349_;
}
}
}
v___jp_334_:
{
lean_object* v_pivot_336_; lean_object* v___x_337_; lean_object* v_fst_338_; lean_object* v_snd_339_; uint8_t v___x_340_; 
v_pivot_336_ = lean_array_fget(v___y_335_, v_hi_333_);
lean_inc_n(v_lo_332_, 2);
v___x_337_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg(v_hi_333_, v_pivot_336_, v___y_335_, v_lo_332_, v_lo_332_);
lean_dec(v_pivot_336_);
v_fst_338_ = lean_ctor_get(v___x_337_, 0);
lean_inc(v_fst_338_);
v_snd_339_ = lean_ctor_get(v___x_337_, 1);
lean_inc(v_snd_339_);
lean_dec_ref(v___x_337_);
v___x_340_ = lean_nat_dec_le(v_hi_333_, v_fst_338_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_341_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(v_n_330_, v_snd_339_, v_lo_332_, v_fst_338_);
v___x_342_ = lean_unsigned_to_nat(1u);
v___x_343_ = lean_nat_add(v_fst_338_, v___x_342_);
lean_dec(v_fst_338_);
v_as_331_ = v___x_341_;
v_lo_332_ = v___x_343_;
goto _start;
}
else
{
lean_dec(v_fst_338_);
lean_dec(v_lo_332_);
return v_snd_339_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg___boxed(lean_object* v_n_365_, lean_object* v_as_366_, lean_object* v_lo_367_, lean_object* v_hi_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(v_n_365_, v_as_366_, v_lo_367_, v_hi_368_);
lean_dec(v_hi_368_);
lean_dec(v_n_365_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4(lean_object* v_xs_370_){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_371_ = lean_array_get_size(v_xs_370_);
v___x_372_ = lean_unsigned_to_nat(0u);
v___x_373_ = lean_nat_dec_eq(v___x_371_, v___x_372_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___y_377_; uint8_t v___x_381_; 
v___x_374_ = lean_unsigned_to_nat(1u);
v___x_375_ = lean_nat_sub(v___x_371_, v___x_374_);
v___x_381_ = lean_nat_dec_le(v___x_372_, v___x_375_);
if (v___x_381_ == 0)
{
lean_inc(v___x_375_);
v___y_377_ = v___x_375_;
goto v___jp_376_;
}
else
{
v___y_377_ = v___x_372_;
goto v___jp_376_;
}
v___jp_376_:
{
uint8_t v___x_378_; 
v___x_378_ = lean_nat_dec_le(v___y_377_, v___x_375_);
if (v___x_378_ == 0)
{
lean_object* v___x_379_; 
lean_dec(v___x_375_);
lean_inc(v___y_377_);
v___x_379_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(v___x_371_, v_xs_370_, v___y_377_, v___y_377_);
lean_dec(v___y_377_);
return v___x_379_;
}
else
{
lean_object* v___x_380_; 
v___x_380_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(v___x_371_, v_xs_370_, v___y_377_, v___x_375_);
lean_dec(v___x_375_);
return v___x_380_;
}
}
}
else
{
return v_xs_370_;
}
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_382_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__0);
v___x_384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
return v___x_384_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_385_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1);
v___x_386_ = lean_unsigned_to_nat(0u);
v___x_387_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_387_, 0, v___x_386_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
lean_ctor_set(v___x_387_, 2, v___x_386_);
lean_ctor_set(v___x_387_, 3, v___x_386_);
lean_ctor_set(v___x_387_, 4, v___x_385_);
lean_ctor_set(v___x_387_, 5, v___x_385_);
lean_ctor_set(v___x_387_, 6, v___x_385_);
lean_ctor_set(v___x_387_, 7, v___x_385_);
lean_ctor_set(v___x_387_, 8, v___x_385_);
lean_ctor_set(v___x_387_, 9, v___x_385_);
return v___x_387_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; 
v___x_388_ = lean_unsigned_to_nat(32u);
v___x_389_ = lean_mk_empty_array_with_capacity(v___x_388_);
v___x_390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
return v___x_390_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4(void){
_start:
{
size_t v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_391_ = ((size_t)5ULL);
v___x_392_ = lean_unsigned_to_nat(0u);
v___x_393_ = lean_unsigned_to_nat(32u);
v___x_394_ = lean_mk_empty_array_with_capacity(v___x_393_);
v___x_395_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__3);
v___x_396_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_396_, 0, v___x_395_);
lean_ctor_set(v___x_396_, 1, v___x_394_);
lean_ctor_set(v___x_396_, 2, v___x_392_);
lean_ctor_set(v___x_396_, 3, v___x_392_);
lean_ctor_set_usize(v___x_396_, 4, v___x_391_);
return v___x_396_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5(void){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_397_ = lean_box(1);
v___x_398_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__4);
v___x_399_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__1);
v___x_400_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v___x_398_);
lean_ctor_set(v___x_400_, 2, v___x_397_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2(lean_object* v_msgData_401_, lean_object* v___y_402_, lean_object* v___y_403_){
_start:
{
lean_object* v___x_405_; lean_object* v_env_406_; lean_object* v_options_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_405_ = lean_st_ref_get(v___y_403_);
v_env_406_ = lean_ctor_get(v___x_405_, 0);
lean_inc_ref(v_env_406_);
lean_dec(v___x_405_);
v_options_407_ = lean_ctor_get(v___y_402_, 2);
v___x_408_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__2);
v___x_409_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___closed__5);
lean_inc_ref(v_options_407_);
v___x_410_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_410_, 0, v_env_406_);
lean_ctor_set(v___x_410_, 1, v___x_408_);
lean_ctor_set(v___x_410_, 2, v___x_409_);
lean_ctor_set(v___x_410_, 3, v_options_407_);
v___x_411_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_410_);
lean_ctor_set(v___x_411_, 1, v_msgData_401_);
v___x_412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_412_, 0, v___x_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2___boxed(lean_object* v_msgData_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2(v_msgData_413_, v___y_414_, v___y_415_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
return v_res_417_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0(void){
_start:
{
lean_object* v___x_418_; double v___x_419_; 
v___x_418_ = lean_unsigned_to_nat(0u);
v___x_419_ = lean_float_of_nat(v___x_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1(lean_object* v_cls_423_, lean_object* v_msg_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v_ref_428_; lean_object* v___x_429_; lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_474_; 
v_ref_428_ = lean_ctor_get(v___y_425_, 5);
v___x_429_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1_spec__2(v_msg_424_, v___y_425_, v___y_426_);
v_a_430_ = lean_ctor_get(v___x_429_, 0);
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_429_);
if (v_isSharedCheck_474_ == 0)
{
v___x_432_ = v___x_429_;
v_isShared_433_ = v_isSharedCheck_474_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_429_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_474_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_434_; lean_object* v_traceState_435_; lean_object* v_env_436_; lean_object* v_nextMacroScope_437_; lean_object* v_ngen_438_; lean_object* v_auxDeclNGen_439_; lean_object* v_cache_440_; lean_object* v_messages_441_; lean_object* v_infoState_442_; lean_object* v_snapshotTasks_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_473_; 
v___x_434_ = lean_st_ref_take(v___y_426_);
v_traceState_435_ = lean_ctor_get(v___x_434_, 4);
v_env_436_ = lean_ctor_get(v___x_434_, 0);
v_nextMacroScope_437_ = lean_ctor_get(v___x_434_, 1);
v_ngen_438_ = lean_ctor_get(v___x_434_, 2);
v_auxDeclNGen_439_ = lean_ctor_get(v___x_434_, 3);
v_cache_440_ = lean_ctor_get(v___x_434_, 5);
v_messages_441_ = lean_ctor_get(v___x_434_, 6);
v_infoState_442_ = lean_ctor_get(v___x_434_, 7);
v_snapshotTasks_443_ = lean_ctor_get(v___x_434_, 8);
v_isSharedCheck_473_ = !lean_is_exclusive(v___x_434_);
if (v_isSharedCheck_473_ == 0)
{
v___x_445_ = v___x_434_;
v_isShared_446_ = v_isSharedCheck_473_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_snapshotTasks_443_);
lean_inc(v_infoState_442_);
lean_inc(v_messages_441_);
lean_inc(v_cache_440_);
lean_inc(v_traceState_435_);
lean_inc(v_auxDeclNGen_439_);
lean_inc(v_ngen_438_);
lean_inc(v_nextMacroScope_437_);
lean_inc(v_env_436_);
lean_dec(v___x_434_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_473_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
uint64_t v_tid_447_; lean_object* v_traces_448_; lean_object* v___x_450_; uint8_t v_isShared_451_; uint8_t v_isSharedCheck_472_; 
v_tid_447_ = lean_ctor_get_uint64(v_traceState_435_, sizeof(void*)*1);
v_traces_448_ = lean_ctor_get(v_traceState_435_, 0);
v_isSharedCheck_472_ = !lean_is_exclusive(v_traceState_435_);
if (v_isSharedCheck_472_ == 0)
{
v___x_450_ = v_traceState_435_;
v_isShared_451_ = v_isSharedCheck_472_;
goto v_resetjp_449_;
}
else
{
lean_inc(v_traces_448_);
lean_dec(v_traceState_435_);
v___x_450_ = lean_box(0);
v_isShared_451_ = v_isSharedCheck_472_;
goto v_resetjp_449_;
}
v_resetjp_449_:
{
lean_object* v___x_452_; double v___x_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_462_; 
v___x_452_ = lean_box(0);
v___x_453_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__0);
v___x_454_ = 0;
v___x_455_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__1));
v___x_456_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_456_, 0, v_cls_423_);
lean_ctor_set(v___x_456_, 1, v___x_452_);
lean_ctor_set(v___x_456_, 2, v___x_455_);
lean_ctor_set_float(v___x_456_, sizeof(void*)*3, v___x_453_);
lean_ctor_set_float(v___x_456_, sizeof(void*)*3 + 8, v___x_453_);
lean_ctor_set_uint8(v___x_456_, sizeof(void*)*3 + 16, v___x_454_);
v___x_457_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___closed__2));
v___x_458_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_458_, 0, v___x_456_);
lean_ctor_set(v___x_458_, 1, v_a_430_);
lean_ctor_set(v___x_458_, 2, v___x_457_);
lean_inc(v_ref_428_);
v___x_459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_459_, 0, v_ref_428_);
lean_ctor_set(v___x_459_, 1, v___x_458_);
v___x_460_ = l_Lean_PersistentArray_push___redArg(v_traces_448_, v___x_459_);
if (v_isShared_451_ == 0)
{
lean_ctor_set(v___x_450_, 0, v___x_460_);
v___x_462_ = v___x_450_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v___x_460_);
lean_ctor_set_uint64(v_reuseFailAlloc_471_, sizeof(void*)*1, v_tid_447_);
v___x_462_ = v_reuseFailAlloc_471_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
lean_object* v___x_464_; 
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 4, v___x_462_);
v___x_464_ = v___x_445_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_env_436_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v_nextMacroScope_437_);
lean_ctor_set(v_reuseFailAlloc_470_, 2, v_ngen_438_);
lean_ctor_set(v_reuseFailAlloc_470_, 3, v_auxDeclNGen_439_);
lean_ctor_set(v_reuseFailAlloc_470_, 4, v___x_462_);
lean_ctor_set(v_reuseFailAlloc_470_, 5, v_cache_440_);
lean_ctor_set(v_reuseFailAlloc_470_, 6, v_messages_441_);
lean_ctor_set(v_reuseFailAlloc_470_, 7, v_infoState_442_);
lean_ctor_set(v_reuseFailAlloc_470_, 8, v_snapshotTasks_443_);
v___x_464_ = v_reuseFailAlloc_470_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_468_; 
v___x_465_ = lean_st_ref_set(v___y_426_, v___x_464_);
v___x_466_ = lean_box(0);
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 0, v___x_466_);
v___x_468_ = v___x_432_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_466_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1___boxed(lean_object* v_cls_475_, lean_object* v_msg_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1(v_cls_475_, v_msg_476_, v___y_477_, v___y_478_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6(lean_object* v_traceOpt_497_, uint8_t v_a_498_, lean_object* v_as_499_, size_t v_i_500_, size_t v_stop_501_, lean_object* v_b_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
uint8_t v___x_506_; 
v___x_506_ = lean_usize_dec_eq(v_i_500_, v_stop_501_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; lean_object* v_fst_508_; lean_object* v_traceClass_509_; lean_object* v___y_511_; lean_object* v___y_512_; lean_object* v_name_513_; lean_object* v___y_514_; lean_object* v___y_515_; lean_object* v___y_529_; lean_object* v_name_530_; uint8_t v_scope_531_; lean_object* v___y_532_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_540_; lean_object* v_name_541_; uint8_t v_builder_542_; uint8_t v_scope_543_; lean_object* v___y_544_; lean_object* v_name_555_; lean_object* v_prio_556_; lean_object* v___x_557_; lean_object* v___y_559_; 
v___x_507_ = lean_array_uget_borrowed(v_as_499_, v_i_500_);
v_fst_508_ = lean_ctor_get(v___x_507_, 0);
v_traceClass_509_ = lean_ctor_get(v_traceOpt_497_, 0);
v_name_555_ = lean_ctor_get(v_fst_508_, 1);
v_prio_556_ = lean_ctor_get(v_fst_508_, 3);
v___x_557_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__11));
if (lean_obj_tag(v_prio_556_) == 0)
{
lean_object* v_n_570_; lean_object* v___x_571_; 
v_n_570_ = lean_ctor_get(v_prio_556_, 0);
v___x_571_ = l_Int_repr(v_n_570_);
v___y_559_ = v___x_571_;
goto v___jp_558_;
}
else
{
double v_p_572_; lean_object* v___x_573_; 
v_p_572_ = lean_ctor_get_float(v_prio_556_, 0);
v___x_573_ = lp_aesop_Aesop_Percent_toHumanString(v_p_572_);
v___y_559_ = v___x_573_;
goto v___jp_558_;
}
v___jp_510_:
{
lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; 
v___x_516_ = lean_string_append(v___y_512_, v___y_515_);
v___x_517_ = lean_string_append(v___x_516_, v___y_514_);
v___x_518_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_513_, v_a_498_);
v___x_519_ = lean_string_append(v___x_517_, v___x_518_);
lean_dec_ref(v___x_518_);
v___x_520_ = lean_string_append(v___y_511_, v___x_519_);
lean_dec_ref(v___x_519_);
v___x_521_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
v___x_522_ = l_Lean_MessageData_ofFormat(v___x_521_);
lean_inc(v_traceClass_509_);
v___x_523_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardIndex_trace_spec__1(v_traceClass_509_, v___x_522_, v___y_503_, v___y_504_);
if (lean_obj_tag(v___x_523_) == 0)
{
lean_object* v_a_524_; size_t v___x_525_; size_t v___x_526_; 
v_a_524_ = lean_ctor_get(v___x_523_, 0);
lean_inc(v_a_524_);
lean_dec_ref_known(v___x_523_, 1);
v___x_525_ = ((size_t)1ULL);
v___x_526_ = lean_usize_add(v_i_500_, v___x_525_);
v_i_500_ = v___x_526_;
v_b_502_ = v_a_524_;
goto _start;
}
else
{
lean_dec_ref(v_traceOpt_497_);
return v___x_523_;
}
}
v___jp_528_:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = lean_string_append(v___y_533_, v___y_534_);
v___x_536_ = lean_string_append(v___x_535_, v___y_532_);
if (v_scope_531_ == 0)
{
lean_object* v___x_537_; 
v___x_537_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__0));
v___y_511_ = v___y_529_;
v___y_512_ = v___x_536_;
v_name_513_ = v_name_530_;
v___y_514_ = v___y_532_;
v___y_515_ = v___x_537_;
goto v___jp_510_;
}
else
{
lean_object* v___x_538_; 
v___x_538_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__1));
v___y_511_ = v___y_529_;
v___y_512_ = v___x_536_;
v_name_513_ = v_name_530_;
v___y_514_ = v___y_532_;
v___y_515_ = v___x_538_;
goto v___jp_510_;
}
}
v___jp_539_:
{
lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_545_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__2));
lean_inc_ref(v___y_544_);
v___x_546_ = lean_string_append(v___y_544_, v___x_545_);
switch(v_builder_542_)
{
case 0:
{
lean_object* v___x_547_; 
v___x_547_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__3));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_547_;
goto v___jp_528_;
}
case 1:
{
lean_object* v___x_548_; 
v___x_548_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__4));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_548_;
goto v___jp_528_;
}
case 2:
{
lean_object* v___x_549_; 
v___x_549_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__5));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_549_;
goto v___jp_528_;
}
case 3:
{
lean_object* v___x_550_; 
v___x_550_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__6));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_550_;
goto v___jp_528_;
}
case 4:
{
lean_object* v___x_551_; 
v___x_551_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__7));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_551_;
goto v___jp_528_;
}
case 5:
{
lean_object* v___x_552_; 
v___x_552_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__8));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_552_;
goto v___jp_528_;
}
case 6:
{
lean_object* v___x_553_; 
v___x_553_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__9));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_553_;
goto v___jp_528_;
}
default: 
{
lean_object* v___x_554_; 
v___x_554_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__10));
v___y_529_ = v___y_540_;
v_name_530_ = v_name_541_;
v_scope_531_ = v_scope_543_;
v___y_532_ = v___x_545_;
v___y_533_ = v___x_546_;
v___y_534_ = v___x_554_;
goto v___jp_528_;
}
}
}
v___jp_558_:
{
lean_object* v_name_560_; uint8_t v_builder_561_; uint8_t v_phase_562_; uint8_t v_scope_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; 
v_name_560_ = lean_ctor_get(v_name_555_, 0);
v_builder_561_ = lean_ctor_get_uint8(v_name_555_, sizeof(void*)*1 + 8);
v_phase_562_ = lean_ctor_get_uint8(v_name_555_, sizeof(void*)*1 + 9);
v_scope_563_ = lean_ctor_get_uint8(v_name_555_, sizeof(void*)*1 + 10);
v___x_564_ = lean_string_append(v___x_557_, v___y_559_);
lean_dec_ref(v___y_559_);
v___x_565_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__12));
v___x_566_ = lean_string_append(v___x_564_, v___x_565_);
switch(v_phase_562_)
{
case 0:
{
lean_object* v___x_567_; 
v___x_567_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__13));
lean_inc(v_name_560_);
v___y_540_ = v___x_566_;
v_name_541_ = v_name_560_;
v_builder_542_ = v_builder_561_;
v_scope_543_ = v_scope_563_;
v___y_544_ = v___x_567_;
goto v___jp_539_;
}
case 1:
{
lean_object* v___x_568_; 
v___x_568_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__14));
lean_inc(v_name_560_);
v___y_540_ = v___x_566_;
v_name_541_ = v_name_560_;
v_builder_542_ = v_builder_561_;
v_scope_543_ = v_scope_563_;
v___y_544_ = v___x_568_;
goto v___jp_539_;
}
default: 
{
lean_object* v___x_569_; 
v___x_569_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___closed__15));
lean_inc(v_name_560_);
v___y_540_ = v___x_566_;
v_name_541_ = v_name_560_;
v_builder_542_ = v_builder_561_;
v_scope_543_ = v_scope_563_;
v___y_544_ = v___x_569_;
goto v___jp_539_;
}
}
}
}
else
{
lean_object* v___x_574_; 
lean_dec_ref(v_traceOpt_497_);
v___x_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_574_, 0, v_b_502_);
return v___x_574_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6___boxed(lean_object* v_traceOpt_575_, lean_object* v_a_576_, lean_object* v_as_577_, lean_object* v_i_578_, lean_object* v_stop_579_, lean_object* v_b_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_){
_start:
{
uint8_t v_a_4805__boxed_584_; size_t v_i_boxed_585_; size_t v_stop_boxed_586_; lean_object* v_res_587_; 
v_a_4805__boxed_584_ = lean_unbox(v_a_576_);
v_i_boxed_585_ = lean_unbox_usize(v_i_578_);
lean_dec(v_i_578_);
v_stop_boxed_586_ = lean_unbox_usize(v_stop_579_);
lean_dec(v_stop_579_);
v_res_587_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6(v_traceOpt_575_, v_a_4805__boxed_584_, v_as_577_, v_i_boxed_585_, v_stop_boxed_586_, v_b_580_, v___y_581_, v___y_582_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
lean_dec_ref(v_as_577_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace(lean_object* v_traceOpt_593_, lean_object* v_idx_594_, lean_object* v_a_595_, lean_object* v_a_596_){
_start:
{
lean_object* v___x_598_; lean_object* v_a_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_633_; 
v___x_598_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg(v_traceOpt_593_, v_a_595_);
v_a_599_ = lean_ctor_get(v___x_598_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_598_);
if (v_isSharedCheck_633_ == 0)
{
v___x_601_ = v___x_598_;
v_isShared_602_ = v_isSharedCheck_633_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_a_599_);
lean_dec(v___x_598_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_633_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
uint8_t v___x_603_; 
v___x_603_ = lean_unbox(v_a_599_);
if (v___x_603_ == 0)
{
lean_object* v___x_604_; lean_object* v___x_606_; 
lean_dec(v_a_599_);
lean_dec_ref(v_traceOpt_593_);
v___x_604_ = lean_box(0);
if (v_isShared_602_ == 0)
{
lean_ctor_set(v___x_601_, 0, v___x_604_);
v___x_606_ = v___x_601_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_604_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
else
{
lean_object* v_tree_608_; lean_object* v___f_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; uint8_t v___x_617_; 
v_tree_608_ = lean_ctor_get(v_idx_594_, 0);
v___f_609_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_trace___closed__1));
v___x_610_ = lean_unsigned_to_nat(0u);
v___x_611_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_trace___closed__2));
v___x_612_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v___f_609_, v_tree_608_, v___x_611_);
v___x_613_ = lp_aesop_Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4(v___x_612_);
v___x_614_ = lp_aesop_Array_dedupSorted___at___00Aesop_ForwardIndex_trace_spec__5(v___x_613_);
lean_dec_ref(v___x_613_);
v___x_615_ = lean_array_get_size(v___x_614_);
v___x_616_ = lean_box(0);
v___x_617_ = lean_nat_dec_lt(v___x_610_, v___x_615_);
if (v___x_617_ == 0)
{
lean_object* v___x_619_; 
lean_dec_ref(v___x_614_);
lean_dec(v_a_599_);
lean_dec_ref(v_traceOpt_593_);
if (v_isShared_602_ == 0)
{
lean_ctor_set(v___x_601_, 0, v___x_616_);
v___x_619_ = v___x_601_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v___x_616_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
else
{
uint8_t v___x_621_; 
v___x_621_ = lean_nat_dec_le(v___x_615_, v___x_615_);
if (v___x_621_ == 0)
{
if (v___x_617_ == 0)
{
lean_object* v___x_623_; 
lean_dec_ref(v___x_614_);
lean_dec(v_a_599_);
lean_dec_ref(v_traceOpt_593_);
if (v_isShared_602_ == 0)
{
lean_ctor_set(v___x_601_, 0, v___x_616_);
v___x_623_ = v___x_601_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v___x_616_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
else
{
size_t v___x_625_; size_t v___x_626_; uint8_t v___x_627_; lean_object* v___x_628_; 
lean_del_object(v___x_601_);
v___x_625_ = ((size_t)0ULL);
v___x_626_ = lean_usize_of_nat(v___x_615_);
v___x_627_ = lean_unbox(v_a_599_);
lean_dec(v_a_599_);
v___x_628_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6(v_traceOpt_593_, v___x_627_, v___x_614_, v___x_625_, v___x_626_, v___x_616_, v_a_595_, v_a_596_);
lean_dec_ref(v___x_614_);
return v___x_628_;
}
}
else
{
size_t v___x_629_; size_t v___x_630_; uint8_t v___x_631_; lean_object* v___x_632_; 
lean_del_object(v___x_601_);
v___x_629_ = ((size_t)0ULL);
v___x_630_ = lean_usize_of_nat(v___x_615_);
v___x_631_ = lean_unbox(v_a_599_);
lean_dec(v_a_599_);
v___x_632_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardIndex_trace_spec__6(v_traceOpt_593_, v___x_631_, v___x_614_, v___x_629_, v___x_630_, v___x_616_, v_a_595_, v_a_596_);
lean_dec_ref(v___x_614_);
return v___x_632_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_trace___boxed(lean_object* v_traceOpt_634_, lean_object* v_idx_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_aesop_Aesop_ForwardIndex_trace(v_traceOpt_634_, v_idx_635_, v_a_636_, v_a_637_);
lean_dec(v_a_637_);
lean_dec_ref(v_a_636_);
lean_dec_ref(v_idx_635_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0(lean_object* v_opt_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___redArg(v_opt_640_, v___y_641_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0___boxed(lean_object* v_opt_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardIndex_trace_spec__0(v_opt_645_, v___y_646_, v___y_647_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec_ref(v_opt_645_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2(lean_object* v_00_u03c3_650_, lean_object* v_00_u03b1_651_, lean_object* v_f_652_, lean_object* v_x_653_, lean_object* v_x_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___redArg(v_f_652_, v_x_653_, v_x_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2___boxed(lean_object* v_00_u03c3_656_, lean_object* v_00_u03b1_657_, lean_object* v_f_658_, lean_object* v_x_659_, lean_object* v_x_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_aesop_Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2(v_00_u03c3_656_, v_00_u03b1_657_, v_f_658_, v_x_659_, v_x_660_);
lean_dec_ref(v_x_660_);
return v_res_661_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___redArg(lean_object* v_map_662_, lean_object* v_f_663_, lean_object* v_init_664_){
_start:
{
lean_object* v___x_665_; 
v___x_665_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_663_, v_map_662_, v_init_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___redArg___boxed(lean_object* v_map_666_, lean_object* v_f_667_, lean_object* v_init_668_){
_start:
{
lean_object* v_res_669_; 
v_res_669_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___redArg(v_map_666_, v_f_667_, v_init_668_);
lean_dec_ref(v_map_666_);
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3(lean_object* v_00_u03c3_670_, lean_object* v_00_u03b2_671_, lean_object* v_map_672_, lean_object* v_f_673_, lean_object* v_init_674_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_673_, v_map_672_, v_init_674_);
return v___x_675_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3___boxed(lean_object* v_00_u03c3_676_, lean_object* v_00_u03b2_677_, lean_object* v_map_678_, lean_object* v_f_679_, lean_object* v_init_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3(v_00_u03c3_676_, v_00_u03b2_677_, v_map_678_, v_f_679_, v_init_680_);
lean_dec_ref(v_map_678_);
return v_res_681_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4(lean_object* v_00_u03b1_682_, lean_object* v_00_u03c3_683_, lean_object* v_f_684_, lean_object* v_as_685_, size_t v_i_686_, size_t v_stop_687_, lean_object* v_b_688_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___redArg(v_f_684_, v_as_685_, v_i_686_, v_stop_687_, v_b_688_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4___boxed(lean_object* v_00_u03b1_690_, lean_object* v_00_u03c3_691_, lean_object* v_f_692_, lean_object* v_as_693_, lean_object* v_i_694_, lean_object* v_stop_695_, lean_object* v_b_696_){
_start:
{
size_t v_i_boxed_697_; size_t v_stop_boxed_698_; lean_object* v_res_699_; 
v_i_boxed_697_ = lean_unbox_usize(v_i_694_);
lean_dec(v_i_694_);
v_stop_boxed_698_ = lean_unbox_usize(v_stop_695_);
lean_dec(v_stop_695_);
v_res_699_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__4(v_00_u03b1_690_, v_00_u03c3_691_, v_f_692_, v_as_693_, v_i_boxed_697_, v_stop_boxed_698_, v_b_696_);
lean_dec_ref(v_as_693_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5(lean_object* v_00_u03b1_700_, lean_object* v_00_u03c3_701_, lean_object* v_f_702_, lean_object* v_as_703_, size_t v_i_704_, size_t v_stop_705_, lean_object* v_b_706_){
_start:
{
lean_object* v___x_707_; 
v___x_707_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___redArg(v_f_702_, v_as_703_, v_i_704_, v_stop_705_, v_b_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5___boxed(lean_object* v_00_u03b1_708_, lean_object* v_00_u03c3_709_, lean_object* v_f_710_, lean_object* v_as_711_, lean_object* v_i_712_, lean_object* v_stop_713_, lean_object* v_b_714_){
_start:
{
size_t v_i_boxed_715_; size_t v_stop_boxed_716_; lean_object* v_res_717_; 
v_i_boxed_715_ = lean_unbox_usize(v_i_712_);
lean_dec(v_i_712_);
v_stop_boxed_716_ = lean_unbox_usize(v_stop_713_);
lean_dec(v_stop_713_);
v_res_717_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_DiscrTree_Trie_foldValuesM___at___00Aesop_ForwardIndex_trace_spec__2_spec__5(v_00_u03b1_708_, v_00_u03c3_709_, v_f_710_, v_as_711_, v_i_boxed_715_, v_stop_boxed_716_, v_b_714_);
lean_dec_ref(v_as_711_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7(lean_object* v_00_u03c3_718_, lean_object* v_00_u03b1_719_, lean_object* v_00_u03b2_720_, lean_object* v_f_721_, lean_object* v_x_722_, lean_object* v_x_723_){
_start:
{
lean_object* v___x_724_; 
v___x_724_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_721_, v_x_722_, v_x_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___boxed(lean_object* v_00_u03c3_725_, lean_object* v_00_u03b1_726_, lean_object* v_00_u03b2_727_, lean_object* v_f_728_, lean_object* v_x_729_, lean_object* v_x_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7(v_00_u03c3_725_, v_00_u03b1_726_, v_00_u03b2_727_, v_f_728_, v_x_729_, v_x_730_);
lean_dec_ref(v_x_729_);
return v_res_731_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9(lean_object* v_n_732_, lean_object* v_as_733_, lean_object* v_lo_734_, lean_object* v_hi_735_, lean_object* v_w_736_, lean_object* v_hlo_737_, lean_object* v_hhi_738_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___redArg(v_n_732_, v_as_733_, v_lo_734_, v_hi_735_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9___boxed(lean_object* v_n_740_, lean_object* v_as_741_, lean_object* v_lo_742_, lean_object* v_hi_743_, lean_object* v_w_744_, lean_object* v_hlo_745_, lean_object* v_hhi_746_){
_start:
{
lean_object* v_res_747_; 
v_res_747_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9(v_n_740_, v_as_741_, v_lo_742_, v_hi_743_, v_w_744_, v_hlo_745_, v_hhi_746_);
lean_dec(v_hi_743_);
lean_dec(v_n_740_);
return v_res_747_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8(lean_object* v_00_u03b1_748_, lean_object* v_00_u03b2_749_, lean_object* v_00_u03c3_750_, lean_object* v_f_751_, lean_object* v_as_752_, size_t v_i_753_, size_t v_stop_754_, lean_object* v_b_755_){
_start:
{
lean_object* v___x_756_; 
v___x_756_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___redArg(v_f_751_, v_as_752_, v_i_753_, v_stop_754_, v_b_755_);
return v___x_756_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8___boxed(lean_object* v_00_u03b1_757_, lean_object* v_00_u03b2_758_, lean_object* v_00_u03c3_759_, lean_object* v_f_760_, lean_object* v_as_761_, lean_object* v_i_762_, lean_object* v_stop_763_, lean_object* v_b_764_){
_start:
{
size_t v_i_boxed_765_; size_t v_stop_boxed_766_; lean_object* v_res_767_; 
v_i_boxed_765_ = lean_unbox_usize(v_i_762_);
lean_dec(v_i_762_);
v_stop_boxed_766_ = lean_unbox_usize(v_stop_763_);
lean_dec(v_stop_763_);
v_res_767_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__8(v_00_u03b1_757_, v_00_u03b2_758_, v_00_u03c3_759_, v_f_760_, v_as_761_, v_i_boxed_765_, v_stop_boxed_766_, v_b_764_);
lean_dec_ref(v_as_761_);
return v_res_767_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9(lean_object* v_00_u03c3_768_, lean_object* v_00_u03b1_769_, lean_object* v_00_u03b2_770_, lean_object* v_f_771_, lean_object* v_keys_772_, lean_object* v_vals_773_, lean_object* v_heq_774_, lean_object* v_i_775_, lean_object* v_acc_776_){
_start:
{
lean_object* v___x_777_; 
v___x_777_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___redArg(v_f_771_, v_keys_772_, v_vals_773_, v_i_775_, v_acc_776_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9___boxed(lean_object* v_00_u03c3_778_, lean_object* v_00_u03b1_779_, lean_object* v_00_u03b2_780_, lean_object* v_f_781_, lean_object* v_keys_782_, lean_object* v_vals_783_, lean_object* v_heq_784_, lean_object* v_i_785_, lean_object* v_acc_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7_spec__9(v_00_u03c3_778_, v_00_u03b1_779_, v_00_u03b2_780_, v_f_781_, v_keys_782_, v_vals_783_, v_heq_784_, v_i_785_, v_acc_786_);
lean_dec_ref(v_vals_783_);
lean_dec_ref(v_keys_782_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12(lean_object* v_n_788_, lean_object* v_lo_789_, lean_object* v_hi_790_, lean_object* v_hhi_791_, lean_object* v_pivot_792_, lean_object* v_as_793_, lean_object* v_i_794_, lean_object* v_k_795_, lean_object* v_ilo_796_, lean_object* v_ik_797_, lean_object* v_w_798_){
_start:
{
lean_object* v___x_799_; 
v___x_799_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___redArg(v_hi_790_, v_pivot_792_, v_as_793_, v_i_794_, v_k_795_);
return v___x_799_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12___boxed(lean_object* v_n_800_, lean_object* v_lo_801_, lean_object* v_hi_802_, lean_object* v_hhi_803_, lean_object* v_pivot_804_, lean_object* v_as_805_, lean_object* v_i_806_, lean_object* v_k_807_, lean_object* v_ilo_808_, lean_object* v_ik_809_, lean_object* v_w_810_){
_start:
{
lean_object* v_res_811_; 
v_res_811_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_ForwardIndex_trace_spec__4_spec__9_spec__12(v_n_800_, v_lo_801_, v_hi_802_, v_hhi_803_, v_pivot_804_, v_as_805_, v_i_806_, v_k_807_, v_ilo_808_, v_ik_809_, v_w_810_);
lean_dec_ref(v_pivot_804_);
lean_dec(v_hi_802_);
lean_dec(v_lo_801_);
lean_dec(v_n_800_);
return v_res_811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_812_, lean_object* v_vals_813_, lean_object* v_i_814_, lean_object* v_k_815_){
_start:
{
uint8_t v___y_821_; lean_object* v___x_824_; uint8_t v___x_825_; 
v___x_824_ = lean_array_get_size(v_keys_812_);
v___x_825_ = lean_nat_dec_lt(v_i_814_, v___x_824_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; 
lean_dec(v_i_814_);
v___x_826_ = lean_box(0);
return v___x_826_;
}
else
{
lean_object* v_name_827_; uint8_t v_builder_828_; uint8_t v_phase_829_; uint8_t v_scope_830_; uint64_t v_hash_831_; lean_object* v_k_x27_832_; uint8_t v___y_834_; uint8_t v_builder_841_; uint64_t v_hash_842_; uint8_t v___x_843_; 
v_name_827_ = lean_ctor_get(v_k_815_, 0);
v_builder_828_ = lean_ctor_get_uint8(v_k_815_, sizeof(void*)*1 + 8);
v_phase_829_ = lean_ctor_get_uint8(v_k_815_, sizeof(void*)*1 + 9);
v_scope_830_ = lean_ctor_get_uint8(v_k_815_, sizeof(void*)*1 + 10);
v_hash_831_ = lean_ctor_get_uint64(v_k_815_, sizeof(void*)*1);
v_k_x27_832_ = lean_array_fget_borrowed(v_keys_812_, v_i_814_);
v_builder_841_ = lean_ctor_get_uint8(v_k_x27_832_, sizeof(void*)*1 + 8);
v_hash_842_ = lean_ctor_get_uint64(v_k_x27_832_, sizeof(void*)*1);
v___x_843_ = lean_uint64_dec_eq(v_hash_831_, v_hash_842_);
if (v___x_843_ == 0)
{
v___y_834_ = v___x_843_;
goto v___jp_833_;
}
else
{
uint8_t v___x_844_; 
v___x_844_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_828_, v_builder_841_);
v___y_834_ = v___x_844_;
goto v___jp_833_;
}
v___jp_833_:
{
if (v___y_834_ == 0)
{
goto v___jp_816_;
}
else
{
lean_object* v_name_835_; uint8_t v_phase_836_; uint8_t v_scope_837_; uint8_t v___x_838_; 
v_name_835_ = lean_ctor_get(v_k_x27_832_, 0);
v_phase_836_ = lean_ctor_get_uint8(v_k_x27_832_, sizeof(void*)*1 + 9);
v_scope_837_ = lean_ctor_get_uint8(v_k_x27_832_, sizeof(void*)*1 + 10);
v___x_838_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_829_, v_phase_836_);
if (v___x_838_ == 0)
{
v___y_821_ = v___x_838_;
goto v___jp_820_;
}
else
{
uint8_t v___x_839_; 
v___x_839_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_830_, v_scope_837_);
if (v___x_839_ == 0)
{
v___y_821_ = v___x_839_;
goto v___jp_820_;
}
else
{
uint8_t v___x_840_; 
v___x_840_ = lean_name_eq(v_name_827_, v_name_835_);
v___y_821_ = v___x_840_;
goto v___jp_820_;
}
}
}
}
}
v___jp_816_:
{
lean_object* v___x_817_; lean_object* v___x_818_; 
v___x_817_ = lean_unsigned_to_nat(1u);
v___x_818_ = lean_nat_add(v_i_814_, v___x_817_);
lean_dec(v_i_814_);
v_i_814_ = v___x_818_;
goto _start;
}
v___jp_820_:
{
if (v___y_821_ == 0)
{
goto v___jp_816_;
}
else
{
lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_822_ = lean_array_fget_borrowed(v_vals_813_, v_i_814_);
lean_dec(v_i_814_);
lean_inc(v___x_822_);
v___x_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_823_, 0, v___x_822_);
return v___x_823_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_845_, lean_object* v_vals_846_, lean_object* v_i_847_, lean_object* v_k_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg(v_keys_845_, v_vals_846_, v_i_847_, v_k_848_);
lean_dec_ref(v_k_848_);
lean_dec_ref(v_vals_846_);
lean_dec_ref(v_keys_845_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg(lean_object* v_x_850_, size_t v_x_851_, lean_object* v_x_852_){
_start:
{
if (lean_obj_tag(v_x_850_) == 0)
{
lean_object* v_es_853_; lean_object* v___x_854_; size_t v___x_855_; size_t v___x_856_; lean_object* v_j_857_; lean_object* v___x_858_; 
v_es_853_ = lean_ctor_get(v_x_850_, 0);
v___x_854_ = lean_box(2);
v___x_855_ = ((size_t)31ULL);
v___x_856_ = lean_usize_land(v_x_851_, v___x_855_);
v_j_857_ = lean_usize_to_nat(v___x_856_);
v___x_858_ = lean_array_get_borrowed(v___x_854_, v_es_853_, v_j_857_);
lean_dec(v_j_857_);
switch(lean_obj_tag(v___x_858_))
{
case 0:
{
lean_object* v_key_859_; lean_object* v_val_860_; uint8_t v___y_862_; lean_object* v_name_865_; uint8_t v_builder_866_; uint8_t v_phase_867_; uint8_t v_scope_868_; uint64_t v_hash_869_; lean_object* v_name_870_; uint8_t v_builder_871_; uint8_t v_phase_872_; uint8_t v_scope_873_; uint64_t v_hash_874_; uint8_t v___y_876_; uint8_t v___x_881_; 
v_key_859_ = lean_ctor_get(v___x_858_, 0);
v_val_860_ = lean_ctor_get(v___x_858_, 1);
v_name_865_ = lean_ctor_get(v_x_852_, 0);
v_builder_866_ = lean_ctor_get_uint8(v_x_852_, sizeof(void*)*1 + 8);
v_phase_867_ = lean_ctor_get_uint8(v_x_852_, sizeof(void*)*1 + 9);
v_scope_868_ = lean_ctor_get_uint8(v_x_852_, sizeof(void*)*1 + 10);
v_hash_869_ = lean_ctor_get_uint64(v_x_852_, sizeof(void*)*1);
v_name_870_ = lean_ctor_get(v_key_859_, 0);
v_builder_871_ = lean_ctor_get_uint8(v_key_859_, sizeof(void*)*1 + 8);
v_phase_872_ = lean_ctor_get_uint8(v_key_859_, sizeof(void*)*1 + 9);
v_scope_873_ = lean_ctor_get_uint8(v_key_859_, sizeof(void*)*1 + 10);
v_hash_874_ = lean_ctor_get_uint64(v_key_859_, sizeof(void*)*1);
v___x_881_ = lean_uint64_dec_eq(v_hash_869_, v_hash_874_);
if (v___x_881_ == 0)
{
v___y_876_ = v___x_881_;
goto v___jp_875_;
}
else
{
uint8_t v___x_882_; 
v___x_882_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_866_, v_builder_871_);
v___y_876_ = v___x_882_;
goto v___jp_875_;
}
v___jp_861_:
{
if (v___y_862_ == 0)
{
lean_object* v___x_863_; 
v___x_863_ = lean_box(0);
return v___x_863_;
}
else
{
lean_object* v___x_864_; 
lean_inc(v_val_860_);
v___x_864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_864_, 0, v_val_860_);
return v___x_864_;
}
}
v___jp_875_:
{
if (v___y_876_ == 0)
{
lean_object* v___x_877_; 
v___x_877_ = lean_box(0);
return v___x_877_;
}
else
{
uint8_t v___x_878_; 
v___x_878_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_867_, v_phase_872_);
if (v___x_878_ == 0)
{
v___y_862_ = v___x_878_;
goto v___jp_861_;
}
else
{
uint8_t v___x_879_; 
v___x_879_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_868_, v_scope_873_);
if (v___x_879_ == 0)
{
v___y_862_ = v___x_879_;
goto v___jp_861_;
}
else
{
uint8_t v___x_880_; 
v___x_880_ = lean_name_eq(v_name_865_, v_name_870_);
v___y_862_ = v___x_880_;
goto v___jp_861_;
}
}
}
}
}
case 1:
{
lean_object* v_node_883_; size_t v___x_884_; size_t v___x_885_; 
v_node_883_ = lean_ctor_get(v___x_858_, 0);
v___x_884_ = ((size_t)5ULL);
v___x_885_ = lean_usize_shift_right(v_x_851_, v___x_884_);
v_x_850_ = v_node_883_;
v_x_851_ = v___x_885_;
goto _start;
}
default: 
{
lean_object* v___x_887_; 
v___x_887_ = lean_box(0);
return v___x_887_;
}
}
}
else
{
lean_object* v_ks_888_; lean_object* v_vs_889_; lean_object* v___x_890_; lean_object* v___x_891_; 
v_ks_888_ = lean_ctor_get(v_x_850_, 0);
v_vs_889_ = lean_ctor_get(v_x_850_, 1);
v___x_890_ = lean_unsigned_to_nat(0u);
v___x_891_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg(v_ks_888_, v_vs_889_, v___x_890_, v_x_852_);
return v___x_891_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg___boxed(lean_object* v_x_892_, lean_object* v_x_893_, lean_object* v_x_894_){
_start:
{
size_t v_x_1459__boxed_895_; lean_object* v_res_896_; 
v_x_1459__boxed_895_ = lean_unbox_usize(v_x_893_);
lean_dec(v_x_893_);
v_res_896_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg(v_x_892_, v_x_1459__boxed_895_, v_x_894_);
lean_dec_ref(v_x_894_);
lean_dec_ref(v_x_892_);
return v_res_896_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(lean_object* v_x_897_, lean_object* v_x_898_){
_start:
{
uint64_t v_hash_899_; size_t v___x_900_; lean_object* v___x_901_; 
v_hash_899_ = lean_ctor_get_uint64(v_x_898_, sizeof(void*)*1);
v___x_900_ = lean_uint64_to_usize(v_hash_899_);
v___x_901_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg(v_x_897_, v___x_900_, v_x_898_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg___boxed(lean_object* v_x_902_, lean_object* v_x_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(v_x_902_, v_x_903_);
lean_dec_ref(v_x_903_);
lean_dec_ref(v_x_902_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11___redArg(lean_object* v_x_905_, lean_object* v_x_906_, lean_object* v_x_907_, lean_object* v_x_908_){
_start:
{
lean_object* v_ks_909_; lean_object* v_vs_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_949_; 
v_ks_909_ = lean_ctor_get(v_x_905_, 0);
v_vs_910_ = lean_ctor_get(v_x_905_, 1);
v_isSharedCheck_949_ = !lean_is_exclusive(v_x_905_);
if (v_isSharedCheck_949_ == 0)
{
v___x_912_ = v_x_905_;
v_isShared_913_ = v_isSharedCheck_949_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_vs_910_);
lean_inc(v_ks_909_);
lean_dec(v_x_905_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_949_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
uint8_t v___y_922_; lean_object* v___x_926_; uint8_t v___x_927_; 
v___x_926_ = lean_array_get_size(v_ks_909_);
v___x_927_ = lean_nat_dec_lt(v_x_906_, v___x_926_);
if (v___x_927_ == 0)
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
lean_del_object(v___x_912_);
lean_dec(v_x_906_);
v___x_928_ = lean_array_push(v_ks_909_, v_x_907_);
v___x_929_ = lean_array_push(v_vs_910_, v_x_908_);
v___x_930_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_928_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
return v___x_930_;
}
else
{
lean_object* v_name_931_; uint8_t v_builder_932_; uint8_t v_phase_933_; uint8_t v_scope_934_; uint64_t v_hash_935_; lean_object* v_k_x27_936_; uint8_t v___y_938_; uint8_t v_builder_945_; uint64_t v_hash_946_; uint8_t v___x_947_; 
v_name_931_ = lean_ctor_get(v_x_907_, 0);
v_builder_932_ = lean_ctor_get_uint8(v_x_907_, sizeof(void*)*1 + 8);
v_phase_933_ = lean_ctor_get_uint8(v_x_907_, sizeof(void*)*1 + 9);
v_scope_934_ = lean_ctor_get_uint8(v_x_907_, sizeof(void*)*1 + 10);
v_hash_935_ = lean_ctor_get_uint64(v_x_907_, sizeof(void*)*1);
v_k_x27_936_ = lean_array_fget_borrowed(v_ks_909_, v_x_906_);
v_builder_945_ = lean_ctor_get_uint8(v_k_x27_936_, sizeof(void*)*1 + 8);
v_hash_946_ = lean_ctor_get_uint64(v_k_x27_936_, sizeof(void*)*1);
v___x_947_ = lean_uint64_dec_eq(v_hash_935_, v_hash_946_);
if (v___x_947_ == 0)
{
v___y_938_ = v___x_947_;
goto v___jp_937_;
}
else
{
uint8_t v___x_948_; 
v___x_948_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_932_, v_builder_945_);
v___y_938_ = v___x_948_;
goto v___jp_937_;
}
v___jp_937_:
{
if (v___y_938_ == 0)
{
goto v___jp_914_;
}
else
{
lean_object* v_name_939_; uint8_t v_phase_940_; uint8_t v_scope_941_; uint8_t v___x_942_; 
v_name_939_ = lean_ctor_get(v_k_x27_936_, 0);
v_phase_940_ = lean_ctor_get_uint8(v_k_x27_936_, sizeof(void*)*1 + 9);
v_scope_941_ = lean_ctor_get_uint8(v_k_x27_936_, sizeof(void*)*1 + 10);
v___x_942_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_933_, v_phase_940_);
if (v___x_942_ == 0)
{
v___y_922_ = v___x_942_;
goto v___jp_921_;
}
else
{
uint8_t v___x_943_; 
v___x_943_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_934_, v_scope_941_);
if (v___x_943_ == 0)
{
v___y_922_ = v___x_943_;
goto v___jp_921_;
}
else
{
uint8_t v___x_944_; 
v___x_944_ = lean_name_eq(v_name_931_, v_name_939_);
v___y_922_ = v___x_944_;
goto v___jp_921_;
}
}
}
}
}
v___jp_914_:
{
lean_object* v___x_916_; 
if (v_isShared_913_ == 0)
{
v___x_916_ = v___x_912_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_920_; 
v_reuseFailAlloc_920_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_920_, 0, v_ks_909_);
lean_ctor_set(v_reuseFailAlloc_920_, 1, v_vs_910_);
v___x_916_ = v_reuseFailAlloc_920_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
lean_object* v___x_917_; lean_object* v___x_918_; 
v___x_917_ = lean_unsigned_to_nat(1u);
v___x_918_ = lean_nat_add(v_x_906_, v___x_917_);
lean_dec(v_x_906_);
v_x_905_ = v___x_916_;
v_x_906_ = v___x_918_;
goto _start;
}
}
v___jp_921_:
{
if (v___y_922_ == 0)
{
goto v___jp_914_;
}
else
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
lean_del_object(v___x_912_);
v___x_923_ = lean_array_fset(v_ks_909_, v_x_906_, v_x_907_);
v___x_924_ = lean_array_fset(v_vs_910_, v_x_906_, v_x_908_);
lean_dec(v_x_906_);
v___x_925_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_925_, 0, v___x_923_);
lean_ctor_set(v___x_925_, 1, v___x_924_);
return v___x_925_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4___redArg(lean_object* v_n_950_, lean_object* v_k_951_, lean_object* v_v_952_){
_start:
{
lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_953_ = lean_unsigned_to_nat(0u);
v___x_954_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11___redArg(v_n_950_, v___x_953_, v_k_951_, v_v_952_);
return v___x_954_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(lean_object* v_x_956_, size_t v_x_957_, size_t v_x_958_, lean_object* v_x_959_, lean_object* v_x_960_){
_start:
{
if (lean_obj_tag(v_x_956_) == 0)
{
lean_object* v_es_961_; size_t v___x_962_; size_t v___x_963_; lean_object* v_j_964_; lean_object* v___x_965_; uint8_t v___x_966_; 
v_es_961_ = lean_ctor_get(v_x_956_, 0);
v___x_962_ = ((size_t)31ULL);
v___x_963_ = lean_usize_land(v_x_957_, v___x_962_);
v_j_964_ = lean_usize_to_nat(v___x_963_);
v___x_965_ = lean_array_get_size(v_es_961_);
v___x_966_ = lean_nat_dec_lt(v_j_964_, v___x_965_);
if (v___x_966_ == 0)
{
lean_dec(v_j_964_);
lean_dec(v_x_960_);
lean_dec_ref(v_x_959_);
return v_x_956_;
}
else
{
lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_1024_; 
lean_inc_ref(v_es_961_);
v_isSharedCheck_1024_ = !lean_is_exclusive(v_x_956_);
if (v_isSharedCheck_1024_ == 0)
{
lean_object* v_unused_1025_; 
v_unused_1025_ = lean_ctor_get(v_x_956_, 0);
lean_dec(v_unused_1025_);
v___x_968_ = v_x_956_;
v_isShared_969_ = v_isSharedCheck_1024_;
goto v_resetjp_967_;
}
else
{
lean_dec(v_x_956_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_1024_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v_v_970_; lean_object* v___x_971_; lean_object* v_xs_x27_972_; lean_object* v___y_974_; 
v_v_970_ = lean_array_fget(v_es_961_, v_j_964_);
v___x_971_ = lean_box(0);
v_xs_x27_972_ = lean_array_fset(v_es_961_, v_j_964_, v___x_971_);
switch(lean_obj_tag(v_v_970_))
{
case 0:
{
lean_object* v_key_979_; lean_object* v_val_980_; lean_object* v___x_982_; uint8_t v_isShared_983_; uint8_t v_isSharedCheck_1009_; 
v_key_979_ = lean_ctor_get(v_v_970_, 0);
v_val_980_ = lean_ctor_get(v_v_970_, 1);
v_isSharedCheck_1009_ = !lean_is_exclusive(v_v_970_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_982_ = v_v_970_;
v_isShared_983_ = v_isSharedCheck_1009_;
goto v_resetjp_981_;
}
else
{
lean_inc(v_val_980_);
lean_inc(v_key_979_);
lean_dec(v_v_970_);
v___x_982_ = lean_box(0);
v_isShared_983_ = v_isSharedCheck_1009_;
goto v_resetjp_981_;
}
v_resetjp_981_:
{
uint8_t v___y_988_; lean_object* v_name_992_; uint8_t v_builder_993_; uint8_t v_phase_994_; uint8_t v_scope_995_; uint64_t v_hash_996_; lean_object* v_name_997_; uint8_t v_builder_998_; uint8_t v_phase_999_; uint8_t v_scope_1000_; uint64_t v_hash_1001_; uint8_t v___y_1003_; uint8_t v___x_1007_; 
v_name_992_ = lean_ctor_get(v_x_959_, 0);
v_builder_993_ = lean_ctor_get_uint8(v_x_959_, sizeof(void*)*1 + 8);
v_phase_994_ = lean_ctor_get_uint8(v_x_959_, sizeof(void*)*1 + 9);
v_scope_995_ = lean_ctor_get_uint8(v_x_959_, sizeof(void*)*1 + 10);
v_hash_996_ = lean_ctor_get_uint64(v_x_959_, sizeof(void*)*1);
v_name_997_ = lean_ctor_get(v_key_979_, 0);
v_builder_998_ = lean_ctor_get_uint8(v_key_979_, sizeof(void*)*1 + 8);
v_phase_999_ = lean_ctor_get_uint8(v_key_979_, sizeof(void*)*1 + 9);
v_scope_1000_ = lean_ctor_get_uint8(v_key_979_, sizeof(void*)*1 + 10);
v_hash_1001_ = lean_ctor_get_uint64(v_key_979_, sizeof(void*)*1);
v___x_1007_ = lean_uint64_dec_eq(v_hash_996_, v_hash_1001_);
if (v___x_1007_ == 0)
{
v___y_1003_ = v___x_1007_;
goto v___jp_1002_;
}
else
{
uint8_t v___x_1008_; 
v___x_1008_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_993_, v_builder_998_);
v___y_1003_ = v___x_1008_;
goto v___jp_1002_;
}
v___jp_984_:
{
lean_object* v___x_985_; lean_object* v___x_986_; 
v___x_985_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_979_, v_val_980_, v_x_959_, v_x_960_);
v___x_986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_986_, 0, v___x_985_);
v___y_974_ = v___x_986_;
goto v___jp_973_;
}
v___jp_987_:
{
if (v___y_988_ == 0)
{
lean_del_object(v___x_982_);
goto v___jp_984_;
}
else
{
lean_object* v___x_990_; 
lean_dec(v_val_980_);
lean_dec(v_key_979_);
if (v_isShared_983_ == 0)
{
lean_ctor_set(v___x_982_, 1, v_x_960_);
lean_ctor_set(v___x_982_, 0, v_x_959_);
v___x_990_ = v___x_982_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v_x_959_);
lean_ctor_set(v_reuseFailAlloc_991_, 1, v_x_960_);
v___x_990_ = v_reuseFailAlloc_991_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
v___y_974_ = v___x_990_;
goto v___jp_973_;
}
}
}
v___jp_1002_:
{
if (v___y_1003_ == 0)
{
lean_del_object(v___x_982_);
goto v___jp_984_;
}
else
{
uint8_t v___x_1004_; 
v___x_1004_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_994_, v_phase_999_);
if (v___x_1004_ == 0)
{
v___y_988_ = v___x_1004_;
goto v___jp_987_;
}
else
{
uint8_t v___x_1005_; 
v___x_1005_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_995_, v_scope_1000_);
if (v___x_1005_ == 0)
{
v___y_988_ = v___x_1005_;
goto v___jp_987_;
}
else
{
uint8_t v___x_1006_; 
v___x_1006_ = lean_name_eq(v_name_992_, v_name_997_);
v___y_988_ = v___x_1006_;
goto v___jp_987_;
}
}
}
}
}
}
case 1:
{
lean_object* v_node_1010_; lean_object* v___x_1012_; uint8_t v_isShared_1013_; uint8_t v_isSharedCheck_1022_; 
v_node_1010_ = lean_ctor_get(v_v_970_, 0);
v_isSharedCheck_1022_ = !lean_is_exclusive(v_v_970_);
if (v_isSharedCheck_1022_ == 0)
{
v___x_1012_ = v_v_970_;
v_isShared_1013_ = v_isSharedCheck_1022_;
goto v_resetjp_1011_;
}
else
{
lean_inc(v_node_1010_);
lean_dec(v_v_970_);
v___x_1012_ = lean_box(0);
v_isShared_1013_ = v_isSharedCheck_1022_;
goto v_resetjp_1011_;
}
v_resetjp_1011_:
{
size_t v___x_1014_; size_t v___x_1015_; size_t v___x_1016_; size_t v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1020_; 
v___x_1014_ = ((size_t)5ULL);
v___x_1015_ = lean_usize_shift_right(v_x_957_, v___x_1014_);
v___x_1016_ = ((size_t)1ULL);
v___x_1017_ = lean_usize_add(v_x_958_, v___x_1016_);
v___x_1018_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(v_node_1010_, v___x_1015_, v___x_1017_, v_x_959_, v_x_960_);
if (v_isShared_1013_ == 0)
{
lean_ctor_set(v___x_1012_, 0, v___x_1018_);
v___x_1020_ = v___x_1012_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1021_; 
v_reuseFailAlloc_1021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1021_, 0, v___x_1018_);
v___x_1020_ = v_reuseFailAlloc_1021_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
v___y_974_ = v___x_1020_;
goto v___jp_973_;
}
}
}
default: 
{
lean_object* v___x_1023_; 
v___x_1023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1023_, 0, v_x_959_);
lean_ctor_set(v___x_1023_, 1, v_x_960_);
v___y_974_ = v___x_1023_;
goto v___jp_973_;
}
}
v___jp_973_:
{
lean_object* v___x_975_; lean_object* v___x_977_; 
v___x_975_ = lean_array_fset(v_xs_x27_972_, v_j_964_, v___y_974_);
lean_dec(v_j_964_);
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_977_ = v___x_968_;
goto v_reusejp_976_;
}
else
{
lean_object* v_reuseFailAlloc_978_; 
v_reuseFailAlloc_978_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_978_, 0, v___x_975_);
v___x_977_ = v_reuseFailAlloc_978_;
goto v_reusejp_976_;
}
v_reusejp_976_:
{
return v___x_977_;
}
}
}
}
}
else
{
lean_object* v_ks_1026_; lean_object* v_vs_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1047_; 
v_ks_1026_ = lean_ctor_get(v_x_956_, 0);
v_vs_1027_ = lean_ctor_get(v_x_956_, 1);
v_isSharedCheck_1047_ = !lean_is_exclusive(v_x_956_);
if (v_isSharedCheck_1047_ == 0)
{
v___x_1029_ = v_x_956_;
v_isShared_1030_ = v_isSharedCheck_1047_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_vs_1027_);
lean_inc(v_ks_1026_);
lean_dec(v_x_956_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1047_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1032_; 
if (v_isShared_1030_ == 0)
{
v___x_1032_ = v___x_1029_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1046_; 
v_reuseFailAlloc_1046_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1046_, 0, v_ks_1026_);
lean_ctor_set(v_reuseFailAlloc_1046_, 1, v_vs_1027_);
v___x_1032_ = v_reuseFailAlloc_1046_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
lean_object* v_newNode_1033_; uint8_t v___y_1035_; size_t v___x_1041_; uint8_t v___x_1042_; 
v_newNode_1033_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4___redArg(v___x_1032_, v_x_959_, v_x_960_);
v___x_1041_ = ((size_t)7ULL);
v___x_1042_ = lean_usize_dec_le(v___x_1041_, v_x_958_);
if (v___x_1042_ == 0)
{
lean_object* v___x_1043_; lean_object* v___x_1044_; uint8_t v___x_1045_; 
v___x_1043_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1033_);
v___x_1044_ = lean_unsigned_to_nat(4u);
v___x_1045_ = lean_nat_dec_lt(v___x_1043_, v___x_1044_);
lean_dec(v___x_1043_);
v___y_1035_ = v___x_1045_;
goto v___jp_1034_;
}
else
{
v___y_1035_ = v___x_1042_;
goto v___jp_1034_;
}
v___jp_1034_:
{
if (v___y_1035_ == 0)
{
lean_object* v_ks_1036_; lean_object* v_vs_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v_ks_1036_ = lean_ctor_get(v_newNode_1033_, 0);
lean_inc_ref(v_ks_1036_);
v_vs_1037_ = lean_ctor_get(v_newNode_1033_, 1);
lean_inc_ref(v_vs_1037_);
lean_dec_ref(v_newNode_1033_);
v___x_1038_ = lean_unsigned_to_nat(0u);
v___x_1039_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___closed__0);
v___x_1040_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg(v_x_958_, v_ks_1036_, v_vs_1037_, v___x_1038_, v___x_1039_);
lean_dec_ref(v_vs_1037_);
lean_dec_ref(v_ks_1036_);
return v___x_1040_;
}
else
{
return v_newNode_1033_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg(size_t v_depth_1048_, lean_object* v_keys_1049_, lean_object* v_vals_1050_, lean_object* v_i_1051_, lean_object* v_entries_1052_){
_start:
{
lean_object* v___x_1053_; uint8_t v___x_1054_; 
v___x_1053_ = lean_array_get_size(v_keys_1049_);
v___x_1054_ = lean_nat_dec_lt(v_i_1051_, v___x_1053_);
if (v___x_1054_ == 0)
{
lean_dec(v_i_1051_);
return v_entries_1052_;
}
else
{
lean_object* v_k_1055_; uint64_t v_hash_1056_; lean_object* v_v_1057_; size_t v_h_1058_; size_t v___x_1059_; lean_object* v___x_1060_; size_t v___x_1061_; size_t v___x_1062_; size_t v___x_1063_; size_t v_h_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v_k_1055_ = lean_array_fget_borrowed(v_keys_1049_, v_i_1051_);
v_hash_1056_ = lean_ctor_get_uint64(v_k_1055_, sizeof(void*)*1);
v_v_1057_ = lean_array_fget_borrowed(v_vals_1050_, v_i_1051_);
v_h_1058_ = lean_uint64_to_usize(v_hash_1056_);
v___x_1059_ = ((size_t)5ULL);
v___x_1060_ = lean_unsigned_to_nat(1u);
v___x_1061_ = ((size_t)1ULL);
v___x_1062_ = lean_usize_sub(v_depth_1048_, v___x_1061_);
v___x_1063_ = lean_usize_mul(v___x_1059_, v___x_1062_);
v_h_1064_ = lean_usize_shift_right(v_h_1058_, v___x_1063_);
v___x_1065_ = lean_nat_add(v_i_1051_, v___x_1060_);
lean_dec(v_i_1051_);
lean_inc(v_v_1057_);
lean_inc(v_k_1055_);
v___x_1066_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(v_entries_1052_, v_h_1064_, v_depth_1048_, v_k_1055_, v_v_1057_);
v_i_1051_ = v___x_1065_;
v_entries_1052_ = v___x_1066_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_depth_1068_, lean_object* v_keys_1069_, lean_object* v_vals_1070_, lean_object* v_i_1071_, lean_object* v_entries_1072_){
_start:
{
size_t v_depth_boxed_1073_; lean_object* v_res_1074_; 
v_depth_boxed_1073_ = lean_unbox_usize(v_depth_1068_);
lean_dec(v_depth_1068_);
v_res_1074_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg(v_depth_boxed_1073_, v_keys_1069_, v_vals_1070_, v_i_1071_, v_entries_1072_);
lean_dec_ref(v_vals_1070_);
lean_dec_ref(v_keys_1069_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg___boxed(lean_object* v_x_1075_, lean_object* v_x_1076_, lean_object* v_x_1077_, lean_object* v_x_1078_, lean_object* v_x_1079_){
_start:
{
size_t v_x_1619__boxed_1080_; size_t v_x_1620__boxed_1081_; lean_object* v_res_1082_; 
v_x_1619__boxed_1080_ = lean_unbox_usize(v_x_1076_);
lean_dec(v_x_1076_);
v_x_1620__boxed_1081_ = lean_unbox_usize(v_x_1077_);
lean_dec(v_x_1077_);
v_res_1082_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(v_x_1075_, v_x_1619__boxed_1080_, v_x_1620__boxed_1081_, v_x_1078_, v_x_1079_);
return v_res_1082_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(lean_object* v_x_1083_, lean_object* v_x_1084_, lean_object* v_x_1085_){
_start:
{
uint64_t v_hash_1086_; size_t v___x_1087_; size_t v___x_1088_; lean_object* v___x_1089_; 
v_hash_1086_ = lean_ctor_get_uint64(v_x_1084_, sizeof(void*)*1);
v___x_1087_ = lean_uint64_to_usize(v_hash_1086_);
v___x_1088_ = ((size_t)1ULL);
v___x_1089_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(v_x_1083_, v___x_1087_, v___x_1088_, v_x_1084_, v_x_1085_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__0(lean_object* v_map_1090_, lean_object* v_k_1091_, lean_object* v_v_u2082_1092_){
_start:
{
lean_object* v___x_1093_; 
v___x_1093_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(v_map_1090_, v_k_1091_);
if (lean_obj_tag(v___x_1093_) == 0)
{
lean_object* v___x_1094_; 
v___x_1094_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(v_map_1090_, v_k_1091_, v_v_u2082_1092_);
return v___x_1094_;
}
else
{
lean_object* v_val_1095_; lean_object* v___x_1096_; 
lean_dec_ref(v_v_u2082_1092_);
v_val_1095_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_val_1095_);
lean_dec_ref_known(v___x_1093_, 1);
v___x_1096_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(v_map_1090_, v_k_1091_, v_val_1095_);
return v___x_1096_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15___redArg(lean_object* v_x_1097_, lean_object* v_x_1098_, lean_object* v_x_1099_, lean_object* v_x_1100_){
_start:
{
lean_object* v_ks_1101_; lean_object* v_vs_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1143_; 
v_ks_1101_ = lean_ctor_get(v_x_1097_, 0);
v_vs_1102_ = lean_ctor_get(v_x_1097_, 1);
v_isSharedCheck_1143_ = !lean_is_exclusive(v_x_1097_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1104_ = v_x_1097_;
v_isShared_1105_ = v_isSharedCheck_1143_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_vs_1102_);
lean_inc(v_ks_1101_);
lean_dec(v_x_1097_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1143_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
uint8_t v___y_1114_; lean_object* v___x_1118_; uint8_t v___x_1119_; 
v___x_1118_ = lean_array_get_size(v_ks_1101_);
v___x_1119_ = lean_nat_dec_lt(v_x_1098_, v___x_1118_);
if (v___x_1119_ == 0)
{
lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
lean_del_object(v___x_1104_);
lean_dec(v_x_1098_);
v___x_1120_ = lean_array_push(v_ks_1101_, v_x_1099_);
v___x_1121_ = lean_array_push(v_vs_1102_, v_x_1100_);
v___x_1122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1120_);
lean_ctor_set(v___x_1122_, 1, v___x_1121_);
return v___x_1122_;
}
else
{
lean_object* v_name_1123_; lean_object* v_k_x27_1124_; lean_object* v_name_1125_; lean_object* v_name_1126_; uint8_t v_builder_1127_; uint8_t v_phase_1128_; uint8_t v_scope_1129_; uint64_t v_hash_1130_; lean_object* v_name_1131_; uint8_t v_builder_1132_; uint8_t v_phase_1133_; uint8_t v_scope_1134_; uint64_t v_hash_1135_; uint8_t v___y_1137_; uint8_t v___x_1141_; 
v_name_1123_ = lean_ctor_get(v_x_1099_, 1);
v_k_x27_1124_ = lean_array_fget_borrowed(v_ks_1101_, v_x_1098_);
v_name_1125_ = lean_ctor_get(v_k_x27_1124_, 1);
v_name_1126_ = lean_ctor_get(v_name_1123_, 0);
v_builder_1127_ = lean_ctor_get_uint8(v_name_1123_, sizeof(void*)*1 + 8);
v_phase_1128_ = lean_ctor_get_uint8(v_name_1123_, sizeof(void*)*1 + 9);
v_scope_1129_ = lean_ctor_get_uint8(v_name_1123_, sizeof(void*)*1 + 10);
v_hash_1130_ = lean_ctor_get_uint64(v_name_1123_, sizeof(void*)*1);
v_name_1131_ = lean_ctor_get(v_name_1125_, 0);
v_builder_1132_ = lean_ctor_get_uint8(v_name_1125_, sizeof(void*)*1 + 8);
v_phase_1133_ = lean_ctor_get_uint8(v_name_1125_, sizeof(void*)*1 + 9);
v_scope_1134_ = lean_ctor_get_uint8(v_name_1125_, sizeof(void*)*1 + 10);
v_hash_1135_ = lean_ctor_get_uint64(v_name_1125_, sizeof(void*)*1);
v___x_1141_ = lean_uint64_dec_eq(v_hash_1130_, v_hash_1135_);
if (v___x_1141_ == 0)
{
v___y_1137_ = v___x_1141_;
goto v___jp_1136_;
}
else
{
uint8_t v___x_1142_; 
v___x_1142_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_1127_, v_builder_1132_);
v___y_1137_ = v___x_1142_;
goto v___jp_1136_;
}
v___jp_1136_:
{
if (v___y_1137_ == 0)
{
goto v___jp_1106_;
}
else
{
uint8_t v___x_1138_; 
v___x_1138_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_1128_, v_phase_1133_);
if (v___x_1138_ == 0)
{
v___y_1114_ = v___x_1138_;
goto v___jp_1113_;
}
else
{
uint8_t v___x_1139_; 
v___x_1139_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_1129_, v_scope_1134_);
if (v___x_1139_ == 0)
{
v___y_1114_ = v___x_1139_;
goto v___jp_1113_;
}
else
{
uint8_t v___x_1140_; 
v___x_1140_ = lean_name_eq(v_name_1126_, v_name_1131_);
v___y_1114_ = v___x_1140_;
goto v___jp_1113_;
}
}
}
}
}
v___jp_1106_:
{
lean_object* v___x_1108_; 
if (v_isShared_1105_ == 0)
{
v___x_1108_ = v___x_1104_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v_ks_1101_);
lean_ctor_set(v_reuseFailAlloc_1112_, 1, v_vs_1102_);
v___x_1108_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
lean_object* v___x_1109_; lean_object* v___x_1110_; 
v___x_1109_ = lean_unsigned_to_nat(1u);
v___x_1110_ = lean_nat_add(v_x_1098_, v___x_1109_);
lean_dec(v_x_1098_);
v_x_1097_ = v___x_1108_;
v_x_1098_ = v___x_1110_;
goto _start;
}
}
v___jp_1113_:
{
if (v___y_1114_ == 0)
{
goto v___jp_1106_;
}
else
{
lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; 
lean_del_object(v___x_1104_);
v___x_1115_ = lean_array_fset(v_ks_1101_, v_x_1098_, v_x_1099_);
v___x_1116_ = lean_array_fset(v_vs_1102_, v_x_1098_, v_x_1100_);
lean_dec(v_x_1098_);
v___x_1117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1115_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
return v___x_1117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8___redArg(lean_object* v_n_1144_, lean_object* v_k_1145_, lean_object* v_v_1146_){
_start:
{
lean_object* v___x_1147_; lean_object* v___x_1148_; 
v___x_1147_ = lean_unsigned_to_nat(0u);
v___x_1148_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15___redArg(v_n_1144_, v___x_1147_, v_k_1145_, v_v_1146_);
return v___x_1148_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_1149_; 
v___x_1149_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(lean_object* v_x_1150_, size_t v_x_1151_, size_t v_x_1152_, lean_object* v_x_1153_, lean_object* v_x_1154_){
_start:
{
if (lean_obj_tag(v_x_1150_) == 0)
{
lean_object* v_es_1155_; size_t v___x_1156_; size_t v___x_1157_; lean_object* v_j_1158_; lean_object* v___x_1159_; uint8_t v___x_1160_; 
v_es_1155_ = lean_ctor_get(v_x_1150_, 0);
v___x_1156_ = ((size_t)31ULL);
v___x_1157_ = lean_usize_land(v_x_1151_, v___x_1156_);
v_j_1158_ = lean_usize_to_nat(v___x_1157_);
v___x_1159_ = lean_array_get_size(v_es_1155_);
v___x_1160_ = lean_nat_dec_lt(v_j_1158_, v___x_1159_);
if (v___x_1160_ == 0)
{
lean_dec(v_j_1158_);
lean_dec(v_x_1154_);
lean_dec_ref(v_x_1153_);
return v_x_1150_;
}
else
{
lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1220_; 
lean_inc_ref(v_es_1155_);
v_isSharedCheck_1220_ = !lean_is_exclusive(v_x_1150_);
if (v_isSharedCheck_1220_ == 0)
{
lean_object* v_unused_1221_; 
v_unused_1221_ = lean_ctor_get(v_x_1150_, 0);
lean_dec(v_unused_1221_);
v___x_1162_ = v_x_1150_;
v_isShared_1163_ = v_isSharedCheck_1220_;
goto v_resetjp_1161_;
}
else
{
lean_dec(v_x_1150_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1220_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v_v_1164_; lean_object* v___x_1165_; lean_object* v_xs_x27_1166_; lean_object* v___y_1168_; 
v_v_1164_ = lean_array_fget(v_es_1155_, v_j_1158_);
v___x_1165_ = lean_box(0);
v_xs_x27_1166_ = lean_array_fset(v_es_1155_, v_j_1158_, v___x_1165_);
switch(lean_obj_tag(v_v_1164_))
{
case 0:
{
lean_object* v_key_1173_; lean_object* v_val_1174_; lean_object* v___x_1176_; uint8_t v_isShared_1177_; uint8_t v_isSharedCheck_1205_; 
v_key_1173_ = lean_ctor_get(v_v_1164_, 0);
v_val_1174_ = lean_ctor_get(v_v_1164_, 1);
v_isSharedCheck_1205_ = !lean_is_exclusive(v_v_1164_);
if (v_isSharedCheck_1205_ == 0)
{
v___x_1176_ = v_v_1164_;
v_isShared_1177_ = v_isSharedCheck_1205_;
goto v_resetjp_1175_;
}
else
{
lean_inc(v_val_1174_);
lean_inc(v_key_1173_);
lean_dec(v_v_1164_);
v___x_1176_ = lean_box(0);
v_isShared_1177_ = v_isSharedCheck_1205_;
goto v_resetjp_1175_;
}
v_resetjp_1175_:
{
uint8_t v___y_1182_; lean_object* v_name_1186_; lean_object* v_name_1187_; lean_object* v_name_1188_; uint8_t v_builder_1189_; uint8_t v_phase_1190_; uint8_t v_scope_1191_; uint64_t v_hash_1192_; lean_object* v_name_1193_; uint8_t v_builder_1194_; uint8_t v_phase_1195_; uint8_t v_scope_1196_; uint64_t v_hash_1197_; uint8_t v___y_1199_; uint8_t v___x_1203_; 
v_name_1186_ = lean_ctor_get(v_x_1153_, 1);
v_name_1187_ = lean_ctor_get(v_key_1173_, 1);
v_name_1188_ = lean_ctor_get(v_name_1186_, 0);
v_builder_1189_ = lean_ctor_get_uint8(v_name_1186_, sizeof(void*)*1 + 8);
v_phase_1190_ = lean_ctor_get_uint8(v_name_1186_, sizeof(void*)*1 + 9);
v_scope_1191_ = lean_ctor_get_uint8(v_name_1186_, sizeof(void*)*1 + 10);
v_hash_1192_ = lean_ctor_get_uint64(v_name_1186_, sizeof(void*)*1);
v_name_1193_ = lean_ctor_get(v_name_1187_, 0);
v_builder_1194_ = lean_ctor_get_uint8(v_name_1187_, sizeof(void*)*1 + 8);
v_phase_1195_ = lean_ctor_get_uint8(v_name_1187_, sizeof(void*)*1 + 9);
v_scope_1196_ = lean_ctor_get_uint8(v_name_1187_, sizeof(void*)*1 + 10);
v_hash_1197_ = lean_ctor_get_uint64(v_name_1187_, sizeof(void*)*1);
v___x_1203_ = lean_uint64_dec_eq(v_hash_1192_, v_hash_1197_);
if (v___x_1203_ == 0)
{
v___y_1199_ = v___x_1203_;
goto v___jp_1198_;
}
else
{
uint8_t v___x_1204_; 
v___x_1204_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_1189_, v_builder_1194_);
v___y_1199_ = v___x_1204_;
goto v___jp_1198_;
}
v___jp_1178_:
{
lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1179_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1173_, v_val_1174_, v_x_1153_, v_x_1154_);
v___x_1180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
v___y_1168_ = v___x_1180_;
goto v___jp_1167_;
}
v___jp_1181_:
{
if (v___y_1182_ == 0)
{
lean_del_object(v___x_1176_);
goto v___jp_1178_;
}
else
{
lean_object* v___x_1184_; 
lean_dec(v_val_1174_);
lean_dec(v_key_1173_);
if (v_isShared_1177_ == 0)
{
lean_ctor_set(v___x_1176_, 1, v_x_1154_);
lean_ctor_set(v___x_1176_, 0, v_x_1153_);
v___x_1184_ = v___x_1176_;
goto v_reusejp_1183_;
}
else
{
lean_object* v_reuseFailAlloc_1185_; 
v_reuseFailAlloc_1185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1185_, 0, v_x_1153_);
lean_ctor_set(v_reuseFailAlloc_1185_, 1, v_x_1154_);
v___x_1184_ = v_reuseFailAlloc_1185_;
goto v_reusejp_1183_;
}
v_reusejp_1183_:
{
v___y_1168_ = v___x_1184_;
goto v___jp_1167_;
}
}
}
v___jp_1198_:
{
if (v___y_1199_ == 0)
{
lean_del_object(v___x_1176_);
goto v___jp_1178_;
}
else
{
uint8_t v___x_1200_; 
v___x_1200_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_1190_, v_phase_1195_);
if (v___x_1200_ == 0)
{
v___y_1182_ = v___x_1200_;
goto v___jp_1181_;
}
else
{
uint8_t v___x_1201_; 
v___x_1201_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_1191_, v_scope_1196_);
if (v___x_1201_ == 0)
{
v___y_1182_ = v___x_1201_;
goto v___jp_1181_;
}
else
{
uint8_t v___x_1202_; 
v___x_1202_ = lean_name_eq(v_name_1188_, v_name_1193_);
v___y_1182_ = v___x_1202_;
goto v___jp_1181_;
}
}
}
}
}
}
case 1:
{
lean_object* v_node_1206_; lean_object* v___x_1208_; uint8_t v_isShared_1209_; uint8_t v_isSharedCheck_1218_; 
v_node_1206_ = lean_ctor_get(v_v_1164_, 0);
v_isSharedCheck_1218_ = !lean_is_exclusive(v_v_1164_);
if (v_isSharedCheck_1218_ == 0)
{
v___x_1208_ = v_v_1164_;
v_isShared_1209_ = v_isSharedCheck_1218_;
goto v_resetjp_1207_;
}
else
{
lean_inc(v_node_1206_);
lean_dec(v_v_1164_);
v___x_1208_ = lean_box(0);
v_isShared_1209_ = v_isSharedCheck_1218_;
goto v_resetjp_1207_;
}
v_resetjp_1207_:
{
size_t v___x_1210_; size_t v___x_1211_; size_t v___x_1212_; size_t v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1216_; 
v___x_1210_ = ((size_t)5ULL);
v___x_1211_ = lean_usize_shift_right(v_x_1151_, v___x_1210_);
v___x_1212_ = ((size_t)1ULL);
v___x_1213_ = lean_usize_add(v_x_1152_, v___x_1212_);
v___x_1214_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(v_node_1206_, v___x_1211_, v___x_1213_, v_x_1153_, v_x_1154_);
if (v_isShared_1209_ == 0)
{
lean_ctor_set(v___x_1208_, 0, v___x_1214_);
v___x_1216_ = v___x_1208_;
goto v_reusejp_1215_;
}
else
{
lean_object* v_reuseFailAlloc_1217_; 
v_reuseFailAlloc_1217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1217_, 0, v___x_1214_);
v___x_1216_ = v_reuseFailAlloc_1217_;
goto v_reusejp_1215_;
}
v_reusejp_1215_:
{
v___y_1168_ = v___x_1216_;
goto v___jp_1167_;
}
}
}
default: 
{
lean_object* v___x_1219_; 
v___x_1219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1219_, 0, v_x_1153_);
lean_ctor_set(v___x_1219_, 1, v_x_1154_);
v___y_1168_ = v___x_1219_;
goto v___jp_1167_;
}
}
v___jp_1167_:
{
lean_object* v___x_1169_; lean_object* v___x_1171_; 
v___x_1169_ = lean_array_fset(v_xs_x27_1166_, v_j_1158_, v___y_1168_);
lean_dec(v_j_1158_);
if (v_isShared_1163_ == 0)
{
lean_ctor_set(v___x_1162_, 0, v___x_1169_);
v___x_1171_ = v___x_1162_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v___x_1169_);
v___x_1171_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
return v___x_1171_;
}
}
}
}
}
else
{
lean_object* v_ks_1222_; lean_object* v_vs_1223_; lean_object* v___x_1225_; uint8_t v_isShared_1226_; uint8_t v_isSharedCheck_1243_; 
v_ks_1222_ = lean_ctor_get(v_x_1150_, 0);
v_vs_1223_ = lean_ctor_get(v_x_1150_, 1);
v_isSharedCheck_1243_ = !lean_is_exclusive(v_x_1150_);
if (v_isSharedCheck_1243_ == 0)
{
v___x_1225_ = v_x_1150_;
v_isShared_1226_ = v_isSharedCheck_1243_;
goto v_resetjp_1224_;
}
else
{
lean_inc(v_vs_1223_);
lean_inc(v_ks_1222_);
lean_dec(v_x_1150_);
v___x_1225_ = lean_box(0);
v_isShared_1226_ = v_isSharedCheck_1243_;
goto v_resetjp_1224_;
}
v_resetjp_1224_:
{
lean_object* v___x_1228_; 
if (v_isShared_1226_ == 0)
{
v___x_1228_ = v___x_1225_;
goto v_reusejp_1227_;
}
else
{
lean_object* v_reuseFailAlloc_1242_; 
v_reuseFailAlloc_1242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1242_, 0, v_ks_1222_);
lean_ctor_set(v_reuseFailAlloc_1242_, 1, v_vs_1223_);
v___x_1228_ = v_reuseFailAlloc_1242_;
goto v_reusejp_1227_;
}
v_reusejp_1227_:
{
lean_object* v_newNode_1229_; uint8_t v___y_1231_; size_t v___x_1237_; uint8_t v___x_1238_; 
v_newNode_1229_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8___redArg(v___x_1228_, v_x_1153_, v_x_1154_);
v___x_1237_ = ((size_t)7ULL);
v___x_1238_ = lean_usize_dec_le(v___x_1237_, v_x_1152_);
if (v___x_1238_ == 0)
{
lean_object* v___x_1239_; lean_object* v___x_1240_; uint8_t v___x_1241_; 
v___x_1239_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1229_);
v___x_1240_ = lean_unsigned_to_nat(4u);
v___x_1241_ = lean_nat_dec_lt(v___x_1239_, v___x_1240_);
lean_dec(v___x_1239_);
v___y_1231_ = v___x_1241_;
goto v___jp_1230_;
}
else
{
v___y_1231_ = v___x_1238_;
goto v___jp_1230_;
}
v___jp_1230_:
{
if (v___y_1231_ == 0)
{
lean_object* v_ks_1232_; lean_object* v_vs_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; 
v_ks_1232_ = lean_ctor_get(v_newNode_1229_, 0);
lean_inc_ref(v_ks_1232_);
v_vs_1233_ = lean_ctor_get(v_newNode_1229_, 1);
lean_inc_ref(v_vs_1233_);
lean_dec_ref(v_newNode_1229_);
v___x_1234_ = lean_unsigned_to_nat(0u);
v___x_1235_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___closed__0);
v___x_1236_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg(v_x_1152_, v_ks_1232_, v_vs_1233_, v___x_1234_, v___x_1235_);
lean_dec_ref(v_vs_1233_);
lean_dec_ref(v_ks_1232_);
return v___x_1236_;
}
else
{
return v_newNode_1229_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg(size_t v_depth_1244_, lean_object* v_keys_1245_, lean_object* v_vals_1246_, lean_object* v_i_1247_, lean_object* v_entries_1248_){
_start:
{
lean_object* v___x_1249_; uint8_t v___x_1250_; 
v___x_1249_ = lean_array_get_size(v_keys_1245_);
v___x_1250_ = lean_nat_dec_lt(v_i_1247_, v___x_1249_);
if (v___x_1250_ == 0)
{
lean_dec(v_i_1247_);
return v_entries_1248_;
}
else
{
lean_object* v_k_1251_; lean_object* v_name_1252_; uint64_t v_hash_1253_; lean_object* v_v_1254_; size_t v_h_1255_; size_t v___x_1256_; lean_object* v___x_1257_; size_t v___x_1258_; size_t v___x_1259_; size_t v___x_1260_; size_t v_h_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
v_k_1251_ = lean_array_fget_borrowed(v_keys_1245_, v_i_1247_);
v_name_1252_ = lean_ctor_get(v_k_1251_, 1);
v_hash_1253_ = lean_ctor_get_uint64(v_name_1252_, sizeof(void*)*1);
v_v_1254_ = lean_array_fget_borrowed(v_vals_1246_, v_i_1247_);
v_h_1255_ = lean_uint64_to_usize(v_hash_1253_);
v___x_1256_ = ((size_t)5ULL);
v___x_1257_ = lean_unsigned_to_nat(1u);
v___x_1258_ = ((size_t)1ULL);
v___x_1259_ = lean_usize_sub(v_depth_1244_, v___x_1258_);
v___x_1260_ = lean_usize_mul(v___x_1256_, v___x_1259_);
v_h_1261_ = lean_usize_shift_right(v_h_1255_, v___x_1260_);
v___x_1262_ = lean_nat_add(v_i_1247_, v___x_1257_);
lean_dec(v_i_1247_);
lean_inc(v_v_1254_);
lean_inc(v_k_1251_);
v___x_1263_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(v_entries_1248_, v_h_1261_, v_depth_1244_, v_k_1251_, v_v_1254_);
v_i_1247_ = v___x_1262_;
v_entries_1248_ = v___x_1263_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg___boxed(lean_object* v_depth_1265_, lean_object* v_keys_1266_, lean_object* v_vals_1267_, lean_object* v_i_1268_, lean_object* v_entries_1269_){
_start:
{
size_t v_depth_boxed_1270_; lean_object* v_res_1271_; 
v_depth_boxed_1270_ = lean_unbox_usize(v_depth_1265_);
lean_dec(v_depth_1265_);
v_res_1271_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg(v_depth_boxed_1270_, v_keys_1266_, v_vals_1267_, v_i_1268_, v_entries_1269_);
lean_dec_ref(v_vals_1267_);
lean_dec_ref(v_keys_1266_);
return v_res_1271_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg___boxed(lean_object* v_x_1272_, lean_object* v_x_1273_, lean_object* v_x_1274_, lean_object* v_x_1275_, lean_object* v_x_1276_){
_start:
{
size_t v_x_1897__boxed_1277_; size_t v_x_1898__boxed_1278_; lean_object* v_res_1279_; 
v_x_1897__boxed_1277_ = lean_unbox_usize(v_x_1273_);
lean_dec(v_x_1273_);
v_x_1898__boxed_1278_ = lean_unbox_usize(v_x_1274_);
lean_dec(v_x_1274_);
v_res_1279_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(v_x_1272_, v_x_1897__boxed_1277_, v_x_1898__boxed_1278_, v_x_1275_, v_x_1276_);
return v_res_1279_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2___redArg(lean_object* v_x_1280_, lean_object* v_x_1281_, lean_object* v_x_1282_){
_start:
{
lean_object* v_name_1283_; uint64_t v_hash_1284_; size_t v___x_1285_; size_t v___x_1286_; lean_object* v___x_1287_; 
v_name_1283_ = lean_ctor_get(v_x_1281_, 1);
v_hash_1284_ = lean_ctor_get_uint64(v_name_1283_, sizeof(void*)*1);
v___x_1285_ = lean_uint64_to_usize(v_hash_1284_);
v___x_1286_ = ((size_t)1ULL);
v___x_1287_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(v_x_1280_, v___x_1285_, v___x_1286_, v_x_1281_, v_x_1282_);
return v___x_1287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__1(lean_object* v_d_1288_, lean_object* v_a_1289_, lean_object* v_x_1290_){
_start:
{
lean_object* v___x_1291_; lean_object* v___x_1292_; 
v___x_1291_ = lean_box(0);
v___x_1292_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2___redArg(v_d_1288_, v_a_1289_, v___x_1291_);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21___redArg(lean_object* v_x_1293_, lean_object* v_x_1294_, lean_object* v_x_1295_, lean_object* v_x_1296_){
_start:
{
lean_object* v_ks_1297_; lean_object* v_vs_1298_; lean_object* v___x_1300_; uint8_t v_isShared_1301_; uint8_t v_isSharedCheck_1322_; 
v_ks_1297_ = lean_ctor_get(v_x_1293_, 0);
v_vs_1298_ = lean_ctor_get(v_x_1293_, 1);
v_isSharedCheck_1322_ = !lean_is_exclusive(v_x_1293_);
if (v_isSharedCheck_1322_ == 0)
{
v___x_1300_ = v_x_1293_;
v_isShared_1301_ = v_isSharedCheck_1322_;
goto v_resetjp_1299_;
}
else
{
lean_inc(v_vs_1298_);
lean_inc(v_ks_1297_);
lean_dec(v_x_1293_);
v___x_1300_ = lean_box(0);
v_isShared_1301_ = v_isSharedCheck_1322_;
goto v_resetjp_1299_;
}
v_resetjp_1299_:
{
lean_object* v___x_1302_; uint8_t v___x_1303_; 
v___x_1302_ = lean_array_get_size(v_ks_1297_);
v___x_1303_ = lean_nat_dec_lt(v_x_1294_, v___x_1302_);
if (v___x_1303_ == 0)
{
lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1307_; 
lean_dec(v_x_1294_);
v___x_1304_ = lean_array_push(v_ks_1297_, v_x_1295_);
v___x_1305_ = lean_array_push(v_vs_1298_, v_x_1296_);
if (v_isShared_1301_ == 0)
{
lean_ctor_set(v___x_1300_, 1, v___x_1305_);
lean_ctor_set(v___x_1300_, 0, v___x_1304_);
v___x_1307_ = v___x_1300_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1308_; 
v_reuseFailAlloc_1308_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1308_, 0, v___x_1304_);
lean_ctor_set(v_reuseFailAlloc_1308_, 1, v___x_1305_);
v___x_1307_ = v_reuseFailAlloc_1308_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
return v___x_1307_;
}
}
else
{
lean_object* v_k_x27_1309_; uint8_t v___x_1310_; 
v_k_x27_1309_ = lean_array_fget_borrowed(v_ks_1297_, v_x_1294_);
v___x_1310_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_1295_, v_k_x27_1309_);
if (v___x_1310_ == 0)
{
lean_object* v___x_1312_; 
if (v_isShared_1301_ == 0)
{
v___x_1312_ = v___x_1300_;
goto v_reusejp_1311_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v_ks_1297_);
lean_ctor_set(v_reuseFailAlloc_1316_, 1, v_vs_1298_);
v___x_1312_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1311_;
}
v_reusejp_1311_:
{
lean_object* v___x_1313_; lean_object* v___x_1314_; 
v___x_1313_ = lean_unsigned_to_nat(1u);
v___x_1314_ = lean_nat_add(v_x_1294_, v___x_1313_);
lean_dec(v_x_1294_);
v_x_1293_ = v___x_1312_;
v_x_1294_ = v___x_1314_;
goto _start;
}
}
else
{
lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1320_; 
v___x_1317_ = lean_array_fset(v_ks_1297_, v_x_1294_, v_x_1295_);
v___x_1318_ = lean_array_fset(v_vs_1298_, v_x_1294_, v_x_1296_);
lean_dec(v_x_1294_);
if (v_isShared_1301_ == 0)
{
lean_ctor_set(v___x_1300_, 1, v___x_1318_);
lean_ctor_set(v___x_1300_, 0, v___x_1317_);
v___x_1320_ = v___x_1300_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1321_; 
v_reuseFailAlloc_1321_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1321_, 0, v___x_1317_);
lean_ctor_set(v_reuseFailAlloc_1321_, 1, v___x_1318_);
v___x_1320_ = v_reuseFailAlloc_1321_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
return v___x_1320_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15___redArg(lean_object* v_n_1323_, lean_object* v_k_1324_, lean_object* v_v_1325_){
_start:
{
lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1326_ = lean_unsigned_to_nat(0u);
v___x_1327_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21___redArg(v_n_1323_, v___x_1326_, v_k_1324_, v_v_1325_);
return v___x_1327_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_1328_; 
v___x_1328_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(lean_object* v_x_1329_, size_t v_x_1330_, size_t v_x_1331_, lean_object* v_x_1332_, lean_object* v_x_1333_){
_start:
{
if (lean_obj_tag(v_x_1329_) == 0)
{
lean_object* v_es_1334_; size_t v___x_1335_; size_t v___x_1336_; lean_object* v_j_1337_; lean_object* v___x_1338_; uint8_t v___x_1339_; 
v_es_1334_ = lean_ctor_get(v_x_1329_, 0);
v___x_1335_ = ((size_t)31ULL);
v___x_1336_ = lean_usize_land(v_x_1330_, v___x_1335_);
v_j_1337_ = lean_usize_to_nat(v___x_1336_);
v___x_1338_ = lean_array_get_size(v_es_1334_);
v___x_1339_ = lean_nat_dec_lt(v_j_1337_, v___x_1338_);
if (v___x_1339_ == 0)
{
lean_dec(v_j_1337_);
lean_dec(v_x_1333_);
lean_dec(v_x_1332_);
return v_x_1329_;
}
else
{
lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1378_; 
lean_inc_ref(v_es_1334_);
v_isSharedCheck_1378_ = !lean_is_exclusive(v_x_1329_);
if (v_isSharedCheck_1378_ == 0)
{
lean_object* v_unused_1379_; 
v_unused_1379_ = lean_ctor_get(v_x_1329_, 0);
lean_dec(v_unused_1379_);
v___x_1341_ = v_x_1329_;
v_isShared_1342_ = v_isSharedCheck_1378_;
goto v_resetjp_1340_;
}
else
{
lean_dec(v_x_1329_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1378_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
lean_object* v_v_1343_; lean_object* v___x_1344_; lean_object* v_xs_x27_1345_; lean_object* v___y_1347_; 
v_v_1343_ = lean_array_fget(v_es_1334_, v_j_1337_);
v___x_1344_ = lean_box(0);
v_xs_x27_1345_ = lean_array_fset(v_es_1334_, v_j_1337_, v___x_1344_);
switch(lean_obj_tag(v_v_1343_))
{
case 0:
{
lean_object* v_key_1352_; lean_object* v_val_1353_; lean_object* v___x_1355_; uint8_t v_isShared_1356_; uint8_t v_isSharedCheck_1363_; 
v_key_1352_ = lean_ctor_get(v_v_1343_, 0);
v_val_1353_ = lean_ctor_get(v_v_1343_, 1);
v_isSharedCheck_1363_ = !lean_is_exclusive(v_v_1343_);
if (v_isSharedCheck_1363_ == 0)
{
v___x_1355_ = v_v_1343_;
v_isShared_1356_ = v_isSharedCheck_1363_;
goto v_resetjp_1354_;
}
else
{
lean_inc(v_val_1353_);
lean_inc(v_key_1352_);
lean_dec(v_v_1343_);
v___x_1355_ = lean_box(0);
v_isShared_1356_ = v_isSharedCheck_1363_;
goto v_resetjp_1354_;
}
v_resetjp_1354_:
{
uint8_t v___x_1357_; 
v___x_1357_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_1332_, v_key_1352_);
if (v___x_1357_ == 0)
{
lean_object* v___x_1358_; lean_object* v___x_1359_; 
lean_del_object(v___x_1355_);
v___x_1358_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1352_, v_val_1353_, v_x_1332_, v_x_1333_);
v___x_1359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1359_, 0, v___x_1358_);
v___y_1347_ = v___x_1359_;
goto v___jp_1346_;
}
else
{
lean_object* v___x_1361_; 
lean_dec(v_val_1353_);
lean_dec(v_key_1352_);
if (v_isShared_1356_ == 0)
{
lean_ctor_set(v___x_1355_, 1, v_x_1333_);
lean_ctor_set(v___x_1355_, 0, v_x_1332_);
v___x_1361_ = v___x_1355_;
goto v_reusejp_1360_;
}
else
{
lean_object* v_reuseFailAlloc_1362_; 
v_reuseFailAlloc_1362_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1362_, 0, v_x_1332_);
lean_ctor_set(v_reuseFailAlloc_1362_, 1, v_x_1333_);
v___x_1361_ = v_reuseFailAlloc_1362_;
goto v_reusejp_1360_;
}
v_reusejp_1360_:
{
v___y_1347_ = v___x_1361_;
goto v___jp_1346_;
}
}
}
}
case 1:
{
lean_object* v_node_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1376_; 
v_node_1364_ = lean_ctor_get(v_v_1343_, 0);
v_isSharedCheck_1376_ = !lean_is_exclusive(v_v_1343_);
if (v_isSharedCheck_1376_ == 0)
{
v___x_1366_ = v_v_1343_;
v_isShared_1367_ = v_isSharedCheck_1376_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_node_1364_);
lean_dec(v_v_1343_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1376_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
size_t v___x_1368_; size_t v___x_1369_; size_t v___x_1370_; size_t v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1374_; 
v___x_1368_ = ((size_t)5ULL);
v___x_1369_ = lean_usize_shift_right(v_x_1330_, v___x_1368_);
v___x_1370_ = ((size_t)1ULL);
v___x_1371_ = lean_usize_add(v_x_1331_, v___x_1370_);
v___x_1372_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v_node_1364_, v___x_1369_, v___x_1371_, v_x_1332_, v_x_1333_);
if (v_isShared_1367_ == 0)
{
lean_ctor_set(v___x_1366_, 0, v___x_1372_);
v___x_1374_ = v___x_1366_;
goto v_reusejp_1373_;
}
else
{
lean_object* v_reuseFailAlloc_1375_; 
v_reuseFailAlloc_1375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1375_, 0, v___x_1372_);
v___x_1374_ = v_reuseFailAlloc_1375_;
goto v_reusejp_1373_;
}
v_reusejp_1373_:
{
v___y_1347_ = v___x_1374_;
goto v___jp_1346_;
}
}
}
default: 
{
lean_object* v___x_1377_; 
v___x_1377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1377_, 0, v_x_1332_);
lean_ctor_set(v___x_1377_, 1, v_x_1333_);
v___y_1347_ = v___x_1377_;
goto v___jp_1346_;
}
}
v___jp_1346_:
{
lean_object* v___x_1348_; lean_object* v___x_1350_; 
v___x_1348_ = lean_array_fset(v_xs_x27_1345_, v_j_1337_, v___y_1347_);
lean_dec(v_j_1337_);
if (v_isShared_1342_ == 0)
{
lean_ctor_set(v___x_1341_, 0, v___x_1348_);
v___x_1350_ = v___x_1341_;
goto v_reusejp_1349_;
}
else
{
lean_object* v_reuseFailAlloc_1351_; 
v_reuseFailAlloc_1351_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1351_, 0, v___x_1348_);
v___x_1350_ = v_reuseFailAlloc_1351_;
goto v_reusejp_1349_;
}
v_reusejp_1349_:
{
return v___x_1350_;
}
}
}
}
}
else
{
lean_object* v_ks_1380_; lean_object* v_vs_1381_; lean_object* v___x_1383_; uint8_t v_isShared_1384_; uint8_t v_isSharedCheck_1401_; 
v_ks_1380_ = lean_ctor_get(v_x_1329_, 0);
v_vs_1381_ = lean_ctor_get(v_x_1329_, 1);
v_isSharedCheck_1401_ = !lean_is_exclusive(v_x_1329_);
if (v_isSharedCheck_1401_ == 0)
{
v___x_1383_ = v_x_1329_;
v_isShared_1384_ = v_isSharedCheck_1401_;
goto v_resetjp_1382_;
}
else
{
lean_inc(v_vs_1381_);
lean_inc(v_ks_1380_);
lean_dec(v_x_1329_);
v___x_1383_ = lean_box(0);
v_isShared_1384_ = v_isSharedCheck_1401_;
goto v_resetjp_1382_;
}
v_resetjp_1382_:
{
lean_object* v___x_1386_; 
if (v_isShared_1384_ == 0)
{
v___x_1386_ = v___x_1383_;
goto v_reusejp_1385_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v_ks_1380_);
lean_ctor_set(v_reuseFailAlloc_1400_, 1, v_vs_1381_);
v___x_1386_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1385_;
}
v_reusejp_1385_:
{
lean_object* v_newNode_1387_; uint8_t v___y_1389_; size_t v___x_1395_; uint8_t v___x_1396_; 
v_newNode_1387_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15___redArg(v___x_1386_, v_x_1332_, v_x_1333_);
v___x_1395_ = ((size_t)7ULL);
v___x_1396_ = lean_usize_dec_le(v___x_1395_, v_x_1331_);
if (v___x_1396_ == 0)
{
lean_object* v___x_1397_; lean_object* v___x_1398_; uint8_t v___x_1399_; 
v___x_1397_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1387_);
v___x_1398_ = lean_unsigned_to_nat(4u);
v___x_1399_ = lean_nat_dec_lt(v___x_1397_, v___x_1398_);
lean_dec(v___x_1397_);
v___y_1389_ = v___x_1399_;
goto v___jp_1388_;
}
else
{
v___y_1389_ = v___x_1396_;
goto v___jp_1388_;
}
v___jp_1388_:
{
if (v___y_1389_ == 0)
{
lean_object* v_ks_1390_; lean_object* v_vs_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; 
v_ks_1390_ = lean_ctor_get(v_newNode_1387_, 0);
lean_inc_ref(v_ks_1390_);
v_vs_1391_ = lean_ctor_get(v_newNode_1387_, 1);
lean_inc_ref(v_vs_1391_);
lean_dec_ref(v_newNode_1387_);
v___x_1392_ = lean_unsigned_to_nat(0u);
v___x_1393_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___closed__0);
v___x_1394_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg(v_x_1331_, v_ks_1390_, v_vs_1391_, v___x_1392_, v___x_1393_);
lean_dec_ref(v_vs_1391_);
lean_dec_ref(v_ks_1390_);
return v___x_1394_;
}
else
{
return v_newNode_1387_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg(size_t v_depth_1402_, lean_object* v_keys_1403_, lean_object* v_vals_1404_, lean_object* v_i_1405_, lean_object* v_entries_1406_){
_start:
{
lean_object* v___x_1407_; uint8_t v___x_1408_; 
v___x_1407_ = lean_array_get_size(v_keys_1403_);
v___x_1408_ = lean_nat_dec_lt(v_i_1405_, v___x_1407_);
if (v___x_1408_ == 0)
{
lean_dec(v_i_1405_);
return v_entries_1406_;
}
else
{
lean_object* v_k_1409_; lean_object* v_v_1410_; uint64_t v___x_1411_; size_t v_h_1412_; size_t v___x_1413_; lean_object* v___x_1414_; size_t v___x_1415_; size_t v___x_1416_; size_t v___x_1417_; size_t v_h_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; 
v_k_1409_ = lean_array_fget_borrowed(v_keys_1403_, v_i_1405_);
v_v_1410_ = lean_array_fget_borrowed(v_vals_1404_, v_i_1405_);
v___x_1411_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_1409_);
v_h_1412_ = lean_uint64_to_usize(v___x_1411_);
v___x_1413_ = ((size_t)5ULL);
v___x_1414_ = lean_unsigned_to_nat(1u);
v___x_1415_ = ((size_t)1ULL);
v___x_1416_ = lean_usize_sub(v_depth_1402_, v___x_1415_);
v___x_1417_ = lean_usize_mul(v___x_1413_, v___x_1416_);
v_h_1418_ = lean_usize_shift_right(v_h_1412_, v___x_1417_);
v___x_1419_ = lean_nat_add(v_i_1405_, v___x_1414_);
lean_dec(v_i_1405_);
lean_inc(v_v_1410_);
lean_inc(v_k_1409_);
v___x_1420_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v_entries_1406_, v_h_1418_, v_depth_1402_, v_k_1409_, v_v_1410_);
v_i_1405_ = v___x_1419_;
v_entries_1406_ = v___x_1420_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg___boxed(lean_object* v_depth_1422_, lean_object* v_keys_1423_, lean_object* v_vals_1424_, lean_object* v_i_1425_, lean_object* v_entries_1426_){
_start:
{
size_t v_depth_boxed_1427_; lean_object* v_res_1428_; 
v_depth_boxed_1427_ = lean_unbox_usize(v_depth_1422_);
lean_dec(v_depth_1422_);
v_res_1428_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg(v_depth_boxed_1427_, v_keys_1423_, v_vals_1424_, v_i_1425_, v_entries_1426_);
lean_dec_ref(v_vals_1424_);
lean_dec_ref(v_keys_1423_);
return v_res_1428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg___boxed(lean_object* v_x_1429_, lean_object* v_x_1430_, lean_object* v_x_1431_, lean_object* v_x_1432_, lean_object* v_x_1433_){
_start:
{
size_t v_x_2164__boxed_1434_; size_t v_x_2165__boxed_1435_; lean_object* v_res_1436_; 
v_x_2164__boxed_1434_ = lean_unbox_usize(v_x_1430_);
lean_dec(v_x_1430_);
v_x_2165__boxed_1435_ = lean_unbox_usize(v_x_1431_);
lean_dec(v_x_1431_);
v_res_1436_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v_x_1429_, v_x_2164__boxed_1434_, v_x_2165__boxed_1435_, v_x_1432_, v_x_1433_);
return v_res_1436_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4___redArg(lean_object* v_x_1437_, lean_object* v_x_1438_, lean_object* v_x_1439_){
_start:
{
uint64_t v___x_1440_; size_t v___x_1441_; size_t v___x_1442_; lean_object* v___x_1443_; 
v___x_1440_ = l_Lean_Meta_DiscrTree_Key_hash(v_x_1438_);
v___x_1441_ = lean_uint64_to_usize(v___x_1440_);
v___x_1442_ = ((size_t)1ULL);
v___x_1443_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v_x_1437_, v___x_1441_, v___x_1442_, v_x_1438_, v_x_1439_);
return v___x_1443_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg(lean_object* v_keys_1444_, lean_object* v_vals_1445_, lean_object* v_i_1446_, lean_object* v_k_1447_){
_start:
{
lean_object* v___x_1448_; uint8_t v___x_1449_; 
v___x_1448_ = lean_array_get_size(v_keys_1444_);
v___x_1449_ = lean_nat_dec_lt(v_i_1446_, v___x_1448_);
if (v___x_1449_ == 0)
{
lean_object* v___x_1450_; 
lean_dec(v_i_1446_);
v___x_1450_ = lean_box(0);
return v___x_1450_;
}
else
{
lean_object* v_k_x27_1451_; uint8_t v___x_1452_; 
v_k_x27_1451_ = lean_array_fget_borrowed(v_keys_1444_, v_i_1446_);
v___x_1452_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_k_1447_, v_k_x27_1451_);
if (v___x_1452_ == 0)
{
lean_object* v___x_1453_; lean_object* v___x_1454_; 
v___x_1453_ = lean_unsigned_to_nat(1u);
v___x_1454_ = lean_nat_add(v_i_1446_, v___x_1453_);
lean_dec(v_i_1446_);
v_i_1446_ = v___x_1454_;
goto _start;
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1457_; 
v___x_1456_ = lean_array_fget_borrowed(v_vals_1445_, v_i_1446_);
lean_dec(v_i_1446_);
lean_inc(v___x_1456_);
v___x_1457_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1457_, 0, v___x_1456_);
return v___x_1457_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg___boxed(lean_object* v_keys_1458_, lean_object* v_vals_1459_, lean_object* v_i_1460_, lean_object* v_k_1461_){
_start:
{
lean_object* v_res_1462_; 
v_res_1462_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg(v_keys_1458_, v_vals_1459_, v_i_1460_, v_k_1461_);
lean_dec(v_k_1461_);
lean_dec_ref(v_vals_1459_);
lean_dec_ref(v_keys_1458_);
return v_res_1462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg(lean_object* v_x_1463_, size_t v_x_1464_, lean_object* v_x_1465_){
_start:
{
if (lean_obj_tag(v_x_1463_) == 0)
{
lean_object* v_es_1466_; lean_object* v___x_1467_; size_t v___x_1468_; size_t v___x_1469_; lean_object* v_j_1470_; lean_object* v___x_1471_; 
v_es_1466_ = lean_ctor_get(v_x_1463_, 0);
v___x_1467_ = lean_box(2);
v___x_1468_ = ((size_t)31ULL);
v___x_1469_ = lean_usize_land(v_x_1464_, v___x_1468_);
v_j_1470_ = lean_usize_to_nat(v___x_1469_);
v___x_1471_ = lean_array_get_borrowed(v___x_1467_, v_es_1466_, v_j_1470_);
lean_dec(v_j_1470_);
switch(lean_obj_tag(v___x_1471_))
{
case 0:
{
lean_object* v_key_1472_; lean_object* v_val_1473_; uint8_t v___x_1474_; 
v_key_1472_ = lean_ctor_get(v___x_1471_, 0);
v_val_1473_ = lean_ctor_get(v___x_1471_, 1);
v___x_1474_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_1465_, v_key_1472_);
if (v___x_1474_ == 0)
{
lean_object* v___x_1475_; 
v___x_1475_ = lean_box(0);
return v___x_1475_;
}
else
{
lean_object* v___x_1476_; 
lean_inc(v_val_1473_);
v___x_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1476_, 0, v_val_1473_);
return v___x_1476_;
}
}
case 1:
{
lean_object* v_node_1477_; size_t v___x_1478_; size_t v___x_1479_; 
v_node_1477_ = lean_ctor_get(v___x_1471_, 0);
v___x_1478_ = ((size_t)5ULL);
v___x_1479_ = lean_usize_shift_right(v_x_1464_, v___x_1478_);
v_x_1463_ = v_node_1477_;
v_x_1464_ = v___x_1479_;
goto _start;
}
default: 
{
lean_object* v___x_1481_; 
v___x_1481_ = lean_box(0);
return v___x_1481_;
}
}
}
else
{
lean_object* v_ks_1482_; lean_object* v_vs_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; 
v_ks_1482_ = lean_ctor_get(v_x_1463_, 0);
v_vs_1483_ = lean_ctor_get(v_x_1463_, 1);
v___x_1484_ = lean_unsigned_to_nat(0u);
v___x_1485_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg(v_ks_1482_, v_vs_1483_, v___x_1484_, v_x_1465_);
return v___x_1485_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg___boxed(lean_object* v_x_1486_, lean_object* v_x_1487_, lean_object* v_x_1488_){
_start:
{
size_t v_x_2352__boxed_1489_; lean_object* v_res_1490_; 
v_x_2352__boxed_1489_ = lean_unbox_usize(v_x_1487_);
lean_dec(v_x_1487_);
v_res_1490_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg(v_x_1486_, v_x_2352__boxed_1489_, v_x_1488_);
lean_dec(v_x_1488_);
lean_dec_ref(v_x_1486_);
return v_res_1490_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg(lean_object* v_x_1491_, lean_object* v_x_1492_){
_start:
{
uint64_t v___x_1493_; size_t v___x_1494_; lean_object* v___x_1495_; 
v___x_1493_ = l_Lean_Meta_DiscrTree_Key_hash(v_x_1492_);
v___x_1494_ = lean_uint64_to_usize(v___x_1493_);
v___x_1495_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg(v_x_1491_, v___x_1494_, v_x_1492_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg___boxed(lean_object* v_x_1496_, lean_object* v_x_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg(v_x_1496_, v_x_1497_);
lean_dec(v_x_1497_);
lean_dec_ref(v_x_1496_);
return v_res_1498_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge___lam__2(lean_object* v_map_1499_, lean_object* v_k_1500_, lean_object* v_v_u2082_1501_){
_start:
{
lean_object* v___x_1502_; 
v___x_1502_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg(v_map_1499_, v_k_1500_);
if (lean_obj_tag(v___x_1502_) == 0)
{
lean_object* v___x_1503_; 
v___x_1503_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4___redArg(v_map_1499_, v_k_1500_, v_v_u2082_1501_);
return v___x_1503_;
}
else
{
lean_object* v_val_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; 
v_val_1504_ = lean_ctor_get(v___x_1502_, 0);
lean_inc(v_val_1504_);
lean_dec_ref_known(v___x_1502_, 1);
v___x_1505_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(v_val_1504_, v_v_u2082_1501_);
v___x_1506_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4___redArg(v_map_1499_, v_k_1500_, v___x_1505_);
return v___x_1506_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg___lam__0(lean_object* v_f_1507_, lean_object* v_x1_1508_, lean_object* v_x2_1509_, lean_object* v_x3_1510_){
_start:
{
lean_object* v___x_1511_; 
v___x_1511_ = lean_apply_3(v_f_1507_, v_x1_1508_, v_x2_1509_, v_x3_1510_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg(lean_object* v_map_1512_, lean_object* v_f_1513_, lean_object* v_init_1514_){
_start:
{
lean_object* v___f_1515_; lean_object* v___x_1516_; 
v___f_1515_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1515_, 0, v_f_1513_);
v___x_1516_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v___f_1515_, v_map_1512_, v_init_1514_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg___boxed(lean_object* v_map_1517_, lean_object* v_f_1518_, lean_object* v_init_1519_){
_start:
{
lean_object* v_res_1520_; 
v_res_1520_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg(v_map_1517_, v_f_1518_, v_init_1519_);
lean_dec_ref(v_map_1517_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg___lam__0(lean_object* v_f_1521_, lean_object* v_x1_1522_, lean_object* v_x2_1523_, lean_object* v_x3_1524_){
_start:
{
lean_object* v___x_1525_; 
v___x_1525_ = lean_apply_3(v_f_1521_, v_x1_1522_, v_x2_1523_, v_x3_1524_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg(lean_object* v_map_1526_, lean_object* v_f_1527_, lean_object* v_init_1528_){
_start:
{
lean_object* v___f_1529_; lean_object* v___x_1530_; 
v___f_1529_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1529_, 0, v_f_1527_);
v___x_1530_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v___f_1529_, v_map_1526_, v_init_1528_);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg___boxed(lean_object* v_map_1531_, lean_object* v_f_1532_, lean_object* v_init_1533_){
_start:
{
lean_object* v_res_1534_; 
v_res_1534_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg(v_map_1531_, v_f_1532_, v_init_1533_);
lean_dec_ref(v_map_1531_);
return v_res_1534_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_merge(lean_object* v_idx_u2081_1538_, lean_object* v_idx_u2082_1539_){
_start:
{
lean_object* v_tree_1540_; lean_object* v_nameToRule_1541_; lean_object* v_constRules_1542_; lean_object* v_tree_1543_; lean_object* v_nameToRule_1544_; lean_object* v_constRules_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1558_; 
v_tree_1540_ = lean_ctor_get(v_idx_u2081_1538_, 0);
lean_inc_ref(v_tree_1540_);
v_nameToRule_1541_ = lean_ctor_get(v_idx_u2081_1538_, 1);
lean_inc_ref(v_nameToRule_1541_);
v_constRules_1542_ = lean_ctor_get(v_idx_u2081_1538_, 2);
lean_inc_ref(v_constRules_1542_);
lean_dec_ref(v_idx_u2081_1538_);
v_tree_1543_ = lean_ctor_get(v_idx_u2082_1539_, 0);
v_nameToRule_1544_ = lean_ctor_get(v_idx_u2082_1539_, 1);
v_constRules_1545_ = lean_ctor_get(v_idx_u2082_1539_, 2);
v_isSharedCheck_1558_ = !lean_is_exclusive(v_idx_u2082_1539_);
if (v_isSharedCheck_1558_ == 0)
{
v___x_1547_ = v_idx_u2082_1539_;
v_isShared_1548_ = v_isSharedCheck_1558_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_constRules_1545_);
lean_inc(v_nameToRule_1544_);
lean_inc(v_tree_1543_);
lean_dec(v_idx_u2082_1539_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1558_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v___f_1549_; lean_object* v___f_1550_; lean_object* v___f_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1556_; 
v___f_1549_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_merge___closed__0));
v___f_1550_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_merge___closed__1));
v___f_1551_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_merge___closed__2));
v___x_1552_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg(v_tree_1543_, v___f_1551_, v_tree_1540_);
lean_dec_ref(v_tree_1543_);
v___x_1553_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg(v_nameToRule_1544_, v___f_1549_, v_nameToRule_1541_);
lean_dec_ref(v_nameToRule_1544_);
v___x_1554_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v___f_1550_, v_constRules_1545_, v_constRules_1542_);
lean_dec_ref(v_constRules_1545_);
if (v_isShared_1548_ == 0)
{
lean_ctor_set(v___x_1547_, 2, v___x_1554_);
lean_ctor_set(v___x_1547_, 1, v___x_1553_);
lean_ctor_set(v___x_1547_, 0, v___x_1552_);
v___x_1556_ = v___x_1547_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1557_; 
v_reuseFailAlloc_1557_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1557_, 0, v___x_1552_);
lean_ctor_set(v_reuseFailAlloc_1557_, 1, v___x_1553_);
lean_ctor_set(v_reuseFailAlloc_1557_, 2, v___x_1554_);
v___x_1556_ = v_reuseFailAlloc_1557_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
return v___x_1556_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0(lean_object* v_00_u03b2_1559_, lean_object* v_x_1560_, lean_object* v_x_1561_){
_start:
{
lean_object* v___x_1562_; 
v___x_1562_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(v_x_1560_, v_x_1561_);
return v___x_1562_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___boxed(lean_object* v_00_u03b2_1563_, lean_object* v_x_1564_, lean_object* v_x_1565_){
_start:
{
lean_object* v_res_1566_; 
v_res_1566_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0(v_00_u03b2_1563_, v_x_1564_, v_x_1565_);
lean_dec_ref(v_x_1565_);
lean_dec_ref(v_x_1564_);
return v_res_1566_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1(lean_object* v_00_u03b2_1567_, lean_object* v_x_1568_, lean_object* v_x_1569_, lean_object* v_x_1570_){
_start:
{
lean_object* v___x_1571_; 
v___x_1571_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(v_x_1568_, v_x_1569_, v_x_1570_);
return v___x_1571_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2(lean_object* v_00_u03b2_1572_, lean_object* v_x_1573_, lean_object* v_x_1574_, lean_object* v_x_1575_){
_start:
{
lean_object* v___x_1576_; 
v___x_1576_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2___redArg(v_x_1573_, v_x_1574_, v_x_1575_);
return v___x_1576_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3(lean_object* v_00_u03b2_1577_, lean_object* v_x_1578_, lean_object* v_x_1579_){
_start:
{
lean_object* v___x_1580_; 
v___x_1580_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___redArg(v_x_1578_, v_x_1579_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3___boxed(lean_object* v_00_u03b2_1581_, lean_object* v_x_1582_, lean_object* v_x_1583_){
_start:
{
lean_object* v_res_1584_; 
v_res_1584_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3(v_00_u03b2_1581_, v_x_1582_, v_x_1583_);
lean_dec(v_x_1583_);
lean_dec_ref(v_x_1582_);
return v_res_1584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4(lean_object* v_00_u03b2_1585_, lean_object* v_x_1586_, lean_object* v_x_1587_, lean_object* v_x_1588_){
_start:
{
lean_object* v___x_1589_; 
v___x_1589_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4___redArg(v_x_1586_, v_x_1587_, v_x_1588_);
return v___x_1589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5(lean_object* v_00_u03c3_1590_, lean_object* v_00_u03b2_1591_, lean_object* v_map_1592_, lean_object* v_f_1593_, lean_object* v_init_1594_){
_start:
{
lean_object* v___x_1595_; 
v___x_1595_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___redArg(v_map_1592_, v_f_1593_, v_init_1594_);
return v___x_1595_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5___boxed(lean_object* v_00_u03c3_1596_, lean_object* v_00_u03b2_1597_, lean_object* v_map_1598_, lean_object* v_f_1599_, lean_object* v_init_1600_){
_start:
{
lean_object* v_res_1601_; 
v_res_1601_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__5(v_00_u03c3_1596_, v_00_u03b2_1597_, v_map_1598_, v_f_1599_, v_init_1600_);
lean_dec_ref(v_map_1598_);
return v_res_1601_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6(lean_object* v_00_u03c3_1602_, lean_object* v_00_u03b2_1603_, lean_object* v_map_1604_, lean_object* v_f_1605_, lean_object* v_init_1606_){
_start:
{
lean_object* v___x_1607_; 
v___x_1607_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___redArg(v_map_1604_, v_f_1605_, v_init_1606_);
return v___x_1607_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6___boxed(lean_object* v_00_u03c3_1608_, lean_object* v_00_u03b2_1609_, lean_object* v_map_1610_, lean_object* v_f_1611_, lean_object* v_init_1612_){
_start:
{
lean_object* v_res_1613_; 
v_res_1613_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6(v_00_u03c3_1608_, v_00_u03b2_1609_, v_map_1610_, v_f_1611_, v_init_1612_);
lean_dec_ref(v_map_1610_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___redArg(lean_object* v_map_1614_, lean_object* v_f_1615_, lean_object* v_init_1616_){
_start:
{
lean_object* v___x_1617_; 
v___x_1617_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_1615_, v_map_1614_, v_init_1616_);
return v___x_1617_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___redArg___boxed(lean_object* v_map_1618_, lean_object* v_f_1619_, lean_object* v_init_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___redArg(v_map_1618_, v_f_1619_, v_init_1620_);
lean_dec_ref(v_map_1618_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7(lean_object* v_00_u03c3_1622_, lean_object* v_00_u03b2_1623_, lean_object* v_map_1624_, lean_object* v_f_1625_, lean_object* v_init_1626_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_1625_, v_map_1624_, v_init_1626_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7___boxed(lean_object* v_00_u03c3_1628_, lean_object* v_00_u03b2_1629_, lean_object* v_map_1630_, lean_object* v_f_1631_, lean_object* v_init_1632_){
_start:
{
lean_object* v_res_1633_; 
v_res_1633_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_merge_spec__7(v_00_u03c3_1628_, v_00_u03b2_1629_, v_map_1630_, v_f_1631_, v_init_1632_);
lean_dec_ref(v_map_1630_);
return v_res_1633_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0(lean_object* v_00_u03b2_1634_, lean_object* v_x_1635_, size_t v_x_1636_, lean_object* v_x_1637_){
_start:
{
lean_object* v___x_1638_; 
v___x_1638_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___redArg(v_x_1635_, v_x_1636_, v_x_1637_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1639_, lean_object* v_x_1640_, lean_object* v_x_1641_, lean_object* v_x_1642_){
_start:
{
size_t v_x_2528__boxed_1643_; lean_object* v_res_1644_; 
v_x_2528__boxed_1643_ = lean_unbox_usize(v_x_1641_);
lean_dec(v_x_1641_);
v_res_1644_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0(v_00_u03b2_1639_, v_x_1640_, v_x_2528__boxed_1643_, v_x_1642_);
lean_dec_ref(v_x_1642_);
lean_dec_ref(v_x_1640_);
return v_res_1644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2(lean_object* v_00_u03b2_1645_, lean_object* v_x_1646_, size_t v_x_1647_, size_t v_x_1648_, lean_object* v_x_1649_, lean_object* v_x_1650_){
_start:
{
lean_object* v___x_1651_; 
v___x_1651_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___redArg(v_x_1646_, v_x_1647_, v_x_1648_, v_x_1649_, v_x_1650_);
return v___x_1651_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1652_, lean_object* v_x_1653_, lean_object* v_x_1654_, lean_object* v_x_1655_, lean_object* v_x_1656_, lean_object* v_x_1657_){
_start:
{
size_t v_x_2539__boxed_1658_; size_t v_x_2540__boxed_1659_; lean_object* v_res_1660_; 
v_x_2539__boxed_1658_ = lean_unbox_usize(v_x_1654_);
lean_dec(v_x_1654_);
v_x_2540__boxed_1659_ = lean_unbox_usize(v_x_1655_);
lean_dec(v_x_1655_);
v_res_1660_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2(v_00_u03b2_1652_, v_x_1653_, v_x_2539__boxed_1658_, v_x_2540__boxed_1659_, v_x_1656_, v_x_1657_);
return v_res_1660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4(lean_object* v_00_u03b2_1661_, lean_object* v_x_1662_, size_t v_x_1663_, size_t v_x_1664_, lean_object* v_x_1665_, lean_object* v_x_1666_){
_start:
{
lean_object* v___x_1667_; 
v___x_1667_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___redArg(v_x_1662_, v_x_1663_, v_x_1664_, v_x_1665_, v_x_1666_);
return v___x_1667_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1668_, lean_object* v_x_1669_, lean_object* v_x_1670_, lean_object* v_x_1671_, lean_object* v_x_1672_, lean_object* v_x_1673_){
_start:
{
size_t v_x_2556__boxed_1674_; size_t v_x_2557__boxed_1675_; lean_object* v_res_1676_; 
v_x_2556__boxed_1674_ = lean_unbox_usize(v_x_1670_);
lean_dec(v_x_1670_);
v_x_2557__boxed_1675_ = lean_unbox_usize(v_x_1671_);
lean_dec(v_x_1671_);
v_res_1676_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4(v_00_u03b2_1668_, v_x_1669_, v_x_2556__boxed_1674_, v_x_2557__boxed_1675_, v_x_1672_, v_x_1673_);
return v_res_1676_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6(lean_object* v_00_u03b2_1677_, lean_object* v_x_1678_, size_t v_x_1679_, lean_object* v_x_1680_){
_start:
{
lean_object* v___x_1681_; 
v___x_1681_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___redArg(v_x_1678_, v_x_1679_, v_x_1680_);
return v___x_1681_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6___boxed(lean_object* v_00_u03b2_1682_, lean_object* v_x_1683_, lean_object* v_x_1684_, lean_object* v_x_1685_){
_start:
{
size_t v_x_2573__boxed_1686_; lean_object* v_res_1687_; 
v_x_2573__boxed_1686_ = lean_unbox_usize(v_x_1684_);
lean_dec(v_x_1684_);
v_res_1687_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6(v_00_u03b2_1682_, v_x_1683_, v_x_2573__boxed_1686_, v_x_1685_);
lean_dec(v_x_1685_);
lean_dec_ref(v_x_1683_);
return v_res_1687_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8(lean_object* v_00_u03b2_1688_, lean_object* v_x_1689_, size_t v_x_1690_, size_t v_x_1691_, lean_object* v_x_1692_, lean_object* v_x_1693_){
_start:
{
lean_object* v___x_1694_; 
v___x_1694_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v_x_1689_, v_x_1690_, v_x_1691_, v_x_1692_, v_x_1693_);
return v___x_1694_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___boxed(lean_object* v_00_u03b2_1695_, lean_object* v_x_1696_, lean_object* v_x_1697_, lean_object* v_x_1698_, lean_object* v_x_1699_, lean_object* v_x_1700_){
_start:
{
size_t v_x_2584__boxed_1701_; size_t v_x_2585__boxed_1702_; lean_object* v_res_1703_; 
v_x_2584__boxed_1701_ = lean_unbox_usize(v_x_1697_);
lean_dec(v_x_1697_);
v_x_2585__boxed_1702_ = lean_unbox_usize(v_x_1698_);
lean_dec(v_x_1698_);
v_res_1703_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8(v_00_u03b2_1695_, v_x_1696_, v_x_2584__boxed_1701_, v_x_2585__boxed_1702_, v_x_1699_, v_x_1700_);
return v_res_1703_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___redArg(lean_object* v_map_1704_, lean_object* v_f_1705_, lean_object* v_init_1706_){
_start:
{
lean_object* v___x_1707_; 
v___x_1707_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_1705_, v_map_1704_, v_init_1706_);
return v___x_1707_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___redArg___boxed(lean_object* v_map_1708_, lean_object* v_f_1709_, lean_object* v_init_1710_){
_start:
{
lean_object* v_res_1711_; 
v_res_1711_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___redArg(v_map_1708_, v_f_1709_, v_init_1710_);
lean_dec_ref(v_map_1708_);
return v_res_1711_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11(lean_object* v_00_u03c3_1712_, lean_object* v_00_u03b2_1713_, lean_object* v_map_1714_, lean_object* v_f_1715_, lean_object* v_init_1716_){
_start:
{
lean_object* v___x_1717_; 
v___x_1717_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v_f_1715_, v_map_1714_, v_init_1716_);
return v___x_1717_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11___boxed(lean_object* v_00_u03c3_1718_, lean_object* v_00_u03b2_1719_, lean_object* v_map_1720_, lean_object* v_f_1721_, lean_object* v_init_1722_){
_start:
{
lean_object* v_res_1723_; 
v_res_1723_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_ForwardIndex_merge_spec__6_spec__11(v_00_u03c3_1718_, v_00_u03b2_1719_, v_map_1720_, v_f_1721_, v_init_1722_);
lean_dec_ref(v_map_1720_);
return v_res_1723_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1724_, lean_object* v_keys_1725_, lean_object* v_vals_1726_, lean_object* v_heq_1727_, lean_object* v_i_1728_, lean_object* v_k_1729_){
_start:
{
lean_object* v___x_1730_; 
v___x_1730_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___redArg(v_keys_1725_, v_vals_1726_, v_i_1728_, v_k_1729_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1731_, lean_object* v_keys_1732_, lean_object* v_vals_1733_, lean_object* v_heq_1734_, lean_object* v_i_1735_, lean_object* v_k_1736_){
_start:
{
lean_object* v_res_1737_; 
v_res_1737_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0_spec__0_spec__1(v_00_u03b2_1731_, v_keys_1732_, v_vals_1733_, v_heq_1734_, v_i_1735_, v_k_1736_);
lean_dec_ref(v_k_1736_);
lean_dec_ref(v_vals_1733_);
lean_dec_ref(v_keys_1732_);
return v_res_1737_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1738_, lean_object* v_n_1739_, lean_object* v_k_1740_, lean_object* v_v_1741_){
_start:
{
lean_object* v___x_1742_; 
v___x_1742_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4___redArg(v_n_1739_, v_k_1740_, v_v_1741_);
return v___x_1742_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_1743_, size_t v_depth_1744_, lean_object* v_keys_1745_, lean_object* v_vals_1746_, lean_object* v_heq_1747_, lean_object* v_i_1748_, lean_object* v_entries_1749_){
_start:
{
lean_object* v___x_1750_; 
v___x_1750_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___redArg(v_depth_1744_, v_keys_1745_, v_vals_1746_, v_i_1748_, v_entries_1749_);
return v___x_1750_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b2_1751_, lean_object* v_depth_1752_, lean_object* v_keys_1753_, lean_object* v_vals_1754_, lean_object* v_heq_1755_, lean_object* v_i_1756_, lean_object* v_entries_1757_){
_start:
{
size_t v_depth_boxed_1758_; lean_object* v_res_1759_; 
v_depth_boxed_1758_ = lean_unbox_usize(v_depth_1752_);
lean_dec(v_depth_1752_);
v_res_1759_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__5(v_00_u03b2_1751_, v_depth_boxed_1758_, v_keys_1753_, v_vals_1754_, v_heq_1755_, v_i_1756_, v_entries_1757_);
lean_dec_ref(v_vals_1754_);
lean_dec_ref(v_keys_1753_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8(lean_object* v_00_u03b2_1760_, lean_object* v_n_1761_, lean_object* v_k_1762_, lean_object* v_v_1763_){
_start:
{
lean_object* v___x_1764_; 
v___x_1764_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8___redArg(v_n_1761_, v_k_1762_, v_v_1763_);
return v___x_1764_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9(lean_object* v_00_u03b2_1765_, size_t v_depth_1766_, lean_object* v_keys_1767_, lean_object* v_vals_1768_, lean_object* v_heq_1769_, lean_object* v_i_1770_, lean_object* v_entries_1771_){
_start:
{
lean_object* v___x_1772_; 
v___x_1772_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___redArg(v_depth_1766_, v_keys_1767_, v_vals_1768_, v_i_1770_, v_entries_1771_);
return v___x_1772_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9___boxed(lean_object* v_00_u03b2_1773_, lean_object* v_depth_1774_, lean_object* v_keys_1775_, lean_object* v_vals_1776_, lean_object* v_heq_1777_, lean_object* v_i_1778_, lean_object* v_entries_1779_){
_start:
{
size_t v_depth_boxed_1780_; lean_object* v_res_1781_; 
v_depth_boxed_1780_ = lean_unbox_usize(v_depth_1774_);
lean_dec(v_depth_1774_);
v_res_1781_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__9(v_00_u03b2_1773_, v_depth_boxed_1780_, v_keys_1775_, v_vals_1776_, v_heq_1777_, v_i_1778_, v_entries_1779_);
lean_dec_ref(v_vals_1776_);
lean_dec_ref(v_keys_1775_);
return v_res_1781_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12(lean_object* v_00_u03b2_1782_, lean_object* v_keys_1783_, lean_object* v_vals_1784_, lean_object* v_heq_1785_, lean_object* v_i_1786_, lean_object* v_k_1787_){
_start:
{
lean_object* v___x_1788_; 
v___x_1788_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___redArg(v_keys_1783_, v_vals_1784_, v_i_1786_, v_k_1787_);
return v___x_1788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12___boxed(lean_object* v_00_u03b2_1789_, lean_object* v_keys_1790_, lean_object* v_vals_1791_, lean_object* v_heq_1792_, lean_object* v_i_1793_, lean_object* v_k_1794_){
_start:
{
lean_object* v_res_1795_; 
v_res_1795_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__3_spec__6_spec__12(v_00_u03b2_1789_, v_keys_1790_, v_vals_1791_, v_heq_1792_, v_i_1793_, v_k_1794_);
lean_dec(v_k_1794_);
lean_dec_ref(v_vals_1791_);
lean_dec_ref(v_keys_1790_);
return v_res_1795_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15(lean_object* v_00_u03b2_1796_, lean_object* v_n_1797_, lean_object* v_k_1798_, lean_object* v_v_1799_){
_start:
{
lean_object* v___x_1800_; 
v___x_1800_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15___redArg(v_n_1797_, v_k_1798_, v_v_1799_);
return v___x_1800_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16(lean_object* v_00_u03b2_1801_, size_t v_depth_1802_, lean_object* v_keys_1803_, lean_object* v_vals_1804_, lean_object* v_heq_1805_, lean_object* v_i_1806_, lean_object* v_entries_1807_){
_start:
{
lean_object* v___x_1808_; 
v___x_1808_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___redArg(v_depth_1802_, v_keys_1803_, v_vals_1804_, v_i_1806_, v_entries_1807_);
return v___x_1808_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16___boxed(lean_object* v_00_u03b2_1809_, lean_object* v_depth_1810_, lean_object* v_keys_1811_, lean_object* v_vals_1812_, lean_object* v_heq_1813_, lean_object* v_i_1814_, lean_object* v_entries_1815_){
_start:
{
size_t v_depth_boxed_1816_; lean_object* v_res_1817_; 
v_depth_boxed_1816_ = lean_unbox_usize(v_depth_1810_);
lean_dec(v_depth_1810_);
v_res_1817_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__16(v_00_u03b2_1809_, v_depth_boxed_1816_, v_keys_1811_, v_vals_1812_, v_heq_1813_, v_i_1814_, v_entries_1815_);
lean_dec_ref(v_vals_1812_);
lean_dec_ref(v_keys_1811_);
return v_res_1817_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11(lean_object* v_00_u03b2_1818_, lean_object* v_x_1819_, lean_object* v_x_1820_, lean_object* v_x_1821_, lean_object* v_x_1822_){
_start:
{
lean_object* v___x_1823_; 
v___x_1823_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1_spec__2_spec__4_spec__11___redArg(v_x_1819_, v_x_1820_, v_x_1821_, v_x_1822_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15(lean_object* v_00_u03b2_1824_, lean_object* v_x_1825_, lean_object* v_x_1826_, lean_object* v_x_1827_, lean_object* v_x_1828_){
_start:
{
lean_object* v___x_1829_; 
v___x_1829_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2_spec__4_spec__8_spec__15___redArg(v_x_1825_, v_x_1826_, v_x_1827_, v_x_1828_);
return v___x_1829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21(lean_object* v_00_u03b2_1830_, lean_object* v_x_1831_, lean_object* v_x_1832_, lean_object* v_x_1833_, lean_object* v_x_1834_){
_start:
{
lean_object* v___x_1835_; 
v___x_1835_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8_spec__15_spec__21___redArg(v_x_1831_, v_x_1832_, v_x_1833_, v_x_1834_);
return v___x_1835_;
}
}
static lean_object* _init_lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1836_; 
v___x_1836_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_1836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2(lean_object* v_msg_1837_){
_start:
{
lean_object* v___x_1838_; lean_object* v___x_1839_; 
v___x_1838_ = lean_obj_once(&lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0, &lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0_once, _init_lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2___closed__0);
v___x_1839_ = lean_panic_fn_borrowed(v___x_1838_, v_msg_1837_);
return v___x_1839_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1_spec__5(lean_object* v_vs_1840_, lean_object* v_v_1841_, lean_object* v_i_1842_){
_start:
{
lean_object* v___x_1847_; uint8_t v___x_1848_; 
v___x_1847_ = lean_array_get_size(v_vs_1840_);
v___x_1848_ = lean_nat_dec_lt(v_i_1842_, v___x_1847_);
if (v___x_1848_ == 0)
{
lean_object* v___x_1849_; 
lean_dec(v_i_1842_);
v___x_1849_ = lean_array_push(v_vs_1840_, v_v_1841_);
return v___x_1849_;
}
else
{
lean_object* v_fst_1850_; lean_object* v_snd_1851_; lean_object* v___x_1852_; lean_object* v_fst_1853_; lean_object* v_snd_1854_; uint8_t v___y_1856_; lean_object* v_name_1859_; lean_object* v_name_1860_; lean_object* v_name_1861_; uint8_t v_builder_1862_; uint8_t v_phase_1863_; uint8_t v_scope_1864_; uint64_t v_hash_1865_; lean_object* v_name_1866_; uint8_t v_builder_1867_; uint8_t v_phase_1868_; uint8_t v_scope_1869_; uint64_t v_hash_1870_; uint8_t v___y_1872_; uint8_t v___x_1876_; 
v_fst_1850_ = lean_ctor_get(v_v_1841_, 0);
v_snd_1851_ = lean_ctor_get(v_v_1841_, 1);
v___x_1852_ = lean_array_fget_borrowed(v_vs_1840_, v_i_1842_);
v_fst_1853_ = lean_ctor_get(v___x_1852_, 0);
v_snd_1854_ = lean_ctor_get(v___x_1852_, 1);
v_name_1859_ = lean_ctor_get(v_fst_1850_, 1);
v_name_1860_ = lean_ctor_get(v_fst_1853_, 1);
v_name_1861_ = lean_ctor_get(v_name_1859_, 0);
v_builder_1862_ = lean_ctor_get_uint8(v_name_1859_, sizeof(void*)*1 + 8);
v_phase_1863_ = lean_ctor_get_uint8(v_name_1859_, sizeof(void*)*1 + 9);
v_scope_1864_ = lean_ctor_get_uint8(v_name_1859_, sizeof(void*)*1 + 10);
v_hash_1865_ = lean_ctor_get_uint64(v_name_1859_, sizeof(void*)*1);
v_name_1866_ = lean_ctor_get(v_name_1860_, 0);
v_builder_1867_ = lean_ctor_get_uint8(v_name_1860_, sizeof(void*)*1 + 8);
v_phase_1868_ = lean_ctor_get_uint8(v_name_1860_, sizeof(void*)*1 + 9);
v_scope_1869_ = lean_ctor_get_uint8(v_name_1860_, sizeof(void*)*1 + 10);
v_hash_1870_ = lean_ctor_get_uint64(v_name_1860_, sizeof(void*)*1);
v___x_1876_ = lean_uint64_dec_eq(v_hash_1865_, v_hash_1870_);
if (v___x_1876_ == 0)
{
v___y_1872_ = v___x_1876_;
goto v___jp_1871_;
}
else
{
uint8_t v___x_1877_; 
v___x_1877_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_1862_, v_builder_1867_);
v___y_1872_ = v___x_1877_;
goto v___jp_1871_;
}
v___jp_1855_:
{
if (v___y_1856_ == 0)
{
goto v___jp_1843_;
}
else
{
uint8_t v___x_1857_; 
v___x_1857_ = lp_aesop_Aesop_instBEqPremiseIndex_beq(v_snd_1851_, v_snd_1854_);
if (v___x_1857_ == 0)
{
goto v___jp_1843_;
}
else
{
lean_object* v___x_1858_; 
v___x_1858_ = lean_array_fset(v_vs_1840_, v_i_1842_, v_v_1841_);
lean_dec(v_i_1842_);
return v___x_1858_;
}
}
}
v___jp_1871_:
{
if (v___y_1872_ == 0)
{
goto v___jp_1843_;
}
else
{
uint8_t v___x_1873_; 
v___x_1873_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_1863_, v_phase_1868_);
if (v___x_1873_ == 0)
{
v___y_1856_ = v___x_1873_;
goto v___jp_1855_;
}
else
{
uint8_t v___x_1874_; 
v___x_1874_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_1864_, v_scope_1869_);
if (v___x_1874_ == 0)
{
v___y_1856_ = v___x_1874_;
goto v___jp_1855_;
}
else
{
uint8_t v___x_1875_; 
v___x_1875_ = lean_name_eq(v_name_1861_, v_name_1866_);
v___y_1856_ = v___x_1875_;
goto v___jp_1855_;
}
}
}
}
}
v___jp_1843_:
{
lean_object* v___x_1844_; lean_object* v___x_1845_; 
v___x_1844_ = lean_unsigned_to_nat(1u);
v___x_1845_ = lean_nat_add(v_i_1842_, v___x_1844_);
lean_dec(v_i_1842_);
v_i_1842_ = v___x_1845_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1(lean_object* v_vs_1878_, lean_object* v_v_1879_){
_start:
{
lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1880_ = lean_unsigned_to_nat(0u);
v___x_1881_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1_spec__5(v_vs_1878_, v_v_1879_, v___x_1880_);
return v___x_1881_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(lean_object* v_a_1882_, lean_object* v_b_1883_){
_start:
{
lean_object* v_fst_1884_; lean_object* v_fst_1885_; uint8_t v___x_1886_; 
v_fst_1884_ = lean_ctor_get(v_a_1882_, 0);
v_fst_1885_ = lean_ctor_get(v_b_1883_, 0);
v___x_1886_ = l_Lean_Meta_DiscrTree_Key_lt(v_fst_1884_, v_fst_1885_);
return v___x_1886_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1___boxed(lean_object* v_a_1887_, lean_object* v_b_1888_){
_start:
{
uint8_t v_res_1889_; lean_object* v_r_1890_; 
v_res_1889_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v_a_1887_, v_b_1888_);
lean_dec_ref(v_b_1888_);
lean_dec_ref(v_a_1887_);
v_r_1890_ = lean_box(v_res_1889_);
return v_r_1890_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(lean_object* v_x_1891_, lean_object* v_keys_1892_, lean_object* v_v_1893_, lean_object* v_k_1894_, lean_object* v_x_1895_){
_start:
{
lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v_c_1898_; lean_object* v___x_1899_; 
v___x_1896_ = lean_unsigned_to_nat(1u);
v___x_1897_ = lean_nat_add(v_x_1891_, v___x_1896_);
v_c_1898_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_1892_, v_v_1893_, v___x_1897_);
lean_dec(v___x_1897_);
v___x_1899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1899_, 0, v_k_1894_);
lean_ctor_set(v___x_1899_, 1, v_c_1898_);
return v___x_1899_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0___boxed(lean_object* v_x_1900_, lean_object* v_keys_1901_, lean_object* v_v_1902_, lean_object* v_k_1903_, lean_object* v_x_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(v_x_1900_, v_keys_1901_, v_v_1902_, v_k_1903_, v_x_1904_);
lean_dec_ref(v_keys_1901_);
lean_dec(v_x_1900_);
return v_res_1905_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg(lean_object* v_x_1908_, lean_object* v_keys_1909_, lean_object* v_v_1910_, lean_object* v_k_1911_, lean_object* v_as_1912_, lean_object* v_k_1913_, lean_object* v_x_1914_, lean_object* v_x_1915_){
_start:
{
lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v_mid_1918_; lean_object* v_midVal_1919_; uint8_t v___x_1920_; 
v___x_1916_ = lean_nat_add(v_x_1914_, v_x_1915_);
v___x_1917_ = lean_unsigned_to_nat(1u);
v_mid_1918_ = lean_nat_shiftr(v___x_1916_, v___x_1917_);
lean_dec(v___x_1916_);
v_midVal_1919_ = lean_array_fget(v_as_1912_, v_mid_1918_);
v___x_1920_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v_midVal_1919_, v_k_1913_);
if (v___x_1920_ == 0)
{
uint8_t v___x_1921_; 
lean_dec(v_x_1915_);
v___x_1921_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v_k_1913_, v_midVal_1919_);
if (v___x_1921_ == 0)
{
lean_object* v___x_1922_; uint8_t v___x_1923_; 
lean_dec(v_x_1914_);
v___x_1922_ = lean_array_get_size(v_as_1912_);
v___x_1923_ = lean_nat_dec_lt(v_mid_1918_, v___x_1922_);
if (v___x_1923_ == 0)
{
lean_dec(v_midVal_1919_);
lean_dec(v_mid_1918_);
lean_dec(v_k_1911_);
lean_dec_ref(v_v_1910_);
return v_as_1912_;
}
else
{
lean_object* v_snd_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1936_; 
v_snd_1924_ = lean_ctor_get(v_midVal_1919_, 1);
v_isSharedCheck_1936_ = !lean_is_exclusive(v_midVal_1919_);
if (v_isSharedCheck_1936_ == 0)
{
lean_object* v_unused_1937_; 
v_unused_1937_ = lean_ctor_get(v_midVal_1919_, 0);
lean_dec(v_unused_1937_);
v___x_1926_ = v_midVal_1919_;
v_isShared_1927_ = v_isSharedCheck_1936_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_snd_1924_);
lean_dec(v_midVal_1919_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1936_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1928_; lean_object* v_xs_x27_1929_; lean_object* v___x_1930_; lean_object* v_c_1931_; lean_object* v___x_1933_; 
v___x_1928_ = lean_box(0);
v_xs_x27_1929_ = lean_array_fset(v_as_1912_, v_mid_1918_, v___x_1928_);
v___x_1930_ = lean_nat_add(v_x_1908_, v___x_1917_);
v_c_1931_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(v_keys_1909_, v_v_1910_, v___x_1930_, v_snd_1924_);
lean_dec(v___x_1930_);
if (v_isShared_1927_ == 0)
{
lean_ctor_set(v___x_1926_, 1, v_c_1931_);
lean_ctor_set(v___x_1926_, 0, v_k_1911_);
v___x_1933_ = v___x_1926_;
goto v_reusejp_1932_;
}
else
{
lean_object* v_reuseFailAlloc_1935_; 
v_reuseFailAlloc_1935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1935_, 0, v_k_1911_);
lean_ctor_set(v_reuseFailAlloc_1935_, 1, v_c_1931_);
v___x_1933_ = v_reuseFailAlloc_1935_;
goto v_reusejp_1932_;
}
v_reusejp_1932_:
{
lean_object* v___x_1934_; 
v___x_1934_ = lean_array_fset(v_xs_x27_1929_, v_mid_1918_, v___x_1933_);
lean_dec(v_mid_1918_);
return v___x_1934_;
}
}
}
}
else
{
lean_dec(v_midVal_1919_);
v_x_1915_ = v_mid_1918_;
goto _start;
}
}
else
{
uint8_t v___x_1939_; 
lean_dec(v_midVal_1919_);
v___x_1939_ = lean_nat_dec_eq(v_mid_1918_, v_x_1914_);
if (v___x_1939_ == 0)
{
lean_dec(v_x_1914_);
v_x_1914_ = v_mid_1918_;
goto _start;
}
else
{
lean_object* v___x_1941_; lean_object* v_c_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v_j_1945_; lean_object* v_as_1946_; lean_object* v___x_1947_; 
lean_dec(v_mid_1918_);
lean_dec(v_x_1915_);
v___x_1941_ = lean_nat_add(v_x_1908_, v___x_1917_);
v_c_1942_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_1909_, v_v_1910_, v___x_1941_);
lean_dec(v___x_1941_);
v___x_1943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1943_, 0, v_k_1911_);
lean_ctor_set(v___x_1943_, 1, v_c_1942_);
v___x_1944_ = lean_nat_add(v_x_1914_, v___x_1917_);
lean_dec(v_x_1914_);
v_j_1945_ = lean_array_get_size(v_as_1912_);
v_as_1946_ = lean_array_push(v_as_1912_, v___x_1943_);
v___x_1947_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_1944_, v_as_1946_, v_j_1945_);
lean_dec(v___x_1944_);
return v___x_1947_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2(lean_object* v_x_1948_, lean_object* v_keys_1949_, lean_object* v_v_1950_, lean_object* v_k_1951_, lean_object* v_as_1952_, lean_object* v_k_1953_){
_start:
{
lean_object* v___x_1954_; lean_object* v___x_1955_; uint8_t v___x_1956_; 
v___x_1954_ = lean_array_get_size(v_as_1952_);
v___x_1955_ = lean_unsigned_to_nat(0u);
v___x_1956_ = lean_nat_dec_eq(v___x_1954_, v___x_1955_);
if (v___x_1956_ == 0)
{
lean_object* v___x_1957_; uint8_t v___x_1958_; 
v___x_1957_ = lean_array_fget_borrowed(v_as_1952_, v___x_1955_);
v___x_1958_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v_k_1953_, v___x_1957_);
if (v___x_1958_ == 0)
{
uint8_t v___x_1959_; 
v___x_1959_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v___x_1957_, v_k_1953_);
if (v___x_1959_ == 0)
{
uint8_t v___x_1960_; 
v___x_1960_ = lean_nat_dec_lt(v___x_1955_, v___x_1954_);
if (v___x_1960_ == 0)
{
lean_dec(v_k_1951_);
lean_dec_ref(v_v_1950_);
return v_as_1952_;
}
else
{
lean_object* v___x_1961_; lean_object* v_xs_x27_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; 
lean_inc(v___x_1957_);
v___x_1961_ = lean_box(0);
v_xs_x27_1962_ = lean_array_fset(v_as_1952_, v___x_1955_, v___x_1961_);
v___x_1963_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v___x_1957_);
v___x_1964_ = lean_array_fset(v_xs_x27_1962_, v___x_1955_, v___x_1963_);
return v___x_1964_;
}
}
else
{
lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; uint8_t v___x_1968_; 
v___x_1965_ = lean_unsigned_to_nat(1u);
v___x_1966_ = lean_nat_sub(v___x_1954_, v___x_1965_);
v___x_1967_ = lean_array_fget_borrowed(v_as_1952_, v___x_1966_);
v___x_1968_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v___x_1967_, v_k_1953_);
if (v___x_1968_ == 0)
{
uint8_t v___x_1969_; 
v___x_1969_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__1(v_k_1953_, v___x_1967_);
if (v___x_1969_ == 0)
{
uint8_t v___x_1970_; 
v___x_1970_ = lean_nat_dec_lt(v___x_1966_, v___x_1954_);
if (v___x_1970_ == 0)
{
lean_dec(v___x_1966_);
lean_dec(v_k_1951_);
lean_dec_ref(v_v_1950_);
return v_as_1952_;
}
else
{
lean_object* v___x_1971_; lean_object* v_xs_x27_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; 
lean_inc(v___x_1967_);
v___x_1971_ = lean_box(0);
v_xs_x27_1972_ = lean_array_fset(v_as_1952_, v___x_1966_, v___x_1971_);
v___x_1973_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v___x_1967_);
v___x_1974_ = lean_array_fset(v_xs_x27_1972_, v___x_1966_, v___x_1973_);
lean_dec(v___x_1966_);
return v___x_1974_;
}
}
else
{
lean_object* v___x_1975_; 
v___x_1975_ = lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v_as_1952_, v_k_1953_, v___x_1955_, v___x_1966_);
return v___x_1975_;
}
}
else
{
lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; 
lean_dec(v___x_1966_);
v___x_1976_ = lean_box(0);
v___x_1977_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v___x_1976_);
v___x_1978_ = lean_array_push(v_as_1952_, v___x_1977_);
return v___x_1978_;
}
}
}
else
{
lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v_as_1981_; lean_object* v___x_1982_; 
v___x_1979_ = lean_box(0);
v___x_1980_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v___x_1979_);
v_as_1981_ = lean_array_push(v_as_1952_, v___x_1980_);
v___x_1982_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_1955_, v_as_1981_, v___x_1954_);
return v___x_1982_;
}
}
else
{
lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
v___x_1983_ = lean_box(0);
v___x_1984_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__0(v_x_1948_, v_keys_1949_, v_v_1950_, v_k_1951_, v___x_1983_);
v___x_1985_ = lean_array_push(v_as_1952_, v___x_1984_);
return v___x_1985_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(lean_object* v_keys_1986_, lean_object* v_v_1987_, lean_object* v_x_1988_, lean_object* v_x_1989_){
_start:
{
lean_object* v_vs_1990_; lean_object* v_children_1991_; lean_object* v___x_1993_; uint8_t v_isShared_1994_; uint8_t v_isSharedCheck_2008_; 
v_vs_1990_ = lean_ctor_get(v_x_1989_, 0);
v_children_1991_ = lean_ctor_get(v_x_1989_, 1);
v_isSharedCheck_2008_ = !lean_is_exclusive(v_x_1989_);
if (v_isSharedCheck_2008_ == 0)
{
v___x_1993_ = v_x_1989_;
v_isShared_1994_ = v_isSharedCheck_2008_;
goto v_resetjp_1992_;
}
else
{
lean_inc(v_children_1991_);
lean_inc(v_vs_1990_);
lean_dec(v_x_1989_);
v___x_1993_ = lean_box(0);
v_isShared_1994_ = v_isSharedCheck_2008_;
goto v_resetjp_1992_;
}
v_resetjp_1992_:
{
lean_object* v___x_1995_; uint8_t v___x_1996_; 
v___x_1995_ = lean_array_get_size(v_keys_1986_);
v___x_1996_ = lean_nat_dec_lt(v_x_1988_, v___x_1995_);
if (v___x_1996_ == 0)
{
lean_object* v___x_1997_; lean_object* v___x_1999_; 
v___x_1997_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__1(v_vs_1990_, v_v_1987_);
if (v_isShared_1994_ == 0)
{
lean_ctor_set(v___x_1993_, 0, v___x_1997_);
v___x_1999_ = v___x_1993_;
goto v_reusejp_1998_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v___x_1997_);
lean_ctor_set(v_reuseFailAlloc_2000_, 1, v_children_1991_);
v___x_1999_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1998_;
}
v_reusejp_1998_:
{
return v___x_1999_;
}
}
else
{
lean_object* v_k_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v_c_2004_; lean_object* v___x_2006_; 
v_k_2001_ = lean_array_fget_borrowed(v_keys_1986_, v_x_1988_);
v___x_2002_ = ((lean_object*)(lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___closed__0));
lean_inc_n(v_k_2001_, 2);
v___x_2003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2003_, 0, v_k_2001_);
lean_ctor_set(v___x_2003_, 1, v___x_2002_);
v_c_2004_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2(v_x_1988_, v_keys_1986_, v_v_1987_, v_k_2001_, v_children_1991_, v___x_2003_);
lean_dec_ref_known(v___x_2003_, 2);
if (v_isShared_1994_ == 0)
{
lean_ctor_set(v___x_1993_, 1, v_c_2004_);
v___x_2006_ = v___x_1993_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v_vs_1990_);
lean_ctor_set(v_reuseFailAlloc_2007_, 1, v_c_2004_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
return v___x_2006_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2(lean_object* v_x_2009_, lean_object* v_keys_2010_, lean_object* v_v_2011_, lean_object* v_k_2012_, lean_object* v_x_2013_){
_start:
{
lean_object* v_snd_2014_; lean_object* v___x_2016_; uint8_t v_isShared_2017_; uint8_t v_isSharedCheck_2024_; 
v_snd_2014_ = lean_ctor_get(v_x_2013_, 1);
v_isSharedCheck_2024_ = !lean_is_exclusive(v_x_2013_);
if (v_isSharedCheck_2024_ == 0)
{
lean_object* v_unused_2025_; 
v_unused_2025_ = lean_ctor_get(v_x_2013_, 0);
lean_dec(v_unused_2025_);
v___x_2016_ = v_x_2013_;
v_isShared_2017_ = v_isSharedCheck_2024_;
goto v_resetjp_2015_;
}
else
{
lean_inc(v_snd_2014_);
lean_dec(v_x_2013_);
v___x_2016_ = lean_box(0);
v_isShared_2017_ = v_isSharedCheck_2024_;
goto v_resetjp_2015_;
}
v_resetjp_2015_:
{
lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v_c_2020_; lean_object* v___x_2022_; 
v___x_2018_ = lean_unsigned_to_nat(1u);
v___x_2019_ = lean_nat_add(v_x_2009_, v___x_2018_);
v_c_2020_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(v_keys_2010_, v_v_2011_, v___x_2019_, v_snd_2014_);
lean_dec(v___x_2019_);
if (v_isShared_2017_ == 0)
{
lean_ctor_set(v___x_2016_, 1, v_c_2020_);
lean_ctor_set(v___x_2016_, 0, v_k_2012_);
v___x_2022_ = v___x_2016_;
goto v_reusejp_2021_;
}
else
{
lean_object* v_reuseFailAlloc_2023_; 
v_reuseFailAlloc_2023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2023_, 0, v_k_2012_);
lean_ctor_set(v_reuseFailAlloc_2023_, 1, v_c_2020_);
v___x_2022_ = v_reuseFailAlloc_2023_;
goto v_reusejp_2021_;
}
v_reusejp_2021_:
{
return v___x_2022_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2___boxed(lean_object* v_x_2026_, lean_object* v_keys_2027_, lean_object* v_v_2028_, lean_object* v_k_2029_, lean_object* v_x_2030_){
_start:
{
lean_object* v_res_2031_; 
v_res_2031_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___lam__2(v_x_2026_, v_keys_2027_, v_v_2028_, v_k_2029_, v_x_2030_);
lean_dec_ref(v_keys_2027_);
lean_dec(v_x_2026_);
return v_res_2031_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0___boxed(lean_object* v_keys_2032_, lean_object* v_v_2033_, lean_object* v_x_2034_, lean_object* v_x_2035_){
_start:
{
lean_object* v_res_2036_; 
v_res_2036_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(v_keys_2032_, v_v_2033_, v_x_2034_, v_x_2035_);
lean_dec(v_x_2034_);
lean_dec_ref(v_keys_2032_);
return v_res_2036_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object* v_x_2037_, lean_object* v_keys_2038_, lean_object* v_v_2039_, lean_object* v_k_2040_, lean_object* v_as_2041_, lean_object* v_k_2042_, lean_object* v_x_2043_, lean_object* v_x_2044_){
_start:
{
lean_object* v_res_2045_; 
v_res_2045_ = lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg(v_x_2037_, v_keys_2038_, v_v_2039_, v_k_2040_, v_as_2041_, v_k_2042_, v_x_2043_, v_x_2044_);
lean_dec_ref(v_k_2042_);
lean_dec_ref(v_keys_2038_);
lean_dec(v_x_2037_);
return v_res_2045_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2___boxed(lean_object* v_x_2046_, lean_object* v_keys_2047_, lean_object* v_v_2048_, lean_object* v_k_2049_, lean_object* v_as_2050_, lean_object* v_k_2051_){
_start:
{
lean_object* v_res_2052_; 
v_res_2052_ = lp_aesop_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2(v_x_2046_, v_keys_2047_, v_v_2048_, v_k_2049_, v_as_2050_, v_k_2051_);
lean_dec_ref(v_k_2051_);
lean_dec_ref(v_keys_2047_);
lean_dec(v_x_2046_);
return v_res_2052_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(lean_object* v_keys_2053_, lean_object* v_v_2054_, lean_object* v_x_2055_){
_start:
{
if (lean_obj_tag(v_x_2055_) == 0)
{
lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; 
v___x_2056_ = lean_unsigned_to_nat(1u);
v___x_2057_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_2053_, v_v_2054_, v___x_2056_);
v___x_2058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2058_, 0, v___x_2057_);
return v___x_2058_;
}
else
{
lean_object* v_val_2059_; lean_object* v___x_2061_; uint8_t v_isShared_2062_; uint8_t v_isSharedCheck_2068_; 
v_val_2059_ = lean_ctor_get(v_x_2055_, 0);
v_isSharedCheck_2068_ = !lean_is_exclusive(v_x_2055_);
if (v_isSharedCheck_2068_ == 0)
{
v___x_2061_ = v_x_2055_;
v_isShared_2062_ = v_isSharedCheck_2068_;
goto v_resetjp_2060_;
}
else
{
lean_inc(v_val_2059_);
lean_dec(v_x_2055_);
v___x_2061_ = lean_box(0);
v_isShared_2062_ = v_isSharedCheck_2068_;
goto v_resetjp_2060_;
}
v_resetjp_2060_:
{
lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2066_; 
v___x_2063_ = lean_unsigned_to_nat(1u);
v___x_2064_ = lp_aesop___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0(v_keys_2053_, v_v_2054_, v___x_2063_, v_val_2059_);
if (v_isShared_2062_ == 0)
{
lean_ctor_set(v___x_2061_, 0, v___x_2064_);
v___x_2066_ = v___x_2061_;
goto v_reusejp_2065_;
}
else
{
lean_object* v_reuseFailAlloc_2067_; 
v_reuseFailAlloc_2067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2067_, 0, v___x_2064_);
v___x_2066_ = v_reuseFailAlloc_2067_;
goto v_reusejp_2065_;
}
v_reusejp_2065_:
{
return v___x_2066_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0___boxed(lean_object* v_keys_2069_, lean_object* v_v_2070_, lean_object* v_x_2071_){
_start:
{
lean_object* v_res_2072_; 
v_res_2072_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2069_, v_v_2070_, v_x_2071_);
lean_dec_ref(v_keys_2069_);
return v_res_2072_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10(lean_object* v_xs_2073_, lean_object* v_v_2074_, lean_object* v_i_2075_){
_start:
{
lean_object* v___x_2076_; uint8_t v___x_2077_; 
v___x_2076_ = lean_array_get_size(v_xs_2073_);
v___x_2077_ = lean_nat_dec_lt(v_i_2075_, v___x_2076_);
if (v___x_2077_ == 0)
{
lean_object* v___x_2078_; 
lean_dec(v_i_2075_);
v___x_2078_ = lean_box(0);
return v___x_2078_;
}
else
{
lean_object* v___x_2079_; uint8_t v___x_2080_; 
v___x_2079_ = lean_array_fget_borrowed(v_xs_2073_, v_i_2075_);
v___x_2080_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v___x_2079_, v_v_2074_);
if (v___x_2080_ == 0)
{
lean_object* v___x_2081_; lean_object* v___x_2082_; 
v___x_2081_ = lean_unsigned_to_nat(1u);
v___x_2082_ = lean_nat_add(v_i_2075_, v___x_2081_);
lean_dec(v_i_2075_);
v_i_2075_ = v___x_2082_;
goto _start;
}
else
{
lean_object* v___x_2084_; 
v___x_2084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2084_, 0, v_i_2075_);
return v___x_2084_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10___boxed(lean_object* v_xs_2085_, lean_object* v_v_2086_, lean_object* v_i_2087_){
_start:
{
lean_object* v_res_2088_; 
v_res_2088_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10(v_xs_2085_, v_v_2086_, v_i_2087_);
lean_dec(v_v_2086_);
lean_dec_ref(v_xs_2085_);
return v_res_2088_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4(lean_object* v_xs_2089_, lean_object* v_v_2090_){
_start:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; 
v___x_2091_ = lean_unsigned_to_nat(0u);
v___x_2092_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4_spec__10(v_xs_2089_, v_v_2090_, v___x_2091_);
return v___x_2092_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4___boxed(lean_object* v_xs_2093_, lean_object* v_v_2094_){
_start:
{
lean_object* v_res_2095_; 
v_res_2095_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4(v_xs_2093_, v_v_2094_);
lean_dec(v_v_2094_);
lean_dec_ref(v_xs_2093_);
return v_res_2095_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1(lean_object* v_keys_2096_, lean_object* v_v_2097_, lean_object* v_x_2098_, size_t v_x_2099_, size_t v_x_2100_, lean_object* v_x_2101_){
_start:
{
if (lean_obj_tag(v_x_2098_) == 0)
{
lean_object* v_es_2102_; size_t v___x_2103_; size_t v___x_2104_; lean_object* v_j_2105_; lean_object* v___x_2106_; uint8_t v___x_2107_; 
v_es_2102_ = lean_ctor_get(v_x_2098_, 0);
v___x_2103_ = ((size_t)31ULL);
v___x_2104_ = lean_usize_land(v_x_2099_, v___x_2103_);
v_j_2105_ = lean_usize_to_nat(v___x_2104_);
v___x_2106_ = lean_array_get_size(v_es_2102_);
v___x_2107_ = lean_nat_dec_lt(v_j_2105_, v___x_2106_);
if (v___x_2107_ == 0)
{
lean_dec(v_j_2105_);
lean_dec(v_x_2101_);
lean_dec_ref(v_v_2097_);
return v_x_2098_;
}
else
{
lean_object* v___x_2109_; uint8_t v_isShared_2110_; uint8_t v_isSharedCheck_2175_; 
lean_inc_ref(v_es_2102_);
v_isSharedCheck_2175_ = !lean_is_exclusive(v_x_2098_);
if (v_isSharedCheck_2175_ == 0)
{
lean_object* v_unused_2176_; 
v_unused_2176_ = lean_ctor_get(v_x_2098_, 0);
lean_dec(v_unused_2176_);
v___x_2109_ = v_x_2098_;
v_isShared_2110_ = v_isSharedCheck_2175_;
goto v_resetjp_2108_;
}
else
{
lean_dec(v_x_2098_);
v___x_2109_ = lean_box(0);
v_isShared_2110_ = v_isSharedCheck_2175_;
goto v_resetjp_2108_;
}
v_resetjp_2108_:
{
lean_object* v_v_2111_; lean_object* v___x_2112_; lean_object* v_xs_x27_2113_; lean_object* v___y_2115_; 
v_v_2111_ = lean_array_fget(v_es_2102_, v_j_2105_);
v___x_2112_ = lean_box(0);
v_xs_x27_2113_ = lean_array_fset(v_es_2102_, v_j_2105_, v___x_2112_);
switch(lean_obj_tag(v_v_2111_))
{
case 0:
{
lean_object* v_key_2120_; lean_object* v_val_2121_; uint8_t v___x_2122_; 
v_key_2120_ = lean_ctor_get(v_v_2111_, 0);
v_val_2121_ = lean_ctor_get(v_v_2111_, 1);
v___x_2122_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_2101_, v_key_2120_);
if (v___x_2122_ == 0)
{
lean_object* v___x_2123_; lean_object* v___x_2124_; 
v___x_2123_ = lean_box(0);
v___x_2124_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2096_, v_v_2097_, v___x_2123_);
if (lean_obj_tag(v___x_2124_) == 0)
{
lean_dec(v_x_2101_);
v___y_2115_ = v_v_2111_;
goto v___jp_2114_;
}
else
{
lean_object* v_val_2125_; lean_object* v___x_2127_; uint8_t v_isShared_2128_; uint8_t v_isSharedCheck_2133_; 
lean_inc(v_val_2121_);
lean_inc(v_key_2120_);
lean_dec_ref_known(v_v_2111_, 2);
v_val_2125_ = lean_ctor_get(v___x_2124_, 0);
v_isSharedCheck_2133_ = !lean_is_exclusive(v___x_2124_);
if (v_isSharedCheck_2133_ == 0)
{
v___x_2127_ = v___x_2124_;
v_isShared_2128_ = v_isSharedCheck_2133_;
goto v_resetjp_2126_;
}
else
{
lean_inc(v_val_2125_);
lean_dec(v___x_2124_);
v___x_2127_ = lean_box(0);
v_isShared_2128_ = v_isSharedCheck_2133_;
goto v_resetjp_2126_;
}
v_resetjp_2126_:
{
lean_object* v___x_2129_; lean_object* v___x_2131_; 
v___x_2129_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2120_, v_val_2121_, v_x_2101_, v_val_2125_);
if (v_isShared_2128_ == 0)
{
lean_ctor_set(v___x_2127_, 0, v___x_2129_);
v___x_2131_ = v___x_2127_;
goto v_reusejp_2130_;
}
else
{
lean_object* v_reuseFailAlloc_2132_; 
v_reuseFailAlloc_2132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2132_, 0, v___x_2129_);
v___x_2131_ = v_reuseFailAlloc_2132_;
goto v_reusejp_2130_;
}
v_reusejp_2130_:
{
v___y_2115_ = v___x_2131_;
goto v___jp_2114_;
}
}
}
}
else
{
lean_object* v___x_2135_; uint8_t v_isShared_2136_; uint8_t v_isSharedCheck_2144_; 
lean_inc(v_val_2121_);
v_isSharedCheck_2144_ = !lean_is_exclusive(v_v_2111_);
if (v_isSharedCheck_2144_ == 0)
{
lean_object* v_unused_2145_; lean_object* v_unused_2146_; 
v_unused_2145_ = lean_ctor_get(v_v_2111_, 1);
lean_dec(v_unused_2145_);
v_unused_2146_ = lean_ctor_get(v_v_2111_, 0);
lean_dec(v_unused_2146_);
v___x_2135_ = v_v_2111_;
v_isShared_2136_ = v_isSharedCheck_2144_;
goto v_resetjp_2134_;
}
else
{
lean_dec(v_v_2111_);
v___x_2135_ = lean_box(0);
v_isShared_2136_ = v_isSharedCheck_2144_;
goto v_resetjp_2134_;
}
v_resetjp_2134_:
{
lean_object* v___x_2137_; lean_object* v___x_2138_; 
v___x_2137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2137_, 0, v_val_2121_);
v___x_2138_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2096_, v_v_2097_, v___x_2137_);
if (lean_obj_tag(v___x_2138_) == 0)
{
lean_object* v___x_2139_; 
lean_del_object(v___x_2135_);
lean_dec(v_x_2101_);
v___x_2139_ = lean_box(2);
v___y_2115_ = v___x_2139_;
goto v___jp_2114_;
}
else
{
lean_object* v_val_2140_; lean_object* v___x_2142_; 
v_val_2140_ = lean_ctor_get(v___x_2138_, 0);
lean_inc(v_val_2140_);
lean_dec_ref_known(v___x_2138_, 1);
if (v_isShared_2136_ == 0)
{
lean_ctor_set(v___x_2135_, 1, v_val_2140_);
lean_ctor_set(v___x_2135_, 0, v_x_2101_);
v___x_2142_ = v___x_2135_;
goto v_reusejp_2141_;
}
else
{
lean_object* v_reuseFailAlloc_2143_; 
v_reuseFailAlloc_2143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2143_, 0, v_x_2101_);
lean_ctor_set(v_reuseFailAlloc_2143_, 1, v_val_2140_);
v___x_2142_ = v_reuseFailAlloc_2143_;
goto v_reusejp_2141_;
}
v_reusejp_2141_:
{
v___y_2115_ = v___x_2142_;
goto v___jp_2114_;
}
}
}
}
}
case 1:
{
lean_object* v_node_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2170_; 
v_node_2147_ = lean_ctor_get(v_v_2111_, 0);
v_isSharedCheck_2170_ = !lean_is_exclusive(v_v_2111_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2149_ = v_v_2111_;
v_isShared_2150_ = v_isSharedCheck_2170_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_node_2147_);
lean_dec(v_v_2111_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2170_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
size_t v___x_2151_; size_t v___x_2152_; size_t v___x_2153_; size_t v___x_2154_; lean_object* v_newNode_2155_; lean_object* v___x_2156_; 
v___x_2151_ = ((size_t)5ULL);
v___x_2152_ = lean_usize_shift_right(v_x_2099_, v___x_2151_);
v___x_2153_ = ((size_t)1ULL);
v___x_2154_ = lean_usize_add(v_x_2100_, v___x_2153_);
v_newNode_2155_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1(v_keys_2096_, v_v_2097_, v_node_2147_, v___x_2152_, v___x_2154_, v_x_2101_);
lean_inc_ref(v_newNode_2155_);
v___x_2156_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_2155_);
if (lean_obj_tag(v___x_2156_) == 0)
{
lean_object* v___x_2158_; 
if (v_isShared_2150_ == 0)
{
lean_ctor_set(v___x_2149_, 0, v_newNode_2155_);
v___x_2158_ = v___x_2149_;
goto v_reusejp_2157_;
}
else
{
lean_object* v_reuseFailAlloc_2159_; 
v_reuseFailAlloc_2159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2159_, 0, v_newNode_2155_);
v___x_2158_ = v_reuseFailAlloc_2159_;
goto v_reusejp_2157_;
}
v_reusejp_2157_:
{
v___y_2115_ = v___x_2158_;
goto v___jp_2114_;
}
}
else
{
lean_object* v_val_2160_; lean_object* v_fst_2161_; lean_object* v_snd_2162_; lean_object* v___x_2164_; uint8_t v_isShared_2165_; uint8_t v_isSharedCheck_2169_; 
lean_dec_ref(v_newNode_2155_);
lean_del_object(v___x_2149_);
v_val_2160_ = lean_ctor_get(v___x_2156_, 0);
lean_inc(v_val_2160_);
lean_dec_ref_known(v___x_2156_, 1);
v_fst_2161_ = lean_ctor_get(v_val_2160_, 0);
v_snd_2162_ = lean_ctor_get(v_val_2160_, 1);
v_isSharedCheck_2169_ = !lean_is_exclusive(v_val_2160_);
if (v_isSharedCheck_2169_ == 0)
{
v___x_2164_ = v_val_2160_;
v_isShared_2165_ = v_isSharedCheck_2169_;
goto v_resetjp_2163_;
}
else
{
lean_inc(v_snd_2162_);
lean_inc(v_fst_2161_);
lean_dec(v_val_2160_);
v___x_2164_ = lean_box(0);
v_isShared_2165_ = v_isSharedCheck_2169_;
goto v_resetjp_2163_;
}
v_resetjp_2163_:
{
lean_object* v___x_2167_; 
if (v_isShared_2165_ == 0)
{
v___x_2167_ = v___x_2164_;
goto v_reusejp_2166_;
}
else
{
lean_object* v_reuseFailAlloc_2168_; 
v_reuseFailAlloc_2168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2168_, 0, v_fst_2161_);
lean_ctor_set(v_reuseFailAlloc_2168_, 1, v_snd_2162_);
v___x_2167_ = v_reuseFailAlloc_2168_;
goto v_reusejp_2166_;
}
v_reusejp_2166_:
{
v___y_2115_ = v___x_2167_;
goto v___jp_2114_;
}
}
}
}
}
default: 
{
lean_object* v___x_2171_; lean_object* v___x_2172_; 
v___x_2171_ = lean_box(0);
v___x_2172_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2096_, v_v_2097_, v___x_2171_);
if (lean_obj_tag(v___x_2172_) == 0)
{
lean_dec(v_x_2101_);
v___y_2115_ = v_v_2111_;
goto v___jp_2114_;
}
else
{
lean_object* v_val_2173_; lean_object* v___x_2174_; 
v_val_2173_ = lean_ctor_get(v___x_2172_, 0);
lean_inc(v_val_2173_);
lean_dec_ref_known(v___x_2172_, 1);
v___x_2174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2174_, 0, v_x_2101_);
lean_ctor_set(v___x_2174_, 1, v_val_2173_);
v___y_2115_ = v___x_2174_;
goto v___jp_2114_;
}
}
}
v___jp_2114_:
{
lean_object* v___x_2116_; lean_object* v___x_2118_; 
v___x_2116_ = lean_array_fset(v_xs_x27_2113_, v_j_2105_, v___y_2115_);
lean_dec(v_j_2105_);
if (v_isShared_2110_ == 0)
{
lean_ctor_set(v___x_2109_, 0, v___x_2116_);
v___x_2118_ = v___x_2109_;
goto v_reusejp_2117_;
}
else
{
lean_object* v_reuseFailAlloc_2119_; 
v_reuseFailAlloc_2119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2119_, 0, v___x_2116_);
v___x_2118_ = v_reuseFailAlloc_2119_;
goto v_reusejp_2117_;
}
v_reusejp_2117_:
{
return v___x_2118_;
}
}
}
}
}
else
{
lean_object* v_ks_2177_; lean_object* v_vs_2178_; lean_object* v___x_2180_; uint8_t v_isShared_2181_; uint8_t v_isSharedCheck_2211_; 
v_ks_2177_ = lean_ctor_get(v_x_2098_, 0);
v_vs_2178_ = lean_ctor_get(v_x_2098_, 1);
v_isSharedCheck_2211_ = !lean_is_exclusive(v_x_2098_);
if (v_isSharedCheck_2211_ == 0)
{
v___x_2180_ = v_x_2098_;
v_isShared_2181_ = v_isSharedCheck_2211_;
goto v_resetjp_2179_;
}
else
{
lean_inc(v_vs_2178_);
lean_inc(v_ks_2177_);
lean_dec(v_x_2098_);
v___x_2180_ = lean_box(0);
v_isShared_2181_ = v_isSharedCheck_2211_;
goto v_resetjp_2179_;
}
v_resetjp_2179_:
{
lean_object* v___x_2182_; 
v___x_2182_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1_spec__4(v_ks_2177_, v_x_2101_);
if (lean_obj_tag(v___x_2182_) == 0)
{
lean_object* v___x_2184_; 
if (v_isShared_2181_ == 0)
{
v___x_2184_ = v___x_2180_;
goto v_reusejp_2183_;
}
else
{
lean_object* v_reuseFailAlloc_2189_; 
v_reuseFailAlloc_2189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2189_, 0, v_ks_2177_);
lean_ctor_set(v_reuseFailAlloc_2189_, 1, v_vs_2178_);
v___x_2184_ = v_reuseFailAlloc_2189_;
goto v_reusejp_2183_;
}
v_reusejp_2183_:
{
lean_object* v___x_2185_; lean_object* v___x_2186_; 
v___x_2185_ = lean_box(0);
v___x_2186_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2096_, v_v_2097_, v___x_2185_);
if (lean_obj_tag(v___x_2186_) == 0)
{
lean_dec(v_x_2101_);
return v___x_2184_;
}
else
{
lean_object* v_val_2187_; lean_object* v___x_2188_; 
v_val_2187_ = lean_ctor_get(v___x_2186_, 0);
lean_inc(v_val_2187_);
lean_dec_ref_known(v___x_2186_, 1);
v___x_2188_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__4_spec__8___redArg(v___x_2184_, v_x_2099_, v_x_2100_, v_x_2101_, v_val_2187_);
return v___x_2188_;
}
}
}
else
{
lean_object* v_val_2190_; lean_object* v___x_2192_; uint8_t v_isShared_2193_; uint8_t v_isSharedCheck_2210_; 
v_val_2190_ = lean_ctor_get(v___x_2182_, 0);
v_isSharedCheck_2210_ = !lean_is_exclusive(v___x_2182_);
if (v_isSharedCheck_2210_ == 0)
{
v___x_2192_ = v___x_2182_;
v_isShared_2193_ = v_isSharedCheck_2210_;
goto v_resetjp_2191_;
}
else
{
lean_inc(v_val_2190_);
lean_dec(v___x_2182_);
v___x_2192_ = lean_box(0);
v_isShared_2193_ = v_isSharedCheck_2210_;
goto v_resetjp_2191_;
}
v_resetjp_2191_:
{
lean_object* v_v_x27_2194_; lean_object* v_keys_2195_; lean_object* v_vals_2196_; lean_object* v___x_2198_; 
v_v_x27_2194_ = lean_array_fget(v_vs_2178_, v_val_2190_);
lean_inc(v_val_2190_);
v_keys_2195_ = l_Array_eraseIdx___redArg(v_ks_2177_, v_val_2190_);
v_vals_2196_ = l_Array_eraseIdx___redArg(v_vs_2178_, v_val_2190_);
if (v_isShared_2193_ == 0)
{
lean_ctor_set(v___x_2192_, 0, v_v_x27_2194_);
v___x_2198_ = v___x_2192_;
goto v_reusejp_2197_;
}
else
{
lean_object* v_reuseFailAlloc_2209_; 
v_reuseFailAlloc_2209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2209_, 0, v_v_x27_2194_);
v___x_2198_ = v_reuseFailAlloc_2209_;
goto v_reusejp_2197_;
}
v_reusejp_2197_:
{
lean_object* v___x_2199_; 
v___x_2199_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___lam__0(v_keys_2096_, v_v_2097_, v___x_2198_);
if (lean_obj_tag(v___x_2199_) == 0)
{
lean_object* v___x_2201_; 
lean_dec(v_x_2101_);
if (v_isShared_2181_ == 0)
{
lean_ctor_set(v___x_2180_, 1, v_vals_2196_);
lean_ctor_set(v___x_2180_, 0, v_keys_2195_);
v___x_2201_ = v___x_2180_;
goto v_reusejp_2200_;
}
else
{
lean_object* v_reuseFailAlloc_2202_; 
v_reuseFailAlloc_2202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2202_, 0, v_keys_2195_);
lean_ctor_set(v_reuseFailAlloc_2202_, 1, v_vals_2196_);
v___x_2201_ = v_reuseFailAlloc_2202_;
goto v_reusejp_2200_;
}
v_reusejp_2200_:
{
return v___x_2201_;
}
}
else
{
lean_object* v_val_2203_; lean_object* v_keys_2204_; lean_object* v_vals_2205_; lean_object* v___x_2207_; 
v_val_2203_ = lean_ctor_get(v___x_2199_, 0);
lean_inc(v_val_2203_);
lean_dec_ref_known(v___x_2199_, 1);
v_keys_2204_ = lean_array_push(v_keys_2195_, v_x_2101_);
v_vals_2205_ = lean_array_push(v_vals_2196_, v_val_2203_);
if (v_isShared_2181_ == 0)
{
lean_ctor_set(v___x_2180_, 1, v_vals_2205_);
lean_ctor_set(v___x_2180_, 0, v_keys_2204_);
v___x_2207_ = v___x_2180_;
goto v_reusejp_2206_;
}
else
{
lean_object* v_reuseFailAlloc_2208_; 
v_reuseFailAlloc_2208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2208_, 0, v_keys_2204_);
lean_ctor_set(v_reuseFailAlloc_2208_, 1, v_vals_2205_);
v___x_2207_ = v_reuseFailAlloc_2208_;
goto v_reusejp_2206_;
}
v_reusejp_2206_:
{
return v___x_2207_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1___boxed(lean_object* v_keys_2212_, lean_object* v_v_2213_, lean_object* v_x_2214_, lean_object* v_x_2215_, lean_object* v_x_2216_, lean_object* v_x_2217_){
_start:
{
size_t v_x_2615__boxed_2218_; size_t v_x_2616__boxed_2219_; lean_object* v_res_2220_; 
v_x_2615__boxed_2218_ = lean_unbox_usize(v_x_2215_);
lean_dec(v_x_2215_);
v_x_2616__boxed_2219_ = lean_unbox_usize(v_x_2216_);
lean_dec(v_x_2216_);
v_res_2220_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1(v_keys_2212_, v_v_2213_, v_x_2214_, v_x_2615__boxed_2218_, v_x_2616__boxed_2219_, v_x_2217_);
lean_dec_ref(v_keys_2212_);
return v_res_2220_;
}
}
static lean_object* _init_lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3(void){
_start:
{
lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; 
v___x_2224_ = ((lean_object*)(lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__2));
v___x_2225_ = lean_unsigned_to_nat(23u);
v___x_2226_ = lean_unsigned_to_nat(166u);
v___x_2227_ = ((lean_object*)(lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__1));
v___x_2228_ = ((lean_object*)(lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__0));
v___x_2229_ = l_mkPanicMessageWithDecl(v___x_2228_, v___x_2227_, v___x_2226_, v___x_2225_, v___x_2224_);
return v___x_2229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0(lean_object* v_d_2230_, lean_object* v_keys_2231_, lean_object* v_v_2232_){
_start:
{
lean_object* v___x_2233_; lean_object* v___x_2234_; uint8_t v___x_2235_; 
v___x_2233_ = lean_array_get_size(v_keys_2231_);
v___x_2234_ = lean_unsigned_to_nat(0u);
v___x_2235_ = lean_nat_dec_eq(v___x_2233_, v___x_2234_);
if (v___x_2235_ == 0)
{
lean_object* v___x_2236_; lean_object* v_k_2237_; uint64_t v___x_2238_; size_t v_h_2239_; size_t v___x_2240_; lean_object* v___x_2241_; 
v___x_2236_ = lean_box(0);
v_k_2237_ = lean_array_get_borrowed(v___x_2236_, v_keys_2231_, v___x_2234_);
v___x_2238_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_2237_);
v_h_2239_ = lean_uint64_to_usize(v___x_2238_);
v___x_2240_ = ((size_t)1ULL);
lean_inc(v_k_2237_);
v___x_2241_ = lp_aesop_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__1(v_keys_2231_, v_v_2232_, v_d_2230_, v_h_2239_, v___x_2240_, v_k_2237_);
return v___x_2241_;
}
else
{
lean_object* v___x_2242_; lean_object* v___x_2243_; 
lean_dec_ref(v_v_2232_);
lean_dec_ref(v_d_2230_);
v___x_2242_ = lean_obj_once(&lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3, &lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3_once, _init_lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___closed__3);
v___x_2243_ = lp_aesop_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__2(v___x_2242_);
return v___x_2243_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0___boxed(lean_object* v_d_2244_, lean_object* v_keys_2245_, lean_object* v_v_2246_){
_start:
{
lean_object* v_res_2247_; 
v_res_2247_ = lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0(v_d_2244_, v_keys_2245_, v_v_2246_);
lean_dec_ref(v_keys_2245_);
return v_res_2247_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1(lean_object* v_r_2248_, lean_object* v_as_2249_, size_t v_sz_2250_, size_t v_i_2251_, lean_object* v_b_2252_){
_start:
{
lean_object* v_a_2254_; uint8_t v___x_2258_; 
v___x_2258_ = lean_usize_dec_lt(v_i_2251_, v_sz_2250_);
if (v___x_2258_ == 0)
{
lean_dec_ref(v_r_2248_);
return v_b_2252_;
}
else
{
lean_object* v_a_2259_; lean_object* v_typeDiscrTreeKeys_x3f_2260_; 
v_a_2259_ = lean_array_uget_borrowed(v_as_2249_, v_i_2251_);
v_typeDiscrTreeKeys_x3f_2260_ = lean_ctor_get(v_a_2259_, 0);
if (lean_obj_tag(v_typeDiscrTreeKeys_x3f_2260_) == 1)
{
lean_object* v_premiseIndex_2261_; lean_object* v_val_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; 
v_premiseIndex_2261_ = lean_ctor_get(v_a_2259_, 2);
v_val_2262_ = lean_ctor_get(v_typeDiscrTreeKeys_x3f_2260_, 0);
lean_inc(v_premiseIndex_2261_);
lean_inc_ref(v_r_2248_);
v___x_2263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2263_, 0, v_r_2248_);
lean_ctor_set(v___x_2263_, 1, v_premiseIndex_2261_);
v___x_2264_ = lp_aesop_Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0(v_b_2252_, v_val_2262_, v___x_2263_);
v_a_2254_ = v___x_2264_;
goto v___jp_2253_;
}
else
{
v_a_2254_ = v_b_2252_;
goto v___jp_2253_;
}
}
v___jp_2253_:
{
size_t v___x_2255_; size_t v___x_2256_; 
v___x_2255_ = ((size_t)1ULL);
v___x_2256_ = lean_usize_add(v_i_2251_, v___x_2255_);
v_i_2251_ = v___x_2256_;
v_b_2252_ = v_a_2254_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1___boxed(lean_object* v_r_2265_, lean_object* v_as_2266_, lean_object* v_sz_2267_, lean_object* v_i_2268_, lean_object* v_b_2269_){
_start:
{
size_t v_sz_boxed_2270_; size_t v_i_boxed_2271_; lean_object* v_res_2272_; 
v_sz_boxed_2270_ = lean_unbox_usize(v_sz_2267_);
lean_dec(v_sz_2267_);
v_i_boxed_2271_ = lean_unbox_usize(v_i_2268_);
lean_dec(v_i_2268_);
v_res_2272_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1(v_r_2265_, v_as_2266_, v_sz_boxed_2270_, v_i_boxed_2271_, v_b_2269_);
lean_dec_ref(v_as_2266_);
return v_res_2272_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2(lean_object* v_r_2273_, lean_object* v_as_2274_, size_t v_sz_2275_, size_t v_i_2276_, lean_object* v_b_2277_){
_start:
{
uint8_t v___x_2278_; 
v___x_2278_ = lean_usize_dec_lt(v_i_2276_, v_sz_2275_);
if (v___x_2278_ == 0)
{
lean_dec_ref(v_r_2273_);
return v_b_2277_;
}
else
{
lean_object* v_a_2279_; size_t v_sz_2280_; size_t v___x_2281_; lean_object* v___x_2282_; size_t v___x_2283_; size_t v___x_2284_; 
v_a_2279_ = lean_array_uget_borrowed(v_as_2274_, v_i_2276_);
v_sz_2280_ = lean_array_size(v_a_2279_);
v___x_2281_ = ((size_t)0ULL);
lean_inc_ref(v_r_2273_);
v___x_2282_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__1(v_r_2273_, v_a_2279_, v_sz_2280_, v___x_2281_, v_b_2277_);
v___x_2283_ = ((size_t)1ULL);
v___x_2284_ = lean_usize_add(v_i_2276_, v___x_2283_);
v_i_2276_ = v___x_2284_;
v_b_2277_ = v___x_2282_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2___boxed(lean_object* v_r_2286_, lean_object* v_as_2287_, lean_object* v_sz_2288_, lean_object* v_i_2289_, lean_object* v_b_2290_){
_start:
{
size_t v_sz_boxed_2291_; size_t v_i_boxed_2292_; lean_object* v_res_2293_; 
v_sz_boxed_2291_ = lean_unbox_usize(v_sz_2288_);
lean_dec(v_sz_2288_);
v_i_boxed_2292_ = lean_unbox_usize(v_i_2289_);
lean_dec(v_i_2289_);
v_res_2293_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2(v_r_2286_, v_as_2287_, v_sz_boxed_2291_, v_i_boxed_2292_, v_b_2290_);
lean_dec_ref(v_as_2287_);
return v_res_2293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_insert(lean_object* v_r_2294_, lean_object* v_idx_2295_){
_start:
{
lean_object* v_toForwardRuleInfo_2296_; lean_object* v_name_2297_; uint8_t v___x_2298_; 
v_toForwardRuleInfo_2296_ = lean_ctor_get(v_r_2294_, 0);
v_name_2297_ = lean_ctor_get(v_r_2294_, 1);
v___x_2298_ = lp_aesop_Aesop_ForwardRuleInfo_isConstant(v_toForwardRuleInfo_2296_);
if (v___x_2298_ == 0)
{
lean_object* v_tree_2299_; lean_object* v_nameToRule_2300_; lean_object* v_constRules_2301_; lean_object* v___x_2303_; uint8_t v_isShared_2304_; uint8_t v_isSharedCheck_2313_; 
lean_inc_ref(v_name_2297_);
v_tree_2299_ = lean_ctor_get(v_idx_2295_, 0);
v_nameToRule_2300_ = lean_ctor_get(v_idx_2295_, 1);
v_constRules_2301_ = lean_ctor_get(v_idx_2295_, 2);
v_isSharedCheck_2313_ = !lean_is_exclusive(v_idx_2295_);
if (v_isSharedCheck_2313_ == 0)
{
v___x_2303_ = v_idx_2295_;
v_isShared_2304_ = v_isSharedCheck_2313_;
goto v_resetjp_2302_;
}
else
{
lean_inc(v_constRules_2301_);
lean_inc(v_nameToRule_2300_);
lean_inc(v_tree_2299_);
lean_dec(v_idx_2295_);
v___x_2303_ = lean_box(0);
v_isShared_2304_ = v_isSharedCheck_2313_;
goto v_resetjp_2302_;
}
v_resetjp_2302_:
{
lean_object* v_slotClusters_2305_; size_t v_sz_2306_; size_t v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2311_; 
v_slotClusters_2305_ = lean_ctor_get(v_toForwardRuleInfo_2296_, 2);
v_sz_2306_ = lean_array_size(v_slotClusters_2305_);
v___x_2307_ = ((size_t)0ULL);
lean_inc_ref(v_r_2294_);
v___x_2308_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardIndex_insert_spec__2(v_r_2294_, v_slotClusters_2305_, v_sz_2306_, v___x_2307_, v_tree_2299_);
v___x_2309_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(v_nameToRule_2300_, v_name_2297_, v_r_2294_);
if (v_isShared_2304_ == 0)
{
lean_ctor_set(v___x_2303_, 1, v___x_2309_);
lean_ctor_set(v___x_2303_, 0, v___x_2308_);
v___x_2311_ = v___x_2303_;
goto v_reusejp_2310_;
}
else
{
lean_object* v_reuseFailAlloc_2312_; 
v_reuseFailAlloc_2312_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2312_, 0, v___x_2308_);
lean_ctor_set(v_reuseFailAlloc_2312_, 1, v___x_2309_);
lean_ctor_set(v_reuseFailAlloc_2312_, 2, v_constRules_2301_);
v___x_2311_ = v_reuseFailAlloc_2312_;
goto v_reusejp_2310_;
}
v_reusejp_2310_:
{
return v___x_2311_;
}
}
}
else
{
lean_object* v_tree_2314_; lean_object* v_nameToRule_2315_; lean_object* v_constRules_2316_; lean_object* v___x_2318_; uint8_t v_isShared_2319_; uint8_t v_isSharedCheck_2326_; 
v_tree_2314_ = lean_ctor_get(v_idx_2295_, 0);
v_nameToRule_2315_ = lean_ctor_get(v_idx_2295_, 1);
v_constRules_2316_ = lean_ctor_get(v_idx_2295_, 2);
v_isSharedCheck_2326_ = !lean_is_exclusive(v_idx_2295_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2318_ = v_idx_2295_;
v_isShared_2319_ = v_isSharedCheck_2326_;
goto v_resetjp_2317_;
}
else
{
lean_inc(v_constRules_2316_);
lean_inc(v_nameToRule_2315_);
lean_inc(v_tree_2314_);
lean_dec(v_idx_2295_);
v___x_2318_ = lean_box(0);
v_isShared_2319_ = v_isSharedCheck_2326_;
goto v_resetjp_2317_;
}
v_resetjp_2317_:
{
lean_object* v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2324_; 
lean_inc_ref(v_r_2294_);
lean_inc_ref(v_name_2297_);
v___x_2320_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__1___redArg(v_nameToRule_2315_, v_name_2297_, v_r_2294_);
v___x_2321_ = lean_box(0);
v___x_2322_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_ForwardIndex_merge_spec__2___redArg(v_constRules_2316_, v_r_2294_, v___x_2321_);
if (v_isShared_2319_ == 0)
{
lean_ctor_set(v___x_2318_, 2, v___x_2322_);
lean_ctor_set(v___x_2318_, 1, v___x_2320_);
v___x_2324_ = v___x_2318_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v_tree_2314_);
lean_ctor_set(v_reuseFailAlloc_2325_, 1, v___x_2320_);
lean_ctor_set(v_reuseFailAlloc_2325_, 2, v___x_2322_);
v___x_2324_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
return v___x_2324_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7(lean_object* v_x_2327_, lean_object* v_keys_2328_, lean_object* v_v_2329_, lean_object* v_k_2330_, lean_object* v_as_2331_, lean_object* v_k_2332_, lean_object* v_x_2333_, lean_object* v_x_2334_, lean_object* v_x_2335_, lean_object* v_x_2336_){
_start:
{
lean_object* v___x_2337_; 
v___x_2337_ = lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___redArg(v_x_2327_, v_keys_2328_, v_v_2329_, v_k_2330_, v_as_2331_, v_k_2332_, v_x_2333_, v_x_2334_);
return v___x_2337_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7___boxed(lean_object* v_x_2338_, lean_object* v_keys_2339_, lean_object* v_v_2340_, lean_object* v_k_2341_, lean_object* v_as_2342_, lean_object* v_k_2343_, lean_object* v_x_2344_, lean_object* v_x_2345_, lean_object* v_x_2346_, lean_object* v_x_2347_){
_start:
{
lean_object* v_res_2348_; 
v_res_2348_ = lp_aesop___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00Aesop_ForwardIndex_insert_spec__0_spec__0_spec__2_spec__7(v_x_2338_, v_keys_2339_, v_v_2340_, v_k_2341_, v_as_2342_, v_k_2343_, v_x_2344_, v_x_2345_, v_x_2346_, v_x_2347_);
lean_dec_ref(v_k_2343_);
lean_dec_ref(v_keys_2339_);
lean_dec(v_x_2338_);
return v_res_2348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_get(lean_object* v_idx_2349_, lean_object* v_e_2350_, lean_object* v_a_2351_, lean_object* v_a_2352_, lean_object* v_a_2353_, lean_object* v_a_2354_){
_start:
{
lean_object* v_tree_2356_; lean_object* v___x_2357_; 
v_tree_2356_ = lean_ctor_get(v_idx_2349_, 0);
lean_inc_ref(v_tree_2356_);
lean_dec_ref(v_idx_2349_);
v___x_2357_ = lp_aesop_Aesop_getUnify___redArg(v_tree_2356_, v_e_2350_, v_a_2351_, v_a_2352_, v_a_2353_, v_a_2354_);
return v___x_2357_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_get___boxed(lean_object* v_idx_2358_, lean_object* v_e_2359_, lean_object* v_a_2360_, lean_object* v_a_2361_, lean_object* v_a_2362_, lean_object* v_a_2363_, lean_object* v_a_2364_){
_start:
{
lean_object* v_res_2365_; 
v_res_2365_ = lp_aesop_Aesop_ForwardIndex_get(v_idx_2358_, v_e_2359_, v_a_2360_, v_a_2361_, v_a_2362_, v_a_2363_);
lean_dec(v_a_2363_);
lean_dec_ref(v_a_2362_);
lean_dec(v_a_2361_);
lean_dec_ref(v_a_2360_);
return v_res_2365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getRuleWithName_x3f(lean_object* v_n_2366_, lean_object* v_idx_2367_){
_start:
{
lean_object* v_nameToRule_2368_; lean_object* v___x_2369_; 
v_nameToRule_2368_ = lean_ctor_get(v_idx_2367_, 1);
v___x_2369_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_ForwardIndex_merge_spec__0___redArg(v_nameToRule_2368_, v_n_2366_);
return v___x_2369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getRuleWithName_x3f___boxed(lean_object* v_n_2370_, lean_object* v_idx_2371_){
_start:
{
lean_object* v_res_2372_; 
v_res_2372_ = lp_aesop_Aesop_ForwardIndex_getRuleWithName_x3f(v_n_2370_, v_idx_2371_);
lean_dec_ref(v_idx_2371_);
lean_dec_ref(v_n_2370_);
return v_res_2372_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0(lean_object* v_d_2375_, lean_object* v_a_2376_, lean_object* v_x_2377_){
_start:
{
lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; 
v___x_2378_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___lam__0___closed__0));
v___x_2379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2379_, 0, v_a_2376_);
lean_ctor_set(v___x_2379_, 1, v___x_2378_);
v___x_2380_ = lean_array_push(v_d_2375_, v___x_2379_);
return v___x_2380_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches(lean_object* v_idx_2384_){
_start:
{
lean_object* v_constRules_2385_; lean_object* v___f_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; 
v_constRules_2385_ = lean_ctor_get(v_idx_2384_, 2);
v___f_2386_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__0));
v___x_2387_ = ((lean_object*)(lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___closed__1));
v___x_2388_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_ForwardIndex_trace_spec__3_spec__7___redArg(v___f_2386_, v_constRules_2385_, v___x_2387_);
return v___x_2388_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardIndex_getConstRuleMatches___boxed(lean_object* v_idx_2389_){
_start:
{
lean_object* v_res_2390_; 
v_res_2390_ = lp_aesop_Aesop_ForwardIndex_getConstRuleMatches(v_idx_2389_);
lean_dec_ref(v_idx_2389_);
return v_res_2390_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Index_Forward(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedForwardIndex_default = _init_lp_aesop_Aesop_instInhabitedForwardIndex_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardIndex_default);
lp_aesop_Aesop_instInhabitedForwardIndex = _init_lp_aesop_Aesop_instInhabitedForwardIndex();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardIndex);
lp_aesop_Aesop_ForwardIndex_instEmptyCollection = _init_lp_aesop_Aesop_ForwardIndex_instEmptyCollection();
lean_mark_persistent(lp_aesop_Aesop_ForwardIndex_instEmptyCollection);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Index_Forward(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Index_Forward(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Index_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Index_Forward(builtin);
}
#ifdef __cplusplus
}
#endif
