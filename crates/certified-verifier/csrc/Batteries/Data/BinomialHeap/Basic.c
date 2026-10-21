// Lean compiler output
// Module: Batteries.Data.BinomialHeap.Basic
// Imports: public import Init public meta import Init public import Batteries.Classes.Order public import Batteries.Control.ForInStep.Basic
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_push___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftl(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_batteries_ForInStep_run___boxed(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "Batteries.BinomialHeap.Imp.HeapNode.nil"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2;
static lean_once_cell_t lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3;
static const lean_string_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Batteries.BinomialHeap.Imp.HeapNode.node"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Batteries.BinomialHeap.Imp.Heap.nil"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Batteries.BinomialHeap.Imp.Heap.cons"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__2_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__3 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__3_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__4_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__5 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__5_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__6 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__0_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__1_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__7 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__7_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__2_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__3_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__4_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__8 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__8_value),((lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_push___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_ForInStep_run___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__0_value;
static const lean_string_object lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(v_x_4_);
lean_dec(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(lean_object* v_00_u03b1_6_, lean_object* v_x_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___redArg(v_x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx___boxed(lean_object* v_00_u03b1_9_, lean_object* v_x_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorIdx(v_00_u03b1_9_, v_x_10_);
lean_dec(v_x_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(lean_object* v_t_12_, lean_object* v_k_13_){
_start:
{
if (lean_obj_tag(v_t_12_) == 0)
{
return v_k_13_;
}
else
{
lean_object* v_a_14_; lean_object* v_child_15_; lean_object* v_sibling_16_; lean_object* v___x_17_; 
v_a_14_ = lean_ctor_get(v_t_12_, 0);
lean_inc(v_a_14_);
v_child_15_ = lean_ctor_get(v_t_12_, 1);
lean_inc(v_child_15_);
v_sibling_16_ = lean_ctor_get(v_t_12_, 2);
lean_inc(v_sibling_16_);
lean_dec_ref_known(v_t_12_, 3);
v___x_17_ = lean_apply_3(v_k_13_, v_a_14_, v_child_15_, v_sibling_16_);
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(lean_object* v_00_u03b1_18_, lean_object* v_motive_19_, lean_object* v_ctorIdx_20_, lean_object* v_t_21_, lean_object* v_h_22_, lean_object* v_k_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(v_t_21_, v_k_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___boxed(lean_object* v_00_u03b1_25_, lean_object* v_motive_26_, lean_object* v_ctorIdx_27_, lean_object* v_t_28_, lean_object* v_h_29_, lean_object* v_k_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim(v_00_u03b1_25_, v_motive_26_, v_ctorIdx_27_, v_t_28_, v_h_29_, v_k_30_);
lean_dec(v_ctorIdx_27_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim___redArg(lean_object* v_t_32_, lean_object* v_nil_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(v_t_32_, v_nil_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_nil_elim(lean_object* v_00_u03b1_35_, lean_object* v_motive_36_, lean_object* v_t_37_, lean_object* v_h_38_, lean_object* v_nil_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(v_t_37_, v_nil_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim___redArg(lean_object* v_t_41_, lean_object* v_node_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(v_t_41_, v_node_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_node_elim(lean_object* v_00_u03b1_44_, lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_node_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_ctorElim___redArg(v_t_46_, v_node_48_);
return v___x_49_;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = lean_unsigned_to_nat(2u);
v___x_54_ = lean_nat_to_int(v___x_53_);
return v___x_54_;
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_unsigned_to_nat(1u);
v___x_56_ = lean_nat_to_int(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(lean_object* v_inst_63_, lean_object* v_x_64_, lean_object* v_prec_65_){
_start:
{
lean_object* v___y_67_; 
if (lean_obj_tag(v_x_64_) == 0)
{
lean_object* v___x_73_; uint8_t v___x_74_; 
lean_dec_ref(v_inst_63_);
v___x_73_ = lean_unsigned_to_nat(1024u);
v___x_74_ = lean_nat_dec_le(v___x_73_, v_prec_65_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; 
v___x_75_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2);
v___y_67_ = v___x_75_;
goto v___jp_66_;
}
else
{
lean_object* v___x_76_; 
v___x_76_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3);
v___y_67_ = v___x_76_;
goto v___jp_66_;
}
}
else
{
lean_object* v_a_77_; lean_object* v_child_78_; lean_object* v_sibling_79_; lean_object* v___x_80_; lean_object* v___y_82_; uint8_t v___x_97_; 
v_a_77_ = lean_ctor_get(v_x_64_, 0);
lean_inc(v_a_77_);
v_child_78_ = lean_ctor_get(v_x_64_, 1);
lean_inc(v_child_78_);
v_sibling_79_ = lean_ctor_get(v_x_64_, 2);
lean_inc(v_sibling_79_);
lean_dec_ref_known(v_x_64_, 3);
v___x_80_ = lean_unsigned_to_nat(1024u);
v___x_97_ = lean_nat_dec_le(v___x_80_, v_prec_65_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; 
v___x_98_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2);
v___y_82_ = v___x_98_;
goto v___jp_81_;
}
else
{
lean_object* v___x_99_; 
v___x_99_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3);
v___y_82_ = v___x_99_;
goto v___jp_81_;
}
v___jp_81_:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_83_ = lean_box(1);
v___x_84_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__6));
lean_inc_ref_n(v_inst_63_, 2);
v___x_85_ = lean_apply_2(v_inst_63_, v_a_77_, v___x_80_);
v___x_86_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_84_);
lean_ctor_set(v___x_86_, 1, v___x_85_);
v___x_87_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_83_);
v___x_88_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(v_inst_63_, v_child_78_, v___x_80_);
v___x_89_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_87_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v___x_83_);
v___x_91_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(v_inst_63_, v_sibling_79_, v___x_80_);
v___x_92_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_90_);
lean_ctor_set(v___x_92_, 1, v___x_91_);
lean_inc(v___y_82_);
v___x_93_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_93_, 0, v___y_82_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
v___x_94_ = 0;
v___x_95_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_95_, 0, v___x_93_);
lean_ctor_set_uint8(v___x_95_, sizeof(void*)*1, v___x_94_);
v___x_96_ = l_Repr_addAppParen(v___x_95_, v_prec_65_);
return v___x_96_;
}
}
v___jp_66_:
{
lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_68_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__1));
lean_inc(v___y_67_);
v___x_69_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_69_, 0, v___y_67_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
v___x_70_ = 0;
v___x_71_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_71_, 0, v___x_69_);
lean_ctor_set_uint8(v___x_71_, sizeof(void*)*1, v___x_70_);
v___x_72_ = l_Repr_addAppParen(v___x_71_, v_prec_65_);
return v___x_72_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___boxed(lean_object* v_inst_100_, lean_object* v_x_101_, lean_object* v_prec_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(v_inst_100_, v_x_101_, v_prec_102_);
lean_dec(v_prec_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(lean_object* v_00_u03b1_104_, lean_object* v_inst_105_, lean_object* v_x_106_, lean_object* v_prec_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(v_inst_105_, v_x_106_, v_prec_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_x_111_, lean_object* v_prec_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr(v_00_u03b1_109_, v_inst_110_, v_x_111_, v_prec_112_);
lean_dec(v_prec_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode___redArg(lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed), 4, 2);
lean_closure_set(v___x_115_, 0, lean_box(0));
lean_closure_set(v___x_115_, 1, v_inst_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___boxed), 4, 2);
lean_closure_set(v___x_118_, 0, lean_box(0));
lean_closure_set(v___x_118_, 1, v_inst_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(lean_object* v_x_119_){
_start:
{
if (lean_obj_tag(v_x_119_) == 0)
{
lean_object* v___x_120_; 
v___x_120_ = lean_unsigned_to_nat(0u);
return v___x_120_;
}
else
{
lean_object* v_child_121_; lean_object* v_sibling_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v_child_121_ = lean_ctor_get(v_x_119_, 1);
v_sibling_122_ = lean_ctor_get(v_x_119_, 2);
v___x_123_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(v_child_121_);
v___x_124_ = lean_unsigned_to_nat(1u);
v___x_125_ = lean_nat_add(v___x_123_, v___x_124_);
lean_dec(v___x_123_);
v___x_126_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(v_sibling_122_);
v___x_127_ = lean_nat_add(v___x_125_, v___x_126_);
lean_dec(v___x_126_);
lean_dec(v___x_125_);
return v___x_127_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg___boxed(lean_object* v_x_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(v_x_128_);
lean_dec(v_x_128_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(lean_object* v_00_u03b1_130_, lean_object* v_x_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(v_x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___boxed(lean_object* v_00_u03b1_133_, lean_object* v_x_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize(v_00_u03b1_133_, v_x_134_);
lean_dec(v_x_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter___redArg(lean_object* v_x_136_, lean_object* v_h__1_137_, lean_object* v_h__2_138_){
_start:
{
if (lean_obj_tag(v_x_136_) == 0)
{
lean_object* v___x_139_; lean_object* v___x_140_; 
lean_dec(v_h__2_138_);
v___x_139_ = lean_box(0);
v___x_140_ = lean_apply_1(v_h__1_137_, v___x_139_);
return v___x_140_;
}
else
{
lean_object* v_a_141_; lean_object* v_child_142_; lean_object* v_sibling_143_; lean_object* v___x_144_; 
lean_dec(v_h__1_137_);
v_a_141_ = lean_ctor_get(v_x_136_, 0);
lean_inc(v_a_141_);
v_child_142_ = lean_ctor_get(v_x_136_, 1);
lean_inc(v_child_142_);
v_sibling_143_ = lean_ctor_get(v_x_136_, 2);
lean_inc(v_sibling_143_);
lean_dec_ref_known(v_x_136_, 3);
v___x_144_ = lean_apply_3(v_h__2_138_, v_a_141_, v_child_142_, v_sibling_143_);
return v___x_144_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_realSize_match__1_splitter(lean_object* v_00_u03b1_145_, lean_object* v_motive_146_, lean_object* v_x_147_, lean_object* v_h__1_148_, lean_object* v_h__2_149_){
_start:
{
if (lean_obj_tag(v_x_147_) == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec(v_h__2_149_);
v___x_150_ = lean_box(0);
v___x_151_ = lean_apply_1(v_h__1_148_, v___x_150_);
return v___x_151_;
}
else
{
lean_object* v_a_152_; lean_object* v_child_153_; lean_object* v_sibling_154_; lean_object* v___x_155_; 
lean_dec(v_h__1_148_);
v_a_152_ = lean_ctor_get(v_x_147_, 0);
lean_inc(v_a_152_);
v_child_153_ = lean_ctor_get(v_x_147_, 1);
lean_inc(v_child_153_);
v_sibling_154_ = lean_ctor_get(v_x_147_, 2);
lean_inc(v_sibling_154_);
lean_dec_ref_known(v_x_147_, 3);
v___x_155_ = lean_apply_3(v_h__2_149_, v_a_152_, v_child_153_, v_sibling_154_);
return v___x_155_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(lean_object* v_a_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = lean_box(0);
v___x_158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_158_, 0, v_a_156_);
lean_ctor_set(v___x_158_, 1, v___x_157_);
lean_ctor_set(v___x_158_, 2, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton(lean_object* v_00_u03b1_159_, lean_object* v_a_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_singleton___redArg(v_a_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(lean_object* v_x_162_){
_start:
{
if (lean_obj_tag(v_x_162_) == 0)
{
lean_object* v___x_163_; 
v___x_163_ = lean_unsigned_to_nat(0u);
return v___x_163_;
}
else
{
lean_object* v_sibling_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v_sibling_164_ = lean_ctor_get(v_x_162_, 2);
v___x_165_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(v_sibling_164_);
v___x_166_ = lean_unsigned_to_nat(1u);
v___x_167_ = lean_nat_add(v___x_165_, v___x_166_);
lean_dec(v___x_165_);
return v___x_167_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg___boxed(lean_object* v_x_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(v_x_168_);
lean_dec(v_x_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(lean_object* v_00_u03b1_170_, lean_object* v_x_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(v_x_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___boxed(lean_object* v_00_u03b1_173_, lean_object* v_x_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank(v_00_u03b1_173_, v_x_174_);
lean_dec(v_x_174_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
if (lean_obj_tag(v_a_176_) == 0)
{
return v_a_177_;
}
else
{
lean_object* v_sibling_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v_sibling_178_ = lean_ctor_get(v_a_176_, 2);
v___x_179_ = lean_unsigned_to_nat(1u);
v___x_180_ = lean_nat_add(v_a_177_, v___x_179_);
lean_dec(v_a_177_);
v_a_176_ = v_sibling_178_;
v_a_177_ = v___x_180_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg___boxed(lean_object* v_a_182_, lean_object* v_a_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(v_a_182_, v_a_183_);
lean_dec(v_a_182_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(lean_object* v_00_u03b1_185_, lean_object* v_a_186_, lean_object* v_a_187_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(v_a_186_, v_a_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___boxed(lean_object* v_00_u03b1_189_, lean_object* v_a_190_, lean_object* v_a_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go(v_00_u03b1_189_, v_a_190_, v_a_191_);
lean_dec(v_a_190_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(lean_object* v_s_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_194_ = lean_unsigned_to_nat(0u);
v___x_195_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(v_s_193_, v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg___boxed(lean_object* v_s_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___redArg(v_s_196_);
lean_dec(v_s_196_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(lean_object* v_00_u03b1_198_, lean_object* v_s_199_){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = lean_unsigned_to_nat(0u);
v___x_201_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR_go___redArg(v_s_199_, v___x_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR___boxed(lean_object* v_00_u03b1_202_, lean_object* v_s_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rankTR(v_00_u03b1_202_, v_s_203_);
lean_dec(v_s_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter___redArg(lean_object* v_x_205_, lean_object* v_x_206_, lean_object* v_h__1_207_, lean_object* v_h__2_208_){
_start:
{
if (lean_obj_tag(v_x_205_) == 0)
{
lean_object* v___x_209_; 
lean_dec(v_h__2_208_);
v___x_209_ = lean_apply_1(v_h__1_207_, v_x_206_);
return v___x_209_;
}
else
{
lean_object* v_a_210_; lean_object* v_child_211_; lean_object* v_sibling_212_; lean_object* v___x_213_; 
lean_dec(v_h__1_207_);
v_a_210_ = lean_ctor_get(v_x_205_, 0);
lean_inc(v_a_210_);
v_child_211_ = lean_ctor_get(v_x_205_, 1);
lean_inc(v_child_211_);
v_sibling_212_ = lean_ctor_get(v_x_205_, 2);
lean_inc(v_sibling_212_);
lean_dec_ref_known(v_x_205_, 3);
v___x_213_ = lean_apply_4(v_h__2_208_, v_a_210_, v_child_211_, v_sibling_212_, v_x_206_);
return v___x_213_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_rankTR_go_match__1_splitter(lean_object* v_00_u03b1_214_, lean_object* v_motive_215_, lean_object* v_x_216_, lean_object* v_x_217_, lean_object* v_h__1_218_, lean_object* v_h__2_219_){
_start:
{
if (lean_obj_tag(v_x_216_) == 0)
{
lean_object* v___x_220_; 
lean_dec(v_h__2_219_);
v___x_220_ = lean_apply_1(v_h__1_218_, v_x_217_);
return v___x_220_;
}
else
{
lean_object* v_a_221_; lean_object* v_child_222_; lean_object* v_sibling_223_; lean_object* v___x_224_; 
lean_dec(v_h__1_218_);
v_a_221_ = lean_ctor_get(v_x_216_, 0);
lean_inc(v_a_221_);
v_child_222_ = lean_ctor_get(v_x_216_, 1);
lean_inc(v_child_222_);
v_sibling_223_ = lean_ctor_get(v_x_216_, 2);
lean_inc(v_sibling_223_);
lean_dec_ref_known(v_x_216_, 3);
v___x_224_ = lean_apply_4(v_h__2_219_, v_a_221_, v_child_222_, v_sibling_223_, v_x_217_);
return v___x_224_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(lean_object* v_x_225_){
_start:
{
if (lean_obj_tag(v_x_225_) == 0)
{
lean_object* v___x_226_; 
v___x_226_ = lean_unsigned_to_nat(0u);
return v___x_226_;
}
else
{
lean_object* v___x_227_; 
v___x_227_ = lean_unsigned_to_nat(1u);
return v___x_227_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg___boxed(lean_object* v_x_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(v_x_228_);
lean_dec(v_x_228_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(lean_object* v_00_u03b1_230_, lean_object* v_x_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___redArg(v_x_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx___boxed(lean_object* v_00_u03b1_233_, lean_object* v_x_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorIdx(v_00_u03b1_233_, v_x_234_);
lean_dec(v_x_234_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(lean_object* v_t_236_, lean_object* v_k_237_){
_start:
{
if (lean_obj_tag(v_t_236_) == 0)
{
return v_k_237_;
}
else
{
lean_object* v_rank_238_; lean_object* v_val_239_; lean_object* v_node_240_; lean_object* v_next_241_; lean_object* v___x_242_; 
v_rank_238_ = lean_ctor_get(v_t_236_, 0);
lean_inc(v_rank_238_);
v_val_239_ = lean_ctor_get(v_t_236_, 1);
lean_inc(v_val_239_);
v_node_240_ = lean_ctor_get(v_t_236_, 2);
lean_inc(v_node_240_);
v_next_241_ = lean_ctor_get(v_t_236_, 3);
lean_inc(v_next_241_);
lean_dec_ref_known(v_t_236_, 4);
v___x_242_ = lean_apply_4(v_k_237_, v_rank_238_, v_val_239_, v_node_240_, v_next_241_);
return v___x_242_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(lean_object* v_00_u03b1_243_, lean_object* v_motive_244_, lean_object* v_ctorIdx_245_, lean_object* v_t_246_, lean_object* v_h_247_, lean_object* v_k_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(v_t_246_, v_k_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___boxed(lean_object* v_00_u03b1_250_, lean_object* v_motive_251_, lean_object* v_ctorIdx_252_, lean_object* v_t_253_, lean_object* v_h_254_, lean_object* v_k_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim(v_00_u03b1_250_, v_motive_251_, v_ctorIdx_252_, v_t_253_, v_h_254_, v_k_255_);
lean_dec(v_ctorIdx_252_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim___redArg(lean_object* v_t_257_, lean_object* v_nil_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(v_t_257_, v_nil_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_nil_elim(lean_object* v_00_u03b1_260_, lean_object* v_motive_261_, lean_object* v_t_262_, lean_object* v_h_263_, lean_object* v_nil_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(v_t_262_, v_nil_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim___redArg(lean_object* v_t_266_, lean_object* v_cons_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(v_t_266_, v_cons_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_cons_elim(lean_object* v_00_u03b1_269_, lean_object* v_motive_270_, lean_object* v_t_271_, lean_object* v_h_272_, lean_object* v_cons_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_ctorElim___redArg(v_t_271_, v_cons_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(lean_object* v_inst_284_, lean_object* v_x_285_, lean_object* v_prec_286_){
_start:
{
lean_object* v___y_288_; 
if (lean_obj_tag(v_x_285_) == 0)
{
lean_object* v___x_294_; uint8_t v___x_295_; 
lean_dec_ref(v_inst_284_);
v___x_294_ = lean_unsigned_to_nat(1024u);
v___x_295_ = lean_nat_dec_le(v___x_294_, v_prec_286_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; 
v___x_296_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2);
v___y_288_ = v___x_296_;
goto v___jp_287_;
}
else
{
lean_object* v___x_297_; 
v___x_297_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3);
v___y_288_ = v___x_297_;
goto v___jp_287_;
}
}
else
{
lean_object* v_rank_298_; lean_object* v_val_299_; lean_object* v_node_300_; lean_object* v_next_301_; lean_object* v___x_302_; lean_object* v___y_304_; uint8_t v___x_323_; 
v_rank_298_ = lean_ctor_get(v_x_285_, 0);
lean_inc(v_rank_298_);
v_val_299_ = lean_ctor_get(v_x_285_, 1);
lean_inc(v_val_299_);
v_node_300_ = lean_ctor_get(v_x_285_, 2);
lean_inc(v_node_300_);
v_next_301_ = lean_ctor_get(v_x_285_, 3);
lean_inc(v_next_301_);
lean_dec_ref_known(v_x_285_, 4);
v___x_302_ = lean_unsigned_to_nat(1024u);
v___x_323_ = lean_nat_dec_le(v___x_302_, v_prec_286_);
if (v___x_323_ == 0)
{
lean_object* v___x_324_; 
v___x_324_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__2);
v___y_304_ = v___x_324_;
goto v___jp_303_;
}
else
{
lean_object* v___x_325_; 
v___x_325_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg___closed__3);
v___y_304_ = v___x_325_;
goto v___jp_303_;
}
v___jp_303_:
{
lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; uint8_t v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_305_ = lean_box(1);
v___x_306_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__4));
v___x_307_ = l_Nat_reprFast(v_rank_298_);
v___x_308_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_308_, 0, v___x_307_);
v___x_309_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_306_);
lean_ctor_set(v___x_309_, 1, v___x_308_);
v___x_310_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v___x_305_);
lean_inc_ref_n(v_inst_284_, 2);
v___x_311_ = lean_apply_2(v_inst_284_, v_val_299_, v___x_302_);
v___x_312_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_310_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
v___x_313_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
lean_ctor_set(v___x_313_, 1, v___x_305_);
v___x_314_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeapNode_repr___redArg(v_inst_284_, v_node_300_, v___x_302_);
v___x_315_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_313_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
v___x_316_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v___x_305_);
v___x_317_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(v_inst_284_, v_next_301_, v___x_302_);
v___x_318_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_316_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
lean_inc(v___y_304_);
v___x_319_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_319_, 0, v___y_304_);
lean_ctor_set(v___x_319_, 1, v___x_318_);
v___x_320_ = 0;
v___x_321_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_321_, 0, v___x_319_);
lean_ctor_set_uint8(v___x_321_, sizeof(void*)*1, v___x_320_);
v___x_322_ = l_Repr_addAppParen(v___x_321_, v_prec_286_);
return v___x_322_;
}
}
v___jp_287_:
{
lean_object* v___x_289_; lean_object* v___x_290_; uint8_t v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_289_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___closed__1));
lean_inc(v___y_288_);
v___x_290_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_290_, 0, v___y_288_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
v___x_291_ = 0;
v___x_292_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_292_, 0, v___x_290_);
lean_ctor_set_uint8(v___x_292_, sizeof(void*)*1, v___x_291_);
v___x_293_ = l_Repr_addAppParen(v___x_292_, v_prec_286_);
return v___x_293_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg___boxed(lean_object* v_inst_326_, lean_object* v_x_327_, lean_object* v_prec_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(v_inst_326_, v_x_327_, v_prec_328_);
lean_dec(v_prec_328_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(lean_object* v_00_u03b1_330_, lean_object* v_inst_331_, lean_object* v_x_332_, lean_object* v_prec_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___redArg(v_inst_331_, v_x_332_, v_prec_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed(lean_object* v_00_u03b1_335_, lean_object* v_inst_336_, lean_object* v_x_337_, lean_object* v_prec_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr(v_00_u03b1_335_, v_inst_336_, v_x_337_, v_prec_338_);
lean_dec(v_prec_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap___redArg(lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed), 4, 2);
lean_closure_set(v___x_341_, 0, lean_box(0));
lean_closure_set(v___x_341_, 1, v_inst_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap(lean_object* v_00_u03b1_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_instReprHeap_repr___boxed), 4, 2);
lean_closure_set(v___x_344_, 0, lean_box(0));
lean_closure_set(v___x_344_, 1, v_inst_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(lean_object* v_x_345_){
_start:
{
if (lean_obj_tag(v_x_345_) == 0)
{
lean_object* v___x_346_; 
v___x_346_ = lean_unsigned_to_nat(0u);
return v___x_346_;
}
else
{
lean_object* v_node_347_; lean_object* v_next_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v_node_347_ = lean_ctor_get(v_x_345_, 2);
v_next_348_ = lean_ctor_get(v_x_345_, 3);
v___x_349_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_realSize___redArg(v_node_347_);
v___x_350_ = lean_unsigned_to_nat(1u);
v___x_351_ = lean_nat_add(v___x_349_, v___x_350_);
lean_dec(v___x_349_);
v___x_352_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(v_next_348_);
v___x_353_ = lean_nat_add(v___x_351_, v___x_352_);
lean_dec(v___x_352_);
lean_dec(v___x_351_);
return v___x_353_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg___boxed(lean_object* v_x_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(v_x_354_);
lean_dec(v_x_354_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(lean_object* v_00_u03b1_356_, lean_object* v_x_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___redArg(v_x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize___boxed(lean_object* v_00_u03b1_359_, lean_object* v_x_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_realSize(v_00_u03b1_359_, v_x_360_);
lean_dec(v_x_360_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter___redArg(lean_object* v_x_362_, lean_object* v_h__1_363_, lean_object* v_h__2_364_){
_start:
{
if (lean_obj_tag(v_x_362_) == 0)
{
lean_object* v___x_365_; lean_object* v___x_366_; 
lean_dec(v_h__2_364_);
v___x_365_ = lean_box(0);
v___x_366_ = lean_apply_1(v_h__1_363_, v___x_365_);
return v___x_366_;
}
else
{
lean_object* v_rank_367_; lean_object* v_val_368_; lean_object* v_node_369_; lean_object* v_next_370_; lean_object* v___x_371_; 
lean_dec(v_h__1_363_);
v_rank_367_ = lean_ctor_get(v_x_362_, 0);
lean_inc(v_rank_367_);
v_val_368_ = lean_ctor_get(v_x_362_, 1);
lean_inc(v_val_368_);
v_node_369_ = lean_ctor_get(v_x_362_, 2);
lean_inc(v_node_369_);
v_next_370_ = lean_ctor_get(v_x_362_, 3);
lean_inc(v_next_370_);
lean_dec_ref_known(v_x_362_, 4);
v___x_371_ = lean_apply_4(v_h__2_364_, v_rank_367_, v_val_368_, v_node_369_, v_next_370_);
return v___x_371_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_realSize_match__1_splitter(lean_object* v_00_u03b1_372_, lean_object* v_motive_373_, lean_object* v_x_374_, lean_object* v_h__1_375_, lean_object* v_h__2_376_){
_start:
{
if (lean_obj_tag(v_x_374_) == 0)
{
lean_object* v___x_377_; lean_object* v___x_378_; 
lean_dec(v_h__2_376_);
v___x_377_ = lean_box(0);
v___x_378_ = lean_apply_1(v_h__1_375_, v___x_377_);
return v___x_378_;
}
else
{
lean_object* v_rank_379_; lean_object* v_val_380_; lean_object* v_node_381_; lean_object* v_next_382_; lean_object* v___x_383_; 
lean_dec(v_h__1_375_);
v_rank_379_ = lean_ctor_get(v_x_374_, 0);
lean_inc(v_rank_379_);
v_val_380_ = lean_ctor_get(v_x_374_, 1);
lean_inc(v_val_380_);
v_node_381_ = lean_ctor_get(v_x_374_, 2);
lean_inc(v_node_381_);
v_next_382_ = lean_ctor_get(v_x_374_, 3);
lean_inc(v_next_382_);
lean_dec_ref_known(v_x_374_, 4);
v___x_383_ = lean_apply_4(v_h__2_376_, v_rank_379_, v_val_380_, v_node_381_, v_next_382_);
return v___x_383_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(lean_object* v_x_384_){
_start:
{
if (lean_obj_tag(v_x_384_) == 0)
{
lean_object* v___x_385_; 
v___x_385_ = lean_unsigned_to_nat(0u);
return v___x_385_;
}
else
{
lean_object* v_rank_386_; lean_object* v_next_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v_rank_386_ = lean_ctor_get(v_x_384_, 0);
v_next_387_ = lean_ctor_get(v_x_384_, 3);
v___x_388_ = lean_unsigned_to_nat(1u);
v___x_389_ = lean_nat_shiftl(v___x_388_, v_rank_386_);
v___x_390_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(v_next_387_);
v___x_391_ = lean_nat_add(v___x_389_, v___x_390_);
lean_dec(v___x_390_);
lean_dec(v___x_389_);
return v___x_391_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg___boxed(lean_object* v_x_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(v_x_392_);
lean_dec(v_x_392_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(lean_object* v_00_u03b1_394_, lean_object* v_x_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(v_x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___boxed(lean_object* v_00_u03b1_397_, lean_object* v_x_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size(v_00_u03b1_397_, v_x_398_);
lean_dec(v_x_398_);
return v_res_399_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(lean_object* v_x_400_){
_start:
{
if (lean_obj_tag(v_x_400_) == 0)
{
uint8_t v___x_401_; 
v___x_401_ = 1;
return v___x_401_;
}
else
{
uint8_t v___x_402_; 
v___x_402_ = 0;
return v___x_402_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg___boxed(lean_object* v_x_403_){
_start:
{
uint8_t v_res_404_; lean_object* v_r_405_; 
v_res_404_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___redArg(v_x_403_);
lean_dec(v_x_403_);
v_r_405_ = lean_box(v_res_404_);
return v_r_405_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(lean_object* v_00_u03b1_406_, lean_object* v_x_407_){
_start:
{
if (lean_obj_tag(v_x_407_) == 0)
{
uint8_t v___x_408_; 
v___x_408_ = 1;
return v___x_408_;
}
else
{
uint8_t v___x_409_; 
v___x_409_ = 0;
return v___x_409_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty___boxed(lean_object* v_00_u03b1_410_, lean_object* v_x_411_){
_start:
{
uint8_t v_res_412_; lean_object* v_r_413_; 
v_res_412_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_isEmpty(v_00_u03b1_410_, v_x_411_);
lean_dec(v_x_411_);
v_r_413_ = lean_box(v_res_412_);
return v_r_413_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton___redArg(lean_object* v_a_414_){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_415_ = lean_unsigned_to_nat(0u);
v___x_416_ = lean_box(0);
v___x_417_ = lean_box(0);
v___x_418_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_418_, 0, v___x_415_);
lean_ctor_set(v___x_418_, 1, v_a_414_);
lean_ctor_set(v___x_418_, 2, v___x_416_);
lean_ctor_set(v___x_418_, 3, v___x_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_singleton(lean_object* v_00_u03b1_419_, lean_object* v_a_420_){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_421_ = lean_unsigned_to_nat(0u);
v___x_422_ = lean_box(0);
v___x_423_ = lean_box(0);
v___x_424_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_424_, 0, v___x_421_);
lean_ctor_set(v___x_424_, 1, v_a_420_);
lean_ctor_set(v___x_424_, 2, v___x_422_);
lean_ctor_set(v___x_424_, 3, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___redArg(lean_object* v_n_425_, lean_object* v_rank_426_){
_start:
{
uint8_t v___x_427_; 
v___x_427_ = lean_nat_dec_lt(v_n_425_, v_rank_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___redArg___boxed(lean_object* v_n_428_, lean_object* v_rank_429_){
_start:
{
uint8_t v_res_430_; lean_object* v_r_431_; 
v_res_430_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___redArg(v_n_428_, v_rank_429_);
lean_dec(v_rank_429_);
lean_dec(v_n_428_);
v_r_431_ = lean_box(v_res_430_);
return v_r_431_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2(lean_object* v_00_u03b1_432_, lean_object* v_n_433_, lean_object* v_rank_434_, lean_object* v_val_435_, lean_object* v_node_436_, lean_object* v_next_437_){
_start:
{
uint8_t v___x_438_; 
v___x_438_ = lean_nat_dec_lt(v_n_433_, v_rank_434_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2___boxed(lean_object* v_00_u03b1_439_, lean_object* v_n_440_, lean_object* v_rank_441_, lean_object* v_val_442_, lean_object* v_node_443_, lean_object* v_next_444_){
_start:
{
uint8_t v_res_445_; lean_object* v_r_446_; 
v_res_445_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___aux__2(v_00_u03b1_439_, v_n_440_, v_rank_441_, v_val_442_, v_node_443_, v_next_444_);
lean_dec(v_next_444_);
lean_dec(v_node_443_);
lean_dec(v_val_442_);
lean_dec(v_rank_441_);
lean_dec(v_n_440_);
v_r_446_ = lean_box(v_res_445_);
return v_r_446_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object* v_s_447_, lean_object* v_n_448_){
_start:
{
if (lean_obj_tag(v_s_447_) == 0)
{
uint8_t v___x_449_; 
v___x_449_ = 1;
return v___x_449_;
}
else
{
lean_object* v_rank_450_; uint8_t v___x_451_; 
v_rank_450_ = lean_ctor_get(v_s_447_, 0);
v___x_451_ = lean_nat_dec_lt(v_n_448_, v_rank_450_);
return v___x_451_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg___boxed(lean_object* v_s_452_, lean_object* v_n_453_){
_start:
{
uint8_t v_res_454_; lean_object* v_r_455_; 
v_res_454_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_s_452_, v_n_453_);
lean_dec(v_n_453_);
lean_dec(v_s_452_);
v_r_455_ = lean_box(v_res_454_);
return v_r_455_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(lean_object* v_00_u03b1_456_, lean_object* v_s_457_, lean_object* v_n_458_){
_start:
{
uint8_t v___x_459_; 
v___x_459_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_s_457_, v_n_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___boxed(lean_object* v_00_u03b1_460_, lean_object* v_s_461_, lean_object* v_n_462_){
_start:
{
uint8_t v_res_463_; lean_object* v_r_464_; 
v_res_463_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT(v_00_u03b1_460_, v_s_461_, v_n_462_);
lean_dec(v_n_462_);
lean_dec(v_s_461_);
v_r_464_ = lean_box(v_res_463_);
return v_r_464_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(lean_object* v_x_465_){
_start:
{
if (lean_obj_tag(v_x_465_) == 0)
{
lean_object* v___x_466_; 
v___x_466_ = lean_unsigned_to_nat(0u);
return v___x_466_;
}
else
{
lean_object* v_next_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v_next_467_ = lean_ctor_get(v_x_465_, 3);
v___x_468_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(v_next_467_);
v___x_469_ = lean_unsigned_to_nat(1u);
v___x_470_ = lean_nat_add(v___x_468_, v___x_469_);
lean_dec(v___x_468_);
return v___x_470_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg___boxed(lean_object* v_x_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(v_x_471_);
lean_dec(v_x_471_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(lean_object* v_00_u03b1_473_, lean_object* v_x_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___redArg(v_x_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_length___boxed(lean_object* v_00_u03b1_476_, lean_object* v_x_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_length(v_00_u03b1_476_, v_x_477_);
lean_dec(v_x_477_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine___redArg(lean_object* v_le_479_, lean_object* v_a_u2081_480_, lean_object* v_a_u2082_481_, lean_object* v_n_u2081_482_, lean_object* v_n_u2082_483_){
_start:
{
lean_object* v___x_484_; uint8_t v___x_485_; 
lean_inc(v_a_u2082_481_);
lean_inc(v_a_u2081_480_);
v___x_484_ = lean_apply_2(v_le_479_, v_a_u2081_480_, v_a_u2082_481_);
v___x_485_ = lean_unbox(v___x_484_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_486_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_486_, 0, v_a_u2081_480_);
lean_ctor_set(v___x_486_, 1, v_n_u2081_482_);
lean_ctor_set(v___x_486_, 2, v_n_u2082_483_);
v___x_487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_487_, 0, v_a_u2082_481_);
lean_ctor_set(v___x_487_, 1, v___x_486_);
return v___x_487_;
}
else
{
lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_488_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_488_, 0, v_a_u2082_481_);
lean_ctor_set(v___x_488_, 1, v_n_u2082_483_);
lean_ctor_set(v___x_488_, 2, v_n_u2081_482_);
v___x_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_489_, 0, v_a_u2081_480_);
lean_ctor_set(v___x_489_, 1, v___x_488_);
return v___x_489_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_combine(lean_object* v_00_u03b1_490_, lean_object* v_le_491_, lean_object* v_a_u2081_492_, lean_object* v_a_u2082_493_, lean_object* v_n_u2081_494_, lean_object* v_n_u2082_495_){
_start:
{
lean_object* v___x_496_; uint8_t v___x_497_; 
lean_inc(v_a_u2082_493_);
lean_inc(v_a_u2081_492_);
v___x_496_ = lean_apply_2(v_le_491_, v_a_u2081_492_, v_a_u2082_493_);
v___x_497_ = lean_unbox(v___x_496_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_498_, 0, v_a_u2081_492_);
lean_ctor_set(v___x_498_, 1, v_n_u2081_494_);
lean_ctor_set(v___x_498_, 2, v_n_u2082_495_);
v___x_499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_499_, 0, v_a_u2082_493_);
lean_ctor_set(v___x_499_, 1, v___x_498_);
return v___x_499_;
}
else
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_500_, 0, v_a_u2082_493_);
lean_ctor_set(v___x_500_, 1, v_n_u2082_495_);
lean_ctor_set(v___x_500_, 2, v_n_u2081_494_);
v___x_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_501_, 0, v_a_u2081_492_);
lean_ctor_set(v___x_501_, 1, v___x_500_);
return v___x_501_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(lean_object* v_le_502_, lean_object* v_x_503_, lean_object* v_x_504_){
_start:
{
if (lean_obj_tag(v_x_503_) == 0)
{
lean_dec_ref(v_le_502_);
return v_x_504_;
}
else
{
if (lean_obj_tag(v_x_504_) == 0)
{
lean_dec_ref(v_le_502_);
return v_x_503_;
}
else
{
lean_object* v_rank_505_; lean_object* v_val_506_; lean_object* v_node_507_; lean_object* v_next_508_; lean_object* v_rank_509_; lean_object* v_val_510_; lean_object* v_node_511_; lean_object* v_next_512_; lean_object* v_fst_514_; lean_object* v_snd_515_; uint8_t v___x_529_; 
v_rank_505_ = lean_ctor_get(v_x_503_, 0);
v_val_506_ = lean_ctor_get(v_x_503_, 1);
v_node_507_ = lean_ctor_get(v_x_503_, 2);
v_next_508_ = lean_ctor_get(v_x_503_, 3);
v_rank_509_ = lean_ctor_get(v_x_504_, 0);
v_val_510_ = lean_ctor_get(v_x_504_, 1);
v_node_511_ = lean_ctor_get(v_x_504_, 2);
v_next_512_ = lean_ctor_get(v_x_504_, 3);
v___x_529_ = lean_nat_dec_lt(v_rank_505_, v_rank_509_);
if (v___x_529_ == 0)
{
lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_542_; 
lean_inc(v_next_512_);
lean_inc(v_node_511_);
lean_inc(v_val_510_);
lean_inc(v_rank_509_);
v_isSharedCheck_542_ = !lean_is_exclusive(v_x_504_);
if (v_isSharedCheck_542_ == 0)
{
lean_object* v_unused_543_; lean_object* v_unused_544_; lean_object* v_unused_545_; lean_object* v_unused_546_; 
v_unused_543_ = lean_ctor_get(v_x_504_, 3);
lean_dec(v_unused_543_);
v_unused_544_ = lean_ctor_get(v_x_504_, 2);
lean_dec(v_unused_544_);
v_unused_545_ = lean_ctor_get(v_x_504_, 1);
lean_dec(v_unused_545_);
v_unused_546_ = lean_ctor_get(v_x_504_, 0);
lean_dec(v_unused_546_);
v___x_531_ = v_x_504_;
v_isShared_532_ = v_isSharedCheck_542_;
goto v_resetjp_530_;
}
else
{
lean_dec(v_x_504_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_542_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
uint8_t v___x_533_; 
v___x_533_ = lean_nat_dec_lt(v_rank_509_, v_rank_505_);
if (v___x_533_ == 0)
{
lean_object* v___x_534_; uint8_t v___x_535_; 
lean_inc(v_next_508_);
lean_inc(v_node_507_);
lean_inc_n(v_val_506_, 2);
lean_inc(v_rank_505_);
lean_del_object(v___x_531_);
lean_dec(v_rank_509_);
lean_dec_ref_known(v_x_503_, 4);
lean_inc_ref(v_le_502_);
lean_inc(v_val_510_);
v___x_534_ = lean_apply_2(v_le_502_, v_val_506_, v_val_510_);
v___x_535_ = lean_unbox(v___x_534_);
if (v___x_535_ == 0)
{
lean_object* v___x_536_; 
v___x_536_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_536_, 0, v_val_506_);
lean_ctor_set(v___x_536_, 1, v_node_507_);
lean_ctor_set(v___x_536_, 2, v_node_511_);
v_fst_514_ = v_val_510_;
v_snd_515_ = v___x_536_;
goto v___jp_513_;
}
else
{
lean_object* v___x_537_; 
v___x_537_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_537_, 0, v_val_510_);
lean_ctor_set(v___x_537_, 1, v_node_511_);
lean_ctor_set(v___x_537_, 2, v_node_507_);
v_fst_514_ = v_val_506_;
v_snd_515_ = v___x_537_;
goto v___jp_513_;
}
}
else
{
lean_object* v___x_538_; lean_object* v___x_540_; 
v___x_538_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_502_, v_x_503_, v_next_512_);
if (v_isShared_532_ == 0)
{
lean_ctor_set(v___x_531_, 3, v___x_538_);
v___x_540_ = v___x_531_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_rank_509_);
lean_ctor_set(v_reuseFailAlloc_541_, 1, v_val_510_);
lean_ctor_set(v_reuseFailAlloc_541_, 2, v_node_511_);
lean_ctor_set(v_reuseFailAlloc_541_, 3, v___x_538_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
else
{
lean_object* v___x_548_; uint8_t v_isShared_549_; uint8_t v_isSharedCheck_554_; 
lean_inc(v_next_508_);
lean_inc(v_node_507_);
lean_inc(v_val_506_);
lean_inc(v_rank_505_);
v_isSharedCheck_554_ = !lean_is_exclusive(v_x_503_);
if (v_isSharedCheck_554_ == 0)
{
lean_object* v_unused_555_; lean_object* v_unused_556_; lean_object* v_unused_557_; lean_object* v_unused_558_; 
v_unused_555_ = lean_ctor_get(v_x_503_, 3);
lean_dec(v_unused_555_);
v_unused_556_ = lean_ctor_get(v_x_503_, 2);
lean_dec(v_unused_556_);
v_unused_557_ = lean_ctor_get(v_x_503_, 1);
lean_dec(v_unused_557_);
v_unused_558_ = lean_ctor_get(v_x_503_, 0);
lean_dec(v_unused_558_);
v___x_548_ = v_x_503_;
v_isShared_549_ = v_isSharedCheck_554_;
goto v_resetjp_547_;
}
else
{
lean_dec(v_x_503_);
v___x_548_ = lean_box(0);
v_isShared_549_ = v_isSharedCheck_554_;
goto v_resetjp_547_;
}
v_resetjp_547_:
{
lean_object* v___x_550_; lean_object* v___x_552_; 
v___x_550_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_502_, v_next_508_, v_x_504_);
if (v_isShared_549_ == 0)
{
lean_ctor_set(v___x_548_, 3, v___x_550_);
v___x_552_ = v___x_548_;
goto v_reusejp_551_;
}
else
{
lean_object* v_reuseFailAlloc_553_; 
v_reuseFailAlloc_553_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_553_, 0, v_rank_505_);
lean_ctor_set(v_reuseFailAlloc_553_, 1, v_val_506_);
lean_ctor_set(v_reuseFailAlloc_553_, 2, v_node_507_);
lean_ctor_set(v_reuseFailAlloc_553_, 3, v___x_550_);
v___x_552_ = v_reuseFailAlloc_553_;
goto v_reusejp_551_;
}
v_reusejp_551_:
{
return v___x_552_;
}
}
}
v___jp_513_:
{
lean_object* v___x_516_; lean_object* v_r_517_; uint8_t v___x_518_; 
v___x_516_ = lean_unsigned_to_nat(1u);
v_r_517_ = lean_nat_add(v_rank_505_, v___x_516_);
lean_dec(v_rank_505_);
v___x_518_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_508_, v_r_517_);
if (v___x_518_ == 0)
{
uint8_t v___x_519_; 
v___x_519_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_512_, v_r_517_);
if (v___x_519_ == 0)
{
lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_520_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_502_, v_next_508_, v_next_512_);
v___x_521_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_521_, 0, v_r_517_);
lean_ctor_set(v___x_521_, 1, v_fst_514_);
lean_ctor_set(v___x_521_, 2, v_snd_515_);
lean_ctor_set(v___x_521_, 3, v___x_520_);
return v___x_521_;
}
else
{
lean_object* v___x_522_; 
v___x_522_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_522_, 0, v_r_517_);
lean_ctor_set(v___x_522_, 1, v_fst_514_);
lean_ctor_set(v___x_522_, 2, v_snd_515_);
lean_ctor_set(v___x_522_, 3, v_next_512_);
v_x_503_ = v_next_508_;
v_x_504_ = v___x_522_;
goto _start;
}
}
else
{
uint8_t v___x_524_; 
v___x_524_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_512_, v_r_517_);
if (v___x_524_ == 0)
{
lean_object* v___x_525_; 
v___x_525_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_525_, 0, v_r_517_);
lean_ctor_set(v___x_525_, 1, v_fst_514_);
lean_ctor_set(v___x_525_, 2, v_snd_515_);
lean_ctor_set(v___x_525_, 3, v_next_508_);
v_x_503_ = v___x_525_;
v_x_504_ = v_next_512_;
goto _start;
}
else
{
lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_527_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_502_, v_next_508_, v_next_512_);
v___x_528_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_528_, 0, v_r_517_);
lean_ctor_set(v___x_528_, 1, v_fst_514_);
lean_ctor_set(v___x_528_, 2, v_snd_515_);
lean_ctor_set(v___x_528_, 3, v___x_527_);
return v___x_528_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge(lean_object* v_00_u03b1_559_, lean_object* v_le_560_, lean_object* v_x_561_, lean_object* v_x_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_560_, v_x_561_, v_x_562_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter___redArg(lean_object* v_x_564_, lean_object* v_x_565_, lean_object* v_h__1_566_, lean_object* v_h__2_567_, lean_object* v_h__3_568_){
_start:
{
if (lean_obj_tag(v_x_564_) == 0)
{
lean_object* v___x_569_; 
lean_dec(v_h__3_568_);
lean_dec(v_h__2_567_);
v___x_569_ = lean_apply_1(v_h__1_566_, v_x_565_);
return v___x_569_;
}
else
{
lean_dec(v_h__1_566_);
if (lean_obj_tag(v_x_565_) == 0)
{
lean_object* v___x_570_; 
lean_dec(v_h__3_568_);
v___x_570_ = lean_apply_2(v_h__2_567_, v_x_564_, lean_box(0));
return v___x_570_;
}
else
{
lean_object* v_rank_571_; lean_object* v_val_572_; lean_object* v_node_573_; lean_object* v_next_574_; lean_object* v_rank_575_; lean_object* v_val_576_; lean_object* v_node_577_; lean_object* v_next_578_; lean_object* v___x_579_; 
lean_dec(v_h__2_567_);
v_rank_571_ = lean_ctor_get(v_x_564_, 0);
lean_inc(v_rank_571_);
v_val_572_ = lean_ctor_get(v_x_564_, 1);
lean_inc(v_val_572_);
v_node_573_ = lean_ctor_get(v_x_564_, 2);
lean_inc(v_node_573_);
v_next_574_ = lean_ctor_get(v_x_564_, 3);
lean_inc(v_next_574_);
lean_dec_ref_known(v_x_564_, 4);
v_rank_575_ = lean_ctor_get(v_x_565_, 0);
lean_inc(v_rank_575_);
v_val_576_ = lean_ctor_get(v_x_565_, 1);
lean_inc(v_val_576_);
v_node_577_ = lean_ctor_get(v_x_565_, 2);
lean_inc(v_node_577_);
v_next_578_ = lean_ctor_get(v_x_565_, 3);
lean_inc(v_next_578_);
lean_dec_ref_known(v_x_565_, 4);
v___x_579_ = lean_apply_8(v_h__3_568_, v_rank_571_, v_val_572_, v_node_573_, v_next_574_, v_rank_575_, v_val_576_, v_node_577_, v_next_578_);
return v___x_579_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__3_splitter(lean_object* v_00_u03b1_580_, lean_object* v_motive_581_, lean_object* v_x_582_, lean_object* v_x_583_, lean_object* v_h__1_584_, lean_object* v_h__2_585_, lean_object* v_h__3_586_){
_start:
{
if (lean_obj_tag(v_x_582_) == 0)
{
lean_object* v___x_587_; 
lean_dec(v_h__3_586_);
lean_dec(v_h__2_585_);
v___x_587_ = lean_apply_1(v_h__1_584_, v_x_583_);
return v___x_587_;
}
else
{
lean_dec(v_h__1_584_);
if (lean_obj_tag(v_x_583_) == 0)
{
lean_object* v___x_588_; 
lean_dec(v_h__3_586_);
v___x_588_ = lean_apply_2(v_h__2_585_, v_x_582_, lean_box(0));
return v___x_588_;
}
else
{
lean_object* v_rank_589_; lean_object* v_val_590_; lean_object* v_node_591_; lean_object* v_next_592_; lean_object* v_rank_593_; lean_object* v_val_594_; lean_object* v_node_595_; lean_object* v_next_596_; lean_object* v___x_597_; 
lean_dec(v_h__2_585_);
v_rank_589_ = lean_ctor_get(v_x_582_, 0);
lean_inc(v_rank_589_);
v_val_590_ = lean_ctor_get(v_x_582_, 1);
lean_inc(v_val_590_);
v_node_591_ = lean_ctor_get(v_x_582_, 2);
lean_inc(v_node_591_);
v_next_592_ = lean_ctor_get(v_x_582_, 3);
lean_inc(v_next_592_);
lean_dec_ref_known(v_x_582_, 4);
v_rank_593_ = lean_ctor_get(v_x_583_, 0);
lean_inc(v_rank_593_);
v_val_594_ = lean_ctor_get(v_x_583_, 1);
lean_inc(v_val_594_);
v_node_595_ = lean_ctor_get(v_x_583_, 2);
lean_inc(v_node_595_);
v_next_596_ = lean_ctor_get(v_x_583_, 3);
lean_inc(v_next_596_);
lean_dec_ref_known(v_x_583_, 4);
v___x_597_ = lean_apply_8(v_h__3_586_, v_rank_589_, v_val_590_, v_node_591_, v_next_592_, v_rank_593_, v_val_594_, v_node_595_, v_next_596_);
return v___x_597_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter___redArg(lean_object* v_x_598_, lean_object* v_h__1_599_){
_start:
{
lean_object* v_fst_600_; lean_object* v_snd_601_; lean_object* v___x_602_; 
v_fst_600_ = lean_ctor_get(v_x_598_, 0);
lean_inc(v_fst_600_);
v_snd_601_ = lean_ctor_get(v_x_598_, 1);
lean_inc(v_snd_601_);
lean_dec_ref(v_x_598_);
v___x_602_ = lean_apply_2(v_h__1_599_, v_fst_600_, v_snd_601_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_merge_match__1_splitter(lean_object* v_00_u03b1_603_, lean_object* v_motive_604_, lean_object* v_x_605_, lean_object* v_h__1_606_){
_start:
{
lean_object* v_fst_607_; lean_object* v_snd_608_; lean_object* v___x_609_; 
v_fst_607_ = lean_ctor_get(v_x_605_, 0);
lean_inc(v_fst_607_);
v_snd_608_ = lean_ctor_get(v_x_605_, 1);
lean_inc(v_snd_608_);
lean_dec_ref(v_x_605_);
v___x_609_ = lean_apply_2(v_h__1_606_, v_fst_607_, v_snd_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(lean_object* v_a_610_, lean_object* v_a_611_, lean_object* v_a_612_){
_start:
{
if (lean_obj_tag(v_a_610_) == 0)
{
lean_dec(v_a_611_);
return v_a_612_;
}
else
{
lean_object* v_a_613_; lean_object* v_child_614_; lean_object* v_sibling_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; 
v_a_613_ = lean_ctor_get(v_a_610_, 0);
v_child_614_ = lean_ctor_get(v_a_610_, 1);
v_sibling_615_ = lean_ctor_get(v_a_610_, 2);
v___x_616_ = lean_unsigned_to_nat(1u);
v___x_617_ = lean_nat_sub(v_a_611_, v___x_616_);
lean_dec(v_a_611_);
lean_inc(v_child_614_);
lean_inc(v_a_613_);
lean_inc(v___x_617_);
v___x_618_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_618_, 0, v___x_617_);
lean_ctor_set(v___x_618_, 1, v_a_613_);
lean_ctor_set(v___x_618_, 2, v_child_614_);
lean_ctor_set(v___x_618_, 3, v_a_612_);
v_a_610_ = v_sibling_615_;
v_a_611_ = v___x_617_;
v_a_612_ = v___x_618_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg___boxed(lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(v_a_620_, v_a_621_, v_a_622_);
lean_dec(v_a_620_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(lean_object* v_00_u03b1_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(v_a_625_, v_a_626_, v_a_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___boxed(lean_object* v_00_u03b1_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go(v_00_u03b1_629_, v_a_630_, v_a_631_, v_a_632_);
lean_dec(v_a_630_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(lean_object* v_s_634_){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_rank___redArg(v_s_634_);
v___x_636_ = lean_box(0);
v___x_637_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap_go___redArg(v_s_634_, v___x_635_, v___x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg___boxed(lean_object* v_s_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(v_s_638_);
lean_dec(v_s_638_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(lean_object* v_00_u03b1_640_, lean_object* v_s_641_){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(v_s_641_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___boxed(lean_object* v_00_u03b1_643_, lean_object* v_s_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap(v_00_u03b1_643_, v_s_644_);
lean_dec(v_s_644_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(lean_object* v_le_646_, lean_object* v_a_647_, lean_object* v_x_648_){
_start:
{
if (lean_obj_tag(v_x_648_) == 0)
{
lean_dec_ref(v_le_646_);
return v_a_647_;
}
else
{
lean_object* v_val_649_; lean_object* v_next_650_; lean_object* v___x_651_; uint8_t v___x_652_; 
v_val_649_ = lean_ctor_get(v_x_648_, 1);
lean_inc_n(v_val_649_, 2);
v_next_650_ = lean_ctor_get(v_x_648_, 3);
lean_inc(v_next_650_);
lean_dec_ref_known(v_x_648_, 4);
lean_inc_ref(v_le_646_);
lean_inc(v_a_647_);
v___x_651_ = lean_apply_2(v_le_646_, v_a_647_, v_val_649_);
v___x_652_ = lean_unbox(v___x_651_);
if (v___x_652_ == 0)
{
lean_dec(v_a_647_);
v_a_647_ = v_val_649_;
v_x_648_ = v_next_650_;
goto _start;
}
else
{
lean_dec(v_val_649_);
v_x_648_ = v_next_650_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD(lean_object* v_00_u03b1_655_, lean_object* v_le_656_, lean_object* v_a_657_, lean_object* v_x_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_656_, v_a_657_, v_x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f___redArg(lean_object* v_le_660_, lean_object* v_x_661_){
_start:
{
if (lean_obj_tag(v_x_661_) == 0)
{
lean_object* v___x_662_; 
lean_dec_ref(v_le_660_);
v___x_662_ = lean_box(0);
return v___x_662_;
}
else
{
lean_object* v_val_663_; lean_object* v_next_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v_val_663_ = lean_ctor_get(v_x_661_, 1);
lean_inc(v_val_663_);
v_next_664_ = lean_ctor_get(v_x_661_, 3);
lean_inc(v_next_664_);
lean_dec_ref_known(v_x_661_, 4);
v___x_665_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_660_, v_val_663_, v_next_664_);
v___x_666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_666_, 0, v___x_665_);
return v___x_666_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_head_x3f(lean_object* v_00_u03b1_667_, lean_object* v_le_668_, lean_object* v_x_669_){
_start:
{
if (lean_obj_tag(v_x_669_) == 0)
{
lean_object* v___x_670_; 
lean_dec_ref(v_le_668_);
v___x_670_ = lean_box(0);
return v___x_670_;
}
else
{
lean_object* v_val_671_; lean_object* v_next_672_; lean_object* v___x_673_; lean_object* v___x_674_; 
v_val_671_ = lean_ctor_get(v_x_669_, 1);
lean_inc(v_val_671_);
v_next_672_ = lean_ctor_get(v_x_669_, 3);
lean_inc(v_next_672_);
lean_dec_ref_known(v_x_669_, 4);
v___x_673_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_668_, v_val_671_, v_next_672_);
v___x_674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_674_, 0, v___x_673_);
return v___x_674_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0(lean_object* v_rank_675_, lean_object* v_val_676_, lean_object* v_node_677_, lean_object* v_next_678_){
_start:
{
lean_object* v___x_679_; 
v___x_679_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_679_, 0, v_rank_675_);
lean_ctor_set(v___x_679_, 1, v_val_676_);
lean_ctor_set(v___x_679_, 2, v_node_677_);
lean_ctor_set(v___x_679_, 3, v_next_678_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(lean_object* v_le_680_, lean_object* v_k_681_, lean_object* v_x_682_, lean_object* v_x_683_){
_start:
{
if (lean_obj_tag(v_x_682_) == 0)
{
lean_dec_ref(v_k_681_);
lean_dec_ref(v_le_680_);
return v_x_683_;
}
else
{
lean_object* v_rank_684_; lean_object* v_val_685_; lean_object* v_node_686_; lean_object* v_next_687_; lean_object* v_val_688_; lean_object* v___f_689_; lean_object* v___x_690_; lean_object* v___x_691_; uint8_t v___x_692_; 
v_rank_684_ = lean_ctor_get(v_x_682_, 0);
lean_inc(v_rank_684_);
v_val_685_ = lean_ctor_get(v_x_682_, 1);
lean_inc_n(v_val_685_, 3);
v_node_686_ = lean_ctor_get(v_x_682_, 2);
lean_inc_n(v_node_686_, 2);
v_next_687_ = lean_ctor_get(v_x_682_, 3);
lean_inc(v_next_687_);
lean_dec_ref_known(v_x_682_, 4);
v_val_688_ = lean_ctor_get(v_x_683_, 1);
v___f_689_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0), 4, 3);
lean_closure_set(v___f_689_, 0, v_rank_684_);
lean_closure_set(v___f_689_, 1, v_val_685_);
lean_closure_set(v___f_689_, 2, v_node_686_);
lean_inc_ref(v_k_681_);
v___x_690_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_690_, 0, lean_box(0));
lean_closure_set(v___x_690_, 1, lean_box(0));
lean_closure_set(v___x_690_, 2, lean_box(0));
lean_closure_set(v___x_690_, 3, v_k_681_);
lean_closure_set(v___x_690_, 4, v___f_689_);
lean_inc_ref(v_le_680_);
lean_inc(v_val_688_);
v___x_691_ = lean_apply_2(v_le_680_, v_val_688_, v_val_685_);
v___x_692_ = lean_unbox(v___x_691_);
if (v___x_692_ == 0)
{
lean_object* v___x_694_; uint8_t v_isShared_695_; uint8_t v_isSharedCheck_700_; 
v_isSharedCheck_700_ = !lean_is_exclusive(v_x_683_);
if (v_isSharedCheck_700_ == 0)
{
lean_object* v_unused_701_; lean_object* v_unused_702_; lean_object* v_unused_703_; lean_object* v_unused_704_; 
v_unused_701_ = lean_ctor_get(v_x_683_, 3);
lean_dec(v_unused_701_);
v_unused_702_ = lean_ctor_get(v_x_683_, 2);
lean_dec(v_unused_702_);
v_unused_703_ = lean_ctor_get(v_x_683_, 1);
lean_dec(v_unused_703_);
v_unused_704_ = lean_ctor_get(v_x_683_, 0);
lean_dec(v_unused_704_);
v___x_694_ = v_x_683_;
v_isShared_695_ = v_isSharedCheck_700_;
goto v_resetjp_693_;
}
else
{
lean_dec(v_x_683_);
v___x_694_ = lean_box(0);
v_isShared_695_ = v_isSharedCheck_700_;
goto v_resetjp_693_;
}
v_resetjp_693_:
{
lean_object* v___x_697_; 
lean_inc(v_next_687_);
if (v_isShared_695_ == 0)
{
lean_ctor_set(v___x_694_, 3, v_next_687_);
lean_ctor_set(v___x_694_, 2, v_node_686_);
lean_ctor_set(v___x_694_, 1, v_val_685_);
lean_ctor_set(v___x_694_, 0, v_k_681_);
v___x_697_ = v___x_694_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v_k_681_);
lean_ctor_set(v_reuseFailAlloc_699_, 1, v_val_685_);
lean_ctor_set(v_reuseFailAlloc_699_, 2, v_node_686_);
lean_ctor_set(v_reuseFailAlloc_699_, 3, v_next_687_);
v___x_697_ = v_reuseFailAlloc_699_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
v_k_681_ = v___x_690_;
v_x_682_ = v_next_687_;
v_x_683_ = v___x_697_;
goto _start;
}
}
}
else
{
lean_dec(v_node_686_);
lean_dec(v_val_685_);
lean_dec_ref(v_k_681_);
v_k_681_ = v___x_690_;
v_x_682_ = v_next_687_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin(lean_object* v_00_u03b1_706_, lean_object* v_le_707_, lean_object* v_k_708_, lean_object* v_x_709_, lean_object* v_x_710_){
_start:
{
lean_object* v___x_711_; 
v___x_711_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg(v_le_707_, v_k_708_, v_x_709_, v_x_710_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(lean_object* v___y_712_){
_start:
{
lean_inc(v___y_712_);
return v___y_712_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0___boxed(lean_object* v___y_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___lam__0(v___y_713_);
lean_dec(v___y_713_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0(lean_object* v_rank_715_, lean_object* v_val_716_, lean_object* v_node_717_, lean_object* v_k_718_, lean_object* v___y_719_){
_start:
{
lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_720_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_720_, 0, v_rank_715_);
lean_ctor_set(v___x_720_, 1, v_val_716_);
lean_ctor_set(v___x_720_, 2, v_node_717_);
lean_ctor_set(v___x_720_, 3, v___y_719_);
v___x_721_ = lean_apply_1(v_k_718_, v___x_720_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(lean_object* v_le_722_, lean_object* v_k_723_, lean_object* v_x_724_, lean_object* v_x_725_){
_start:
{
if (lean_obj_tag(v_x_724_) == 0)
{
lean_dec_ref(v_k_723_);
lean_dec_ref(v_le_722_);
return v_x_725_;
}
else
{
lean_object* v_rank_726_; lean_object* v_val_727_; lean_object* v_node_728_; lean_object* v_next_729_; lean_object* v_val_730_; lean_object* v___f_731_; lean_object* v___x_732_; uint8_t v___x_733_; 
v_rank_726_ = lean_ctor_get(v_x_724_, 0);
lean_inc(v_rank_726_);
v_val_727_ = lean_ctor_get(v_x_724_, 1);
lean_inc_n(v_val_727_, 3);
v_node_728_ = lean_ctor_get(v_x_724_, 2);
lean_inc_n(v_node_728_, 2);
v_next_729_ = lean_ctor_get(v_x_724_, 3);
lean_inc(v_next_729_);
lean_dec_ref_known(v_x_724_, 4);
v_val_730_ = lean_ctor_get(v_x_725_, 1);
lean_inc_ref(v_k_723_);
v___f_731_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(v___f_731_, 0, v_rank_726_);
lean_closure_set(v___f_731_, 1, v_val_727_);
lean_closure_set(v___f_731_, 2, v_node_728_);
lean_closure_set(v___f_731_, 3, v_k_723_);
lean_inc_ref(v_le_722_);
lean_inc(v_val_730_);
v___x_732_ = lean_apply_2(v_le_722_, v_val_730_, v_val_727_);
v___x_733_ = lean_unbox(v___x_732_);
if (v___x_733_ == 0)
{
lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_741_; 
v_isSharedCheck_741_ = !lean_is_exclusive(v_x_725_);
if (v_isSharedCheck_741_ == 0)
{
lean_object* v_unused_742_; lean_object* v_unused_743_; lean_object* v_unused_744_; lean_object* v_unused_745_; 
v_unused_742_ = lean_ctor_get(v_x_725_, 3);
lean_dec(v_unused_742_);
v_unused_743_ = lean_ctor_get(v_x_725_, 2);
lean_dec(v_unused_743_);
v_unused_744_ = lean_ctor_get(v_x_725_, 1);
lean_dec(v_unused_744_);
v_unused_745_ = lean_ctor_get(v_x_725_, 0);
lean_dec(v_unused_745_);
v___x_735_ = v_x_725_;
v_isShared_736_ = v_isSharedCheck_741_;
goto v_resetjp_734_;
}
else
{
lean_dec(v_x_725_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_741_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_738_; 
lean_inc(v_next_729_);
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 3, v_next_729_);
lean_ctor_set(v___x_735_, 2, v_node_728_);
lean_ctor_set(v___x_735_, 1, v_val_727_);
lean_ctor_set(v___x_735_, 0, v_k_723_);
v___x_738_ = v___x_735_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_k_723_);
lean_ctor_set(v_reuseFailAlloc_740_, 1, v_val_727_);
lean_ctor_set(v_reuseFailAlloc_740_, 2, v_node_728_);
lean_ctor_set(v_reuseFailAlloc_740_, 3, v_next_729_);
v___x_738_ = v_reuseFailAlloc_740_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
v_k_723_ = v___f_731_;
v_x_724_ = v_next_729_;
v_x_725_ = v___x_738_;
goto _start;
}
}
}
else
{
lean_dec(v_node_728_);
lean_dec(v_val_727_);
lean_dec_ref(v_k_723_);
v_k_723_ = v___f_731_;
v_x_724_ = v_next_729_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(lean_object* v_le_747_, lean_object* v_x_748_, lean_object* v_x_749_){
_start:
{
if (lean_obj_tag(v_x_748_) == 0)
{
lean_dec_ref(v_le_747_);
return v_x_749_;
}
else
{
if (lean_obj_tag(v_x_749_) == 0)
{
lean_dec_ref(v_le_747_);
return v_x_748_;
}
else
{
lean_object* v_rank_750_; lean_object* v_val_751_; lean_object* v_node_752_; lean_object* v_next_753_; lean_object* v_rank_754_; lean_object* v_val_755_; lean_object* v_node_756_; lean_object* v_next_757_; lean_object* v_fst_759_; lean_object* v_snd_760_; uint8_t v___x_774_; 
v_rank_750_ = lean_ctor_get(v_x_748_, 0);
v_val_751_ = lean_ctor_get(v_x_748_, 1);
v_node_752_ = lean_ctor_get(v_x_748_, 2);
v_next_753_ = lean_ctor_get(v_x_748_, 3);
v_rank_754_ = lean_ctor_get(v_x_749_, 0);
v_val_755_ = lean_ctor_get(v_x_749_, 1);
v_node_756_ = lean_ctor_get(v_x_749_, 2);
v_next_757_ = lean_ctor_get(v_x_749_, 3);
v___x_774_ = lean_nat_dec_lt(v_rank_750_, v_rank_754_);
if (v___x_774_ == 0)
{
lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_787_; 
lean_inc(v_next_757_);
lean_inc(v_node_756_);
lean_inc(v_val_755_);
lean_inc(v_rank_754_);
v_isSharedCheck_787_ = !lean_is_exclusive(v_x_749_);
if (v_isSharedCheck_787_ == 0)
{
lean_object* v_unused_788_; lean_object* v_unused_789_; lean_object* v_unused_790_; lean_object* v_unused_791_; 
v_unused_788_ = lean_ctor_get(v_x_749_, 3);
lean_dec(v_unused_788_);
v_unused_789_ = lean_ctor_get(v_x_749_, 2);
lean_dec(v_unused_789_);
v_unused_790_ = lean_ctor_get(v_x_749_, 1);
lean_dec(v_unused_790_);
v_unused_791_ = lean_ctor_get(v_x_749_, 0);
lean_dec(v_unused_791_);
v___x_776_ = v_x_749_;
v_isShared_777_ = v_isSharedCheck_787_;
goto v_resetjp_775_;
}
else
{
lean_dec(v_x_749_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_787_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
uint8_t v___x_778_; 
v___x_778_ = lean_nat_dec_lt(v_rank_754_, v_rank_750_);
if (v___x_778_ == 0)
{
lean_object* v___x_779_; uint8_t v___x_780_; 
lean_inc(v_next_753_);
lean_inc(v_node_752_);
lean_inc_n(v_val_751_, 2);
lean_inc(v_rank_750_);
lean_del_object(v___x_776_);
lean_dec(v_rank_754_);
lean_dec_ref_known(v_x_748_, 4);
lean_inc_ref(v_le_747_);
lean_inc(v_val_755_);
v___x_779_ = lean_apply_2(v_le_747_, v_val_751_, v_val_755_);
v___x_780_ = lean_unbox(v___x_779_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; 
v___x_781_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_781_, 0, v_val_751_);
lean_ctor_set(v___x_781_, 1, v_node_752_);
lean_ctor_set(v___x_781_, 2, v_node_756_);
v_fst_759_ = v_val_755_;
v_snd_760_ = v___x_781_;
goto v___jp_758_;
}
else
{
lean_object* v___x_782_; 
v___x_782_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_782_, 0, v_val_755_);
lean_ctor_set(v___x_782_, 1, v_node_756_);
lean_ctor_set(v___x_782_, 2, v_node_752_);
v_fst_759_ = v_val_751_;
v_snd_760_ = v___x_782_;
goto v___jp_758_;
}
}
else
{
lean_object* v___x_783_; lean_object* v___x_785_; 
v___x_783_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_747_, v_x_748_, v_next_757_);
if (v_isShared_777_ == 0)
{
lean_ctor_set(v___x_776_, 3, v___x_783_);
v___x_785_ = v___x_776_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_rank_754_);
lean_ctor_set(v_reuseFailAlloc_786_, 1, v_val_755_);
lean_ctor_set(v_reuseFailAlloc_786_, 2, v_node_756_);
lean_ctor_set(v_reuseFailAlloc_786_, 3, v___x_783_);
v___x_785_ = v_reuseFailAlloc_786_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
return v___x_785_;
}
}
}
}
else
{
lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_799_; 
lean_inc(v_next_753_);
lean_inc(v_node_752_);
lean_inc(v_val_751_);
lean_inc(v_rank_750_);
v_isSharedCheck_799_ = !lean_is_exclusive(v_x_748_);
if (v_isSharedCheck_799_ == 0)
{
lean_object* v_unused_800_; lean_object* v_unused_801_; lean_object* v_unused_802_; lean_object* v_unused_803_; 
v_unused_800_ = lean_ctor_get(v_x_748_, 3);
lean_dec(v_unused_800_);
v_unused_801_ = lean_ctor_get(v_x_748_, 2);
lean_dec(v_unused_801_);
v_unused_802_ = lean_ctor_get(v_x_748_, 1);
lean_dec(v_unused_802_);
v_unused_803_ = lean_ctor_get(v_x_748_, 0);
lean_dec(v_unused_803_);
v___x_793_ = v_x_748_;
v_isShared_794_ = v_isSharedCheck_799_;
goto v_resetjp_792_;
}
else
{
lean_dec(v_x_748_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_799_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_795_; lean_object* v___x_797_; 
v___x_795_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_747_, v_next_753_, v_x_749_);
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 3, v___x_795_);
v___x_797_ = v___x_793_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_rank_750_);
lean_ctor_set(v_reuseFailAlloc_798_, 1, v_val_751_);
lean_ctor_set(v_reuseFailAlloc_798_, 2, v_node_752_);
lean_ctor_set(v_reuseFailAlloc_798_, 3, v___x_795_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
}
v___jp_758_:
{
lean_object* v___x_761_; lean_object* v_r_762_; uint8_t v___x_763_; 
v___x_761_ = lean_unsigned_to_nat(1u);
v_r_762_ = lean_nat_add(v_rank_750_, v___x_761_);
lean_dec(v_rank_750_);
v___x_763_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_753_, v_r_762_);
if (v___x_763_ == 0)
{
uint8_t v___x_764_; 
v___x_764_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_757_, v_r_762_);
if (v___x_764_ == 0)
{
lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_765_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_747_, v_next_753_, v_next_757_);
v___x_766_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_766_, 0, v_r_762_);
lean_ctor_set(v___x_766_, 1, v_fst_759_);
lean_ctor_set(v___x_766_, 2, v_snd_760_);
lean_ctor_set(v___x_766_, 3, v___x_765_);
return v___x_766_;
}
else
{
lean_object* v___x_767_; 
v___x_767_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_767_, 0, v_r_762_);
lean_ctor_set(v___x_767_, 1, v_fst_759_);
lean_ctor_set(v___x_767_, 2, v_snd_760_);
lean_ctor_set(v___x_767_, 3, v_next_757_);
v_x_748_ = v_next_753_;
v_x_749_ = v___x_767_;
goto _start;
}
}
else
{
uint8_t v___x_769_; 
v___x_769_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_757_, v_r_762_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; 
v___x_770_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_770_, 0, v_r_762_);
lean_ctor_set(v___x_770_, 1, v_fst_759_);
lean_ctor_set(v___x_770_, 2, v_snd_760_);
lean_ctor_set(v___x_770_, 3, v_next_753_);
v_x_748_ = v___x_770_;
v_x_749_ = v_next_757_;
goto _start;
}
else
{
lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_772_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_747_, v_next_753_, v_next_757_);
v___x_773_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_773_, 0, v_r_762_);
lean_ctor_set(v___x_773_, 1, v_fst_759_);
lean_ctor_set(v___x_773_, 2, v_snd_760_);
lean_ctor_set(v___x_773_, 3, v___x_772_);
return v___x_773_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object* v_le_805_, lean_object* v_x_806_){
_start:
{
if (lean_obj_tag(v_x_806_) == 0)
{
lean_object* v___x_807_; 
lean_dec_ref(v_le_805_);
v___x_807_ = lean_box(0);
return v___x_807_;
}
else
{
lean_object* v_rank_808_; lean_object* v_val_809_; lean_object* v_node_810_; lean_object* v_next_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_830_; 
v_rank_808_ = lean_ctor_get(v_x_806_, 0);
v_val_809_ = lean_ctor_get(v_x_806_, 1);
v_node_810_ = lean_ctor_get(v_x_806_, 2);
v_next_811_ = lean_ctor_get(v_x_806_, 3);
v_isSharedCheck_830_ = !lean_is_exclusive(v_x_806_);
if (v_isSharedCheck_830_ == 0)
{
v___x_813_ = v_x_806_;
v_isShared_814_ = v_isSharedCheck_830_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_next_811_);
lean_inc(v_node_810_);
lean_inc(v_val_809_);
lean_inc(v_rank_808_);
lean_dec(v_x_806_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_830_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v___f_815_; lean_object* v___f_816_; lean_object* v___x_818_; 
v___f_815_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg___closed__0));
lean_inc(v_node_810_);
lean_inc(v_val_809_);
v___f_816_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___redArg___lam__0), 4, 3);
lean_closure_set(v___f_816_, 0, v_rank_808_);
lean_closure_set(v___f_816_, 1, v_val_809_);
lean_closure_set(v___f_816_, 2, v_node_810_);
lean_inc(v_next_811_);
if (v_isShared_814_ == 0)
{
lean_ctor_set_tag(v___x_813_, 0);
lean_ctor_set(v___x_813_, 0, v___f_815_);
v___x_818_ = v___x_813_;
goto v_reusejp_817_;
}
else
{
lean_object* v_reuseFailAlloc_829_; 
v_reuseFailAlloc_829_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_829_, 0, v___f_815_);
lean_ctor_set(v_reuseFailAlloc_829_, 1, v_val_809_);
lean_ctor_set(v_reuseFailAlloc_829_, 2, v_node_810_);
lean_ctor_set(v_reuseFailAlloc_829_, 3, v_next_811_);
v___x_818_ = v_reuseFailAlloc_829_;
goto v_reusejp_817_;
}
v_reusejp_817_:
{
lean_object* v___x_819_; lean_object* v_before_820_; lean_object* v_val_821_; lean_object* v_node_822_; lean_object* v_next_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; 
lean_inc_ref(v_le_805_);
v___x_819_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(v_le_805_, v___f_816_, v_next_811_, v___x_818_);
v_before_820_ = lean_ctor_get(v___x_819_, 0);
lean_inc_ref(v_before_820_);
v_val_821_ = lean_ctor_get(v___x_819_, 1);
lean_inc(v_val_821_);
v_node_822_ = lean_ctor_get(v___x_819_, 2);
lean_inc(v_node_822_);
v_next_823_ = lean_ctor_get(v___x_819_, 3);
lean_inc(v_next_823_);
lean_dec_ref(v___x_819_);
v___x_824_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_toHeap___redArg(v_node_822_);
lean_dec(v_node_822_);
v___x_825_ = lean_apply_1(v_before_820_, v_next_823_);
v___x_826_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_805_, v___x_824_, v___x_825_);
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v_val_821_);
lean_ctor_set(v___x_827_, 1, v___x_826_);
v___x_828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_828_, 0, v___x_827_);
return v___x_828_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin(lean_object* v_00_u03b1_831_, lean_object* v_le_832_, lean_object* v_x_833_){
_start:
{
lean_object* v___x_834_; 
v___x_834_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_832_, v_x_833_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0(lean_object* v_00_u03b1_835_, lean_object* v_le_836_, lean_object* v_k_837_, lean_object* v_x_838_, lean_object* v_x_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg(v_le_836_, v_k_837_, v_x_838_, v_x_839_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1(lean_object* v_00_u03b1_841_, lean_object* v_le_842_, lean_object* v_x_843_, lean_object* v_x_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_842_, v_x_843_, v_x_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f___redArg(lean_object* v_le_846_, lean_object* v_h_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_846_, v_h_847_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v___x_849_; 
v___x_849_ = lean_box(0);
return v___x_849_;
}
else
{
lean_object* v_val_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_858_; 
v_val_850_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_858_ == 0)
{
v___x_852_ = v___x_848_;
v_isShared_853_ = v_isSharedCheck_858_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_val_850_);
lean_dec(v___x_848_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_858_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v_snd_854_; lean_object* v___x_856_; 
v_snd_854_ = lean_ctor_get(v_val_850_, 1);
lean_inc(v_snd_854_);
lean_dec(v_val_850_);
if (v_isShared_853_ == 0)
{
lean_ctor_set(v___x_852_, 0, v_snd_854_);
v___x_856_ = v___x_852_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_snd_854_);
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
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail_x3f(lean_object* v_00_u03b1_859_, lean_object* v_le_860_, lean_object* v_h_861_){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_860_, v_h_861_);
if (lean_obj_tag(v___x_862_) == 0)
{
lean_object* v___x_863_; 
v___x_863_ = lean_box(0);
return v___x_863_;
}
else
{
lean_object* v_val_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_872_; 
v_val_864_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_872_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_872_ == 0)
{
v___x_866_ = v___x_862_;
v_isShared_867_ = v_isSharedCheck_872_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_val_864_);
lean_dec(v___x_862_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_872_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v_snd_868_; lean_object* v___x_870_; 
v_snd_868_ = lean_ctor_get(v_val_864_, 1);
lean_inc(v_snd_868_);
lean_dec(v_val_864_);
if (v_isShared_867_ == 0)
{
lean_ctor_set(v___x_866_, 0, v_snd_868_);
v___x_870_ = v___x_866_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_871_; 
v_reuseFailAlloc_871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_871_, 0, v_snd_868_);
v___x_870_ = v_reuseFailAlloc_871_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
return v___x_870_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail___redArg(lean_object* v_le_873_, lean_object* v_h_874_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_873_, v_h_874_);
if (lean_obj_tag(v___x_875_) == 0)
{
lean_object* v___x_876_; 
v___x_876_ = lean_box(0);
return v___x_876_;
}
else
{
lean_object* v_val_877_; lean_object* v_snd_878_; 
v_val_877_ = lean_ctor_get(v___x_875_, 0);
lean_inc(v_val_877_);
lean_dec_ref_known(v___x_875_, 1);
v_snd_878_ = lean_ctor_get(v_val_877_, 1);
lean_inc(v_snd_878_);
lean_dec(v_val_877_);
return v_snd_878_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_tail(lean_object* v_00_u03b1_879_, lean_object* v_le_880_, lean_object* v_h_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_880_, v_h_881_);
if (lean_obj_tag(v___x_882_) == 0)
{
lean_object* v___x_883_; 
v___x_883_ = lean_box(0);
return v___x_883_;
}
else
{
lean_object* v_val_884_; lean_object* v_snd_885_; 
v_val_884_ = lean_ctor_get(v___x_882_, 0);
lean_inc(v_val_884_);
lean_dec_ref_known(v___x_882_, 1);
v_snd_885_ = lean_ctor_get(v_val_884_, 1);
lean_inc(v_snd_885_);
lean_dec(v_val_884_);
return v_snd_885_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter___redArg(lean_object* v_x_886_, lean_object* v_x_887_, lean_object* v_h__1_888_, lean_object* v_h__2_889_){
_start:
{
if (lean_obj_tag(v_x_886_) == 0)
{
lean_object* v___x_890_; 
lean_dec(v_h__2_889_);
v___x_890_ = lean_apply_1(v_h__1_888_, v_x_887_);
return v___x_890_;
}
else
{
lean_object* v_rank_891_; lean_object* v_val_892_; lean_object* v_node_893_; lean_object* v_next_894_; lean_object* v___x_895_; 
lean_dec(v_h__1_888_);
v_rank_891_ = lean_ctor_get(v_x_886_, 0);
lean_inc(v_rank_891_);
v_val_892_ = lean_ctor_get(v_x_886_, 1);
lean_inc(v_val_892_);
v_node_893_ = lean_ctor_get(v_x_886_, 2);
lean_inc(v_node_893_);
v_next_894_ = lean_ctor_get(v_x_886_, 3);
lean_inc(v_next_894_);
lean_dec_ref_known(v_x_886_, 4);
v___x_895_ = lean_apply_5(v_h__2_889_, v_rank_891_, v_val_892_, v_node_893_, v_next_894_, v_x_887_);
return v___x_895_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_findMin_match__1_splitter(lean_object* v_00_u03b1_896_, lean_object* v_motive_897_, lean_object* v_x_898_, lean_object* v_x_899_, lean_object* v_h__1_900_, lean_object* v_h__2_901_){
_start:
{
if (lean_obj_tag(v_x_898_) == 0)
{
lean_object* v___x_902_; 
lean_dec(v_h__2_901_);
v___x_902_ = lean_apply_1(v_h__1_900_, v_x_899_);
return v___x_902_;
}
else
{
lean_object* v_rank_903_; lean_object* v_val_904_; lean_object* v_node_905_; lean_object* v_next_906_; lean_object* v___x_907_; 
lean_dec(v_h__1_900_);
v_rank_903_ = lean_ctor_get(v_x_898_, 0);
lean_inc(v_rank_903_);
v_val_904_ = lean_ctor_get(v_x_898_, 1);
lean_inc(v_val_904_);
v_node_905_ = lean_ctor_get(v_x_898_, 2);
lean_inc(v_node_905_);
v_next_906_ = lean_ctor_get(v_x_898_, 3);
lean_inc(v_next_906_);
lean_dec_ref_known(v_x_898_, 4);
v___x_907_ = lean_apply_5(v_h__2_901_, v_rank_903_, v_val_904_, v_node_905_, v_next_906_, v_x_899_);
return v___x_907_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter___redArg(lean_object* v_x_908_, lean_object* v_x_909_, lean_object* v_x_910_, lean_object* v_h__1_911_, lean_object* v_h__2_912_){
_start:
{
if (lean_obj_tag(v_x_908_) == 0)
{
lean_object* v___x_913_; 
lean_dec(v_h__2_912_);
v___x_913_ = lean_apply_2(v_h__1_911_, v_x_909_, v_x_910_);
return v___x_913_;
}
else
{
lean_object* v_a_914_; lean_object* v_child_915_; lean_object* v_sibling_916_; lean_object* v___x_917_; 
lean_dec(v_h__1_911_);
v_a_914_ = lean_ctor_get(v_x_908_, 0);
lean_inc(v_a_914_);
v_child_915_ = lean_ctor_get(v_x_908_, 1);
lean_inc(v_child_915_);
v_sibling_916_ = lean_ctor_get(v_x_908_, 2);
lean_inc(v_sibling_916_);
lean_dec_ref_known(v_x_908_, 3);
v___x_917_ = lean_apply_5(v_h__2_912_, v_a_914_, v_child_915_, v_sibling_916_, v_x_909_, v_x_910_);
return v___x_917_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_HeapNode_toHeap_go_match__1_splitter(lean_object* v_00_u03b1_918_, lean_object* v_motive_919_, lean_object* v_x_920_, lean_object* v_x_921_, lean_object* v_x_922_, lean_object* v_h__1_923_, lean_object* v_h__2_924_){
_start:
{
if (lean_obj_tag(v_x_920_) == 0)
{
lean_object* v___x_925_; 
lean_dec(v_h__2_924_);
v___x_925_ = lean_apply_2(v_h__1_923_, v_x_921_, v_x_922_);
return v___x_925_;
}
else
{
lean_object* v_a_926_; lean_object* v_child_927_; lean_object* v_sibling_928_; lean_object* v___x_929_; 
lean_dec(v_h__1_923_);
v_a_926_ = lean_ctor_get(v_x_920_, 0);
lean_inc(v_a_926_);
v_child_927_ = lean_ctor_get(v_x_920_, 1);
lean_inc(v_child_927_);
v_sibling_928_ = lean_ctor_get(v_x_920_, 2);
lean_inc(v_sibling_928_);
lean_dec_ref_known(v_x_920_, 3);
v___x_929_ = lean_apply_5(v_h__2_924_, v_a_926_, v_child_927_, v_sibling_928_, v_x_921_, v_x_922_);
return v___x_929_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(lean_object* v_inst_930_, lean_object* v_le_931_, lean_object* v_s_932_, lean_object* v_init_933_, lean_object* v_f_934_){
_start:
{
lean_object* v_toApplicative_935_; lean_object* v_toBind_936_; lean_object* v_toPure_937_; lean_object* v___x_938_; 
v_toApplicative_935_ = lean_ctor_get(v_inst_930_, 0);
v_toBind_936_ = lean_ctor_get(v_inst_930_, 1);
lean_inc(v_toBind_936_);
v_toPure_937_ = lean_ctor_get(v_toApplicative_935_, 1);
lean_inc_ref(v_le_931_);
v___x_938_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_931_, v_s_932_);
if (lean_obj_tag(v___x_938_) == 0)
{
lean_object* v___x_939_; 
lean_inc(v_toPure_937_);
lean_dec(v_toBind_936_);
lean_dec(v_f_934_);
lean_dec_ref(v_le_931_);
lean_dec_ref(v_inst_930_);
v___x_939_ = lean_apply_2(v_toPure_937_, lean_box(0), v_init_933_);
return v___x_939_;
}
else
{
lean_object* v_val_940_; lean_object* v_fst_941_; lean_object* v_snd_942_; lean_object* v___f_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v_val_940_ = lean_ctor_get(v___x_938_, 0);
lean_inc(v_val_940_);
lean_dec_ref_known(v___x_938_, 1);
v_fst_941_ = lean_ctor_get(v_val_940_, 0);
lean_inc(v_fst_941_);
v_snd_942_ = lean_ctor_get(v_val_940_, 1);
lean_inc(v_snd_942_);
lean_dec(v_val_940_);
lean_inc(v_f_934_);
v___f_943_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0), 5, 4);
lean_closure_set(v___f_943_, 0, v_inst_930_);
lean_closure_set(v___f_943_, 1, v_le_931_);
lean_closure_set(v___f_943_, 2, v_snd_942_);
lean_closure_set(v___f_943_, 3, v_f_934_);
v___x_944_ = lean_apply_2(v_f_934_, v_init_933_, v_fst_941_);
v___x_945_ = lean_apply_4(v_toBind_936_, lean_box(0), lean_box(0), v___x_944_, v___f_943_);
return v___x_945_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg___lam__0(lean_object* v_inst_946_, lean_object* v_le_947_, lean_object* v_snd_948_, lean_object* v_f_949_, lean_object* v_____do__lift_950_){
_start:
{
lean_object* v___x_951_; 
v___x_951_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v_inst_946_, v_le_947_, v_snd_948_, v_____do__lift_950_, v_f_949_);
return v___x_951_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM(lean_object* v_m_952_, lean_object* v_00_u03b1_953_, lean_object* v_00_u03b2_954_, lean_object* v_inst_955_, lean_object* v_le_956_, lean_object* v_s_957_, lean_object* v_init_958_, lean_object* v_f_959_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v_inst_955_, v_le_956_, v_s_957_, v_init_958_, v_f_959_);
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter___redArg(lean_object* v_x_961_, lean_object* v_h__1_962_, lean_object* v_h__2_963_){
_start:
{
if (lean_obj_tag(v_x_961_) == 0)
{
lean_object* v___x_964_; 
lean_dec(v_h__2_963_);
v___x_964_ = lean_apply_1(v_h__1_962_, lean_box(0));
return v___x_964_;
}
else
{
lean_object* v_val_965_; lean_object* v_fst_966_; lean_object* v_snd_967_; lean_object* v___x_968_; 
lean_dec(v_h__1_962_);
v_val_965_ = lean_ctor_get(v_x_961_, 0);
lean_inc(v_val_965_);
lean_dec_ref_known(v_x_961_, 1);
v_fst_966_ = lean_ctor_get(v_val_965_, 0);
lean_inc(v_fst_966_);
v_snd_967_ = lean_ctor_get(v_val_965_, 1);
lean_inc(v_snd_967_);
lean_dec(v_val_965_);
v___x_968_ = lean_apply_3(v_h__2_963_, v_fst_966_, v_snd_967_, lean_box(0));
return v___x_968_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_BinomialHeap_Basic_0__Batteries_BinomialHeap_Imp_Heap_foldM_match__1_splitter(lean_object* v_00_u03b1_969_, lean_object* v_motive_970_, lean_object* v_x_971_, lean_object* v_h__1_972_, lean_object* v_h__2_973_){
_start:
{
if (lean_obj_tag(v_x_971_) == 0)
{
lean_object* v___x_974_; 
lean_dec(v_h__2_973_);
v___x_974_ = lean_apply_1(v_h__1_972_, lean_box(0));
return v___x_974_;
}
else
{
lean_object* v_val_975_; lean_object* v_fst_976_; lean_object* v_snd_977_; lean_object* v___x_978_; 
lean_dec(v_h__1_972_);
v_val_975_ = lean_ctor_get(v_x_971_, 0);
lean_inc(v_val_975_);
lean_dec_ref_known(v_x_971_, 1);
v_fst_976_ = lean_ctor_get(v_val_975_, 0);
lean_inc(v_fst_976_);
v_snd_977_ = lean_ctor_get(v_val_975_, 1);
lean_inc(v_snd_977_);
lean_dec(v_val_975_);
v___x_978_ = lean_apply_3(v_h__2_973_, v_fst_976_, v_snd_977_, lean_box(0));
return v___x_978_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg(lean_object* v_le_998_, lean_object* v_s_999_, lean_object* v_init_1000_, lean_object* v_f_1001_){
_start:
{
lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_1002_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1003_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1002_, v_le_998_, v_s_999_, v_init_1000_, v_f_1001_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold(lean_object* v_00_u03b1_1004_, lean_object* v_00_u03b2_1005_, lean_object* v_le_1006_, lean_object* v_s_1007_, lean_object* v_init_1008_, lean_object* v_f_1009_){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1011_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1010_, v_le_1006_, v_s_1007_, v_init_1008_, v_f_1009_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg(lean_object* v_le_1015_, lean_object* v_s_1016_){
_start:
{
lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; 
v___x_1017_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1018_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1019_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1020_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1019_, v_le_1015_, v_s_1016_, v___x_1017_, v___x_1018_);
return v___x_1020_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray(lean_object* v_00_u03b1_1021_, lean_object* v_le_1022_, lean_object* v_s_1023_){
_start:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; 
v___x_1024_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1025_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1026_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1027_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1026_, v_le_1022_, v_s_1023_, v___x_1024_, v___x_1025_);
return v___x_1027_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList___redArg(lean_object* v_le_1028_, lean_object* v_s_1029_){
_start:
{
lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; 
v___x_1030_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1031_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1032_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1033_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1032_, v_le_1028_, v_s_1029_, v___x_1030_, v___x_1031_);
v___x_1034_ = lean_array_to_list(v___x_1033_);
return v___x_1034_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toList(lean_object* v_00_u03b1_1035_, lean_object* v_le_1036_, lean_object* v_s_1037_){
_start:
{
lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; 
v___x_1038_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1039_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1040_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1041_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1040_, v_le_1036_, v_s_1037_, v___x_1038_, v___x_1039_);
v___x_1042_ = lean_array_to_list(v___x_1041_);
return v___x_1042_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0(lean_object* v_join_1043_, lean_object* v_a_1044_, lean_object* v_____do__lift_1045_, lean_object* v_____do__lift_1046_){
_start:
{
lean_object* v___x_1047_; 
v___x_1047_ = lean_apply_3(v_join_1043_, v_a_1044_, v_____do__lift_1045_, v_____do__lift_1046_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(lean_object* v_inst_1048_, lean_object* v_nil_1049_, lean_object* v_join_1050_, lean_object* v_x_1051_){
_start:
{
if (lean_obj_tag(v_x_1051_) == 0)
{
lean_object* v_toApplicative_1052_; lean_object* v_toPure_1053_; lean_object* v___x_1054_; 
v_toApplicative_1052_ = lean_ctor_get(v_inst_1048_, 0);
lean_inc_ref(v_toApplicative_1052_);
lean_dec(v_join_1050_);
lean_dec_ref(v_inst_1048_);
v_toPure_1053_ = lean_ctor_get(v_toApplicative_1052_, 1);
lean_inc(v_toPure_1053_);
lean_dec_ref(v_toApplicative_1052_);
v___x_1054_ = lean_apply_2(v_toPure_1053_, lean_box(0), v_nil_1049_);
return v___x_1054_;
}
else
{
lean_object* v_toBind_1055_; lean_object* v_a_1056_; lean_object* v_child_1057_; lean_object* v_sibling_1058_; lean_object* v___f_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; 
v_toBind_1055_ = lean_ctor_get(v_inst_1048_, 1);
lean_inc_n(v_toBind_1055_, 2);
v_a_1056_ = lean_ctor_get(v_x_1051_, 0);
lean_inc(v_a_1056_);
v_child_1057_ = lean_ctor_get(v_x_1051_, 1);
lean_inc(v_child_1057_);
v_sibling_1058_ = lean_ctor_get(v_x_1051_, 2);
lean_inc(v_sibling_1058_);
lean_dec_ref_known(v_x_1051_, 3);
lean_inc(v_nil_1049_);
lean_inc_ref(v_inst_1048_);
lean_inc(v_join_1050_);
v___f_1059_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1), 7, 6);
lean_closure_set(v___f_1059_, 0, v_join_1050_);
lean_closure_set(v___f_1059_, 1, v_a_1056_);
lean_closure_set(v___f_1059_, 2, v_inst_1048_);
lean_closure_set(v___f_1059_, 3, v_nil_1049_);
lean_closure_set(v___f_1059_, 4, v_sibling_1058_);
lean_closure_set(v___f_1059_, 5, v_toBind_1055_);
v___x_1060_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(v_inst_1048_, v_nil_1049_, v_join_1050_, v_child_1057_);
v___x_1061_ = lean_apply_4(v_toBind_1055_, lean_box(0), lean_box(0), v___x_1060_, v___f_1059_);
return v___x_1061_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__1(lean_object* v_join_1062_, lean_object* v_a_1063_, lean_object* v_inst_1064_, lean_object* v_nil_1065_, lean_object* v_sibling_1066_, lean_object* v_toBind_1067_, lean_object* v_____do__lift_1068_){
_start:
{
lean_object* v___f_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; 
lean_inc(v_join_1062_);
v___f_1069_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1069_, 0, v_join_1062_);
lean_closure_set(v___f_1069_, 1, v_a_1063_);
lean_closure_set(v___f_1069_, 2, v_____do__lift_1068_);
v___x_1070_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(v_inst_1064_, v_nil_1065_, v_join_1062_, v_sibling_1066_);
v___x_1071_ = lean_apply_4(v_toBind_1067_, lean_box(0), lean_box(0), v___x_1070_, v___f_1069_);
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM(lean_object* v_m_1072_, lean_object* v_00_u03b2_1073_, lean_object* v_00_u03b1_1074_, lean_object* v_inst_1075_, lean_object* v_nil_1076_, lean_object* v_join_1077_, lean_object* v_x_1078_){
_start:
{
lean_object* v___x_1079_; 
v___x_1079_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(v_inst_1075_, v_nil_1076_, v_join_1077_, v_x_1078_);
return v___x_1079_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0(lean_object* v_join_1080_, lean_object* v_val_1081_, lean_object* v_____do__lift_1082_, lean_object* v_____do__lift_1083_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lean_apply_3(v_join_1080_, v_val_1081_, v_____do__lift_1082_, v_____do__lift_1083_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(lean_object* v_inst_1085_, lean_object* v_nil_1086_, lean_object* v_join_1087_, lean_object* v_x_1088_){
_start:
{
if (lean_obj_tag(v_x_1088_) == 0)
{
lean_object* v_toApplicative_1089_; lean_object* v_toPure_1090_; lean_object* v___x_1091_; 
v_toApplicative_1089_ = lean_ctor_get(v_inst_1085_, 0);
lean_inc_ref(v_toApplicative_1089_);
lean_dec(v_join_1087_);
lean_dec_ref(v_inst_1085_);
v_toPure_1090_ = lean_ctor_get(v_toApplicative_1089_, 1);
lean_inc(v_toPure_1090_);
lean_dec_ref(v_toApplicative_1089_);
v___x_1091_ = lean_apply_2(v_toPure_1090_, lean_box(0), v_nil_1086_);
return v___x_1091_;
}
else
{
lean_object* v_toBind_1092_; lean_object* v_val_1093_; lean_object* v_node_1094_; lean_object* v_next_1095_; lean_object* v___f_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; 
v_toBind_1092_ = lean_ctor_get(v_inst_1085_, 1);
lean_inc_n(v_toBind_1092_, 2);
v_val_1093_ = lean_ctor_get(v_x_1088_, 1);
lean_inc(v_val_1093_);
v_node_1094_ = lean_ctor_get(v_x_1088_, 2);
lean_inc(v_node_1094_);
v_next_1095_ = lean_ctor_get(v_x_1088_, 3);
lean_inc(v_next_1095_);
lean_dec_ref_known(v_x_1088_, 4);
lean_inc(v_nil_1086_);
lean_inc_ref(v_inst_1085_);
lean_inc(v_join_1087_);
v___f_1096_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1), 7, 6);
lean_closure_set(v___f_1096_, 0, v_join_1087_);
lean_closure_set(v___f_1096_, 1, v_val_1093_);
lean_closure_set(v___f_1096_, 2, v_inst_1085_);
lean_closure_set(v___f_1096_, 3, v_nil_1086_);
lean_closure_set(v___f_1096_, 4, v_next_1095_);
lean_closure_set(v___f_1096_, 5, v_toBind_1092_);
v___x_1097_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___redArg(v_inst_1085_, v_nil_1086_, v_join_1087_, v_node_1094_);
v___x_1098_ = lean_apply_4(v_toBind_1092_, lean_box(0), lean_box(0), v___x_1097_, v___f_1096_);
return v___x_1098_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__1(lean_object* v_join_1099_, lean_object* v_val_1100_, lean_object* v_inst_1101_, lean_object* v_nil_1102_, lean_object* v_next_1103_, lean_object* v_toBind_1104_, lean_object* v_____do__lift_1105_){
_start:
{
lean_object* v___f_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; 
lean_inc(v_join_1099_);
v___f_1106_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1106_, 0, v_join_1099_);
lean_closure_set(v___f_1106_, 1, v_val_1100_);
lean_closure_set(v___f_1106_, 2, v_____do__lift_1105_);
v___x_1107_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(v_inst_1101_, v_nil_1102_, v_join_1099_, v_next_1103_);
v___x_1108_ = lean_apply_4(v_toBind_1104_, lean_box(0), lean_box(0), v___x_1107_, v___f_1106_);
return v___x_1108_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM(lean_object* v_m_1109_, lean_object* v_00_u03b2_1110_, lean_object* v_00_u03b1_1111_, lean_object* v_inst_1112_, lean_object* v_nil_1113_, lean_object* v_join_1114_, lean_object* v_x_1115_){
_start:
{
lean_object* v___x_1116_; 
v___x_1116_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(v_inst_1112_, v_nil_1113_, v_join_1114_, v_x_1115_);
return v___x_1116_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree___redArg(lean_object* v_nil_1117_, lean_object* v_join_1118_, lean_object* v_s_1119_){
_start:
{
lean_object* v___x_1120_; lean_object* v___x_1121_; 
v___x_1120_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1121_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(v___x_1120_, v_nil_1117_, v_join_1118_, v_s_1119_);
return v___x_1121_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTree(lean_object* v_00_u03b2_1122_, lean_object* v_00_u03b1_1123_, lean_object* v_nil_1124_, lean_object* v_join_1125_, lean_object* v_s_1126_){
_start:
{
lean_object* v___x_1127_; lean_object* v___x_1128_; 
v___x_1127_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1128_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___redArg(v___x_1127_, v_nil_1124_, v_join_1125_, v_s_1126_);
return v___x_1128_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(lean_object* v___y_1129_){
_start:
{
lean_inc(v___y_1129_);
return v___y_1129_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0___boxed(lean_object* v___y_1130_){
_start:
{
lean_object* v_res_1131_; 
v_res_1131_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___lam__0(v___y_1130_);
lean_dec(v___y_1130_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(lean_object* v_nil_1132_, lean_object* v_x_1133_, lean_object* v___y_1134_){
_start:
{
if (lean_obj_tag(v_x_1133_) == 0)
{
lean_object* v___x_1135_; 
v___x_1135_ = lean_apply_1(v_nil_1132_, v___y_1134_);
return v___x_1135_;
}
else
{
lean_object* v_a_1136_; lean_object* v_child_1137_; lean_object* v_sibling_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; 
v_a_1136_ = lean_ctor_get(v_x_1133_, 0);
v_child_1137_ = lean_ctor_get(v_x_1133_, 1);
v_sibling_1138_ = lean_ctor_get(v_x_1133_, 2);
lean_inc_ref(v_nil_1132_);
v___x_1139_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(v_nil_1132_, v_sibling_1138_, v___y_1134_);
v___x_1140_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(v_nil_1132_, v_child_1137_, v___x_1139_);
lean_inc(v_a_1136_);
v___x_1141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1141_, 0, v_a_1136_);
lean_ctor_set(v___x_1141_, 1, v___x_1140_);
return v___x_1141_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg___boxed(lean_object* v_nil_1142_, lean_object* v_x_1143_, lean_object* v___y_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(v_nil_1142_, v_x_1143_, v___y_1144_);
lean_dec(v_x_1143_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(lean_object* v_nil_1146_, lean_object* v_x_1147_, lean_object* v___y_1148_){
_start:
{
if (lean_obj_tag(v_x_1147_) == 0)
{
lean_object* v___x_1149_; 
v___x_1149_ = lean_apply_1(v_nil_1146_, v___y_1148_);
return v___x_1149_;
}
else
{
lean_object* v_val_1150_; lean_object* v_node_1151_; lean_object* v_next_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; 
v_val_1150_ = lean_ctor_get(v_x_1147_, 1);
v_node_1151_ = lean_ctor_get(v_x_1147_, 2);
v_next_1152_ = lean_ctor_get(v_x_1147_, 3);
lean_inc_ref(v_nil_1146_);
v___x_1153_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(v_nil_1146_, v_next_1152_, v___y_1148_);
v___x_1154_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(v_nil_1146_, v_node_1151_, v___x_1153_);
lean_inc(v_val_1150_);
v___x_1155_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1155_, 0, v_val_1150_);
lean_ctor_set(v___x_1155_, 1, v___x_1154_);
return v___x_1155_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg___boxed(lean_object* v_nil_1156_, lean_object* v_x_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(v_nil_1156_, v_x_1157_, v___y_1158_);
lean_dec(v_x_1157_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(lean_object* v_s_1161_){
_start:
{
lean_object* v___f_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; 
v___f_1162_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___closed__0));
v___x_1163_ = lean_box(0);
v___x_1164_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(v___f_1162_, v_s_1161_, v___x_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg___boxed(lean_object* v_s_1165_){
_start:
{
lean_object* v_res_1166_; 
v_res_1166_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(v_s_1165_);
lean_dec(v_s_1165_);
return v_res_1166_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(lean_object* v_00_u03b1_1167_, lean_object* v_s_1168_){
_start:
{
lean_object* v___x_1169_; 
v___x_1169_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(v_s_1168_);
return v___x_1169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___boxed(lean_object* v_00_u03b1_1170_, lean_object* v_s_1171_){
_start:
{
lean_object* v_res_1172_; 
v_res_1172_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered(v_00_u03b1_1170_, v_s_1171_);
lean_dec(v_s_1171_);
return v_res_1172_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(lean_object* v_00_u03b1_1173_, lean_object* v_nil_1174_, lean_object* v_x_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v___x_1177_; 
v___x_1177_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___redArg(v_nil_1174_, v_x_1175_, v___y_1176_);
return v___x_1177_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0___boxed(lean_object* v_00_u03b1_1178_, lean_object* v_nil_1179_, lean_object* v_x_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0(v_00_u03b1_1178_, v_nil_1179_, v_x_1180_, v___y_1181_);
lean_dec(v_x_1180_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(lean_object* v_00_u03b1_1183_, lean_object* v_nil_1184_, lean_object* v_x_1185_, lean_object* v___y_1186_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___redArg(v_nil_1184_, v_x_1185_, v___y_1186_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1188_, lean_object* v_nil_1189_, lean_object* v_x_1190_, lean_object* v___y_1191_){
_start:
{
lean_object* v_res_1192_; 
v_res_1192_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toListUnordered_spec__0_spec__0(v_00_u03b1_1188_, v_nil_1189_, v_x_1190_, v___y_1191_);
lean_dec(v_x_1190_);
return v_res_1192_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(lean_object* v___y_1193_){
_start:
{
lean_inc_ref(v___y_1193_);
return v___y_1193_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0___boxed(lean_object* v___y_1194_){
_start:
{
lean_object* v_res_1195_; 
v_res_1195_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___lam__0(v___y_1194_);
lean_dec_ref(v___y_1194_);
return v_res_1195_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(lean_object* v_nil_1196_, lean_object* v_x_1197_, lean_object* v___y_1198_){
_start:
{
if (lean_obj_tag(v_x_1197_) == 0)
{
lean_object* v___x_1199_; 
v___x_1199_ = lean_apply_1(v_nil_1196_, v___y_1198_);
return v___x_1199_;
}
else
{
lean_object* v_a_1200_; lean_object* v_child_1201_; lean_object* v_sibling_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; 
v_a_1200_ = lean_ctor_get(v_x_1197_, 0);
lean_inc(v_a_1200_);
v_child_1201_ = lean_ctor_get(v_x_1197_, 1);
lean_inc(v_child_1201_);
v_sibling_1202_ = lean_ctor_get(v_x_1197_, 2);
lean_inc(v_sibling_1202_);
lean_dec_ref_known(v_x_1197_, 3);
v___x_1203_ = lean_array_push(v___y_1198_, v_a_1200_);
lean_inc_ref(v_nil_1196_);
v___x_1204_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(v_nil_1196_, v_child_1201_, v___x_1203_);
v_x_1197_ = v_sibling_1202_;
v___y_1198_ = v___x_1204_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(lean_object* v_nil_1206_, lean_object* v_x_1207_, lean_object* v___y_1208_){
_start:
{
if (lean_obj_tag(v_x_1207_) == 0)
{
lean_object* v___x_1209_; 
v___x_1209_ = lean_apply_1(v_nil_1206_, v___y_1208_);
return v___x_1209_;
}
else
{
lean_object* v_val_1210_; lean_object* v_node_1211_; lean_object* v_next_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; 
v_val_1210_ = lean_ctor_get(v_x_1207_, 1);
lean_inc(v_val_1210_);
v_node_1211_ = lean_ctor_get(v_x_1207_, 2);
lean_inc(v_node_1211_);
v_next_1212_ = lean_ctor_get(v_x_1207_, 3);
lean_inc(v_next_1212_);
lean_dec_ref_known(v_x_1207_, 4);
v___x_1213_ = lean_array_push(v___y_1208_, v_val_1210_);
lean_inc_ref(v_nil_1206_);
v___x_1214_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(v_nil_1206_, v_node_1211_, v___x_1213_);
v_x_1207_ = v_next_1212_;
v___y_1208_ = v___x_1214_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(lean_object* v_s_1217_){
_start:
{
lean_object* v___f_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___f_1218_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg___closed__0));
v___x_1219_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1220_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(v___f_1218_, v_s_1217_, v___x_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered(lean_object* v_00_u03b1_1221_, lean_object* v_s_1222_){
_start:
{
lean_object* v___x_1223_; 
v___x_1223_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(v_s_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0(lean_object* v_00_u03b1_1224_, lean_object* v_nil_1225_, lean_object* v_x_1226_, lean_object* v___y_1227_){
_start:
{
lean_object* v___x_1228_; 
v___x_1228_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0___redArg(v_nil_1225_, v_x_1226_, v___y_1227_);
return v___x_1228_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0(lean_object* v_00_u03b1_1229_, lean_object* v_nil_1230_, lean_object* v_x_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v___x_1233_; 
v___x_1233_ = lp_batteries_Batteries_BinomialHeap_Imp_HeapNode_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_foldTreeM___at___00Batteries_BinomialHeap_Imp_Heap_toArrayUnordered_spec__0_spec__0___redArg(v_nil_1230_, v_x_1231_, v___y_1232_);
return v___x_1233_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(lean_object* v_le_1234_, lean_object* v_k_1235_, lean_object* v_res_1236_, lean_object* v_s_1237_, lean_object* v_hr_1238_){
_start:
{
if (lean_obj_tag(v_s_1237_) == 0)
{
lean_dec_ref(v_res_1236_);
lean_dec_ref(v_k_1235_);
lean_dec_ref(v_le_1234_);
return v_hr_1238_;
}
else
{
lean_object* v_rank_1239_; lean_object* v_val_1240_; lean_object* v_node_1241_; lean_object* v_next_1242_; lean_object* v_val_1243_; lean_object* v___x_1244_; uint8_t v___x_1245_; 
v_rank_1239_ = lean_ctor_get(v_s_1237_, 0);
lean_inc(v_rank_1239_);
v_val_1240_ = lean_ctor_get(v_s_1237_, 1);
lean_inc_n(v_val_1240_, 2);
v_node_1241_ = lean_ctor_get(v_s_1237_, 2);
lean_inc(v_node_1241_);
v_next_1242_ = lean_ctor_get(v_s_1237_, 3);
lean_inc(v_next_1242_);
lean_dec_ref_known(v_s_1237_, 4);
v_val_1243_ = lean_ctor_get(v_res_1236_, 1);
lean_inc_ref(v_le_1234_);
lean_inc(v_val_1243_);
v___x_1244_ = lean_apply_2(v_le_1234_, v_val_1243_, v_val_1240_);
v___x_1245_ = lean_unbox(v___x_1244_);
if (v___x_1245_ == 0)
{
lean_object* v___x_1247_; uint8_t v_isShared_1248_; uint8_t v_isSharedCheck_1254_; 
lean_dec(v_hr_1238_);
v_isSharedCheck_1254_ = !lean_is_exclusive(v_res_1236_);
if (v_isSharedCheck_1254_ == 0)
{
lean_object* v_unused_1255_; lean_object* v_unused_1256_; lean_object* v_unused_1257_; lean_object* v_unused_1258_; 
v_unused_1255_ = lean_ctor_get(v_res_1236_, 3);
lean_dec(v_unused_1255_);
v_unused_1256_ = lean_ctor_get(v_res_1236_, 2);
lean_dec(v_unused_1256_);
v_unused_1257_ = lean_ctor_get(v_res_1236_, 1);
lean_dec(v_unused_1257_);
v_unused_1258_ = lean_ctor_get(v_res_1236_, 0);
lean_dec(v_unused_1258_);
v___x_1247_ = v_res_1236_;
v_isShared_1248_ = v_isSharedCheck_1254_;
goto v_resetjp_1246_;
}
else
{
lean_dec(v_res_1236_);
v___x_1247_ = lean_box(0);
v_isShared_1248_ = v_isSharedCheck_1254_;
goto v_resetjp_1246_;
}
v_resetjp_1246_:
{
lean_object* v___f_1249_; lean_object* v___x_1251_; 
lean_inc_ref(v_k_1235_);
lean_inc(v_node_1241_);
lean_inc(v_val_1240_);
lean_inc(v_rank_1239_);
v___f_1249_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(v___f_1249_, 0, v_rank_1239_);
lean_closure_set(v___f_1249_, 1, v_val_1240_);
lean_closure_set(v___f_1249_, 2, v_node_1241_);
lean_closure_set(v___f_1249_, 3, v_k_1235_);
lean_inc(v_next_1242_);
if (v_isShared_1248_ == 0)
{
lean_ctor_set(v___x_1247_, 3, v_next_1242_);
lean_ctor_set(v___x_1247_, 2, v_node_1241_);
lean_ctor_set(v___x_1247_, 1, v_val_1240_);
lean_ctor_set(v___x_1247_, 0, v_k_1235_);
v___x_1251_ = v___x_1247_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1253_; 
v_reuseFailAlloc_1253_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1253_, 0, v_k_1235_);
lean_ctor_set(v_reuseFailAlloc_1253_, 1, v_val_1240_);
lean_ctor_set(v_reuseFailAlloc_1253_, 2, v_node_1241_);
lean_ctor_set(v_reuseFailAlloc_1253_, 3, v_next_1242_);
v___x_1251_ = v_reuseFailAlloc_1253_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
v_k_1235_ = v___f_1249_;
v_res_1236_ = v___x_1251_;
v_s_1237_ = v_next_1242_;
v_hr_1238_ = v_rank_1239_;
goto _start;
}
}
}
else
{
lean_object* v___f_1259_; 
v___f_1259_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_findMin___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__0___redArg___lam__0), 5, 4);
lean_closure_set(v___f_1259_, 0, v_rank_1239_);
lean_closure_set(v___f_1259_, 1, v_val_1240_);
lean_closure_set(v___f_1259_, 2, v_node_1241_);
lean_closure_set(v___f_1259_, 3, v_k_1235_);
v_k_1235_ = v___f_1259_;
v_s_1237_ = v_next_1242_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(lean_object* v_00_u03b1_1261_, lean_object* v_le_1262_, lean_object* v_n_1263_, lean_object* v_k_1264_, lean_object* v_res_1265_, lean_object* v_s_1266_, lean_object* v_h_1267_, lean_object* v_hr_1268_, lean_object* v_hk_1269_){
_start:
{
lean_object* v___x_1270_; 
v___x_1270_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___redArg(v_le_1262_, v_k_1264_, v_res_1265_, v_s_1266_, v_hr_1268_);
return v___x_1270_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin___boxed(lean_object* v_00_u03b1_1271_, lean_object* v_le_1272_, lean_object* v_n_1273_, lean_object* v_k_1274_, lean_object* v_res_1275_, lean_object* v_s_1276_, lean_object* v_h_1277_, lean_object* v_hr_1278_, lean_object* v_hk_1279_){
_start:
{
lean_object* v_res_1280_; 
v_res_1280_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_WF_findMin(v_00_u03b1_1271_, v_le_1272_, v_n_1273_, v_k_1274_, v_res_1275_, v_s_1276_, v_h_1277_, v_hr_1278_, v_hk_1279_);
lean_dec(v_n_1273_);
return v_res_1280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap(lean_object* v_00_u03b1_1281_, lean_object* v_le_1282_){
_start:
{
lean_object* v___x_1283_; 
v___x_1283_ = lean_box(0);
return v___x_1283_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_mkBinomialHeap___boxed(lean_object* v_00_u03b1_1284_, lean_object* v_le_1285_){
_start:
{
lean_object* v_res_1286_; 
v_res_1286_ = lp_batteries_Batteries_mkBinomialHeap(v_00_u03b1_1284_, v_le_1285_);
lean_dec_ref(v_le_1285_);
return v_res_1286_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty(lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
lean_object* v___x_1289_; 
v___x_1289_ = lean_box(0);
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_empty___boxed(lean_object* v___y_1290_, lean_object* v___y_1291_){
_start:
{
lean_object* v_res_1292_; 
v_res_1292_ = lp_batteries_Batteries_BinomialHeap_empty(v___y_1290_, v___y_1291_);
lean_dec_ref(v___y_1291_);
return v_res_1292_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection(lean_object* v_00_u03b1_1293_, lean_object* v_le_1294_){
_start:
{
lean_object* v___x_1295_; 
v___x_1295_ = lean_box(0);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instEmptyCollection___boxed(lean_object* v_00_u03b1_1296_, lean_object* v_le_1297_){
_start:
{
lean_object* v_res_1298_; 
v_res_1298_ = lp_batteries_Batteries_BinomialHeap_instEmptyCollection(v_00_u03b1_1296_, v_le_1297_);
lean_dec_ref(v_le_1297_);
return v_res_1298_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited(lean_object* v_00_u03b1_1299_, lean_object* v_le_1300_){
_start:
{
lean_object* v___x_1301_; 
v___x_1301_ = lean_box(0);
return v___x_1301_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instInhabited___boxed(lean_object* v_00_u03b1_1302_, lean_object* v_le_1303_){
_start:
{
lean_object* v_res_1304_; 
v_res_1304_ = lp_batteries_Batteries_BinomialHeap_instInhabited(v_00_u03b1_1302_, v_le_1303_);
lean_dec_ref(v_le_1303_);
return v_res_1304_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(lean_object* v_b_1305_){
_start:
{
if (lean_obj_tag(v_b_1305_) == 0)
{
uint8_t v___x_1306_; 
v___x_1306_ = 1;
return v___x_1306_;
}
else
{
uint8_t v___x_1307_; 
v___x_1307_ = 0;
return v___x_1307_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___redArg___boxed(lean_object* v_b_1308_){
_start:
{
uint8_t v_res_1309_; lean_object* v_r_1310_; 
v_res_1309_ = lp_batteries_Batteries_BinomialHeap_isEmpty___redArg(v_b_1308_);
lean_dec(v_b_1308_);
v_r_1310_ = lean_box(v_res_1309_);
return v_r_1310_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_BinomialHeap_isEmpty(lean_object* v_00_u03b1_1311_, lean_object* v_le_1312_, lean_object* v_b_1313_){
_start:
{
if (lean_obj_tag(v_b_1313_) == 0)
{
uint8_t v___x_1314_; 
v___x_1314_ = 1;
return v___x_1314_;
}
else
{
uint8_t v___x_1315_; 
v___x_1315_ = 0;
return v___x_1315_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_isEmpty___boxed(lean_object* v_00_u03b1_1316_, lean_object* v_le_1317_, lean_object* v_b_1318_){
_start:
{
uint8_t v_res_1319_; lean_object* v_r_1320_; 
v_res_1319_ = lp_batteries_Batteries_BinomialHeap_isEmpty(v_00_u03b1_1316_, v_le_1317_, v_b_1318_);
lean_dec(v_b_1318_);
lean_dec_ref(v_le_1317_);
v_r_1320_ = lean_box(v_res_1319_);
return v_r_1320_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg(lean_object* v_b_1321_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(v_b_1321_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___redArg___boxed(lean_object* v_b_1323_){
_start:
{
lean_object* v_res_1324_; 
v_res_1324_ = lp_batteries_Batteries_BinomialHeap_size___redArg(v_b_1323_);
lean_dec(v_b_1323_);
return v_res_1324_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size(lean_object* v_00_u03b1_1325_, lean_object* v_le_1326_, lean_object* v_b_1327_){
_start:
{
lean_object* v___x_1328_; 
v___x_1328_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_size___redArg(v_b_1327_);
return v___x_1328_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_size___boxed(lean_object* v_00_u03b1_1329_, lean_object* v_le_1330_, lean_object* v_b_1331_){
_start:
{
lean_object* v_res_1332_; 
v_res_1332_ = lp_batteries_Batteries_BinomialHeap_size(v_00_u03b1_1329_, v_le_1330_, v_b_1331_);
lean_dec(v_b_1331_);
lean_dec_ref(v_le_1330_);
return v_res_1332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___redArg(lean_object* v_a_1333_){
_start:
{
lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1334_ = lean_unsigned_to_nat(0u);
v___x_1335_ = lean_box(0);
v___x_1336_ = lean_box(0);
v___x_1337_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1337_, 0, v___x_1334_);
lean_ctor_set(v___x_1337_, 1, v_a_1333_);
lean_ctor_set(v___x_1337_, 2, v___x_1335_);
lean_ctor_set(v___x_1337_, 3, v___x_1336_);
return v___x_1337_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton(lean_object* v_00_u03b1_1338_, lean_object* v_le_1339_, lean_object* v_a_1340_){
_start:
{
lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; 
v___x_1341_ = lean_unsigned_to_nat(0u);
v___x_1342_ = lean_box(0);
v___x_1343_ = lean_box(0);
v___x_1344_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1344_, 0, v___x_1341_);
lean_ctor_set(v___x_1344_, 1, v_a_1340_);
lean_ctor_set(v___x_1344_, 2, v___x_1342_);
lean_ctor_set(v___x_1344_, 3, v___x_1343_);
return v___x_1344_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_singleton___boxed(lean_object* v_00_u03b1_1345_, lean_object* v_le_1346_, lean_object* v_a_1347_){
_start:
{
lean_object* v_res_1348_; 
v_res_1348_ = lp_batteries_Batteries_BinomialHeap_singleton(v_00_u03b1_1345_, v_le_1346_, v_a_1347_);
lean_dec_ref(v_le_1346_);
return v_res_1348_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge___redArg(lean_object* v_le_1349_, lean_object* v_x_1350_, lean_object* v_x_1351_){
_start:
{
lean_object* v___x_1352_; 
v___x_1352_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_1349_, v_x_1350_, v_x_1351_);
return v___x_1352_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_merge(lean_object* v_00_u03b1_1353_, lean_object* v_le_1354_, lean_object* v_x_1355_, lean_object* v_x_1356_){
_start:
{
lean_object* v___x_1357_; 
v___x_1357_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_1354_, v_x_1355_, v_x_1356_);
return v___x_1357_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert___redArg(lean_object* v_le_1358_, lean_object* v_a_1359_, lean_object* v_h_1360_){
_start:
{
lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; 
v___x_1361_ = lean_unsigned_to_nat(0u);
v___x_1362_ = lean_box(0);
v___x_1363_ = lean_box(0);
v___x_1364_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1364_, 0, v___x_1361_);
lean_ctor_set(v___x_1364_, 1, v_a_1359_);
lean_ctor_set(v___x_1364_, 2, v___x_1362_);
lean_ctor_set(v___x_1364_, 3, v___x_1363_);
v___x_1365_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_1358_, v___x_1364_, v_h_1360_);
return v___x_1365_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_insert(lean_object* v_00_u03b1_1366_, lean_object* v_le_1367_, lean_object* v_a_1368_, lean_object* v_h_1369_){
_start:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1370_ = lean_unsigned_to_nat(0u);
v___x_1371_ = lean_box(0);
v___x_1372_ = lean_box(0);
v___x_1373_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1370_);
lean_ctor_set(v___x_1373_, 1, v_a_1368_);
lean_ctor_set(v___x_1373_, 2, v___x_1371_);
lean_ctor_set(v___x_1373_, 3, v___x_1372_);
v___x_1374_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___redArg(v_le_1367_, v___x_1373_, v_h_1369_);
return v___x_1374_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(lean_object* v_le_1375_, lean_object* v_x_1376_, lean_object* v_x_1377_){
_start:
{
if (lean_obj_tag(v_x_1377_) == 0)
{
lean_dec_ref(v_le_1375_);
return v_x_1376_;
}
else
{
lean_object* v_head_1378_; lean_object* v_tail_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; 
v_head_1378_ = lean_ctor_get(v_x_1377_, 0);
v_tail_1379_ = lean_ctor_get(v_x_1377_, 1);
v___x_1380_ = lean_unsigned_to_nat(0u);
v___x_1381_ = lean_box(0);
v___x_1382_ = lean_box(0);
lean_inc(v_head_1378_);
v___x_1383_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1383_, 0, v___x_1380_);
lean_ctor_set(v___x_1383_, 1, v_head_1378_);
lean_ctor_set(v___x_1383_, 2, v___x_1381_);
lean_ctor_set(v___x_1383_, 3, v___x_1382_);
lean_inc_ref(v_le_1375_);
v___x_1384_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_1375_, v___x_1383_, v_x_1376_);
v_x_1376_ = v___x_1384_;
v_x_1377_ = v_tail_1379_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg___boxed(lean_object* v_le_1386_, lean_object* v_x_1387_, lean_object* v_x_1388_){
_start:
{
lean_object* v_res_1389_; 
v_res_1389_ = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(v_le_1386_, v_x_1387_, v_x_1388_);
lean_dec(v_x_1388_);
return v_res_1389_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg(lean_object* v_le_1390_, lean_object* v_as_1391_){
_start:
{
lean_object* v___x_1392_; lean_object* v___x_1393_; 
v___x_1392_ = lean_box(0);
v___x_1393_ = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(v_le_1390_, v___x_1392_, v_as_1391_);
return v___x_1393_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___redArg___boxed(lean_object* v_le_1394_, lean_object* v_as_1395_){
_start:
{
lean_object* v_res_1396_; 
v_res_1396_ = lp_batteries_Batteries_BinomialHeap_ofList___redArg(v_le_1394_, v_as_1395_);
lean_dec(v_as_1395_);
return v_res_1396_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList(lean_object* v_00_u03b1_1397_, lean_object* v_le_1398_, lean_object* v_as_1399_){
_start:
{
lean_object* v___x_1400_; 
v___x_1400_ = lp_batteries_Batteries_BinomialHeap_ofList___redArg(v_le_1398_, v_as_1399_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofList___boxed(lean_object* v_00_u03b1_1401_, lean_object* v_le_1402_, lean_object* v_as_1403_){
_start:
{
lean_object* v_res_1404_; 
v_res_1404_ = lp_batteries_Batteries_BinomialHeap_ofList(v_00_u03b1_1401_, v_le_1402_, v_as_1403_);
lean_dec(v_as_1403_);
return v_res_1404_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(lean_object* v_00_u03b1_1405_, lean_object* v_le_1406_, lean_object* v_x_1407_, lean_object* v_x_1408_){
_start:
{
lean_object* v___x_1409_; 
v___x_1409_ = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___redArg(v_le_1406_, v_x_1407_, v_x_1408_);
return v___x_1409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0___boxed(lean_object* v_00_u03b1_1410_, lean_object* v_le_1411_, lean_object* v_x_1412_, lean_object* v_x_1413_){
_start:
{
lean_object* v_res_1414_; 
v_res_1414_ = lp_batteries_List_foldl___at___00Batteries_BinomialHeap_ofList_spec__0(v_00_u03b1_1410_, v_le_1411_, v_x_1412_, v_x_1413_);
lean_dec(v_x_1413_);
return v_res_1414_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(lean_object* v_le_1415_, lean_object* v_as_1416_, size_t v_i_1417_, size_t v_stop_1418_, lean_object* v_b_1419_){
_start:
{
uint8_t v___x_1420_; 
v___x_1420_ = lean_usize_dec_eq(v_i_1417_, v_stop_1418_);
if (v___x_1420_ == 0)
{
lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; size_t v___x_1427_; size_t v___x_1428_; 
v___x_1421_ = lean_array_uget_borrowed(v_as_1416_, v_i_1417_);
v___x_1422_ = lean_unsigned_to_nat(0u);
v___x_1423_ = lean_box(0);
v___x_1424_ = lean_box(0);
lean_inc(v___x_1421_);
v___x_1425_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1422_);
lean_ctor_set(v___x_1425_, 1, v___x_1421_);
lean_ctor_set(v___x_1425_, 2, v___x_1423_);
lean_ctor_set(v___x_1425_, 3, v___x_1424_);
lean_inc_ref(v_le_1415_);
v___x_1426_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_merge___at___00Batteries_BinomialHeap_Imp_Heap_deleteMin_spec__1___redArg(v_le_1415_, v___x_1425_, v_b_1419_);
v___x_1427_ = ((size_t)1ULL);
v___x_1428_ = lean_usize_add(v_i_1417_, v___x_1427_);
v_i_1417_ = v___x_1428_;
v_b_1419_ = v___x_1426_;
goto _start;
}
else
{
lean_dec_ref(v_le_1415_);
return v_b_1419_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg___boxed(lean_object* v_le_1430_, lean_object* v_as_1431_, lean_object* v_i_1432_, lean_object* v_stop_1433_, lean_object* v_b_1434_){
_start:
{
size_t v_i_boxed_1435_; size_t v_stop_boxed_1436_; lean_object* v_res_1437_; 
v_i_boxed_1435_ = lean_unbox_usize(v_i_1432_);
lean_dec(v_i_1432_);
v_stop_boxed_1436_ = lean_unbox_usize(v_stop_1433_);
lean_dec(v_stop_1433_);
v_res_1437_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(v_le_1430_, v_as_1431_, v_i_boxed_1435_, v_stop_boxed_1436_, v_b_1434_);
lean_dec_ref(v_as_1431_);
return v_res_1437_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg(lean_object* v_le_1438_, lean_object* v_as_1439_){
_start:
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; uint8_t v___x_1443_; 
v___x_1440_ = lean_box(0);
v___x_1441_ = lean_unsigned_to_nat(0u);
v___x_1442_ = lean_array_get_size(v_as_1439_);
v___x_1443_ = lean_nat_dec_lt(v___x_1441_, v___x_1442_);
if (v___x_1443_ == 0)
{
lean_dec_ref(v_le_1438_);
return v___x_1440_;
}
else
{
uint8_t v___x_1444_; 
v___x_1444_ = lean_nat_dec_le(v___x_1442_, v___x_1442_);
if (v___x_1444_ == 0)
{
if (v___x_1443_ == 0)
{
lean_dec_ref(v_le_1438_);
return v___x_1440_;
}
else
{
size_t v___x_1445_; size_t v___x_1446_; lean_object* v___x_1447_; 
v___x_1445_ = ((size_t)0ULL);
v___x_1446_ = lean_usize_of_nat(v___x_1442_);
v___x_1447_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(v_le_1438_, v_as_1439_, v___x_1445_, v___x_1446_, v___x_1440_);
return v___x_1447_;
}
}
else
{
size_t v___x_1448_; size_t v___x_1449_; lean_object* v___x_1450_; 
v___x_1448_ = ((size_t)0ULL);
v___x_1449_ = lean_usize_of_nat(v___x_1442_);
v___x_1450_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(v_le_1438_, v_as_1439_, v___x_1448_, v___x_1449_, v___x_1440_);
return v___x_1450_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___redArg___boxed(lean_object* v_le_1451_, lean_object* v_as_1452_){
_start:
{
lean_object* v_res_1453_; 
v_res_1453_ = lp_batteries_Batteries_BinomialHeap_ofArray___redArg(v_le_1451_, v_as_1452_);
lean_dec_ref(v_as_1452_);
return v_res_1453_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray(lean_object* v_00_u03b1_1454_, lean_object* v_le_1455_, lean_object* v_as_1456_){
_start:
{
lean_object* v___x_1457_; 
v___x_1457_ = lp_batteries_Batteries_BinomialHeap_ofArray___redArg(v_le_1455_, v_as_1456_);
return v___x_1457_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_ofArray___boxed(lean_object* v_00_u03b1_1458_, lean_object* v_le_1459_, lean_object* v_as_1460_){
_start:
{
lean_object* v_res_1461_; 
v_res_1461_ = lp_batteries_Batteries_BinomialHeap_ofArray(v_00_u03b1_1458_, v_le_1459_, v_as_1460_);
lean_dec_ref(v_as_1460_);
return v_res_1461_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(lean_object* v_00_u03b1_1462_, lean_object* v_le_1463_, lean_object* v_as_1464_, size_t v_i_1465_, size_t v_stop_1466_, lean_object* v_b_1467_){
_start:
{
lean_object* v___x_1468_; 
v___x_1468_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___redArg(v_le_1463_, v_as_1464_, v_i_1465_, v_stop_1466_, v_b_1467_);
return v___x_1468_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0___boxed(lean_object* v_00_u03b1_1469_, lean_object* v_le_1470_, lean_object* v_as_1471_, lean_object* v_i_1472_, lean_object* v_stop_1473_, lean_object* v_b_1474_){
_start:
{
size_t v_i_boxed_1475_; size_t v_stop_boxed_1476_; lean_object* v_res_1477_; 
v_i_boxed_1475_ = lean_unbox_usize(v_i_1472_);
lean_dec(v_i_1472_);
v_stop_boxed_1476_ = lean_unbox_usize(v_stop_1473_);
lean_dec(v_stop_1473_);
v_res_1477_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_BinomialHeap_ofArray_spec__0(v_00_u03b1_1469_, v_le_1470_, v_as_1471_, v_i_boxed_1475_, v_stop_boxed_1476_, v_b_1474_);
lean_dec_ref(v_as_1471_);
return v_res_1477_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin___redArg(lean_object* v_le_1478_, lean_object* v_b_1479_){
_start:
{
lean_object* v___x_1480_; 
v___x_1480_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1478_, v_b_1479_);
if (lean_obj_tag(v___x_1480_) == 0)
{
lean_object* v___x_1481_; 
v___x_1481_ = lean_box(0);
return v___x_1481_;
}
else
{
lean_object* v_val_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1498_; 
v_val_1482_ = lean_ctor_get(v___x_1480_, 0);
v_isSharedCheck_1498_ = !lean_is_exclusive(v___x_1480_);
if (v_isSharedCheck_1498_ == 0)
{
v___x_1484_ = v___x_1480_;
v_isShared_1485_ = v_isSharedCheck_1498_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_val_1482_);
lean_dec(v___x_1480_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1498_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v_fst_1486_; lean_object* v_snd_1487_; lean_object* v___x_1489_; uint8_t v_isShared_1490_; uint8_t v_isSharedCheck_1497_; 
v_fst_1486_ = lean_ctor_get(v_val_1482_, 0);
v_snd_1487_ = lean_ctor_get(v_val_1482_, 1);
v_isSharedCheck_1497_ = !lean_is_exclusive(v_val_1482_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1489_ = v_val_1482_;
v_isShared_1490_ = v_isSharedCheck_1497_;
goto v_resetjp_1488_;
}
else
{
lean_inc(v_snd_1487_);
lean_inc(v_fst_1486_);
lean_dec(v_val_1482_);
v___x_1489_ = lean_box(0);
v_isShared_1490_ = v_isSharedCheck_1497_;
goto v_resetjp_1488_;
}
v_resetjp_1488_:
{
lean_object* v___x_1492_; 
if (v_isShared_1490_ == 0)
{
v___x_1492_ = v___x_1489_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_fst_1486_);
lean_ctor_set(v_reuseFailAlloc_1496_, 1, v_snd_1487_);
v___x_1492_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
lean_object* v___x_1494_; 
if (v_isShared_1485_ == 0)
{
lean_ctor_set(v___x_1484_, 0, v___x_1492_);
v___x_1494_ = v___x_1484_;
goto v_reusejp_1493_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v___x_1492_);
v___x_1494_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1493_;
}
v_reusejp_1493_:
{
return v___x_1494_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_deleteMin(lean_object* v_00_u03b1_1499_, lean_object* v_le_1500_, lean_object* v_b_1501_){
_start:
{
lean_object* v___x_1502_; 
v___x_1502_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1500_, v_b_1501_);
if (lean_obj_tag(v___x_1502_) == 0)
{
lean_object* v___x_1503_; 
v___x_1503_ = lean_box(0);
return v___x_1503_;
}
else
{
lean_object* v_val_1504_; lean_object* v___x_1506_; uint8_t v_isShared_1507_; uint8_t v_isSharedCheck_1520_; 
v_val_1504_ = lean_ctor_get(v___x_1502_, 0);
v_isSharedCheck_1520_ = !lean_is_exclusive(v___x_1502_);
if (v_isSharedCheck_1520_ == 0)
{
v___x_1506_ = v___x_1502_;
v_isShared_1507_ = v_isSharedCheck_1520_;
goto v_resetjp_1505_;
}
else
{
lean_inc(v_val_1504_);
lean_dec(v___x_1502_);
v___x_1506_ = lean_box(0);
v_isShared_1507_ = v_isSharedCheck_1520_;
goto v_resetjp_1505_;
}
v_resetjp_1505_:
{
lean_object* v_fst_1508_; lean_object* v_snd_1509_; lean_object* v___x_1511_; uint8_t v_isShared_1512_; uint8_t v_isSharedCheck_1519_; 
v_fst_1508_ = lean_ctor_get(v_val_1504_, 0);
v_snd_1509_ = lean_ctor_get(v_val_1504_, 1);
v_isSharedCheck_1519_ = !lean_is_exclusive(v_val_1504_);
if (v_isSharedCheck_1519_ == 0)
{
v___x_1511_ = v_val_1504_;
v_isShared_1512_ = v_isSharedCheck_1519_;
goto v_resetjp_1510_;
}
else
{
lean_inc(v_snd_1509_);
lean_inc(v_fst_1508_);
lean_dec(v_val_1504_);
v___x_1511_ = lean_box(0);
v_isShared_1512_ = v_isSharedCheck_1519_;
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
lean_object* v_reuseFailAlloc_1518_; 
v_reuseFailAlloc_1518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1518_, 0, v_fst_1508_);
lean_ctor_set(v_reuseFailAlloc_1518_, 1, v_snd_1509_);
v___x_1514_ = v_reuseFailAlloc_1518_;
goto v_reusejp_1513_;
}
v_reusejp_1513_:
{
lean_object* v___x_1516_; 
if (v_isShared_1507_ == 0)
{
lean_ctor_set(v___x_1506_, 0, v___x_1514_);
v___x_1516_ = v___x_1506_;
goto v_reusejp_1515_;
}
else
{
lean_object* v_reuseFailAlloc_1517_; 
v_reuseFailAlloc_1517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1517_, 0, v___x_1514_);
v___x_1516_ = v_reuseFailAlloc_1517_;
goto v_reusejp_1515_;
}
v_reusejp_1515_:
{
return v___x_1516_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream___redArg(lean_object* v_le_1521_){
_start:
{
lean_object* v___x_1522_; 
v___x_1522_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_deleteMin), 3, 2);
lean_closure_set(v___x_1522_, 0, lean_box(0));
lean_closure_set(v___x_1522_, 1, v_le_1521_);
return v___x_1522_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instStream(lean_object* v_00_u03b1_1523_, lean_object* v_le_1524_){
_start:
{
lean_object* v___x_1525_; 
v___x_1525_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_deleteMin), 3, 2);
lean_closure_set(v___x_1525_, 0, lean_box(0));
lean_closure_set(v___x_1525_, 1, v_le_1524_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0(lean_object* v_toPure_1526_, lean_object* v_f_1527_, lean_object* v_x_1528_, lean_object* v_a_1529_){
_start:
{
if (lean_obj_tag(v_x_1528_) == 0)
{
lean_object* v___x_1530_; 
lean_dec(v_a_1529_);
lean_dec(v_f_1527_);
v___x_1530_ = lean_apply_2(v_toPure_1526_, lean_box(0), v_x_1528_);
return v___x_1530_;
}
else
{
lean_object* v_a_1531_; lean_object* v___x_1532_; 
lean_dec(v_toPure_1526_);
v_a_1531_ = lean_ctor_get(v_x_1528_, 0);
lean_inc(v_a_1531_);
lean_dec_ref_known(v_x_1528_, 1);
v___x_1532_ = lean_apply_2(v_f_1527_, v_a_1529_, v_a_1531_);
return v___x_1532_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn___redArg(lean_object* v_le_1534_, lean_object* v_inst_1535_, lean_object* v_b_1536_, lean_object* v_x_1537_, lean_object* v_f_1538_){
_start:
{
lean_object* v_toApplicative_1539_; lean_object* v_toFunctor_1540_; lean_object* v_toPure_1541_; lean_object* v_map_1542_; lean_object* v___f_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; 
v_toApplicative_1539_ = lean_ctor_get(v_inst_1535_, 0);
v_toFunctor_1540_ = lean_ctor_get(v_toApplicative_1539_, 0);
v_toPure_1541_ = lean_ctor_get(v_toApplicative_1539_, 1);
v_map_1542_ = lean_ctor_get(v_toFunctor_1540_, 0);
lean_inc(v_map_1542_);
lean_inc(v_toPure_1541_);
v___f_1543_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_forIn___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1543_, 0, v_toPure_1541_);
lean_closure_set(v___f_1543_, 1, v_f_1538_);
v___x_1544_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_forIn___redArg___closed__0));
v___x_1545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1545_, 0, v_x_1537_);
v___x_1546_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v_inst_1535_, v_le_1534_, v_b_1536_, v___x_1545_, v___f_1543_);
v___x_1547_ = lean_apply_4(v_map_1542_, lean_box(0), lean_box(0), v___x_1544_, v___x_1546_);
return v___x_1547_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_forIn(lean_object* v_00_u03b1_1548_, lean_object* v_le_1549_, lean_object* v_m_1550_, lean_object* v_00_u03b2_1551_, lean_object* v_inst_1552_, lean_object* v_b_1553_, lean_object* v_x_1554_, lean_object* v_f_1555_){
_start:
{
lean_object* v___x_1556_; 
v___x_1556_ = lp_batteries_Batteries_BinomialHeap_forIn___redArg(v_le_1549_, v_inst_1552_, v_b_1553_, v_x_1554_, v_f_1555_);
return v___x_1556_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0(lean_object* v_le_1557_, lean_object* v_inst_1558_, lean_object* v_00_u03b2_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_){
_start:
{
lean_object* v___x_1563_; 
v___x_1563_ = lp_batteries_Batteries_BinomialHeap_forIn___redArg(v_le_1557_, v_inst_1558_, v___y_1560_, v___y_1561_, v___y_1562_);
return v___x_1563_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg(lean_object* v_le_1564_, lean_object* v_inst_1565_){
_start:
{
lean_object* v___f_1566_; 
v___f_1566_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0), 6, 2);
lean_closure_set(v___f_1566_, 0, v_le_1564_);
lean_closure_set(v___f_1566_, 1, v_inst_1565_);
return v___f_1566_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_instForInOfMonad(lean_object* v_00_u03b1_1567_, lean_object* v_le_1568_, lean_object* v_m_1569_, lean_object* v_inst_1570_){
_start:
{
lean_object* v___f_1571_; 
v___f_1571_ = lean_alloc_closure((void*)(lp_batteries_Batteries_BinomialHeap_instForInOfMonad___redArg___lam__0), 6, 2);
lean_closure_set(v___f_1571_, 0, v_le_1568_);
lean_closure_set(v___f_1571_, 1, v_inst_1570_);
return v___f_1571_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f___redArg(lean_object* v_le_1572_, lean_object* v_b_1573_){
_start:
{
if (lean_obj_tag(v_b_1573_) == 0)
{
lean_object* v___x_1574_; 
lean_dec_ref(v_le_1572_);
v___x_1574_ = lean_box(0);
return v___x_1574_;
}
else
{
lean_object* v_val_1575_; lean_object* v_next_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; 
v_val_1575_ = lean_ctor_get(v_b_1573_, 1);
lean_inc(v_val_1575_);
v_next_1576_ = lean_ctor_get(v_b_1573_, 3);
lean_inc(v_next_1576_);
lean_dec_ref_known(v_b_1573_, 4);
v___x_1577_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1572_, v_val_1575_, v_next_1576_);
v___x_1578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1578_, 0, v___x_1577_);
return v___x_1578_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x3f(lean_object* v_00_u03b1_1579_, lean_object* v_le_1580_, lean_object* v_b_1581_){
_start:
{
if (lean_obj_tag(v_b_1581_) == 0)
{
lean_object* v___x_1582_; 
lean_dec_ref(v_le_1580_);
v___x_1582_ = lean_box(0);
return v___x_1582_;
}
else
{
lean_object* v_val_1583_; lean_object* v_next_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; 
v_val_1583_ = lean_ctor_get(v_b_1581_, 1);
lean_inc(v_val_1583_);
v_next_1584_ = lean_ctor_get(v_b_1581_, 3);
lean_inc(v_next_1584_);
lean_dec_ref_known(v_b_1581_, 4);
v___x_1585_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1580_, v_val_1583_, v_next_1584_);
v___x_1586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1586_, 0, v___x_1585_);
return v___x_1586_;
}
}
}
static lean_object* _init_lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3(void){
_start:
{
lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; 
v___x_1590_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__2));
v___x_1591_ = lean_unsigned_to_nat(14u);
v___x_1592_ = lean_unsigned_to_nat(22u);
v___x_1593_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__1));
v___x_1594_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__0));
v___x_1595_ = l_mkPanicMessageWithDecl(v___x_1594_, v___x_1593_, v___x_1592_, v___x_1591_, v___x_1590_);
return v___x_1595_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg(lean_object* v_le_1596_, lean_object* v_inst_1597_, lean_object* v_b_1598_){
_start:
{
if (lean_obj_tag(v_b_1598_) == 0)
{
lean_object* v___x_1599_; lean_object* v___x_1600_; 
lean_dec_ref(v_le_1596_);
v___x_1599_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3);
v___x_1600_ = l_panic___redArg(v_inst_1597_, v___x_1599_);
return v___x_1600_;
}
else
{
lean_object* v_val_1601_; lean_object* v_next_1602_; lean_object* v___x_1603_; 
v_val_1601_ = lean_ctor_get(v_b_1598_, 1);
lean_inc(v_val_1601_);
v_next_1602_ = lean_ctor_get(v_b_1598_, 3);
lean_inc(v_next_1602_);
lean_dec_ref_known(v_b_1598_, 4);
v___x_1603_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1596_, v_val_1601_, v_next_1602_);
return v___x_1603_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___redArg___boxed(lean_object* v_le_1604_, lean_object* v_inst_1605_, lean_object* v_b_1606_){
_start:
{
lean_object* v_res_1607_; 
v_res_1607_ = lp_batteries_Batteries_BinomialHeap_head_x21___redArg(v_le_1604_, v_inst_1605_, v_b_1606_);
lean_dec(v_inst_1605_);
return v_res_1607_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21(lean_object* v_00_u03b1_1608_, lean_object* v_le_1609_, lean_object* v_inst_1610_, lean_object* v_b_1611_){
_start:
{
if (lean_obj_tag(v_b_1611_) == 0)
{
lean_object* v___x_1612_; lean_object* v___x_1613_; 
lean_dec_ref(v_le_1609_);
v___x_1612_ = lean_obj_once(&lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3, &lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3_once, _init_lp_batteries_Batteries_BinomialHeap_head_x21___redArg___closed__3);
v___x_1613_ = l_panic___redArg(v_inst_1610_, v___x_1612_);
return v___x_1613_;
}
else
{
lean_object* v_val_1614_; lean_object* v_next_1615_; lean_object* v___x_1616_; 
v_val_1614_ = lean_ctor_get(v_b_1611_, 1);
lean_inc(v_val_1614_);
v_next_1615_ = lean_ctor_get(v_b_1611_, 3);
lean_inc(v_next_1615_);
lean_dec_ref_known(v_b_1611_, 4);
v___x_1616_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1609_, v_val_1614_, v_next_1615_);
return v___x_1616_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_head_x21___boxed(lean_object* v_00_u03b1_1617_, lean_object* v_le_1618_, lean_object* v_inst_1619_, lean_object* v_b_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_batteries_Batteries_BinomialHeap_head_x21(v_00_u03b1_1617_, v_le_1618_, v_inst_1619_, v_b_1620_);
lean_dec(v_inst_1619_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg(lean_object* v_le_1622_, lean_object* v_inst_1623_, lean_object* v_b_1624_){
_start:
{
if (lean_obj_tag(v_b_1624_) == 0)
{
lean_dec_ref(v_le_1622_);
lean_inc(v_inst_1623_);
return v_inst_1623_;
}
else
{
lean_object* v_val_1625_; lean_object* v_next_1626_; lean_object* v___x_1627_; 
v_val_1625_ = lean_ctor_get(v_b_1624_, 1);
lean_inc(v_val_1625_);
v_next_1626_ = lean_ctor_get(v_b_1624_, 3);
lean_inc(v_next_1626_);
lean_dec_ref_known(v_b_1624_, 4);
v___x_1627_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1622_, v_val_1625_, v_next_1626_);
return v___x_1627_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___redArg___boxed(lean_object* v_le_1628_, lean_object* v_inst_1629_, lean_object* v_b_1630_){
_start:
{
lean_object* v_res_1631_; 
v_res_1631_ = lp_batteries_Batteries_BinomialHeap_headI___redArg(v_le_1628_, v_inst_1629_, v_b_1630_);
lean_dec(v_inst_1629_);
return v_res_1631_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI(lean_object* v_00_u03b1_1632_, lean_object* v_le_1633_, lean_object* v_inst_1634_, lean_object* v_b_1635_){
_start:
{
if (lean_obj_tag(v_b_1635_) == 0)
{
lean_dec_ref(v_le_1633_);
lean_inc(v_inst_1634_);
return v_inst_1634_;
}
else
{
lean_object* v_val_1636_; lean_object* v_next_1637_; lean_object* v___x_1638_; 
v_val_1636_ = lean_ctor_get(v_b_1635_, 1);
lean_inc(v_val_1636_);
v_next_1637_ = lean_ctor_get(v_b_1635_, 3);
lean_inc(v_next_1637_);
lean_dec_ref_known(v_b_1635_, 4);
v___x_1638_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_headD___redArg(v_le_1633_, v_val_1636_, v_next_1637_);
return v___x_1638_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_headI___boxed(lean_object* v_00_u03b1_1639_, lean_object* v_le_1640_, lean_object* v_inst_1641_, lean_object* v_b_1642_){
_start:
{
lean_object* v_res_1643_; 
v_res_1643_ = lp_batteries_Batteries_BinomialHeap_headI(v_00_u03b1_1639_, v_le_1640_, v_inst_1641_, v_b_1642_);
lean_dec(v_inst_1641_);
return v_res_1643_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f___redArg(lean_object* v_le_1644_, lean_object* v_b_1645_){
_start:
{
lean_object* v___x_1646_; 
v___x_1646_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1644_, v_b_1645_);
if (lean_obj_tag(v___x_1646_) == 0)
{
lean_object* v___x_1647_; 
v___x_1647_ = lean_box(0);
return v___x_1647_;
}
else
{
lean_object* v_val_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1656_; 
v_val_1648_ = lean_ctor_get(v___x_1646_, 0);
v_isSharedCheck_1656_ = !lean_is_exclusive(v___x_1646_);
if (v_isSharedCheck_1656_ == 0)
{
v___x_1650_ = v___x_1646_;
v_isShared_1651_ = v_isSharedCheck_1656_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_val_1648_);
lean_dec(v___x_1646_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1656_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v_snd_1652_; lean_object* v___x_1654_; 
v_snd_1652_ = lean_ctor_get(v_val_1648_, 1);
lean_inc(v_snd_1652_);
lean_dec(v_val_1648_);
if (v_isShared_1651_ == 0)
{
lean_ctor_set(v___x_1650_, 0, v_snd_1652_);
v___x_1654_ = v___x_1650_;
goto v_reusejp_1653_;
}
else
{
lean_object* v_reuseFailAlloc_1655_; 
v_reuseFailAlloc_1655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1655_, 0, v_snd_1652_);
v___x_1654_ = v_reuseFailAlloc_1655_;
goto v_reusejp_1653_;
}
v_reusejp_1653_:
{
return v___x_1654_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail_x3f(lean_object* v_00_u03b1_1657_, lean_object* v_le_1658_, lean_object* v_b_1659_){
_start:
{
lean_object* v___x_1660_; 
v___x_1660_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1658_, v_b_1659_);
if (lean_obj_tag(v___x_1660_) == 0)
{
lean_object* v___x_1661_; 
v___x_1661_ = lean_box(0);
return v___x_1661_;
}
else
{
lean_object* v_val_1662_; lean_object* v___x_1664_; uint8_t v_isShared_1665_; uint8_t v_isSharedCheck_1670_; 
v_val_1662_ = lean_ctor_get(v___x_1660_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1660_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1664_ = v___x_1660_;
v_isShared_1665_ = v_isSharedCheck_1670_;
goto v_resetjp_1663_;
}
else
{
lean_inc(v_val_1662_);
lean_dec(v___x_1660_);
v___x_1664_ = lean_box(0);
v_isShared_1665_ = v_isSharedCheck_1670_;
goto v_resetjp_1663_;
}
v_resetjp_1663_:
{
lean_object* v_snd_1666_; lean_object* v___x_1668_; 
v_snd_1666_ = lean_ctor_get(v_val_1662_, 1);
lean_inc(v_snd_1666_);
lean_dec(v_val_1662_);
if (v_isShared_1665_ == 0)
{
lean_ctor_set(v___x_1664_, 0, v_snd_1666_);
v___x_1668_ = v___x_1664_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v_snd_1666_);
v___x_1668_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
return v___x_1668_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail___redArg(lean_object* v_le_1671_, lean_object* v_b_1672_){
_start:
{
lean_object* v___x_1673_; 
v___x_1673_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1671_, v_b_1672_);
if (lean_obj_tag(v___x_1673_) == 0)
{
lean_object* v___x_1674_; 
v___x_1674_ = lean_box(0);
return v___x_1674_;
}
else
{
lean_object* v_val_1675_; lean_object* v_snd_1676_; 
v_val_1675_ = lean_ctor_get(v___x_1673_, 0);
lean_inc(v_val_1675_);
lean_dec_ref_known(v___x_1673_, 1);
v_snd_1676_ = lean_ctor_get(v_val_1675_, 1);
lean_inc(v_snd_1676_);
lean_dec(v_val_1675_);
return v_snd_1676_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_tail(lean_object* v_00_u03b1_1677_, lean_object* v_le_1678_, lean_object* v_b_1679_){
_start:
{
lean_object* v___x_1680_; 
v___x_1680_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v_le_1678_, v_b_1679_);
if (lean_obj_tag(v___x_1680_) == 0)
{
lean_object* v___x_1681_; 
v___x_1681_ = lean_box(0);
return v___x_1681_;
}
else
{
lean_object* v_val_1682_; lean_object* v_snd_1683_; 
v_val_1682_ = lean_ctor_get(v___x_1680_, 0);
lean_inc(v_val_1682_);
lean_dec_ref_known(v___x_1680_, 1);
v_snd_1683_ = lean_ctor_get(v_val_1682_, 1);
lean_inc(v_snd_1683_);
lean_dec(v_val_1682_);
return v_snd_1683_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM___redArg(lean_object* v_le_1684_, lean_object* v_inst_1685_, lean_object* v_b_1686_, lean_object* v_init_1687_, lean_object* v_f_1688_){
_start:
{
lean_object* v___x_1689_; 
v___x_1689_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v_inst_1685_, v_le_1684_, v_b_1686_, v_init_1687_, v_f_1688_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_foldM(lean_object* v_00_u03b1_1690_, lean_object* v_le_1691_, lean_object* v_m_1692_, lean_object* v_00_u03b2_1693_, lean_object* v_inst_1694_, lean_object* v_b_1695_, lean_object* v_init_1696_, lean_object* v_f_1697_){
_start:
{
lean_object* v___x_1698_; 
v___x_1698_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v_inst_1694_, v_le_1691_, v_b_1695_, v_init_1696_, v_f_1697_);
return v___x_1698_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold___redArg(lean_object* v_le_1699_, lean_object* v_b_1700_, lean_object* v_init_1701_, lean_object* v_f_1702_){
_start:
{
lean_object* v___x_1703_; lean_object* v___x_1704_; 
v___x_1703_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1704_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1703_, v_le_1699_, v_b_1700_, v_init_1701_, v_f_1702_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_fold(lean_object* v_00_u03b1_1705_, lean_object* v_le_1706_, lean_object* v_00_u03b2_1707_, lean_object* v_b_1708_, lean_object* v_init_1709_, lean_object* v_f_1710_){
_start:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; 
v___x_1711_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1712_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1711_, v_le_1706_, v_b_1708_, v_init_1709_, v_f_1710_);
return v___x_1712_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList___redArg(lean_object* v_le_1713_, lean_object* v_b_1714_){
_start:
{
lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; 
v___x_1715_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1716_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1717_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1718_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1717_, v_le_1713_, v_b_1714_, v___x_1715_, v___x_1716_);
v___x_1719_ = lean_array_to_list(v___x_1718_);
return v___x_1719_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toList(lean_object* v_00_u03b1_1720_, lean_object* v_le_1721_, lean_object* v_b_1722_){
_start:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1723_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1724_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1725_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1726_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1725_, v_le_1721_, v_b_1722_, v___x_1723_, v___x_1724_);
v___x_1727_ = lean_array_to_list(v___x_1726_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray___redArg(lean_object* v_le_1728_, lean_object* v_b_1729_){
_start:
{
lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1730_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1731_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1732_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1733_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1732_, v_le_1728_, v_b_1729_, v___x_1730_, v___x_1731_);
return v___x_1733_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArray(lean_object* v_00_u03b1_1734_, lean_object* v_le_1735_, lean_object* v_b_1736_){
_start:
{
lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; 
v___x_1737_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__0));
v___x_1738_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArray___redArg___closed__1));
v___x_1739_ = ((lean_object*)(lp_batteries_Batteries_BinomialHeap_Imp_Heap_fold___redArg___closed__9));
v___x_1740_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_foldM___redArg(v___x_1739_, v_le_1735_, v_b_1736_, v___x_1737_, v___x_1738_);
return v___x_1740_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(lean_object* v_b_1741_){
_start:
{
lean_object* v___x_1742_; 
v___x_1742_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(v_b_1741_);
return v___x_1742_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg___boxed(lean_object* v_b_1743_){
_start:
{
lean_object* v_res_1744_; 
v_res_1744_ = lp_batteries_Batteries_BinomialHeap_toListUnordered___redArg(v_b_1743_);
lean_dec(v_b_1743_);
return v_res_1744_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered(lean_object* v_00_u03b1_1745_, lean_object* v_le_1746_, lean_object* v_b_1747_){
_start:
{
lean_object* v___x_1748_; 
v___x_1748_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toListUnordered___redArg(v_b_1747_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toListUnordered___boxed(lean_object* v_00_u03b1_1749_, lean_object* v_le_1750_, lean_object* v_b_1751_){
_start:
{
lean_object* v_res_1752_; 
v_res_1752_ = lp_batteries_Batteries_BinomialHeap_toListUnordered(v_00_u03b1_1749_, v_le_1750_, v_b_1751_);
lean_dec(v_b_1751_);
lean_dec_ref(v_le_1750_);
return v_res_1752_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___redArg(lean_object* v_b_1753_){
_start:
{
lean_object* v___x_1754_; 
v___x_1754_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(v_b_1753_);
return v___x_1754_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered(lean_object* v_00_u03b1_1755_, lean_object* v_le_1756_, lean_object* v_b_1757_){
_start:
{
lean_object* v___x_1758_; 
v___x_1758_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_toArrayUnordered___redArg(v_b_1757_);
return v___x_1758_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_BinomialHeap_toArrayUnordered___boxed(lean_object* v_00_u03b1_1759_, lean_object* v_le_1760_, lean_object* v_b_1761_){
_start:
{
lean_object* v_res_1762_; 
v_res_1762_ = lp_batteries_Batteries_BinomialHeap_toArrayUnordered(v_00_u03b1_1759_, v_le_1760_, v_b_1761_);
lean_dec_ref(v_le_1760_);
return v_res_1762_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Classes_Order(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Classes_Order(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
