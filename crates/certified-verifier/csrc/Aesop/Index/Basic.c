// Lean compiler output
// Module: Aesop.Index.Basic
// Imports: public import Init public meta import Init public import Aesop.Forward.Substitution
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
lean_object* l_Lean_LocalDecl_index(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
lean_object* l_Lean_Meta_DiscrTree_Key_format(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lp_aesop_Aesop_getConclusionDiscrTreeKeys(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_unindexed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_unindexed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_target_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_target_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hyps_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hyps_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_or_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_or_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexingMode_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexingMode;
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__0 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__0_value;
static const lean_ctor_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__0_value)}};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__1 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__1_value;
static const lean_string_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__2 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__2_value;
static const lean_ctor_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__2_value)}};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__3 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__3_value;
static const lean_ctor_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__4 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__4_value;
static const lean_string_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__5 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__5_value;
static const lean_string_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__6 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__6_value;
static lean_once_cell_t lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7;
static lean_once_cell_t lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8;
static const lean_ctor_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__5_value)}};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__9 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__9_value;
static const lean_ctor_object lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__6_value)}};
static const lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__10 = (const lean_object*)&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__2(lean_object*);
static const lean_string_object lp_aesop_Aesop_IndexingMode_format___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "unindexed"};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_IndexingMode_format___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__1 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__1_value;
static const lean_string_object lp_aesop_Aesop_IndexingMode_format___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "target "};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__2 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_IndexingMode_format___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__3 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__3_value;
static const lean_string_object lp_aesop_Aesop_IndexingMode_format___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__4 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_IndexingMode_format___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__5 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__5_value;
static const lean_string_object lp_aesop_Aesop_IndexingMode_format___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hyps "};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__6 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_IndexingMode_format___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__7 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__7_value;
static const lean_string_object lp_aesop_Aesop_IndexingMode_format___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "or "};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__8 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_IndexingMode_format___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_IndexingMode_format___closed__9 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_format___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_format(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Nat_cast___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__1(lean_object*);
static const lean_closure_object lp_aesop_Aesop_IndexingMode_instToFormat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_IndexingMode_format, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_IndexingMode_instToFormat___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexingMode_instToFormat___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_IndexingMode_instToFormat = (const lean_object*)&lp_aesop_Aesop_IndexingMode_instToFormat___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_targetMatchingConclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_targetMatchingConclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_none_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_none_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_target_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_target_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_hyp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_hyp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchLocation_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchLocation;
static const lean_string_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2;
static const lean_string_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "target"};
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5;
static const lean_string_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hyp "};
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_IndexMatchLocation_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_IndexMatchLocation_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_IndexMatchLocation_instBEq = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instBEq___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_IndexMatchLocation_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_IndexMatchLocation_instOrd = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instOrd___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_IndexMatchLocation_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_IndexMatchLocation_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_IndexMatchLocation_instHashable = (const lean_object*)&lp_aesop_Aesop_IndexMatchLocation_instHashable___closed__0_value;
static const lean_array_object lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instLTOfOrd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instLTOfOrd___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
default: 
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorIdx___boxed(lean_object* v_x_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_aesop_Aesop_IndexingMode_ctorIdx(v_x_6_);
lean_dec(v_x_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim___redArg(lean_object* v_t_8_, lean_object* v_k_9_){
_start:
{
if (lean_obj_tag(v_t_8_) == 0)
{
return v_k_9_;
}
else
{
lean_object* v_keys_10_; lean_object* v___x_11_; 
v_keys_10_ = lean_ctor_get(v_t_8_, 0);
lean_inc_ref(v_keys_10_);
lean_dec(v_t_8_);
v___x_11_ = lean_apply_1(v_k_9_, v_keys_10_);
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim(lean_object* v_motive__1_12_, lean_object* v_ctorIdx_13_, lean_object* v_t_14_, lean_object* v_h_15_, lean_object* v_k_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_14_, v_k_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_ctorElim___boxed(lean_object* v_motive__1_18_, lean_object* v_ctorIdx_19_, lean_object* v_t_20_, lean_object* v_h_21_, lean_object* v_k_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_aesop_Aesop_IndexingMode_ctorElim(v_motive__1_18_, v_ctorIdx_19_, v_t_20_, v_h_21_, v_k_22_);
lean_dec(v_ctorIdx_19_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_unindexed_elim___redArg(lean_object* v_t_24_, lean_object* v_unindexed_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_24_, v_unindexed_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_unindexed_elim(lean_object* v_motive__1_27_, lean_object* v_t_28_, lean_object* v_h_29_, lean_object* v_unindexed_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_28_, v_unindexed_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_target_elim___redArg(lean_object* v_t_32_, lean_object* v_target_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_32_, v_target_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_target_elim(lean_object* v_motive__1_35_, lean_object* v_t_36_, lean_object* v_h_37_, lean_object* v_target_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_36_, v_target_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hyps_elim___redArg(lean_object* v_t_40_, lean_object* v_hyps_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_40_, v_hyps_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hyps_elim(lean_object* v_motive__1_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_hyps_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_44_, v_hyps_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_or_elim___redArg(lean_object* v_t_48_, lean_object* v_or_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_48_, v_or_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_or_elim(lean_object* v_motive__1_51_, lean_object* v_t_52_, lean_object* v_h_53_, lean_object* v_or_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_aesop_Aesop_IndexingMode_ctorElim___redArg(v_t_52_, v_or_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndexingMode_default(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_box(0);
return v___x_56_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndexingMode(void){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_box(0);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0_spec__1(lean_object* v_x_58_, lean_object* v_x_59_, lean_object* v_x_60_){
_start:
{
if (lean_obj_tag(v_x_60_) == 0)
{
lean_dec(v_x_58_);
return v_x_59_;
}
else
{
lean_object* v_head_61_; lean_object* v_tail_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_72_; 
v_head_61_ = lean_ctor_get(v_x_60_, 0);
v_tail_62_ = lean_ctor_get(v_x_60_, 1);
v_isSharedCheck_72_ = !lean_is_exclusive(v_x_60_);
if (v_isSharedCheck_72_ == 0)
{
v___x_64_ = v_x_60_;
v_isShared_65_ = v_isSharedCheck_72_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_tail_62_);
lean_inc(v_head_61_);
lean_dec(v_x_60_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_72_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_67_; 
lean_inc(v_x_58_);
if (v_isShared_65_ == 0)
{
lean_ctor_set_tag(v___x_64_, 5);
lean_ctor_set(v___x_64_, 1, v_x_58_);
lean_ctor_set(v___x_64_, 0, v_x_59_);
v___x_67_ = v___x_64_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v_x_59_);
lean_ctor_set(v_reuseFailAlloc_71_, 1, v_x_58_);
v___x_67_ = v_reuseFailAlloc_71_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = l_Lean_Meta_DiscrTree_Key_format(v_head_61_);
v___x_69_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_67_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
v_x_59_ = v___x_69_;
v_x_60_ = v_tail_62_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0(lean_object* v_x_73_, lean_object* v_x_74_){
_start:
{
if (lean_obj_tag(v_x_73_) == 0)
{
lean_object* v___x_75_; 
lean_dec(v_x_74_);
v___x_75_ = lean_box(0);
return v___x_75_;
}
else
{
lean_object* v_tail_76_; 
v_tail_76_ = lean_ctor_get(v_x_73_, 1);
if (lean_obj_tag(v_tail_76_) == 0)
{
lean_object* v_head_77_; lean_object* v___x_78_; 
lean_dec(v_x_74_);
v_head_77_ = lean_ctor_get(v_x_73_, 0);
lean_inc(v_head_77_);
lean_dec_ref_known(v_x_73_, 2);
v___x_78_ = l_Lean_Meta_DiscrTree_Key_format(v_head_77_);
return v___x_78_;
}
else
{
lean_object* v_head_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
lean_inc(v_tail_76_);
v_head_79_ = lean_ctor_get(v_x_73_, 0);
lean_inc(v_head_79_);
lean_dec_ref_known(v_x_73_, 2);
v___x_80_ = l_Lean_Meta_DiscrTree_Key_format(v_head_79_);
v___x_81_ = lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0_spec__1(v_x_74_, v___x_80_, v_tail_76_);
return v___x_81_;
}
}
}
}
static lean_object* _init_lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7(void){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__5));
v___x_94_ = lean_string_length(v___x_93_);
return v___x_94_;
}
}
static lean_object* _init_lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = lean_obj_once(&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7, &lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7_once, _init_lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__7);
v___x_96_ = lean_nat_to_int(v___x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0(lean_object* v_x_101_){
_start:
{
if (lean_obj_tag(v_x_101_) == 0)
{
lean_object* v___x_102_; 
v___x_102_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__1));
return v___x_102_;
}
else
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; lean_object* v___x_112_; 
v___x_103_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__4));
v___x_104_ = lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__0(v_x_101_, v___x_103_);
v___x_105_ = lean_obj_once(&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8, &lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8_once, _init_lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8);
v___x_106_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__9));
v___x_107_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_104_);
v___x_108_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__10));
v___x_109_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_107_);
lean_ctor_set(v___x_109_, 1, v___x_108_);
v___x_110_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_105_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = 0;
v___x_112_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_112_, 0, v___x_110_);
lean_ctor_set_uint8(v___x_112_, sizeof(void*)*1, v___x_111_);
return v___x_112_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4_spec__6(lean_object* v_x_113_, lean_object* v_x_114_, lean_object* v_x_115_){
_start:
{
if (lean_obj_tag(v_x_115_) == 0)
{
lean_dec(v_x_113_);
return v_x_114_;
}
else
{
lean_object* v_head_116_; lean_object* v_tail_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_126_; 
v_head_116_ = lean_ctor_get(v_x_115_, 0);
v_tail_117_ = lean_ctor_get(v_x_115_, 1);
v_isSharedCheck_126_ = !lean_is_exclusive(v_x_115_);
if (v_isSharedCheck_126_ == 0)
{
v___x_119_ = v_x_115_;
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_tail_117_);
lean_inc(v_head_116_);
lean_dec(v_x_115_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_122_; 
lean_inc(v_x_113_);
if (v_isShared_120_ == 0)
{
lean_ctor_set_tag(v___x_119_, 5);
lean_ctor_set(v___x_119_, 1, v_x_113_);
lean_ctor_set(v___x_119_, 0, v_x_114_);
v___x_122_ = v___x_119_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v_x_114_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v_x_113_);
v___x_122_ = v_reuseFailAlloc_125_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
lean_object* v___x_123_; 
v___x_123_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_head_116_);
v_x_114_ = v___x_123_;
v_x_115_ = v_tail_117_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4(lean_object* v_x_127_, lean_object* v_x_128_){
_start:
{
if (lean_obj_tag(v_x_127_) == 0)
{
lean_object* v___x_129_; 
lean_dec(v_x_128_);
v___x_129_ = lean_box(0);
return v___x_129_;
}
else
{
lean_object* v_tail_130_; 
v_tail_130_ = lean_ctor_get(v_x_127_, 1);
if (lean_obj_tag(v_tail_130_) == 0)
{
lean_object* v_head_131_; 
lean_dec(v_x_128_);
v_head_131_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_head_131_);
lean_dec_ref_known(v_x_127_, 2);
return v_head_131_;
}
else
{
lean_object* v_head_132_; lean_object* v___x_133_; 
lean_inc(v_tail_130_);
v_head_132_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_head_132_);
lean_dec_ref_known(v_x_127_, 2);
v___x_133_ = lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4_spec__6(v_x_128_, v_head_132_, v_tail_130_);
return v___x_133_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__2(lean_object* v_x_134_){
_start:
{
if (lean_obj_tag(v_x_134_) == 0)
{
lean_object* v___x_135_; 
v___x_135_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__1));
return v___x_135_;
}
else
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; uint8_t v___x_144_; lean_object* v___x_145_; 
v___x_136_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__4));
v___x_137_ = lp_aesop_Std_Format_joinSep___at___00List_format___at___00Aesop_IndexingMode_format_spec__2_spec__4(v_x_134_, v___x_136_);
v___x_138_ = lean_obj_once(&lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8, &lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8_once, _init_lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__8);
v___x_139_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__9));
v___x_140_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v___x_137_);
v___x_141_ = ((lean_object*)(lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0___closed__10));
v___x_142_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_140_);
lean_ctor_set(v___x_142_, 1, v___x_141_);
v___x_143_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_138_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = 0;
v___x_145_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_145_, 0, v___x_143_);
lean_ctor_set_uint8(v___x_145_, sizeof(void*)*1, v___x_144_);
return v___x_145_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_format(lean_object* v_x_161_){
_start:
{
switch(lean_obj_tag(v_x_161_))
{
case 0:
{
lean_object* v___x_162_; 
v___x_162_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__1));
return v___x_162_;
}
case 1:
{
lean_object* v_keys_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v_keys_163_ = lean_ctor_get(v_x_161_, 0);
lean_inc_ref(v_keys_163_);
lean_dec_ref_known(v_x_161_, 1);
v___x_164_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__3));
v___x_165_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__5));
v___x_166_ = lean_array_to_list(v_keys_163_);
v___x_167_ = lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0(v___x_166_);
v___x_168_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_165_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
v___x_169_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_164_);
lean_ctor_set(v___x_169_, 1, v___x_168_);
return v___x_169_;
}
case 2:
{
lean_object* v_keys_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_keys_170_ = lean_ctor_get(v_x_161_, 0);
lean_inc_ref(v_keys_170_);
lean_dec_ref_known(v_x_161_, 1);
v___x_171_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__7));
v___x_172_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__5));
v___x_173_ = lean_array_to_list(v_keys_170_);
v___x_174_ = lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__0(v___x_173_);
v___x_175_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_172_);
lean_ctor_set(v___x_175_, 1, v___x_174_);
v___x_176_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_171_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
return v___x_176_;
}
default: 
{
lean_object* v_imodes_177_; lean_object* v___x_178_; size_t v_sz_179_; size_t v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v_imodes_177_ = lean_ctor_get(v_x_161_, 0);
lean_inc_ref(v_imodes_177_);
lean_dec_ref_known(v_x_161_, 1);
v___x_178_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__9));
v_sz_179_ = lean_array_size(v_imodes_177_);
v___x_180_ = ((size_t)0ULL);
v___x_181_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1(v_sz_179_, v___x_180_, v_imodes_177_);
v___x_182_ = ((lean_object*)(lp_aesop_Aesop_IndexingMode_format___closed__5));
v___x_183_ = lean_array_to_list(v___x_181_);
v___x_184_ = lp_aesop_List_format___at___00Aesop_IndexingMode_format_spec__2(v___x_183_);
v___x_185_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_182_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
v___x_186_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_178_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
return v___x_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1(size_t v_sz_187_, size_t v_i_188_, lean_object* v_bs_189_){
_start:
{
uint8_t v___x_190_; 
v___x_190_ = lean_usize_dec_lt(v_i_188_, v_sz_187_);
if (v___x_190_ == 0)
{
return v_bs_189_;
}
else
{
lean_object* v_v_191_; lean_object* v___x_192_; lean_object* v_bs_x27_193_; lean_object* v___x_194_; size_t v___x_195_; size_t v___x_196_; lean_object* v___x_197_; 
v_v_191_ = lean_array_uget(v_bs_189_, v_i_188_);
v___x_192_ = lean_unsigned_to_nat(0u);
v_bs_x27_193_ = lean_array_uset(v_bs_189_, v_i_188_, v___x_192_);
v___x_194_ = lp_aesop_Aesop_IndexingMode_format(v_v_191_);
v___x_195_ = ((size_t)1ULL);
v___x_196_ = lean_usize_add(v_i_188_, v___x_195_);
v___x_197_ = lean_array_uset(v_bs_x27_193_, v_i_188_, v___x_194_);
v_i_188_ = v___x_196_;
v_bs_189_ = v___x_197_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1___boxed(lean_object* v_sz_199_, lean_object* v_i_200_, lean_object* v_bs_201_){
_start:
{
size_t v_sz_boxed_202_; size_t v_i_boxed_203_; lean_object* v_res_204_; 
v_sz_boxed_202_ = lean_unbox_usize(v_sz_199_);
lean_dec(v_sz_199_);
v_i_boxed_203_ = lean_unbox_usize(v_i_200_);
lean_dec(v_i_200_);
v_res_204_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_IndexingMode_format_spec__1(v_sz_boxed_202_, v_i_boxed_203_, v_bs_201_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Nat_cast___at___00List_format___at___00Aesop_IndexingMode_format_spec__0_spec__1(lean_object* v_a_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_nat_to_int(v_a_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_targetMatchingConclusion(lean_object* v_type_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_aesop_Aesop_getConclusionDiscrTreeKeys(v_type_209_, v_a_210_, v_a_211_, v_a_212_, v_a_213_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_224_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_224_ == 0)
{
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_220_; lean_object* v___x_222_; 
v___x_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_220_, 0, v_a_216_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_220_);
v___x_222_ = v___x_218_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_220_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
else
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_232_; 
v_a_225_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_232_ == 0)
{
v___x_227_ = v___x_215_;
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_215_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_230_; 
if (v_isShared_228_ == 0)
{
v___x_230_ = v___x_227_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_225_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_targetMatchingConclusion___boxed(lean_object* v_type_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_, lean_object* v_a_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_aesop_Aesop_IndexingMode_targetMatchingConclusion(v_type_233_, v_a_234_, v_a_235_, v_a_236_, v_a_237_);
lean_dec(v_a_237_);
lean_dec_ref(v_a_236_);
lean_dec(v_a_235_);
lean_dec_ref(v_a_234_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg(lean_object* v_x_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = l_Lean_Meta_saveState___redArg(v___y_242_, v___y_244_);
if (lean_obj_tag(v___x_246_) == 0)
{
lean_object* v_a_247_; lean_object* v_r_248_; 
v_a_247_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_a_247_);
lean_dec_ref_known(v___x_246_, 1);
lean_inc(v___y_244_);
lean_inc_ref(v___y_243_);
lean_inc(v___y_242_);
lean_inc_ref(v___y_241_);
v_r_248_ = lean_apply_5(v_x_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_, lean_box(0));
if (lean_obj_tag(v_r_248_) == 0)
{
lean_object* v_a_249_; lean_object* v___x_250_; 
v_a_249_ = lean_ctor_get(v_r_248_, 0);
lean_inc(v_a_249_);
lean_dec_ref_known(v_r_248_, 1);
v___x_250_ = l_Lean_Meta_SavedState_restore___redArg(v_a_247_, v___y_242_, v___y_244_);
lean_dec(v_a_247_);
if (lean_obj_tag(v___x_250_) == 0)
{
lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_257_; 
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_257_ == 0)
{
lean_object* v_unused_258_; 
v_unused_258_ = lean_ctor_get(v___x_250_, 0);
lean_dec(v_unused_258_);
v___x_252_ = v___x_250_;
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
else
{
lean_dec(v___x_250_);
v___x_252_ = lean_box(0);
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
v_resetjp_251_:
{
lean_object* v___x_255_; 
if (v_isShared_253_ == 0)
{
lean_ctor_set(v___x_252_, 0, v_a_249_);
v___x_255_ = v___x_252_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v_a_249_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
else
{
lean_object* v_a_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_266_; 
lean_dec(v_a_249_);
v_a_259_ = lean_ctor_get(v___x_250_, 0);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_266_ == 0)
{
v___x_261_ = v___x_250_;
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_a_259_);
lean_dec(v___x_250_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_264_; 
if (v_isShared_262_ == 0)
{
v___x_264_ = v___x_261_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_a_259_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
else
{
lean_object* v_a_267_; lean_object* v___x_268_; 
v_a_267_ = lean_ctor_get(v_r_248_, 0);
lean_inc(v_a_267_);
lean_dec_ref_known(v_r_248_, 1);
v___x_268_ = l_Lean_Meta_SavedState_restore___redArg(v_a_247_, v___y_242_, v___y_244_);
lean_dec(v_a_247_);
if (lean_obj_tag(v___x_268_) == 0)
{
lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_275_ == 0)
{
lean_object* v_unused_276_; 
v_unused_276_ = lean_ctor_get(v___x_268_, 0);
lean_dec(v_unused_276_);
v___x_270_ = v___x_268_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_dec(v___x_268_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
lean_ctor_set_tag(v___x_270_, 1);
lean_ctor_set(v___x_270_, 0, v_a_267_);
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_267_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
else
{
lean_object* v_a_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_284_; 
lean_dec(v_a_267_);
v_a_277_ = lean_ctor_get(v___x_268_, 0);
v_isSharedCheck_284_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_284_ == 0)
{
v___x_279_ = v___x_268_;
v_isShared_280_ = v_isSharedCheck_284_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_a_277_);
lean_dec(v___x_268_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_284_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v___x_282_; 
if (v_isShared_280_ == 0)
{
v___x_282_ = v___x_279_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v_a_277_);
v___x_282_ = v_reuseFailAlloc_283_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
return v___x_282_;
}
}
}
}
}
else
{
lean_object* v_a_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_292_; 
lean_dec_ref(v_x_240_);
v_a_285_ = lean_ctor_get(v___x_246_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_246_);
if (v_isSharedCheck_292_ == 0)
{
v___x_287_ = v___x_246_;
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_a_285_);
lean_dec(v___x_246_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v_a_285_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg___boxed(lean_object* v_x_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg(v_x_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0(lean_object* v_00_u03b1_300_, lean_object* v_x_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg(v_x_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___boxed(lean_object* v_00_u03b1_308_, lean_object* v_x_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0(v_00_u03b1_308_, v_x_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0(uint8_t v___x_316_, lean_object* v_decl_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_){
_start:
{
lean_object* v_keyedConfig_323_; uint8_t v_trackZetaDelta_324_; lean_object* v_zetaDeltaSet_325_; lean_object* v_lctx_326_; lean_object* v_localInstances_327_; lean_object* v_defEqCtx_x3f_328_; lean_object* v_synthPendingDepth_329_; lean_object* v_customCanUnfoldPredicate_x3f_330_; uint8_t v_univApprox_331_; uint8_t v_inTypeClassResolution_332_; uint8_t v_cacheInferType_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_393_; 
v_keyedConfig_323_ = lean_ctor_get(v___y_318_, 0);
v_trackZetaDelta_324_ = lean_ctor_get_uint8(v___y_318_, sizeof(void*)*7);
v_zetaDeltaSet_325_ = lean_ctor_get(v___y_318_, 1);
v_lctx_326_ = lean_ctor_get(v___y_318_, 2);
v_localInstances_327_ = lean_ctor_get(v___y_318_, 3);
v_defEqCtx_x3f_328_ = lean_ctor_get(v___y_318_, 4);
v_synthPendingDepth_329_ = lean_ctor_get(v___y_318_, 5);
v_customCanUnfoldPredicate_x3f_330_ = lean_ctor_get(v___y_318_, 6);
v_univApprox_331_ = lean_ctor_get_uint8(v___y_318_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_332_ = lean_ctor_get_uint8(v___y_318_, sizeof(void*)*7 + 2);
v_cacheInferType_333_ = lean_ctor_get_uint8(v___y_318_, sizeof(void*)*7 + 3);
v_isSharedCheck_393_ = !lean_is_exclusive(v___y_318_);
if (v_isSharedCheck_393_ == 0)
{
v___x_335_ = v___y_318_;
v_isShared_336_ = v_isSharedCheck_393_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_330_);
lean_inc(v_synthPendingDepth_329_);
lean_inc(v_defEqCtx_x3f_328_);
lean_inc(v_localInstances_327_);
lean_inc(v_lctx_326_);
lean_inc(v_zetaDeltaSet_325_);
lean_inc(v_keyedConfig_323_);
lean_dec(v___y_318_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_393_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_337_; lean_object* v___x_339_; 
v___x_337_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_316_, v_keyedConfig_323_);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 0, v___x_337_);
v___x_339_ = v___x_335_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v___x_337_);
lean_ctor_set(v_reuseFailAlloc_392_, 1, v_zetaDeltaSet_325_);
lean_ctor_set(v_reuseFailAlloc_392_, 2, v_lctx_326_);
lean_ctor_set(v_reuseFailAlloc_392_, 3, v_localInstances_327_);
lean_ctor_set(v_reuseFailAlloc_392_, 4, v_defEqCtx_x3f_328_);
lean_ctor_set(v_reuseFailAlloc_392_, 5, v_synthPendingDepth_329_);
lean_ctor_set(v_reuseFailAlloc_392_, 6, v_customCanUnfoldPredicate_x3f_330_);
lean_ctor_set_uint8(v_reuseFailAlloc_392_, sizeof(void*)*7, v_trackZetaDelta_324_);
lean_ctor_set_uint8(v_reuseFailAlloc_392_, sizeof(void*)*7 + 1, v_univApprox_331_);
lean_ctor_set_uint8(v_reuseFailAlloc_392_, sizeof(void*)*7 + 2, v_inTypeClassResolution_332_);
lean_ctor_set_uint8(v_reuseFailAlloc_392_, sizeof(void*)*7 + 3, v_cacheInferType_333_);
v___x_339_ = v_reuseFailAlloc_392_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
lean_object* v___x_340_; 
v___x_340_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_317_, v___x_339_, v___y_319_, v___y_320_, v___y_321_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_object* v_a_341_; lean_object* v___x_342_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc_n(v_a_341_, 2);
lean_dec_ref_known(v___x_340_, 1);
lean_inc(v___y_321_);
lean_inc_ref(v___y_320_);
lean_inc(v___y_319_);
lean_inc_ref(v___x_339_);
v___x_342_ = lean_infer_type(v_a_341_, v___x_339_, v___y_319_, v___y_320_, v___y_321_);
if (lean_obj_tag(v___x_342_) == 0)
{
lean_object* v_a_343_; uint8_t v___x_344_; lean_object* v___x_345_; 
v_a_343_ = lean_ctor_get(v___x_342_, 0);
lean_inc(v_a_343_);
lean_dec_ref_known(v___x_342_, 1);
v___x_344_ = 0;
v___x_345_ = l_Lean_Meta_forallMetaTelescope(v_a_343_, v___x_344_, v___x_339_, v___y_319_, v___y_320_, v___y_321_);
if (lean_obj_tag(v___x_345_) == 0)
{
lean_object* v_a_346_; lean_object* v_fst_347_; lean_object* v___x_348_; uint8_t v___x_349_; lean_object* v___x_350_; 
v_a_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc(v_a_346_);
lean_dec_ref_known(v___x_345_, 1);
v_fst_347_ = lean_ctor_get(v_a_346_, 0);
lean_inc(v_fst_347_);
lean_dec(v_a_346_);
v___x_348_ = l_Lean_mkAppN(v_a_341_, v_fst_347_);
lean_dec(v_fst_347_);
v___x_349_ = 0;
v___x_350_ = l_Lean_Meta_DiscrTree_mkPath(v___x_348_, v___x_349_, v___x_339_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___x_339_);
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_359_; 
v_a_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_359_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_355_; lean_object* v___x_357_; 
v___x_355_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_355_, 0, v_a_351_);
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 0, v___x_355_);
v___x_357_ = v___x_353_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v___x_355_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
else
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_367_; 
v_a_360_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_367_ == 0)
{
v___x_362_ = v___x_350_;
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_350_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_365_; 
if (v_isShared_363_ == 0)
{
v___x_365_ = v___x_362_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_360_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
}
else
{
lean_object* v_a_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_375_; 
lean_dec(v_a_341_);
lean_dec_ref(v___x_339_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
v_a_368_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_375_ == 0)
{
v___x_370_ = v___x_345_;
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_a_368_);
lean_dec(v___x_345_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_373_; 
if (v_isShared_371_ == 0)
{
v___x_373_ = v___x_370_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_a_368_);
v___x_373_ = v_reuseFailAlloc_374_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
return v___x_373_;
}
}
}
}
else
{
lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
lean_dec(v_a_341_);
lean_dec_ref(v___x_339_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
v_a_376_ = lean_ctor_get(v___x_342_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_342_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v___x_342_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_342_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_a_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
else
{
lean_object* v_a_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_391_; 
lean_dec_ref(v___x_339_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
v_a_384_ = lean_ctor_get(v___x_340_, 0);
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_340_);
if (v_isSharedCheck_391_ == 0)
{
v___x_386_ = v___x_340_;
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_a_384_);
lean_dec(v___x_340_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_389_; 
if (v_isShared_387_ == 0)
{
v___x_389_ = v___x_386_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v_a_384_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0___boxed(lean_object* v___x_394_, lean_object* v_decl_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
uint8_t v___x_1770__boxed_401_; lean_object* v_res_402_; 
v___x_1770__boxed_401_ = lean_unbox(v___x_394_);
v_res_402_ = lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0(v___x_1770__boxed_401_, v_decl_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst(lean_object* v_decl_403_, lean_object* v_a_404_, lean_object* v_a_405_, lean_object* v_a_406_, lean_object* v_a_407_){
_start:
{
uint8_t v___x_409_; lean_object* v___x_410_; lean_object* v___f_411_; lean_object* v___x_412_; 
v___x_409_ = 2;
v___x_410_ = lean_box(v___x_409_);
v___f_411_ = lean_alloc_closure((void*)(lp_aesop_Aesop_IndexingMode_hypsMatchingConst___lam__0___boxed), 7, 2);
lean_closure_set(v___f_411_, 0, v___x_410_);
lean_closure_set(v___f_411_, 1, v_decl_403_);
v___x_412_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_IndexingMode_hypsMatchingConst_spec__0___redArg(v___f_411_, v_a_404_, v_a_405_, v_a_406_, v_a_407_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst___boxed(lean_object* v_decl_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_aesop_Aesop_IndexingMode_hypsMatchingConst(v_decl_413_, v_a_414_, v_a_415_, v_a_416_, v_a_417_);
lean_dec(v_a_417_);
lean_dec_ref(v_a_416_);
lean_dec(v_a_415_);
lean_dec_ref(v_a_414_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorIdx(lean_object* v_x_420_){
_start:
{
switch(lean_obj_tag(v_x_420_))
{
case 0:
{
lean_object* v___x_421_; 
v___x_421_ = lean_unsigned_to_nat(0u);
return v___x_421_;
}
case 1:
{
lean_object* v___x_422_; 
v___x_422_ = lean_unsigned_to_nat(1u);
return v___x_422_;
}
default: 
{
lean_object* v___x_423_; 
v___x_423_ = lean_unsigned_to_nat(2u);
return v___x_423_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorIdx___boxed(lean_object* v_x_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_aesop_Aesop_IndexMatchLocation_ctorIdx(v_x_424_);
lean_dec(v_x_424_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(lean_object* v_t_426_, lean_object* v_k_427_){
_start:
{
if (lean_obj_tag(v_t_426_) == 2)
{
lean_object* v_ldecl_428_; lean_object* v___x_429_; 
v_ldecl_428_ = lean_ctor_get(v_t_426_, 0);
lean_inc_ref(v_ldecl_428_);
lean_dec_ref_known(v_t_426_, 1);
v___x_429_ = lean_apply_1(v_k_427_, v_ldecl_428_);
return v___x_429_;
}
else
{
lean_dec(v_t_426_);
return v_k_427_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim(lean_object* v_motive_430_, lean_object* v_ctorIdx_431_, lean_object* v_t_432_, lean_object* v_h_433_, lean_object* v_k_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_432_, v_k_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_ctorElim___boxed(lean_object* v_motive_436_, lean_object* v_ctorIdx_437_, lean_object* v_t_438_, lean_object* v_h_439_, lean_object* v_k_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim(v_motive_436_, v_ctorIdx_437_, v_t_438_, v_h_439_, v_k_440_);
lean_dec(v_ctorIdx_437_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_none_elim___redArg(lean_object* v_t_442_, lean_object* v_none_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_442_, v_none_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_none_elim(lean_object* v_motive_445_, lean_object* v_t_446_, lean_object* v_h_447_, lean_object* v_none_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_446_, v_none_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_target_elim___redArg(lean_object* v_t_450_, lean_object* v_target_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_450_, v_target_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_target_elim(lean_object* v_motive_453_, lean_object* v_t_454_, lean_object* v_h_455_, lean_object* v_target_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_454_, v_target_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_hyp_elim___redArg(lean_object* v_t_458_, lean_object* v_hyp_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_458_, v_hyp_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_hyp_elim(lean_object* v_motive_461_, lean_object* v_t_462_, lean_object* v_h_463_, lean_object* v_hyp_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_aesop_Aesop_IndexMatchLocation_ctorElim___redArg(v_t_462_, v_hyp_464_);
return v___x_465_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndexMatchLocation_default(void){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lean_box(0);
return v___x_466_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndexMatchLocation(void){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lean_box(0);
return v___x_467_;
}
}
static lean_object* _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2(void){
_start:
{
lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_471_ = ((lean_object*)(lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__1));
v___x_472_ = l_Lean_MessageData_ofFormat(v___x_471_);
return v___x_472_;
}
}
static lean_object* _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_476_ = ((lean_object*)(lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__4));
v___x_477_ = l_Lean_MessageData_ofFormat(v___x_476_);
return v___x_477_;
}
}
static lean_object* _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = ((lean_object*)(lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__6));
v___x_480_ = l_Lean_stringToMessageData(v___x_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0(lean_object* v_x_481_){
_start:
{
switch(lean_obj_tag(v_x_481_))
{
case 0:
{
lean_object* v___x_482_; 
v___x_482_ = lean_obj_once(&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2, &lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2_once, _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__2);
return v___x_482_;
}
case 1:
{
lean_object* v___x_483_; 
v___x_483_ = lean_obj_once(&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5, &lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5_once, _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__5);
return v___x_483_;
}
default: 
{
lean_object* v_ldecl_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v_ldecl_484_ = lean_ctor_get(v_x_481_, 0);
v___x_485_ = lean_obj_once(&lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7, &lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7_once, _init_lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___closed__7);
v___x_486_ = l_Lean_LocalDecl_userName(v_ldecl_484_);
v___x_487_ = l_Lean_MessageData_ofName(v___x_486_);
v___x_488_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_485_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
return v___x_488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0___boxed(lean_object* v_x_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_aesop_Aesop_IndexMatchLocation_instToMessageData___lam__0(v_x_489_);
lean_dec(v_x_489_);
return v_res_490_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0(lean_object* v_x_493_, lean_object* v_x_494_){
_start:
{
switch(lean_obj_tag(v_x_493_))
{
case 0:
{
if (lean_obj_tag(v_x_494_) == 0)
{
uint8_t v___x_495_; 
v___x_495_ = 1;
return v___x_495_;
}
else
{
uint8_t v___x_496_; 
v___x_496_ = 0;
return v___x_496_;
}
}
case 1:
{
if (lean_obj_tag(v_x_494_) == 1)
{
uint8_t v___x_497_; 
v___x_497_ = 1;
return v___x_497_;
}
else
{
uint8_t v___x_498_; 
v___x_498_ = 0;
return v___x_498_;
}
}
default: 
{
if (lean_obj_tag(v_x_494_) == 2)
{
lean_object* v_ldecl_499_; lean_object* v_ldecl_500_; lean_object* v___x_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v_ldecl_499_ = lean_ctor_get(v_x_493_, 0);
v_ldecl_500_ = lean_ctor_get(v_x_494_, 0);
v___x_501_ = l_Lean_LocalDecl_index(v_ldecl_499_);
v___x_502_ = l_Lean_LocalDecl_index(v_ldecl_500_);
v___x_503_ = lean_nat_dec_eq(v___x_501_, v___x_502_);
lean_dec(v___x_502_);
lean_dec(v___x_501_);
return v___x_503_;
}
else
{
uint8_t v___x_504_; 
v___x_504_ = 0;
return v___x_504_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0___boxed(lean_object* v_x_505_, lean_object* v_x_506_){
_start:
{
uint8_t v_res_507_; lean_object* v_r_508_; 
v_res_507_ = lp_aesop_Aesop_IndexMatchLocation_instBEq___lam__0(v_x_505_, v_x_506_);
lean_dec(v_x_506_);
lean_dec(v_x_505_);
v_r_508_ = lean_box(v_res_507_);
return v_r_508_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0(lean_object* v_x_511_, lean_object* v_x_512_){
_start:
{
switch(lean_obj_tag(v_x_511_))
{
case 0:
{
switch(lean_obj_tag(v_x_512_))
{
case 0:
{
uint8_t v___x_513_; 
v___x_513_ = 1;
return v___x_513_;
}
case 1:
{
uint8_t v___x_514_; 
v___x_514_ = 2;
return v___x_514_;
}
default: 
{
uint8_t v___x_515_; 
v___x_515_ = 0;
return v___x_515_;
}
}
}
case 1:
{
if (lean_obj_tag(v_x_512_) == 1)
{
uint8_t v___x_516_; 
v___x_516_ = 1;
return v___x_516_;
}
else
{
uint8_t v___x_517_; 
v___x_517_ = 0;
return v___x_517_;
}
}
default: 
{
if (lean_obj_tag(v_x_512_) == 2)
{
lean_object* v_ldecl_518_; lean_object* v_ldecl_519_; lean_object* v___x_520_; lean_object* v___x_521_; uint8_t v___x_522_; 
v_ldecl_518_ = lean_ctor_get(v_x_511_, 0);
v_ldecl_519_ = lean_ctor_get(v_x_512_, 0);
v___x_520_ = l_Lean_LocalDecl_index(v_ldecl_518_);
v___x_521_ = l_Lean_LocalDecl_index(v_ldecl_519_);
v___x_522_ = lean_nat_dec_lt(v___x_520_, v___x_521_);
if (v___x_522_ == 0)
{
uint8_t v___x_523_; 
v___x_523_ = lean_nat_dec_eq(v___x_520_, v___x_521_);
lean_dec(v___x_521_);
lean_dec(v___x_520_);
if (v___x_523_ == 0)
{
uint8_t v___x_524_; 
v___x_524_ = 2;
return v___x_524_;
}
else
{
uint8_t v___x_525_; 
v___x_525_ = 1;
return v___x_525_;
}
}
else
{
uint8_t v___x_526_; 
lean_dec(v___x_521_);
lean_dec(v___x_520_);
v___x_526_ = 0;
return v___x_526_;
}
}
else
{
uint8_t v___x_527_; 
v___x_527_ = 2;
return v___x_527_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0___boxed(lean_object* v_x_528_, lean_object* v_x_529_){
_start:
{
uint8_t v_res_530_; lean_object* v_r_531_; 
v_res_530_ = lp_aesop_Aesop_IndexMatchLocation_instOrd___lam__0(v_x_528_, v_x_529_);
lean_dec(v_x_529_);
lean_dec(v_x_528_);
v_r_531_ = lean_box(v_res_530_);
return v_r_531_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0(lean_object* v_x_534_){
_start:
{
switch(lean_obj_tag(v_x_534_))
{
case 0:
{
uint64_t v___x_535_; 
v___x_535_ = 7ULL;
return v___x_535_;
}
case 1:
{
uint64_t v___x_536_; 
v___x_536_ = 13ULL;
return v___x_536_;
}
default: 
{
lean_object* v_ldecl_537_; uint64_t v___x_538_; lean_object* v___x_539_; uint64_t v___x_540_; uint64_t v___x_541_; 
v_ldecl_537_ = lean_ctor_get(v_x_534_, 0);
v___x_538_ = 17ULL;
v___x_539_ = l_Lean_LocalDecl_index(v_ldecl_537_);
v___x_540_ = lean_uint64_of_nat(v___x_539_);
lean_dec(v___x_539_);
v___x_541_ = lean_uint64_mix_hash(v___x_538_, v___x_540_);
return v___x_541_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0___boxed(lean_object* v_x_542_){
_start:
{
uint64_t v_res_543_; lean_object* v_r_544_; 
v_res_543_ = lp_aesop_Aesop_IndexMatchLocation_instHashable___lam__0(v_x_542_);
lean_dec(v_x_542_);
v_r_544_ = lean_box_uint64(v_res_543_);
return v_r_544_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(lean_object* v_inst_549_){
_start:
{
lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_550_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg___closed__0));
v___x_551_ = lean_box(0);
v___x_552_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_552_, 0, v_inst_549_);
lean_ctor_set(v___x_552_, 1, v___x_550_);
lean_ctor_set(v___x_552_, 2, v___x_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult_default(lean_object* v_00_u03b1_553_, lean_object* v_inst_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(v_inst_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult___redArg(lean_object* v_inst_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(v_inst_556_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndexMatchResult(lean_object* v_a_558_, lean_object* v_inst_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_aesop_Aesop_instInhabitedIndexMatchResult_default___redArg(v_inst_559_);
return v___x_560_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0(lean_object* v_inst_561_, lean_object* v_r_562_, lean_object* v_s_563_){
_start:
{
lean_object* v_rule_564_; lean_object* v_rule_565_; lean_object* v___x_566_; uint8_t v___x_567_; 
v_rule_564_ = lean_ctor_get(v_r_562_, 0);
lean_inc(v_rule_564_);
lean_dec_ref(v_r_562_);
v_rule_565_ = lean_ctor_get(v_s_563_, 0);
lean_inc(v_rule_565_);
lean_dec_ref(v_s_563_);
v___x_566_ = lean_apply_2(v_inst_561_, v_rule_564_, v_rule_565_);
v___x_567_ = lean_unbox(v___x_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0___boxed(lean_object* v_inst_568_, lean_object* v_r_569_, lean_object* v_s_570_){
_start:
{
uint8_t v_res_571_; lean_object* v_r_572_; 
v_res_571_ = lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0(v_inst_568_, v_r_569_, v_s_570_);
v_r_572_ = lean_box(v_res_571_);
return v_r_572_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd___redArg(lean_object* v_inst_573_){
_start:
{
lean_object* v___f_574_; 
v___f_574_ = lean_alloc_closure((void*)(lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_574_, 0, v_inst_573_);
return v___f_574_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instOrd(lean_object* v_00_u03b1_575_, lean_object* v_inst_576_){
_start:
{
lean_object* v___f_577_; 
v___f_577_ = lean_alloc_closure((void*)(lp_aesop_Aesop_IndexMatchResult_instOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_577_, 0, v_inst_576_);
return v___f_577_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instLTOfOrd(lean_object* v_00_u03b1_578_, lean_object* v_inst_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = lean_box(0);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instLTOfOrd___boxed(lean_object* v_00_u03b1_581_, lean_object* v_inst_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_aesop_Aesop_IndexMatchResult_instLTOfOrd(v_00_u03b1_581_, v_inst_582_);
lean_dec_ref(v_inst_582_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg___lam__0(lean_object* v_inst_584_, lean_object* v_r_585_){
_start:
{
lean_object* v_rule_586_; lean_object* v___x_587_; 
v_rule_586_ = lean_ctor_get(v_r_585_, 0);
lean_inc(v_rule_586_);
lean_dec_ref(v_r_585_);
v___x_587_ = lean_apply_1(v_inst_584_, v_rule_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg(lean_object* v_inst_588_){
_start:
{
lean_object* v___f_589_; 
v___f_589_ = lean_alloc_closure((void*)(lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg___lam__0), 2, 1);
lean_closure_set(v___f_589_, 0, v_inst_588_);
return v___f_589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_IndexMatchResult_instToMessageData(lean_object* v_00_u03b1_590_, lean_object* v_inst_591_){
_start:
{
lean_object* v___f_592_; 
v___f_592_ = lean_alloc_closure((void*)(lp_aesop_Aesop_IndexMatchResult_instToMessageData___redArg___lam__0), 2, 1);
lean_closure_set(v___f_592_, 0, v_inst_591_);
return v___f_592_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Index_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedIndexingMode_default = _init_lp_aesop_Aesop_instInhabitedIndexingMode_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIndexingMode_default);
lp_aesop_Aesop_instInhabitedIndexingMode = _init_lp_aesop_Aesop_instInhabitedIndexingMode();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIndexingMode);
lp_aesop_Aesop_instInhabitedIndexMatchLocation_default = _init_lp_aesop_Aesop_instInhabitedIndexMatchLocation_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIndexMatchLocation_default);
lp_aesop_Aesop_instInhabitedIndexMatchLocation = _init_lp_aesop_Aesop_instInhabitedIndexMatchLocation();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIndexMatchLocation);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Index_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Index_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Index_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Index_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
