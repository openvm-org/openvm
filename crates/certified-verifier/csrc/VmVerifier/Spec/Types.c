// Lean compiler output
// Module: VmVerifier.Spec.Types
// Imports: public import Init public meta import Init public import Fundamentals.Spec.BabyBearExt4.Raw public import Recursion.Spec.Common.VerifierPublicValues public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Main
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_log2(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Std_Format_joinSep___at___00Array_repr___at___00instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lp_workspace_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(lean_object*, lean_object*);
lean_object* lp_workspace_Recursion_Spec_VmSegmentPublicValues_fromList_x3f___redArg(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_constraintEvalCachedIndex;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_unsetPvsAirId;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_symbolicExpressionAirId;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_addressSpaceOffset;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesAddressSpace;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "addrSpaceHeight"};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value),((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6_value;
static lean_once_cell_t lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "addressHeight"};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11_value;
static lean_once_cell_t lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12;
static const lean_string_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value;
static lean_once_cell_t lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14;
static lean_once_cell_t lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_instReprMemoryDimensions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_VmVerifier_instReprMemoryDimensions_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions___closed__0_value;
LEAN_EXPORT const lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions = (const lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_overallHeight(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_labelToIndex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeight_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeight_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeightFromCount(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeightFromCount___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3_value;
static const lean_string_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value;
static lean_once_cell_t lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5;
static lean_once_cell_t lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6;
static const lean_ctor_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value)}};
static const lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8 = (const lean_object*)&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(lean_object*);
static const lean_string_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "authenticationPath"};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value),((lean_object*)&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3_value;
static lean_once_cell_t lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4;
static const lean_string_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "publicValues"};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6_value;
static lean_once_cell_t lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7;
static const lean_string_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "publicValuesCommit"};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value;
static const lean_ctor_object lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value)}};
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_instReprUserPublicValuesProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof___closed__0_value;
LEAN_EXPORT const lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof = (const lean_object*)&lp_workspace_VmVerifier_instReprUserPublicValuesProof___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VerificationBaseline_publicValuesHeight(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VerificationBaseline_publicValuesHeight___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_parseVmProofData_x3f(lean_object*);
static lean_object* _init_lp_workspace_VmVerifier_constraintEvalCachedIndex(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_unsetPvsAirId(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(2u);
return v___x_2_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_symbolicExpressionAirId(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(3u);
return v___x_3_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_addressSpaceOffset(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(1u);
return v___x_4_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_publicValuesAddressSpace(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_unsigned_to_nat(19u);
v___x_20_ = lean_nat_to_int(v___x_19_);
return v___x_20_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = lean_unsigned_to_nat(17u);
v___x_28_ = lean_nat_to_int(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_30_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0));
v___x_31_ = lean_string_length(v___x_30_);
return v___x_31_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_obj_once(&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14, &lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14_once, _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14);
v___x_33_ = lean_nat_to_int(v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg(lean_object* v_x_38_){
_start:
{
lean_object* v_addrSpaceHeight_39_; lean_object* v_addressHeight_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_75_; 
v_addrSpaceHeight_39_ = lean_ctor_get(v_x_38_, 0);
v_addressHeight_40_ = lean_ctor_get(v_x_38_, 1);
v_isSharedCheck_75_ = !lean_is_exclusive(v_x_38_);
if (v_isSharedCheck_75_ == 0)
{
v___x_42_ = v_x_38_;
v_isShared_43_ = v_isSharedCheck_75_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_addressHeight_40_);
lean_inc(v_addrSpaceHeight_39_);
lean_dec(v_x_38_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_75_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_50_; 
v___x_44_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5));
v___x_45_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6));
v___x_46_ = lean_obj_once(&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7, &lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7_once, _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7);
v___x_47_ = l_Nat_reprFast(v_addrSpaceHeight_39_);
v___x_48_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
if (v_isShared_43_ == 0)
{
lean_ctor_set_tag(v___x_42_, 4);
lean_ctor_set(v___x_42_, 1, v___x_48_);
lean_ctor_set(v___x_42_, 0, v___x_46_);
v___x_50_ = v___x_42_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v___x_46_);
lean_ctor_set(v_reuseFailAlloc_74_, 1, v___x_48_);
v___x_50_ = v_reuseFailAlloc_74_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
uint8_t v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_51_ = 0;
v___x_52_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_52_, 0, v___x_50_);
lean_ctor_set_uint8(v___x_52_, sizeof(void*)*1, v___x_51_);
v___x_53_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_53_, 0, v___x_45_);
lean_ctor_set(v___x_53_, 1, v___x_52_);
v___x_54_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9));
v___x_55_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_55_, 0, v___x_53_);
lean_ctor_set(v___x_55_, 1, v___x_54_);
v___x_56_ = lean_box(1);
v___x_57_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_55_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
v___x_58_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11));
v___x_59_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_57_);
lean_ctor_set(v___x_59_, 1, v___x_58_);
v___x_60_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v___x_44_);
v___x_61_ = lean_obj_once(&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12, &lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12_once, _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12);
v___x_62_ = l_Nat_reprFast(v_addressHeight_40_);
v___x_63_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
v___x_64_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_61_);
lean_ctor_set(v___x_64_, 1, v___x_63_);
v___x_65_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set_uint8(v___x_65_, sizeof(void*)*1, v___x_51_);
v___x_66_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_60_);
lean_ctor_set(v___x_66_, 1, v___x_65_);
v___x_67_ = lean_obj_once(&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15, &lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once, _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15);
v___x_68_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16));
v___x_69_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_66_);
v___x_70_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17));
v___x_71_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_69_);
lean_ctor_set(v___x_71_, 1, v___x_70_);
v___x_72_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_67_);
lean_ctor_set(v___x_72_, 1, v___x_71_);
v___x_73_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set_uint8(v___x_73_, sizeof(void*)*1, v___x_51_);
return v___x_73_;
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr(lean_object* v_x_76_, lean_object* v_prec_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg(v_x_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprMemoryDimensions_repr___boxed(lean_object* v_x_79_, lean_object* v_prec_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_workspace_VmVerifier_instReprMemoryDimensions_repr(v_x_79_, v_prec_80_);
lean_dec(v_prec_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_overallHeight(lean_object* v_dimensions_84_){
_start:
{
lean_object* v_addrSpaceHeight_85_; lean_object* v_addressHeight_86_; lean_object* v___x_87_; 
v_addrSpaceHeight_85_ = lean_ctor_get(v_dimensions_84_, 0);
v_addressHeight_86_ = lean_ctor_get(v_dimensions_84_, 1);
v___x_87_ = lean_nat_add(v_addrSpaceHeight_85_, v_addressHeight_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object* v_dimensions_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_workspace_VmVerifier_MemoryDimensions_overallHeight(v_dimensions_88_);
lean_dec_ref(v_dimensions_88_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_labelToIndex(lean_object* v_dimensions_90_, lean_object* v_addressSpace_91_, lean_object* v_blockId_92_){
_start:
{
lean_object* v_addressHeight_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_addressHeight_93_ = lean_ctor_get(v_dimensions_90_, 1);
v___x_94_ = lean_unsigned_to_nat(1u);
v___x_95_ = lean_nat_sub(v_addressSpace_91_, v___x_94_);
v___x_96_ = lean_unsigned_to_nat(2u);
v___x_97_ = lean_nat_pow(v___x_96_, v_addressHeight_93_);
v___x_98_ = lean_nat_mul(v___x_95_, v___x_97_);
lean_dec(v___x_97_);
lean_dec(v___x_95_);
v___x_99_ = lean_nat_add(v___x_98_, v_blockId_92_);
lean_dec(v___x_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object* v_dimensions_100_, lean_object* v_addressSpace_101_, lean_object* v_blockId_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_workspace_VmVerifier_MemoryDimensions_labelToIndex(v_dimensions_100_, v_addressSpace_101_, v_blockId_102_);
lean_dec(v_blockId_102_);
lean_dec(v_addressSpace_101_);
lean_dec_ref(v_dimensions_100_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeight_x3f(lean_object* v_values_104_){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_105_ = l_List_lengthTR___redArg(v_values_104_);
v___x_106_ = lean_unsigned_to_nat(8u);
v___x_107_ = lean_nat_mod(v___x_105_, v___x_106_);
v___x_108_ = lean_unsigned_to_nat(0u);
v___x_109_ = lean_nat_dec_eq(v___x_107_, v___x_108_);
lean_dec(v___x_107_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; 
lean_dec(v___x_105_);
v___x_110_ = lean_box(0);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v_chunks_112_; uint8_t v___x_113_; 
v___x_111_ = lean_unsigned_to_nat(3u);
v_chunks_112_ = lean_nat_shiftr(v___x_105_, v___x_111_);
lean_dec(v___x_105_);
v___x_113_ = lean_nat_dec_lt(v___x_108_, v_chunks_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; 
lean_dec(v_chunks_112_);
v___x_114_ = lean_box(0);
return v___x_114_;
}
else
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_115_ = lean_unsigned_to_nat(2u);
v___x_116_ = lean_nat_log2(v_chunks_112_);
v___x_117_ = lean_nat_pow(v___x_115_, v___x_116_);
v___x_118_ = lean_nat_dec_eq(v___x_117_, v_chunks_112_);
lean_dec(v_chunks_112_);
lean_dec(v___x_117_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; 
lean_dec(v___x_116_);
v___x_119_ = lean_box(0);
return v___x_119_;
}
else
{
lean_object* v___x_120_; 
v___x_120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_120_, 0, v___x_116_);
return v___x_120_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeight_x3f___boxed(lean_object* v_values_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_workspace_VmVerifier_publicValuesHeight_x3f(v_values_121_);
lean_dec(v_values_121_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeightFromCount(lean_object* v_numPublicValues_123_){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_unsigned_to_nat(3u);
v___x_125_ = lean_nat_shiftr(v_numPublicValues_123_, v___x_124_);
v___x_126_ = lean_nat_log2(v___x_125_);
lean_dec(v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_publicValuesHeightFromCount___boxed(lean_object* v_numPublicValues_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_workspace_VmVerifier_publicValuesHeightFromCount(v_numPublicValues_127_);
lean_dec(v_numPublicValues_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_129_, lean_object* v_x_130_, lean_object* v_x_131_){
_start:
{
if (lean_obj_tag(v_x_131_) == 0)
{
lean_dec(v_x_129_);
return v_x_130_;
}
else
{
lean_object* v_head_132_; lean_object* v_tail_133_; lean_object* v___x_135_; uint8_t v_isShared_136_; uint8_t v_isSharedCheck_143_; 
v_head_132_ = lean_ctor_get(v_x_131_, 0);
v_tail_133_ = lean_ctor_get(v_x_131_, 1);
v_isSharedCheck_143_ = !lean_is_exclusive(v_x_131_);
if (v_isSharedCheck_143_ == 0)
{
v___x_135_ = v_x_131_;
v_isShared_136_ = v_isSharedCheck_143_;
goto v_resetjp_134_;
}
else
{
lean_inc(v_tail_133_);
lean_inc(v_head_132_);
lean_dec(v_x_131_);
v___x_135_ = lean_box(0);
v_isShared_136_ = v_isSharedCheck_143_;
goto v_resetjp_134_;
}
v_resetjp_134_:
{
lean_object* v___x_138_; 
lean_inc(v_x_129_);
if (v_isShared_136_ == 0)
{
lean_ctor_set_tag(v___x_135_, 5);
lean_ctor_set(v___x_135_, 1, v_x_129_);
lean_ctor_set(v___x_135_, 0, v_x_130_);
v___x_138_ = v___x_135_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v_x_130_);
lean_ctor_set(v_reuseFailAlloc_142_, 1, v_x_129_);
v___x_138_ = v_reuseFailAlloc_142_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_139_ = lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_132_);
v___x_140_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_138_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v_x_130_ = v___x_140_;
v_x_131_ = v_tail_133_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(lean_object* v_x_144_, lean_object* v_x_145_, lean_object* v_x_146_){
_start:
{
if (lean_obj_tag(v_x_146_) == 0)
{
lean_dec(v_x_144_);
return v_x_145_;
}
else
{
lean_object* v_head_147_; lean_object* v_tail_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_158_; 
v_head_147_ = lean_ctor_get(v_x_146_, 0);
v_tail_148_ = lean_ctor_get(v_x_146_, 1);
v_isSharedCheck_158_ = !lean_is_exclusive(v_x_146_);
if (v_isSharedCheck_158_ == 0)
{
v___x_150_ = v_x_146_;
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_tail_148_);
lean_inc(v_head_147_);
lean_dec(v_x_146_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_158_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
lean_inc(v_x_144_);
if (v_isShared_151_ == 0)
{
lean_ctor_set_tag(v___x_150_, 5);
lean_ctor_set(v___x_150_, 1, v_x_144_);
lean_ctor_set(v___x_150_, 0, v_x_145_);
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_x_145_);
lean_ctor_set(v_reuseFailAlloc_157_, 1, v_x_144_);
v___x_153_ = v_reuseFailAlloc_157_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_147_);
v___x_155_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_153_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = lp_workspace_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(v_x_144_, v___x_155_, v_tail_148_);
return v___x_156_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(lean_object* v_x_159_, lean_object* v_x_160_){
_start:
{
if (lean_obj_tag(v_x_159_) == 0)
{
lean_object* v___x_161_; 
lean_dec(v_x_160_);
v___x_161_ = lean_box(0);
return v___x_161_;
}
else
{
lean_object* v_tail_162_; 
v_tail_162_ = lean_ctor_get(v_x_159_, 1);
if (lean_obj_tag(v_tail_162_) == 0)
{
lean_object* v_head_163_; lean_object* v___x_164_; 
lean_dec(v_x_160_);
v_head_163_ = lean_ctor_get(v_x_159_, 0);
lean_inc(v_head_163_);
lean_dec_ref_known(v_x_159_, 2);
v___x_164_ = lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_163_);
return v___x_164_;
}
else
{
lean_object* v_head_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
lean_inc(v_tail_162_);
v_head_165_ = lean_ctor_get(v_x_159_, 0);
lean_inc(v_head_165_);
lean_dec_ref_known(v_x_159_, 2);
v___x_166_ = lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_165_);
v___x_167_ = lp_workspace_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(v_x_160_, v___x_166_, v_tail_162_);
return v___x_167_;
}
}
}
}
static lean_object* _init_lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2));
v___x_177_ = lean_string_length(v___x_176_);
return v___x_177_;
}
}
static lean_object* _init_lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = lean_obj_once(&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5, &lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5_once, _init_lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5);
v___x_179_ = lean_nat_to_int(v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(lean_object* v_a_184_){
_start:
{
if (lean_obj_tag(v_a_184_) == 0)
{
lean_object* v___x_185_; 
v___x_185_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_185_;
}
else
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; uint8_t v___x_194_; lean_object* v___x_195_; 
v___x_186_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_187_ = lp_workspace_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(v_a_184_, v___x_186_);
v___x_188_ = lean_obj_once(&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_189_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_190_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
lean_ctor_set(v___x_190_, 1, v___x_187_);
v___x_191_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_192_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_190_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
v___x_193_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_188_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = 0;
v___x_195_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_195_, 0, v___x_193_);
lean_ctor_set_uint8(v___x_195_, sizeof(void*)*1, v___x_194_);
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(lean_object* v_a_196_){
_start:
{
if (lean_obj_tag(v_a_196_) == 0)
{
lean_object* v___x_197_; 
v___x_197_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; lean_object* v___x_207_; 
v___x_198_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_199_ = lp_swirl_x2drbr_x2dfv_Std_Format_joinSep___at___00Array_repr___at___00instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0_spec__0_spec__1(v_a_196_, v___x_198_);
v___x_200_ = lean_obj_once(&lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_201_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_202_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v___x_199_);
v___x_203_ = ((lean_object*)(lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_204_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_202_);
lean_ctor_set(v___x_204_, 1, v___x_203_);
v___x_205_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_200_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = 0;
v___x_207_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_207_, 0, v___x_205_);
lean_ctor_set_uint8(v___x_207_, sizeof(void*)*1, v___x_206_);
return v___x_207_;
}
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_unsigned_to_nat(22u);
v___x_218_ = lean_nat_to_int(v___x_217_);
return v___x_218_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_222_ = lean_unsigned_to_nat(16u);
v___x_223_ = lean_nat_to_int(v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg(lean_object* v_x_227_){
_start:
{
lean_object* v_authenticationPath_228_; lean_object* v_publicValues_229_; lean_object* v_publicValuesCommit_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v_authenticationPath_228_ = lean_ctor_get(v_x_227_, 0);
lean_inc(v_authenticationPath_228_);
v_publicValues_229_ = lean_ctor_get(v_x_227_, 1);
lean_inc(v_publicValues_229_);
v_publicValuesCommit_230_ = lean_ctor_get(v_x_227_, 2);
lean_inc_ref(v_publicValuesCommit_230_);
lean_dec_ref(v_x_227_);
v___x_231_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5));
v___x_232_ = ((lean_object*)(lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3));
v___x_233_ = lean_obj_once(&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4, &lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4_once, _init_lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4);
v___x_234_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_authenticationPath_228_);
v___x_235_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_233_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = 0;
v___x_237_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_237_, 0, v___x_235_);
lean_ctor_set_uint8(v___x_237_, sizeof(void*)*1, v___x_236_);
v___x_238_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_232_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9));
v___x_240_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_238_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = lean_box(1);
v___x_242_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = ((lean_object*)(lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6));
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_242_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v___x_231_);
v___x_246_ = lean_obj_once(&lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7, &lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7_once, _init_lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7);
v___x_247_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_publicValues_229_);
v___x_248_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set_uint8(v___x_249_, sizeof(void*)*1, v___x_236_);
v___x_250_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_245_);
lean_ctor_set(v___x_250_, 1, v___x_249_);
v___x_251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___x_239_);
v___x_252_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_241_);
v___x_253_ = ((lean_object*)(lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9));
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_252_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v___x_231_);
v___x_256_ = lp_swirl_x2drbr_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_publicValuesCommit_230_);
v___x_257_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_233_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set_uint8(v___x_258_, sizeof(void*)*1, v___x_236_);
v___x_259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_255_);
lean_ctor_set(v___x_259_, 1, v___x_258_);
v___x_260_ = lean_obj_once(&lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15, &lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once, _init_lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15);
v___x_261_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16));
v___x_262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_259_);
v___x_263_ = ((lean_object*)(lp_workspace_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17));
v___x_264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_262_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_260_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
v___x_266_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_266_, 0, v___x_265_);
lean_ctor_set_uint8(v___x_266_, sizeof(void*)*1, v___x_236_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr(lean_object* v_x_267_, lean_object* v_prec_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___redArg(v_x_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr___boxed(lean_object* v_x_270_, lean_object* v_prec_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_workspace_VmVerifier_instReprUserPublicValuesProof_repr(v_x_270_, v_prec_271_);
lean_dec(v_prec_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(lean_object* v_a_273_, lean_object* v_n_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_a_273_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___boxed(lean_object* v_a_276_, lean_object* v_n_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(v_a_276_, v_n_277_);
lean_dec(v_n_277_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(lean_object* v_a_279_, lean_object* v_n_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_a_279_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___boxed(lean_object* v_a_282_, lean_object* v_n_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_workspace_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(v_a_282_, v_n_283_);
lean_dec(v_n_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VerificationBaseline_publicValuesHeight(lean_object* v_baseline_287_){
_start:
{
lean_object* v_numUserPvs_288_; lean_object* v___x_289_; 
v_numUserPvs_288_ = lean_ctor_get(v_baseline_287_, 4);
v___x_289_ = lp_workspace_VmVerifier_publicValuesHeightFromCount(v_numUserPvs_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VerificationBaseline_publicValuesHeight___boxed(lean_object* v_baseline_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_workspace_VmVerifier_VerificationBaseline_publicValuesHeight(v_baseline_290_);
lean_dec_ref(v_baseline_290_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_parseVmProofData_x3f(lean_object* v_proof_292_){
_start:
{
lean_object* v_inner_293_; lean_object* v_userPublicValuesProof_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_362_; 
v_inner_293_ = lean_ctor_get(v_proof_292_, 0);
v_userPublicValuesProof_294_ = lean_ctor_get(v_proof_292_, 1);
v_isSharedCheck_362_ = !lean_is_exclusive(v_proof_292_);
if (v_isSharedCheck_362_ == 0)
{
v___x_296_ = v_proof_292_;
v_isShared_297_ = v_isSharedCheck_362_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_userPublicValuesProof_294_);
lean_inc(v_inner_293_);
lean_dec(v_proof_292_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_362_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v_traceVdata_298_; lean_object* v_publicValues_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v_traceVdata_298_ = lean_ctor_get(v_inner_293_, 1);
lean_inc(v_traceVdata_298_);
v_publicValues_299_ = lean_ctor_get(v_inner_293_, 2);
lean_inc(v_publicValues_299_);
lean_dec_ref(v_inner_293_);
v___x_300_ = lean_unsigned_to_nat(0u);
v___x_301_ = l_List_get_x3fInternal___redArg(v_publicValues_299_, v___x_300_);
if (lean_obj_tag(v___x_301_) == 0)
{
lean_object* v___x_302_; 
lean_dec(v_publicValues_299_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_302_ = lean_box(0);
return v___x_302_;
}
else
{
lean_object* v_val_303_; lean_object* v___x_304_; 
v_val_303_ = lean_ctor_get(v___x_301_, 0);
lean_inc(v_val_303_);
lean_dec_ref_known(v___x_301_, 1);
v___x_304_ = lp_workspace_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v___x_300_, v_val_303_);
lean_dec(v_val_303_);
if (lean_obj_tag(v___x_304_) == 0)
{
lean_object* v___x_305_; 
lean_dec(v_publicValues_299_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_305_ = lean_box(0);
return v___x_305_;
}
else
{
lean_object* v_val_306_; lean_object* v___x_307_; lean_object* v___x_308_; 
v_val_306_ = lean_ctor_get(v___x_304_, 0);
lean_inc(v_val_306_);
lean_dec_ref_known(v___x_304_, 1);
v___x_307_ = lean_unsigned_to_nat(1u);
v___x_308_ = l_List_get_x3fInternal___redArg(v_publicValues_299_, v___x_307_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v___x_309_; 
lean_dec(v_val_306_);
lean_dec(v_publicValues_299_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_309_ = lean_box(0);
return v___x_309_;
}
else
{
lean_object* v_val_310_; lean_object* v___x_311_; 
v_val_310_ = lean_ctor_get(v___x_308_, 0);
lean_inc(v_val_310_);
lean_dec_ref_known(v___x_308_, 1);
v___x_311_ = lp_workspace_Recursion_Spec_VmSegmentPublicValues_fromList_x3f___redArg(v_val_310_);
if (lean_obj_tag(v___x_311_) == 0)
{
lean_object* v___x_312_; 
lean_dec(v_val_306_);
lean_dec(v_publicValues_299_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_312_ = lean_box(0);
return v___x_312_;
}
else
{
lean_object* v_val_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v_val_313_ = lean_ctor_get(v___x_311_, 0);
lean_inc(v_val_313_);
lean_dec_ref_known(v___x_311_, 1);
v___x_314_ = lean_unsigned_to_nat(2u);
v___x_315_ = l_List_get_x3fInternal___redArg(v_publicValues_299_, v___x_314_);
lean_dec(v_publicValues_299_);
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v___x_316_; 
lean_dec(v_val_313_);
lean_dec(v_val_306_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_316_ = lean_box(0);
return v___x_316_;
}
else
{
lean_object* v_val_317_; uint8_t v___x_318_; 
v_val_317_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_val_317_);
lean_dec_ref_known(v___x_315_, 1);
v___x_318_ = l_List_isEmpty___redArg(v_val_317_);
lean_dec(v_val_317_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; 
lean_dec(v_val_313_);
lean_dec(v_val_306_);
lean_dec(v_traceVdata_298_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_319_ = lean_box(0);
return v___x_319_;
}
else
{
lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_320_ = lean_unsigned_to_nat(3u);
v___x_321_ = l_List_get_x3fInternal___redArg(v_traceVdata_298_, v___x_320_);
lean_dec(v_traceVdata_298_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v___x_322_; 
lean_dec(v_val_313_);
lean_dec(v_val_306_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_322_ = lean_box(0);
return v___x_322_;
}
else
{
lean_object* v_val_323_; 
v_val_323_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_val_323_);
lean_dec_ref_known(v___x_321_, 1);
if (lean_obj_tag(v_val_323_) == 0)
{
lean_object* v___x_324_; 
lean_dec(v_val_313_);
lean_dec(v_val_306_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_324_ = lean_box(0);
return v___x_324_;
}
else
{
lean_object* v_val_325_; lean_object* v_cachedCommitments_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_360_; 
v_val_325_ = lean_ctor_get(v_val_323_, 0);
lean_inc(v_val_325_);
lean_dec_ref_known(v_val_323_, 1);
v_cachedCommitments_326_ = lean_ctor_get(v_val_325_, 1);
v_isSharedCheck_360_ = !lean_is_exclusive(v_val_325_);
if (v_isSharedCheck_360_ == 0)
{
lean_object* v_unused_361_; 
v_unused_361_ = lean_ctor_get(v_val_325_, 0);
lean_dec(v_unused_361_);
v___x_328_ = v_val_325_;
v_isShared_329_ = v_isSharedCheck_360_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_cachedCommitments_326_);
lean_dec(v_val_325_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_360_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; 
v___x_330_ = l_List_get_x3fInternal___redArg(v_cachedCommitments_326_, v___x_300_);
lean_dec(v_cachedCommitments_326_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v___x_331_; 
lean_del_object(v___x_328_);
lean_dec(v_val_313_);
lean_dec(v_val_306_);
lean_del_object(v___x_296_);
lean_dec_ref(v_userPublicValuesProof_294_);
v___x_331_ = lean_box(0);
return v___x_331_;
}
else
{
lean_object* v_val_332_; lean_object* v___x_334_; uint8_t v_isShared_335_; uint8_t v_isSharedCheck_359_; 
v_val_332_ = lean_ctor_get(v___x_330_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_330_);
if (v_isSharedCheck_359_ == 0)
{
v___x_334_ = v___x_330_;
v_isShared_335_ = v_isSharedCheck_359_;
goto v_resetjp_333_;
}
else
{
lean_inc(v_val_332_);
lean_dec(v___x_330_);
v___x_334_ = lean_box(0);
v_isShared_335_ = v_isSharedCheck_359_;
goto v_resetjp_333_;
}
v_resetjp_333_:
{
lean_object* v_internalFlag_336_; lean_object* v_appVkCommit_337_; lean_object* v_leafVkCommit_338_; lean_object* v_internalForLeafVkCommit_339_; lean_object* v_recursionDepth_340_; lean_object* v_internalRecursiveVkCommit_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_358_; 
v_internalFlag_336_ = lean_ctor_get(v_val_306_, 0);
v_appVkCommit_337_ = lean_ctor_get(v_val_306_, 1);
v_leafVkCommit_338_ = lean_ctor_get(v_val_306_, 2);
v_internalForLeafVkCommit_339_ = lean_ctor_get(v_val_306_, 3);
v_recursionDepth_340_ = lean_ctor_get(v_val_306_, 4);
v_internalRecursiveVkCommit_341_ = lean_ctor_get(v_val_306_, 5);
v_isSharedCheck_358_ = !lean_is_exclusive(v_val_306_);
if (v_isSharedCheck_358_ == 0)
{
v___x_343_ = v_val_306_;
v_isShared_344_ = v_isSharedCheck_358_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_internalRecursiveVkCommit_341_);
lean_inc(v_recursionDepth_340_);
lean_inc(v_internalForLeafVkCommit_339_);
lean_inc(v_leafVkCommit_338_);
lean_inc(v_appVkCommit_337_);
lean_inc(v_internalFlag_336_);
lean_dec(v_val_306_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_358_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___x_346_; 
if (v_isShared_344_ == 0)
{
v___x_346_ = v___x_343_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_internalFlag_336_);
lean_ctor_set(v_reuseFailAlloc_357_, 1, v_appVkCommit_337_);
lean_ctor_set(v_reuseFailAlloc_357_, 2, v_leafVkCommit_338_);
lean_ctor_set(v_reuseFailAlloc_357_, 3, v_internalForLeafVkCommit_339_);
lean_ctor_set(v_reuseFailAlloc_357_, 4, v_recursionDepth_340_);
lean_ctor_set(v_reuseFailAlloc_357_, 5, v_internalRecursiveVkCommit_341_);
v___x_346_ = v_reuseFailAlloc_357_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
lean_object* v___x_348_; 
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 1, v_val_313_);
lean_ctor_set(v___x_328_, 0, v___x_346_);
v___x_348_ = v___x_328_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_356_, 1, v_val_313_);
v___x_348_ = v_reuseFailAlloc_356_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
lean_object* v___x_350_; 
if (v_isShared_297_ == 0)
{
lean_ctor_set(v___x_296_, 1, v_val_332_);
lean_ctor_set(v___x_296_, 0, v___x_348_);
v___x_350_ = v___x_296_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v_val_332_);
v___x_350_ = v_reuseFailAlloc_355_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
lean_object* v___x_351_; lean_object* v___x_353_; 
v___x_351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
lean_ctor_set(v___x_351_, 1, v_userPublicValuesProof_294_);
if (v_isShared_335_ == 0)
{
lean_ctor_set(v___x_334_, 0, v___x_351_);
v___x_353_ = v___x_334_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_351_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
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
}
}
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(uint8_t builtin);
lean_object* initialize_workspace_Recursion_Spec_Common_VerifierPublicValues(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_workspace_VmVerifier_Spec_Types(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_workspace_Recursion_Spec_Common_VerifierPublicValues(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_workspace_VmVerifier_constraintEvalCachedIndex = _init_lp_workspace_VmVerifier_constraintEvalCachedIndex();
lean_mark_persistent(lp_workspace_VmVerifier_constraintEvalCachedIndex);
lp_workspace_VmVerifier_unsetPvsAirId = _init_lp_workspace_VmVerifier_unsetPvsAirId();
lean_mark_persistent(lp_workspace_VmVerifier_unsetPvsAirId);
lp_workspace_VmVerifier_symbolicExpressionAirId = _init_lp_workspace_VmVerifier_symbolicExpressionAirId();
lean_mark_persistent(lp_workspace_VmVerifier_symbolicExpressionAirId);
lp_workspace_VmVerifier_addressSpaceOffset = _init_lp_workspace_VmVerifier_addressSpaceOffset();
lean_mark_persistent(lp_workspace_VmVerifier_addressSpaceOffset);
lp_workspace_VmVerifier_publicValuesAddressSpace = _init_lp_workspace_VmVerifier_publicValuesAddressSpace();
lean_mark_persistent(lp_workspace_VmVerifier_publicValuesAddressSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
