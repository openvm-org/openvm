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
lean_object* lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(lean_object*);
lean_object* lp_swirl_x2dfv_Std_Format_joinSep___at___00Array_repr___at___00instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(lean_object*, lean_object*);
lean_object* lp_openvm_x2dfv_Recursion_Spec_VmSegmentPublicValues_fromList_x3f___redArg(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_constraintEvalCachedIndex;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_unsetPvsAirId;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_symbolicExpressionAirId;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_addressSpaceOffset;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesAddressSpace;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "addrSpaceHeight"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__1_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__2_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__4_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__3_value),((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__8_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "addressHeight"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__10_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__13_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions___closed__0_value;
LEAN_EXPORT const lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3_value;
static const lean_string_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value;
static lean_once_cell_t lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5;
static lean_once_cell_t lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6;
static const lean_ctor_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__4_value)}};
static const lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8 = (const lean_object*)&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(lean_object*);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "authenticationPath"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__0_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__1_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__2_value),((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "publicValues"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__5_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "publicValuesCommit"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__8_value)}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof___closed__0_value;
LEAN_EXPORT const lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_parseVmProofData_x3f(lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_constraintEvalCachedIndex(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_unsetPvsAirId(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(2u);
return v___x_2_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_symbolicExpressionAirId(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(3u);
return v___x_3_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_addressSpaceOffset(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(1u);
return v___x_4_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_publicValuesAddressSpace(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_unsigned_to_nat(19u);
v___x_20_ = lean_nat_to_int(v___x_19_);
return v___x_20_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = lean_unsigned_to_nat(17u);
v___x_28_ = lean_nat_to_int(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_30_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__0));
v___x_31_ = lean_string_length(v___x_30_);
return v___x_31_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__14);
v___x_33_ = lean_nat_to_int(v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg(lean_object* v_x_38_){
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
v___x_44_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5));
v___x_45_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__6));
v___x_46_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__7);
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
v___x_54_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9));
v___x_55_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_55_, 0, v___x_53_);
lean_ctor_set(v___x_55_, 1, v___x_54_);
v___x_56_ = lean_box(1);
v___x_57_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_55_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
v___x_58_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__11));
v___x_59_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_57_);
lean_ctor_set(v___x_59_, 1, v___x_58_);
v___x_60_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v___x_44_);
v___x_61_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__12);
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
v___x_67_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15);
v___x_68_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16));
v___x_69_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_66_);
v___x_70_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17));
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr(lean_object* v_x_76_, lean_object* v_prec_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg(v_x_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___boxed(lean_object* v_x_79_, lean_object* v_prec_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr(v_x_79_, v_prec_80_);
lean_dec(v_prec_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(lean_object* v_dimensions_84_){
_start:
{
lean_object* v_addrSpaceHeight_85_; lean_object* v_addressHeight_86_; lean_object* v___x_87_; 
v_addrSpaceHeight_85_ = lean_ctor_get(v_dimensions_84_, 0);
v_addressHeight_86_ = lean_ctor_get(v_dimensions_84_, 1);
v___x_87_ = lean_nat_add(v_addrSpaceHeight_85_, v_addressHeight_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object* v_dimensions_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(v_dimensions_88_);
lean_dec_ref(v_dimensions_88_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(lean_object* v_dimensions_90_, lean_object* v_addressSpace_91_, lean_object* v_blockId_92_){
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object* v_dimensions_100_, lean_object* v_addressSpace_101_, lean_object* v_blockId_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(v_dimensions_100_, v_addressSpace_101_, v_blockId_102_);
lean_dec(v_blockId_102_);
lean_dec(v_addressSpace_101_);
lean_dec_ref(v_dimensions_100_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(lean_object* v_values_104_){
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f___boxed(lean_object* v_values_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(v_values_121_);
lean_dec(v_values_121_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_123_, lean_object* v_x_124_, lean_object* v_x_125_){
_start:
{
if (lean_obj_tag(v_x_125_) == 0)
{
lean_dec(v_x_123_);
return v_x_124_;
}
else
{
lean_object* v_head_126_; lean_object* v_tail_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_137_; 
v_head_126_ = lean_ctor_get(v_x_125_, 0);
v_tail_127_ = lean_ctor_get(v_x_125_, 1);
v_isSharedCheck_137_ = !lean_is_exclusive(v_x_125_);
if (v_isSharedCheck_137_ == 0)
{
v___x_129_ = v_x_125_;
v_isShared_130_ = v_isSharedCheck_137_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_tail_127_);
lean_inc(v_head_126_);
lean_dec(v_x_125_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_137_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_132_; 
lean_inc(v_x_123_);
if (v_isShared_130_ == 0)
{
lean_ctor_set_tag(v___x_129_, 5);
lean_ctor_set(v___x_129_, 1, v_x_123_);
lean_ctor_set(v___x_129_, 0, v_x_124_);
v___x_132_ = v___x_129_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_x_124_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v_x_123_);
v___x_132_ = v_reuseFailAlloc_136_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_126_);
v___x_134_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_132_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
v_x_124_ = v___x_134_;
v_x_125_ = v_tail_127_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(lean_object* v_x_138_, lean_object* v_x_139_, lean_object* v_x_140_){
_start:
{
if (lean_obj_tag(v_x_140_) == 0)
{
lean_dec(v_x_138_);
return v_x_139_;
}
else
{
lean_object* v_head_141_; lean_object* v_tail_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_152_; 
v_head_141_ = lean_ctor_get(v_x_140_, 0);
v_tail_142_ = lean_ctor_get(v_x_140_, 1);
v_isSharedCheck_152_ = !lean_is_exclusive(v_x_140_);
if (v_isSharedCheck_152_ == 0)
{
v___x_144_ = v_x_140_;
v_isShared_145_ = v_isSharedCheck_152_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_tail_142_);
lean_inc(v_head_141_);
lean_dec(v_x_140_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_152_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_147_; 
lean_inc(v_x_138_);
if (v_isShared_145_ == 0)
{
lean_ctor_set_tag(v___x_144_, 5);
lean_ctor_set(v___x_144_, 1, v_x_138_);
lean_ctor_set(v___x_144_, 0, v_x_139_);
v___x_147_ = v___x_144_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_x_139_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v_x_138_);
v___x_147_ = v_reuseFailAlloc_151_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_148_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_141_);
v___x_149_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_147_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = lp_openvm_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(v_x_138_, v___x_149_, v_tail_142_);
return v___x_150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(lean_object* v_x_153_, lean_object* v_x_154_){
_start:
{
if (lean_obj_tag(v_x_153_) == 0)
{
lean_object* v___x_155_; 
lean_dec(v_x_154_);
v___x_155_ = lean_box(0);
return v___x_155_;
}
else
{
lean_object* v_tail_156_; 
v_tail_156_ = lean_ctor_get(v_x_153_, 1);
if (lean_obj_tag(v_tail_156_) == 0)
{
lean_object* v_head_157_; lean_object* v___x_158_; 
lean_dec(v_x_154_);
v_head_157_ = lean_ctor_get(v_x_153_, 0);
lean_inc(v_head_157_);
lean_dec_ref_known(v_x_153_, 2);
v___x_158_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_157_);
return v___x_158_;
}
else
{
lean_object* v_head_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
lean_inc(v_tail_156_);
v_head_159_ = lean_ctor_get(v_x_153_, 0);
lean_inc(v_head_159_);
lean_dec_ref_known(v_x_153_, 2);
v___x_160_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_159_);
v___x_161_ = lp_openvm_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(v_x_154_, v___x_160_, v_tail_156_);
return v___x_161_;
}
}
}
}
static lean_object* _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2));
v___x_171_ = lean_string_length(v___x_170_);
return v___x_171_;
}
}
static lean_object* _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5);
v___x_173_ = lean_nat_to_int(v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(lean_object* v_a_178_){
_start:
{
if (lean_obj_tag(v_a_178_) == 0)
{
lean_object* v___x_179_; 
v___x_179_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_179_;
}
else
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; uint8_t v___x_188_; lean_object* v___x_189_; 
v___x_180_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_181_ = lp_openvm_x2dfv_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(v_a_178_, v___x_180_);
v___x_182_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_183_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_184_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_181_);
v___x_185_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_186_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_184_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
v___x_187_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_187_, 0, v___x_182_);
lean_ctor_set(v___x_187_, 1, v___x_186_);
v___x_188_ = 0;
v___x_189_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_189_, 0, v___x_187_);
lean_ctor_set_uint8(v___x_189_, sizeof(void*)*1, v___x_188_);
return v___x_189_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(lean_object* v_a_190_){
_start:
{
if (lean_obj_tag(v_a_190_) == 0)
{
lean_object* v___x_191_; 
v___x_191_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_191_;
}
else
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; uint8_t v___x_200_; lean_object* v___x_201_; 
v___x_192_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_193_ = lp_swirl_x2dfv_Std_Format_joinSep___at___00Array_repr___at___00instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0_spec__0_spec__1(v_a_190_, v___x_192_);
v___x_194_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_195_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_196_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, v___x_193_);
v___x_197_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_198_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_196_);
lean_ctor_set(v___x_198_, 1, v___x_197_);
v___x_199_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_194_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v___x_200_ = 0;
v___x_201_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_201_, 0, v___x_199_);
lean_ctor_set_uint8(v___x_201_, sizeof(void*)*1, v___x_200_);
return v___x_201_;
}
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lean_unsigned_to_nat(22u);
v___x_212_ = lean_nat_to_int(v___x_211_);
return v___x_212_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_unsigned_to_nat(16u);
v___x_217_ = lean_nat_to_int(v___x_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg(lean_object* v_x_221_){
_start:
{
lean_object* v_authenticationPath_222_; lean_object* v_publicValues_223_; lean_object* v_publicValuesCommit_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; uint8_t v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v_authenticationPath_222_ = lean_ctor_get(v_x_221_, 0);
lean_inc(v_authenticationPath_222_);
v_publicValues_223_ = lean_ctor_get(v_x_221_, 1);
lean_inc(v_publicValues_223_);
v_publicValuesCommit_224_ = lean_ctor_get(v_x_221_, 2);
lean_inc_ref(v_publicValuesCommit_224_);
lean_dec_ref(v_x_221_);
v___x_225_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5));
v___x_226_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3));
v___x_227_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4, &lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4_once, _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4);
v___x_228_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_authenticationPath_222_);
v___x_229_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_227_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = 0;
v___x_231_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_231_, 0, v___x_229_);
lean_ctor_set_uint8(v___x_231_, sizeof(void*)*1, v___x_230_);
v___x_232_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_226_);
lean_ctor_set(v___x_232_, 1, v___x_231_);
v___x_233_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9));
v___x_234_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_232_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = lean_box(1);
v___x_236_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_234_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
v___x_237_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6));
v___x_238_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_236_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
lean_ctor_set(v___x_239_, 1, v___x_225_);
v___x_240_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7, &lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7_once, _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7);
v___x_241_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_publicValues_223_);
v___x_242_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set_uint8(v___x_243_, sizeof(void*)*1, v___x_230_);
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_239_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v___x_233_);
v___x_246_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
lean_ctor_set(v___x_246_, 1, v___x_235_);
v___x_247_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9));
v___x_248_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v___x_225_);
v___x_250_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_publicValuesCommit_224_);
v___x_251_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_227_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set_uint8(v___x_252_, sizeof(void*)*1, v___x_230_);
v___x_253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_249_);
lean_ctor_set(v___x_253_, 1, v___x_252_);
v___x_254_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15);
v___x_255_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16));
v___x_256_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v___x_253_);
v___x_257_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17));
v___x_258_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_256_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
v___x_259_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_254_);
lean_ctor_set(v___x_259_, 1, v___x_258_);
v___x_260_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_260_, 0, v___x_259_);
lean_ctor_set_uint8(v___x_260_, sizeof(void*)*1, v___x_230_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr(lean_object* v_x_261_, lean_object* v_prec_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg(v_x_261_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___boxed(lean_object* v_x_264_, lean_object* v_prec_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr(v_x_264_, v_prec_265_);
lean_dec(v_prec_265_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(lean_object* v_a_267_, lean_object* v_n_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_a_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___boxed(lean_object* v_a_270_, lean_object* v_n_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(v_a_270_, v_n_271_);
lean_dec(v_n_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(lean_object* v_a_273_, lean_object* v_n_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_a_273_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___boxed(lean_object* v_a_276_, lean_object* v_n_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(v_a_276_, v_n_277_);
lean_dec(v_n_277_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_parseVmProofData_x3f(lean_object* v_proof_281_){
_start:
{
lean_object* v_inner_282_; lean_object* v_userPublicValuesProof_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_351_; 
v_inner_282_ = lean_ctor_get(v_proof_281_, 0);
v_userPublicValuesProof_283_ = lean_ctor_get(v_proof_281_, 1);
v_isSharedCheck_351_ = !lean_is_exclusive(v_proof_281_);
if (v_isSharedCheck_351_ == 0)
{
v___x_285_ = v_proof_281_;
v_isShared_286_ = v_isSharedCheck_351_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_userPublicValuesProof_283_);
lean_inc(v_inner_282_);
lean_dec(v_proof_281_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_351_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v_traceVdata_287_; lean_object* v_publicValues_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
v_traceVdata_287_ = lean_ctor_get(v_inner_282_, 1);
lean_inc(v_traceVdata_287_);
v_publicValues_288_ = lean_ctor_get(v_inner_282_, 2);
lean_inc(v_publicValues_288_);
lean_dec_ref(v_inner_282_);
v___x_289_ = lean_unsigned_to_nat(0u);
v___x_290_ = l_List_get_x3fInternal___redArg(v_publicValues_288_, v___x_289_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v___x_291_; 
lean_dec(v_publicValues_288_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_291_ = lean_box(0);
return v___x_291_;
}
else
{
lean_object* v_val_292_; lean_object* v___x_293_; 
v_val_292_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_val_292_);
lean_dec_ref_known(v___x_290_, 1);
v___x_293_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v___x_289_, v_val_292_);
lean_dec(v_val_292_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_object* v___x_294_; 
lean_dec(v_publicValues_288_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_294_ = lean_box(0);
return v___x_294_;
}
else
{
lean_object* v_val_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v_val_295_ = lean_ctor_get(v___x_293_, 0);
lean_inc(v_val_295_);
lean_dec_ref_known(v___x_293_, 1);
v___x_296_ = lean_unsigned_to_nat(1u);
v___x_297_ = l_List_get_x3fInternal___redArg(v_publicValues_288_, v___x_296_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v___x_298_; 
lean_dec(v_val_295_);
lean_dec(v_publicValues_288_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_298_ = lean_box(0);
return v___x_298_;
}
else
{
lean_object* v_val_299_; lean_object* v___x_300_; 
v_val_299_ = lean_ctor_get(v___x_297_, 0);
lean_inc(v_val_299_);
lean_dec_ref_known(v___x_297_, 1);
v___x_300_ = lp_openvm_x2dfv_Recursion_Spec_VmSegmentPublicValues_fromList_x3f___redArg(v_val_299_);
if (lean_obj_tag(v___x_300_) == 0)
{
lean_object* v___x_301_; 
lean_dec(v_val_295_);
lean_dec(v_publicValues_288_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_301_ = lean_box(0);
return v___x_301_;
}
else
{
lean_object* v_val_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v_val_302_ = lean_ctor_get(v___x_300_, 0);
lean_inc(v_val_302_);
lean_dec_ref_known(v___x_300_, 1);
v___x_303_ = lean_unsigned_to_nat(2u);
v___x_304_ = l_List_get_x3fInternal___redArg(v_publicValues_288_, v___x_303_);
lean_dec(v_publicValues_288_);
if (lean_obj_tag(v___x_304_) == 0)
{
lean_object* v___x_305_; 
lean_dec(v_val_302_);
lean_dec(v_val_295_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_305_ = lean_box(0);
return v___x_305_;
}
else
{
lean_object* v_val_306_; uint8_t v___x_307_; 
v_val_306_ = lean_ctor_get(v___x_304_, 0);
lean_inc(v_val_306_);
lean_dec_ref_known(v___x_304_, 1);
v___x_307_ = l_List_isEmpty___redArg(v_val_306_);
lean_dec(v_val_306_);
if (v___x_307_ == 0)
{
lean_object* v___x_308_; 
lean_dec(v_val_302_);
lean_dec(v_val_295_);
lean_dec(v_traceVdata_287_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_308_ = lean_box(0);
return v___x_308_;
}
else
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = lean_unsigned_to_nat(3u);
v___x_310_ = l_List_get_x3fInternal___redArg(v_traceVdata_287_, v___x_309_);
lean_dec(v_traceVdata_287_);
if (lean_obj_tag(v___x_310_) == 0)
{
lean_object* v___x_311_; 
lean_dec(v_val_302_);
lean_dec(v_val_295_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_311_ = lean_box(0);
return v___x_311_;
}
else
{
lean_object* v_val_312_; 
v_val_312_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_val_312_);
lean_dec_ref_known(v___x_310_, 1);
if (lean_obj_tag(v_val_312_) == 0)
{
lean_object* v___x_313_; 
lean_dec(v_val_302_);
lean_dec(v_val_295_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_313_ = lean_box(0);
return v___x_313_;
}
else
{
lean_object* v_val_314_; lean_object* v_cachedCommitments_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_349_; 
v_val_314_ = lean_ctor_get(v_val_312_, 0);
lean_inc(v_val_314_);
lean_dec_ref_known(v_val_312_, 1);
v_cachedCommitments_315_ = lean_ctor_get(v_val_314_, 1);
v_isSharedCheck_349_ = !lean_is_exclusive(v_val_314_);
if (v_isSharedCheck_349_ == 0)
{
lean_object* v_unused_350_; 
v_unused_350_ = lean_ctor_get(v_val_314_, 0);
lean_dec(v_unused_350_);
v___x_317_ = v_val_314_;
v_isShared_318_ = v_isSharedCheck_349_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_cachedCommitments_315_);
lean_dec(v_val_314_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_349_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_319_; 
v___x_319_ = l_List_get_x3fInternal___redArg(v_cachedCommitments_315_, v___x_289_);
lean_dec(v_cachedCommitments_315_);
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v___x_320_; 
lean_del_object(v___x_317_);
lean_dec(v_val_302_);
lean_dec(v_val_295_);
lean_del_object(v___x_285_);
lean_dec_ref(v_userPublicValuesProof_283_);
v___x_320_ = lean_box(0);
return v___x_320_;
}
else
{
lean_object* v_val_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_348_; 
v_val_321_ = lean_ctor_get(v___x_319_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_319_);
if (v_isSharedCheck_348_ == 0)
{
v___x_323_ = v___x_319_;
v_isShared_324_ = v_isSharedCheck_348_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_val_321_);
lean_dec(v___x_319_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_348_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v_internalFlag_325_; lean_object* v_appVkCommit_326_; lean_object* v_leafVkCommit_327_; lean_object* v_internalForLeafVkCommit_328_; lean_object* v_recursionDepth_329_; lean_object* v_internalRecursiveVkCommit_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_347_; 
v_internalFlag_325_ = lean_ctor_get(v_val_295_, 0);
v_appVkCommit_326_ = lean_ctor_get(v_val_295_, 1);
v_leafVkCommit_327_ = lean_ctor_get(v_val_295_, 2);
v_internalForLeafVkCommit_328_ = lean_ctor_get(v_val_295_, 3);
v_recursionDepth_329_ = lean_ctor_get(v_val_295_, 4);
v_internalRecursiveVkCommit_330_ = lean_ctor_get(v_val_295_, 5);
v_isSharedCheck_347_ = !lean_is_exclusive(v_val_295_);
if (v_isSharedCheck_347_ == 0)
{
v___x_332_ = v_val_295_;
v_isShared_333_ = v_isSharedCheck_347_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_internalRecursiveVkCommit_330_);
lean_inc(v_recursionDepth_329_);
lean_inc(v_internalForLeafVkCommit_328_);
lean_inc(v_leafVkCommit_327_);
lean_inc(v_appVkCommit_326_);
lean_inc(v_internalFlag_325_);
lean_dec(v_val_295_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_347_;
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
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v_internalFlag_325_);
lean_ctor_set(v_reuseFailAlloc_346_, 1, v_appVkCommit_326_);
lean_ctor_set(v_reuseFailAlloc_346_, 2, v_leafVkCommit_327_);
lean_ctor_set(v_reuseFailAlloc_346_, 3, v_internalForLeafVkCommit_328_);
lean_ctor_set(v_reuseFailAlloc_346_, 4, v_recursionDepth_329_);
lean_ctor_set(v_reuseFailAlloc_346_, 5, v_internalRecursiveVkCommit_330_);
v___x_335_ = v_reuseFailAlloc_346_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_337_; 
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 1, v_val_302_);
lean_ctor_set(v___x_317_, 0, v___x_335_);
v___x_337_ = v___x_317_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_345_; 
v_reuseFailAlloc_345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_345_, 0, v___x_335_);
lean_ctor_set(v_reuseFailAlloc_345_, 1, v_val_302_);
v___x_337_ = v_reuseFailAlloc_345_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
lean_object* v___x_339_; 
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 1, v_val_321_);
lean_ctor_set(v___x_285_, 0, v___x_337_);
v___x_339_ = v___x_285_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v___x_337_);
lean_ctor_set(v_reuseFailAlloc_344_, 1, v_val_321_);
v___x_339_ = v_reuseFailAlloc_344_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
lean_object* v___x_340_; lean_object* v___x_342_; 
v___x_340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set(v___x_340_, 1, v_userPublicValuesProof_283_);
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 0, v___x_340_);
v___x_342_ = v___x_323_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_340_);
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
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_Recursion_Spec_Common_VerifierPublicValues(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Types(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_Recursion_Spec_Common_VerifierPublicValues(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VmVerifier_constraintEvalCachedIndex = _init_lp_openvm_x2dfv_VmVerifier_constraintEvalCachedIndex();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_constraintEvalCachedIndex);
lp_openvm_x2dfv_VmVerifier_unsetPvsAirId = _init_lp_openvm_x2dfv_VmVerifier_unsetPvsAirId();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_unsetPvsAirId);
lp_openvm_x2dfv_VmVerifier_symbolicExpressionAirId = _init_lp_openvm_x2dfv_VmVerifier_symbolicExpressionAirId();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_symbolicExpressionAirId);
lp_openvm_x2dfv_VmVerifier_addressSpaceOffset = _init_lp_openvm_x2dfv_VmVerifier_addressSpaceOffset();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_addressSpaceOffset);
lp_openvm_x2dfv_VmVerifier_publicValuesAddressSpace = _init_lp_openvm_x2dfv_VmVerifier_publicValuesAddressSpace();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_publicValuesAddressSpace);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
