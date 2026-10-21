// Lean compiler output
// Module: VmVerifier.Spec.Types
// Imports: public import Init public meta import Init public import Fundamentals.Spec.BabyBearExt4.Raw public import Recursion.Spec.Common.VerifierPublicValues public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Main public import VM.Spec.Memory.Spec public import VM.Spec.Registry.Config
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
extern lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig;
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_canonical;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount___boxed(lean_object*);
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VerificationBaseline_publicValuesHeight(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VerificationBaseline_publicValuesHeight___boxed(lean_object*);
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
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_MemoryDimensions_canonical(void){
_start:
{
lean_object* v___x_84_; lean_object* v_addrSpaceHeight_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_84_ = lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig;
v_addrSpaceHeight_85_ = lean_ctor_get(v___x_84_, 6);
v___x_86_ = lean_unsigned_to_nat(26u);
lean_inc(v_addrSpaceHeight_85_);
v___x_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_87_, 0, v_addrSpaceHeight_85_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(lean_object* v_dimensions_88_){
_start:
{
lean_object* v_addrSpaceHeight_89_; lean_object* v_addressHeight_90_; lean_object* v___x_91_; 
v_addrSpaceHeight_89_ = lean_ctor_get(v_dimensions_88_, 0);
v_addressHeight_90_ = lean_ctor_get(v_dimensions_88_, 1);
v___x_91_ = lean_nat_add(v_addrSpaceHeight_89_, v_addressHeight_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight___boxed(lean_object* v_dimensions_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(v_dimensions_92_);
lean_dec_ref(v_dimensions_92_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(lean_object* v_dimensions_94_, lean_object* v_addressSpace_95_, lean_object* v_blockId_96_){
_start:
{
lean_object* v_addressHeight_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_addressHeight_97_ = lean_ctor_get(v_dimensions_94_, 1);
v___x_98_ = lean_unsigned_to_nat(1u);
v___x_99_ = lean_nat_sub(v_addressSpace_95_, v___x_98_);
v___x_100_ = lean_unsigned_to_nat(2u);
v___x_101_ = lean_nat_pow(v___x_100_, v_addressHeight_97_);
v___x_102_ = lean_nat_mul(v___x_99_, v___x_101_);
lean_dec(v___x_101_);
lean_dec(v___x_99_);
v___x_103_ = lean_nat_add(v___x_102_, v_blockId_96_);
lean_dec(v___x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex___boxed(lean_object* v_dimensions_104_, lean_object* v_addressSpace_105_, lean_object* v_blockId_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(v_dimensions_104_, v_addressSpace_105_, v_blockId_106_);
lean_dec(v_blockId_106_);
lean_dec(v_addressSpace_105_);
lean_dec_ref(v_dimensions_104_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(lean_object* v_values_108_){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_109_ = l_List_lengthTR___redArg(v_values_108_);
v___x_110_ = lean_unsigned_to_nat(8u);
v___x_111_ = lean_nat_mod(v___x_109_, v___x_110_);
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = lean_nat_dec_eq(v___x_111_, v___x_112_);
lean_dec(v___x_111_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; 
lean_dec(v___x_109_);
v___x_114_ = lean_box(0);
return v___x_114_;
}
else
{
lean_object* v___x_115_; lean_object* v_chunks_116_; uint8_t v___x_117_; 
v___x_115_ = lean_unsigned_to_nat(3u);
v_chunks_116_ = lean_nat_shiftr(v___x_109_, v___x_115_);
lean_dec(v___x_109_);
v___x_117_ = lean_nat_dec_lt(v___x_112_, v_chunks_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; 
lean_dec(v_chunks_116_);
v___x_118_ = lean_box(0);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_119_ = lean_unsigned_to_nat(2u);
v___x_120_ = lean_nat_log2(v_chunks_116_);
v___x_121_ = lean_nat_pow(v___x_119_, v___x_120_);
v___x_122_ = lean_nat_dec_eq(v___x_121_, v_chunks_116_);
lean_dec(v_chunks_116_);
lean_dec(v___x_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; 
lean_dec(v___x_120_);
v___x_123_ = lean_box(0);
return v___x_123_;
}
else
{
lean_object* v___x_124_; 
v___x_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_124_, 0, v___x_120_);
return v___x_124_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f___boxed(lean_object* v_values_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(v_values_125_);
lean_dec(v_values_125_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount(lean_object* v_numPublicValues_127_){
_start:
{
lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_128_ = lean_unsigned_to_nat(3u);
v___x_129_ = lean_nat_shiftr(v_numPublicValues_127_, v___x_128_);
v___x_130_ = lean_nat_log2(v___x_129_);
lean_dec(v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount___boxed(lean_object* v_numPublicValues_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount(v_numPublicValues_131_);
lean_dec(v_numPublicValues_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_133_, lean_object* v_x_134_, lean_object* v_x_135_){
_start:
{
if (lean_obj_tag(v_x_135_) == 0)
{
lean_dec(v_x_133_);
return v_x_134_;
}
else
{
lean_object* v_head_136_; lean_object* v_tail_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_147_; 
v_head_136_ = lean_ctor_get(v_x_135_, 0);
v_tail_137_ = lean_ctor_get(v_x_135_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v_x_135_);
if (v_isSharedCheck_147_ == 0)
{
v___x_139_ = v_x_135_;
v_isShared_140_ = v_isSharedCheck_147_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_tail_137_);
lean_inc(v_head_136_);
lean_dec(v_x_135_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_147_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v___x_142_; 
lean_inc(v_x_133_);
if (v_isShared_140_ == 0)
{
lean_ctor_set_tag(v___x_139_, 5);
lean_ctor_set(v___x_139_, 1, v_x_133_);
lean_ctor_set(v___x_139_, 0, v_x_134_);
v___x_142_ = v___x_139_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_x_134_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v_x_133_);
v___x_142_ = v_reuseFailAlloc_146_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_143_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_136_);
v___x_144_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_142_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v_x_134_ = v___x_144_;
v_x_135_ = v_tail_137_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(lean_object* v_x_148_, lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
if (lean_obj_tag(v_x_150_) == 0)
{
lean_dec(v_x_148_);
return v_x_149_;
}
else
{
lean_object* v_head_151_; lean_object* v_tail_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_162_; 
v_head_151_ = lean_ctor_get(v_x_150_, 0);
v_tail_152_ = lean_ctor_get(v_x_150_, 1);
v_isSharedCheck_162_ = !lean_is_exclusive(v_x_150_);
if (v_isSharedCheck_162_ == 0)
{
v___x_154_ = v_x_150_;
v_isShared_155_ = v_isSharedCheck_162_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_tail_152_);
lean_inc(v_head_151_);
lean_dec(v_x_150_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_162_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
lean_inc(v_x_148_);
if (v_isShared_155_ == 0)
{
lean_ctor_set_tag(v___x_154_, 5);
lean_ctor_set(v___x_154_, 1, v_x_148_);
lean_ctor_set(v___x_154_, 0, v_x_149_);
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_x_149_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_x_148_);
v___x_157_ = v_reuseFailAlloc_161_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_158_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_151_);
v___x_159_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_157_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = lp_openvm_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1_spec__3(v_x_148_, v___x_159_, v_tail_152_);
return v___x_160_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(lean_object* v_x_163_, lean_object* v_x_164_){
_start:
{
if (lean_obj_tag(v_x_163_) == 0)
{
lean_object* v___x_165_; 
lean_dec(v_x_164_);
v___x_165_ = lean_box(0);
return v___x_165_;
}
else
{
lean_object* v_tail_166_; 
v_tail_166_ = lean_ctor_get(v_x_163_, 1);
if (lean_obj_tag(v_tail_166_) == 0)
{
lean_object* v_head_167_; lean_object* v___x_168_; 
lean_dec(v_x_164_);
v_head_167_ = lean_ctor_get(v_x_163_, 0);
lean_inc(v_head_167_);
lean_dec_ref_known(v_x_163_, 2);
v___x_168_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_167_);
return v___x_168_;
}
else
{
lean_object* v_head_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
lean_inc(v_tail_166_);
v_head_169_ = lean_ctor_get(v_x_163_, 0);
lean_inc(v_head_169_);
lean_dec_ref_known(v_x_163_, 2);
v___x_170_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_head_169_);
v___x_171_ = lp_openvm_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0_spec__1(v_x_164_, v___x_170_, v_tail_166_);
return v___x_171_;
}
}
}
}
static lean_object* _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__2));
v___x_181_ = lean_string_length(v___x_180_);
return v___x_181_;
}
}
static lean_object* _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; 
v___x_182_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__5);
v___x_183_ = lean_nat_to_int(v___x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(lean_object* v_a_188_){
_start:
{
if (lean_obj_tag(v_a_188_) == 0)
{
lean_object* v___x_189_; 
v___x_189_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_189_;
}
else
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; uint8_t v___x_198_; lean_object* v___x_199_; 
v___x_190_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_191_ = lp_openvm_x2dfv_Std_Format_joinSep___at___00List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0_spec__0(v_a_188_, v___x_190_);
v___x_192_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_193_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_194_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v___x_191_);
v___x_195_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_196_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_194_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
v___x_197_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_192_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
v___x_198_ = 0;
v___x_199_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_199_, 0, v___x_197_);
lean_ctor_set_uint8(v___x_199_, sizeof(void*)*1, v___x_198_);
return v___x_199_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(lean_object* v_a_200_){
_start:
{
if (lean_obj_tag(v_a_200_) == 0)
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__1));
return v___x_201_;
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; uint8_t v___x_210_; lean_object* v___x_211_; 
v___x_202_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__3));
v___x_203_ = lp_swirl_x2dfv_Std_Format_joinSep___at___00Array_repr___at___00instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0_spec__0_spec__1(v_a_200_, v___x_202_);
v___x_204_ = lean_obj_once(&lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6, &lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6_once, _init_lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__6);
v___x_205_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__7));
v___x_206_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
lean_ctor_set(v___x_206_, 1, v___x_203_);
v___x_207_ = ((lean_object*)(lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg___closed__8));
v___x_208_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_206_);
lean_ctor_set(v___x_208_, 1, v___x_207_);
v___x_209_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_204_);
lean_ctor_set(v___x_209_, 1, v___x_208_);
v___x_210_ = 0;
v___x_211_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_211_, 0, v___x_209_);
lean_ctor_set_uint8(v___x_211_, sizeof(void*)*1, v___x_210_);
return v___x_211_;
}
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_221_ = lean_unsigned_to_nat(22u);
v___x_222_ = lean_nat_to_int(v___x_221_);
return v___x_222_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_226_ = lean_unsigned_to_nat(16u);
v___x_227_ = lean_nat_to_int(v___x_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg(lean_object* v_x_231_){
_start:
{
lean_object* v_authenticationPath_232_; lean_object* v_publicValues_233_; lean_object* v_publicValuesCommit_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; uint8_t v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
v_authenticationPath_232_ = lean_ctor_get(v_x_231_, 0);
lean_inc(v_authenticationPath_232_);
v_publicValues_233_ = lean_ctor_get(v_x_231_, 1);
lean_inc(v_publicValues_233_);
v_publicValuesCommit_234_ = lean_ctor_get(v_x_231_, 2);
lean_inc_ref(v_publicValuesCommit_234_);
lean_dec_ref(v_x_231_);
v___x_235_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__5));
v___x_236_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__3));
v___x_237_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4, &lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4_once, _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__4);
v___x_238_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_authenticationPath_232_);
v___x_239_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_237_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = 0;
v___x_241_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_241_, 0, v___x_239_);
lean_ctor_set_uint8(v___x_241_, sizeof(void*)*1, v___x_240_);
v___x_242_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_236_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__9));
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_242_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = lean_box(1);
v___x_246_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_244_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__6));
v___x_248_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v___x_235_);
v___x_250_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7, &lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7_once, _init_lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__7);
v___x_251_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_publicValues_233_);
v___x_252_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_250_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
v___x_253_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set_uint8(v___x_253_, sizeof(void*)*1, v___x_240_);
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_249_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v___x_243_);
v___x_256_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v___x_245_);
v___x_257_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg___closed__9));
v___x_258_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_256_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
v___x_259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set(v___x_259_, 1, v___x_235_);
v___x_260_ = lp_swirl_x2dfv_instReprVector_repr___at___00Fundamentals_BabyBearExt4_instReprRaw_repr_spec__0___redArg(v_publicValuesCommit_234_);
v___x_261_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_237_);
lean_ctor_set(v___x_261_, 1, v___x_260_);
v___x_262_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set_uint8(v___x_262_, sizeof(void*)*1, v___x_240_);
v___x_263_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_259_);
lean_ctor_set(v___x_263_, 1, v___x_262_);
v___x_264_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15, &lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15_once, _init_lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__15);
v___x_265_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__16));
v___x_266_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
lean_ctor_set(v___x_266_, 1, v___x_263_);
v___x_267_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_instReprMemoryDimensions_repr___redArg___closed__17));
v___x_268_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
v___x_269_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_264_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
v___x_270_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_270_, 0, v___x_269_);
lean_ctor_set_uint8(v___x_270_, sizeof(void*)*1, v___x_240_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr(lean_object* v_x_271_, lean_object* v_prec_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___redArg(v_x_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr___boxed(lean_object* v_x_274_, lean_object* v_prec_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_openvm_x2dfv_VmVerifier_instReprUserPublicValuesProof_repr(v_x_274_, v_prec_275_);
lean_dec(v_prec_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(lean_object* v_a_277_, lean_object* v_n_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___redArg(v_a_277_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0___boxed(lean_object* v_a_280_, lean_object* v_n_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__0(v_a_280_, v_n_281_);
lean_dec(v_n_281_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(lean_object* v_a_283_, lean_object* v_n_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___redArg(v_a_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1___boxed(lean_object* v_a_286_, lean_object* v_n_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_openvm_x2dfv_List_repr___at___00VmVerifier_instReprUserPublicValuesProof_repr_spec__1(v_a_286_, v_n_287_);
lean_dec(v_n_287_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VerificationBaseline_publicValuesHeight(lean_object* v_baseline_291_){
_start:
{
lean_object* v_numUserPvs_292_; lean_object* v___x_293_; 
v_numUserPvs_292_ = lean_ctor_get(v_baseline_291_, 4);
v___x_293_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeightFromCount(v_numUserPvs_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VerificationBaseline_publicValuesHeight___boxed(lean_object* v_baseline_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_openvm_x2dfv_VmVerifier_VerificationBaseline_publicValuesHeight(v_baseline_294_);
lean_dec_ref(v_baseline_294_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_parseVmProofData_x3f(lean_object* v_proof_296_){
_start:
{
lean_object* v_inner_297_; lean_object* v_userPublicValuesProof_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_366_; 
v_inner_297_ = lean_ctor_get(v_proof_296_, 0);
v_userPublicValuesProof_298_ = lean_ctor_get(v_proof_296_, 1);
v_isSharedCheck_366_ = !lean_is_exclusive(v_proof_296_);
if (v_isSharedCheck_366_ == 0)
{
v___x_300_ = v_proof_296_;
v_isShared_301_ = v_isSharedCheck_366_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_userPublicValuesProof_298_);
lean_inc(v_inner_297_);
lean_dec(v_proof_296_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_366_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v_traceVdata_302_; lean_object* v_publicValues_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v_traceVdata_302_ = lean_ctor_get(v_inner_297_, 1);
lean_inc(v_traceVdata_302_);
v_publicValues_303_ = lean_ctor_get(v_inner_297_, 2);
lean_inc(v_publicValues_303_);
lean_dec_ref(v_inner_297_);
v___x_304_ = lean_unsigned_to_nat(0u);
v___x_305_ = l_List_get_x3fInternal___redArg(v_publicValues_303_, v___x_304_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v___x_306_; 
lean_dec(v_publicValues_303_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_306_ = lean_box(0);
return v___x_306_;
}
else
{
lean_object* v_val_307_; lean_object* v___x_308_; 
v_val_307_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_val_307_);
lean_dec_ref_known(v___x_305_, 1);
v___x_308_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v___x_304_, v_val_307_);
lean_dec(v_val_307_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v___x_309_; 
lean_dec(v_publicValues_303_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_309_ = lean_box(0);
return v___x_309_;
}
else
{
lean_object* v_val_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v_val_310_ = lean_ctor_get(v___x_308_, 0);
lean_inc(v_val_310_);
lean_dec_ref_known(v___x_308_, 1);
v___x_311_ = lean_unsigned_to_nat(1u);
v___x_312_ = l_List_get_x3fInternal___redArg(v_publicValues_303_, v___x_311_);
if (lean_obj_tag(v___x_312_) == 0)
{
lean_object* v___x_313_; 
lean_dec(v_val_310_);
lean_dec(v_publicValues_303_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_313_ = lean_box(0);
return v___x_313_;
}
else
{
lean_object* v_val_314_; lean_object* v___x_315_; 
v_val_314_ = lean_ctor_get(v___x_312_, 0);
lean_inc(v_val_314_);
lean_dec_ref_known(v___x_312_, 1);
v___x_315_ = lp_openvm_x2dfv_Recursion_Spec_VmSegmentPublicValues_fromList_x3f___redArg(v_val_314_);
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v___x_316_; 
lean_dec(v_val_310_);
lean_dec(v_publicValues_303_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_316_ = lean_box(0);
return v___x_316_;
}
else
{
lean_object* v_val_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v_val_317_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_val_317_);
lean_dec_ref_known(v___x_315_, 1);
v___x_318_ = lean_unsigned_to_nat(2u);
v___x_319_ = l_List_get_x3fInternal___redArg(v_publicValues_303_, v___x_318_);
lean_dec(v_publicValues_303_);
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v___x_320_; 
lean_dec(v_val_317_);
lean_dec(v_val_310_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_320_ = lean_box(0);
return v___x_320_;
}
else
{
lean_object* v_val_321_; uint8_t v___x_322_; 
v_val_321_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_val_321_);
lean_dec_ref_known(v___x_319_, 1);
v___x_322_ = l_List_isEmpty___redArg(v_val_321_);
lean_dec(v_val_321_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; 
lean_dec(v_val_317_);
lean_dec(v_val_310_);
lean_dec(v_traceVdata_302_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_323_ = lean_box(0);
return v___x_323_;
}
else
{
lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_324_ = lean_unsigned_to_nat(3u);
v___x_325_ = l_List_get_x3fInternal___redArg(v_traceVdata_302_, v___x_324_);
lean_dec(v_traceVdata_302_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v___x_326_; 
lean_dec(v_val_317_);
lean_dec(v_val_310_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_326_ = lean_box(0);
return v___x_326_;
}
else
{
lean_object* v_val_327_; 
v_val_327_ = lean_ctor_get(v___x_325_, 0);
lean_inc(v_val_327_);
lean_dec_ref_known(v___x_325_, 1);
if (lean_obj_tag(v_val_327_) == 0)
{
lean_object* v___x_328_; 
lean_dec(v_val_317_);
lean_dec(v_val_310_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_328_ = lean_box(0);
return v___x_328_;
}
else
{
lean_object* v_val_329_; lean_object* v_cachedCommitments_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_364_; 
v_val_329_ = lean_ctor_get(v_val_327_, 0);
lean_inc(v_val_329_);
lean_dec_ref_known(v_val_327_, 1);
v_cachedCommitments_330_ = lean_ctor_get(v_val_329_, 1);
v_isSharedCheck_364_ = !lean_is_exclusive(v_val_329_);
if (v_isSharedCheck_364_ == 0)
{
lean_object* v_unused_365_; 
v_unused_365_ = lean_ctor_get(v_val_329_, 0);
lean_dec(v_unused_365_);
v___x_332_ = v_val_329_;
v_isShared_333_ = v_isSharedCheck_364_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_cachedCommitments_330_);
lean_dec(v_val_329_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_364_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_334_; 
v___x_334_ = l_List_get_x3fInternal___redArg(v_cachedCommitments_330_, v___x_304_);
lean_dec(v_cachedCommitments_330_);
if (lean_obj_tag(v___x_334_) == 0)
{
lean_object* v___x_335_; 
lean_del_object(v___x_332_);
lean_dec(v_val_317_);
lean_dec(v_val_310_);
lean_del_object(v___x_300_);
lean_dec_ref(v_userPublicValuesProof_298_);
v___x_335_ = lean_box(0);
return v___x_335_;
}
else
{
lean_object* v_val_336_; lean_object* v___x_338_; uint8_t v_isShared_339_; uint8_t v_isSharedCheck_363_; 
v_val_336_ = lean_ctor_get(v___x_334_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_334_);
if (v_isSharedCheck_363_ == 0)
{
v___x_338_ = v___x_334_;
v_isShared_339_ = v_isSharedCheck_363_;
goto v_resetjp_337_;
}
else
{
lean_inc(v_val_336_);
lean_dec(v___x_334_);
v___x_338_ = lean_box(0);
v_isShared_339_ = v_isSharedCheck_363_;
goto v_resetjp_337_;
}
v_resetjp_337_:
{
lean_object* v_internalFlag_340_; lean_object* v_appVkCommit_341_; lean_object* v_leafVkCommit_342_; lean_object* v_internalForLeafVkCommit_343_; lean_object* v_recursionDepth_344_; lean_object* v_internalRecursiveVkCommit_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_362_; 
v_internalFlag_340_ = lean_ctor_get(v_val_310_, 0);
v_appVkCommit_341_ = lean_ctor_get(v_val_310_, 1);
v_leafVkCommit_342_ = lean_ctor_get(v_val_310_, 2);
v_internalForLeafVkCommit_343_ = lean_ctor_get(v_val_310_, 3);
v_recursionDepth_344_ = lean_ctor_get(v_val_310_, 4);
v_internalRecursiveVkCommit_345_ = lean_ctor_get(v_val_310_, 5);
v_isSharedCheck_362_ = !lean_is_exclusive(v_val_310_);
if (v_isSharedCheck_362_ == 0)
{
v___x_347_ = v_val_310_;
v_isShared_348_ = v_isSharedCheck_362_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_internalRecursiveVkCommit_345_);
lean_inc(v_recursionDepth_344_);
lean_inc(v_internalForLeafVkCommit_343_);
lean_inc(v_leafVkCommit_342_);
lean_inc(v_appVkCommit_341_);
lean_inc(v_internalFlag_340_);
lean_dec(v_val_310_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_362_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_350_; 
if (v_isShared_348_ == 0)
{
v___x_350_ = v___x_347_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_internalFlag_340_);
lean_ctor_set(v_reuseFailAlloc_361_, 1, v_appVkCommit_341_);
lean_ctor_set(v_reuseFailAlloc_361_, 2, v_leafVkCommit_342_);
lean_ctor_set(v_reuseFailAlloc_361_, 3, v_internalForLeafVkCommit_343_);
lean_ctor_set(v_reuseFailAlloc_361_, 4, v_recursionDepth_344_);
lean_ctor_set(v_reuseFailAlloc_361_, 5, v_internalRecursiveVkCommit_345_);
v___x_350_ = v_reuseFailAlloc_361_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
lean_object* v___x_352_; 
if (v_isShared_333_ == 0)
{
lean_ctor_set(v___x_332_, 1, v_val_317_);
lean_ctor_set(v___x_332_, 0, v___x_350_);
v___x_352_ = v___x_332_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_val_317_);
v___x_352_ = v_reuseFailAlloc_360_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
lean_object* v___x_354_; 
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 1, v_val_336_);
lean_ctor_set(v___x_300_, 0, v___x_352_);
v___x_354_ = v___x_300_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_352_);
lean_ctor_set(v_reuseFailAlloc_359_, 1, v_val_336_);
v___x_354_ = v_reuseFailAlloc_359_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
lean_object* v___x_355_; lean_object* v___x_357_; 
v___x_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v_userPublicValuesProof_298_);
if (v_isShared_339_ == 0)
{
lean_ctor_set(v___x_338_, 0, v___x_355_);
v___x_357_ = v___x_338_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(1, 1, 0);
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
lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Spec(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Registry_Config(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Types(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
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
res = initialize_openvm_x2dfv_VM_Spec_Memory_Spec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Registry_Config(builtin);
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
lp_openvm_x2dfv_VmVerifier_MemoryDimensions_canonical = _init_lp_openvm_x2dfv_VmVerifier_MemoryDimensions_canonical();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_MemoryDimensions_canonical);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
