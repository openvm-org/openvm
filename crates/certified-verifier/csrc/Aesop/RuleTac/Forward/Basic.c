// Lean compiler output
// Module: Aesop.RuleTac.Forward.Basic
// Imports: public import Init public meta import Init public import Aesop.Util.Basic import Lean.Meta.Tactic.Clear
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
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_LocalContext_setKind(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_LocalInstances_erase(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_MVarId_tryClearMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_usize_to_nat(size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_components(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Name_ofComponents(lean_object*);
lean_object* l_Lean_LocalContext_findFromUserName_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVarAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
static const lean_string_object lp_aesop_Aesop_forwardHypPrefix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fwd"};
static const lean_object* lp_aesop_Aesop_forwardHypPrefix___closed__0 = (const lean_object*)&lp_aesop_Aesop_forwardHypPrefix___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_forwardHypPrefix___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_forwardHypPrefix___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 83, 158, 36, 61, 4, 76, 239)}};
static const lean_object* lp_aesop_Aesop_forwardHypPrefix___closed__1 = (const lean_object*)&lp_aesop_Aesop_forwardHypPrefix___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_forwardHypPrefix = (const lean_object*)&lp_aesop_Aesop_forwardHypPrefix___closed__1_value;
static const lean_string_object lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "__aesop"};
static const lean_object* lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__0 = (const lean_object*)&lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__0_value),LEAN_SCALAR_PTR_LITERAL(189, 108, 171, 87, 85, 129, 67, 9)}};
static const lean_ctor_object lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_forwardHypPrefix___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 11, 67, 80, 159, 56, 181, 93)}};
static const lean_object* lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1 = (const lean_object*)&lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_forwardImplDetailHypPrefix = (const lean_object*)&lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardImplDetailHypName(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchForwardImplDetailHypName(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardImplDetailHypName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardImplDetailHypName___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardImplDetailHyp(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardImplDetailHyp___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_getForwardImplDetailHyps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_getForwardImplDetailHyps___closed__0 = (const lean_object*)&lp_aesop_Aesop_getForwardImplDetailHyps___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardImplDetailHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardHypData_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardHypData;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardHypData(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardHypData___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardImplDetailHypName(lean_object* v_fwdHypName_10_, lean_object* v_depth_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_12_ = ((lean_object*)(lp_aesop_Aesop_forwardImplDetailHypPrefix));
v___x_13_ = l_Lean_Name_num___override(v___x_12_, v_depth_11_);
v___x_14_ = l_Lean_Name_append(v___x_13_, v_fwdHypName_10_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchForwardImplDetailHypName(lean_object* v_n_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = l_Lean_Name_components(v_n_15_);
if (lean_obj_tag(v___x_16_) == 1)
{
lean_object* v_head_17_; 
v_head_17_ = lean_ctor_get(v___x_16_, 0);
lean_inc(v_head_17_);
if (lean_obj_tag(v_head_17_) == 1)
{
lean_object* v_pre_18_; 
v_pre_18_ = lean_ctor_get(v_head_17_, 0);
if (lean_obj_tag(v_pre_18_) == 0)
{
lean_object* v_tail_19_; lean_object* v_str_20_; lean_object* v___x_21_; uint8_t v___x_22_; 
v_tail_19_ = lean_ctor_get(v___x_16_, 1);
lean_inc(v_tail_19_);
lean_dec_ref_known(v___x_16_, 2);
v_str_20_ = lean_ctor_get(v_head_17_, 1);
lean_inc_ref(v_str_20_);
lean_dec_ref_known(v_head_17_, 2);
v___x_21_ = ((lean_object*)(lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__0));
v___x_22_ = lean_string_dec_eq(v_str_20_, v___x_21_);
lean_dec_ref(v_str_20_);
if (v___x_22_ == 0)
{
lean_object* v___x_23_; 
lean_dec(v_tail_19_);
v___x_23_ = lean_box(0);
return v___x_23_;
}
else
{
if (lean_obj_tag(v_tail_19_) == 1)
{
lean_object* v_head_24_; 
v_head_24_ = lean_ctor_get(v_tail_19_, 0);
lean_inc(v_head_24_);
if (lean_obj_tag(v_head_24_) == 1)
{
lean_object* v_pre_25_; 
v_pre_25_ = lean_ctor_get(v_head_24_, 0);
if (lean_obj_tag(v_pre_25_) == 0)
{
lean_object* v_tail_26_; lean_object* v_str_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
v_tail_26_ = lean_ctor_get(v_tail_19_, 1);
lean_inc(v_tail_26_);
lean_dec_ref_known(v_tail_19_, 2);
v_str_27_ = lean_ctor_get(v_head_24_, 1);
lean_inc_ref(v_str_27_);
lean_dec_ref_known(v_head_24_, 2);
v___x_28_ = ((lean_object*)(lp_aesop_Aesop_forwardHypPrefix___closed__0));
v___x_29_ = lean_string_dec_eq(v_str_27_, v___x_28_);
lean_dec_ref(v_str_27_);
if (v___x_29_ == 0)
{
lean_object* v___x_30_; 
lean_dec(v_tail_26_);
v___x_30_ = lean_box(0);
return v___x_30_;
}
else
{
if (lean_obj_tag(v_tail_26_) == 1)
{
lean_object* v_head_31_; 
v_head_31_ = lean_ctor_get(v_tail_26_, 0);
lean_inc(v_head_31_);
if (lean_obj_tag(v_head_31_) == 2)
{
lean_object* v_pre_32_; 
v_pre_32_ = lean_ctor_get(v_head_31_, 0);
if (lean_obj_tag(v_pre_32_) == 0)
{
lean_object* v_tail_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_43_; 
v_tail_33_ = lean_ctor_get(v_tail_26_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v_tail_26_);
if (v_isSharedCheck_43_ == 0)
{
lean_object* v_unused_44_; 
v_unused_44_ = lean_ctor_get(v_tail_26_, 0);
lean_dec(v_unused_44_);
v___x_35_ = v_tail_26_;
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_tail_33_);
lean_dec(v_tail_26_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v_i_37_; lean_object* v_name_38_; lean_object* v___x_40_; 
v_i_37_ = lean_ctor_get(v_head_31_, 1);
lean_inc(v_i_37_);
lean_dec_ref_known(v_head_31_, 2);
v_name_38_ = lp_aesop_Aesop_Name_ofComponents(v_tail_33_);
if (v_isShared_36_ == 0)
{
lean_ctor_set_tag(v___x_35_, 0);
lean_ctor_set(v___x_35_, 1, v_name_38_);
lean_ctor_set(v___x_35_, 0, v_i_37_);
v___x_40_ = v___x_35_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_i_37_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_name_38_);
v___x_40_ = v_reuseFailAlloc_42_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
lean_object* v___x_41_; 
v___x_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
return v___x_41_;
}
}
}
else
{
lean_object* v___x_45_; 
lean_dec_ref_known(v_head_31_, 2);
lean_dec_ref_known(v_tail_26_, 2);
v___x_45_ = lean_box(0);
return v___x_45_;
}
}
else
{
lean_object* v___x_46_; 
lean_dec_ref_known(v_tail_26_, 2);
lean_dec(v_head_31_);
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
else
{
lean_object* v___x_47_; 
lean_dec(v_tail_26_);
v___x_47_ = lean_box(0);
return v___x_47_;
}
}
}
else
{
lean_object* v___x_48_; 
lean_dec_ref_known(v_head_24_, 2);
lean_dec_ref_known(v_tail_19_, 2);
v___x_48_ = lean_box(0);
return v___x_48_;
}
}
else
{
lean_object* v___x_49_; 
lean_dec_ref_known(v_tail_19_, 2);
lean_dec(v_head_24_);
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
else
{
lean_object* v___x_50_; 
lean_dec(v_tail_19_);
v___x_50_ = lean_box(0);
return v___x_50_;
}
}
}
else
{
lean_object* v___x_51_; 
lean_dec_ref_known(v_head_17_, 2);
lean_dec_ref_known(v___x_16_, 2);
v___x_51_ = lean_box(0);
return v___x_51_;
}
}
else
{
lean_object* v___x_52_; 
lean_dec_ref_known(v___x_16_, 2);
lean_dec(v_head_17_);
v___x_52_ = lean_box(0);
return v___x_52_;
}
}
else
{
lean_object* v___x_53_; 
lean_dec(v___x_16_);
v___x_53_ = lean_box(0);
return v___x_53_;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardImplDetailHypName(lean_object* v_n_54_){
_start:
{
lean_object* v___x_55_; uint8_t v___x_56_; 
v___x_55_ = ((lean_object*)(lp_aesop_Aesop_forwardImplDetailHypPrefix___closed__1));
v___x_56_ = l_Lean_Name_isPrefixOf(v___x_55_, v_n_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardImplDetailHypName___boxed(lean_object* v_n_57_){
_start:
{
uint8_t v_res_58_; lean_object* v_r_59_; 
v_res_58_ = lp_aesop_Aesop_isForwardImplDetailHypName(v_n_57_);
lean_dec(v_n_57_);
v_r_59_ = lean_box(v_res_58_);
return v_r_59_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardImplDetailHyp(lean_object* v_ldecl_60_){
_start:
{
uint8_t v___x_61_; 
v___x_61_ = l_Lean_LocalDecl_isImplementationDetail(v_ldecl_60_);
if (v___x_61_ == 0)
{
return v___x_61_;
}
else
{
lean_object* v___x_62_; uint8_t v___x_63_; 
v___x_62_ = l_Lean_LocalDecl_userName(v_ldecl_60_);
v___x_63_ = lp_aesop_Aesop_isForwardImplDetailHypName(v___x_62_);
lean_dec(v___x_62_);
return v___x_63_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardImplDetailHyp___boxed(lean_object* v_ldecl_64_){
_start:
{
uint8_t v_res_65_; lean_object* v_r_66_; 
v_res_65_ = lp_aesop_Aesop_isForwardImplDetailHyp(v_ldecl_64_);
lean_dec_ref(v_ldecl_64_);
v_r_66_ = lean_box(v_res_65_);
return v_r_66_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(lean_object* v_as_67_, size_t v_sz_68_, size_t v_i_69_, lean_object* v_b_70_){
_start:
{
uint8_t v___x_72_; 
v___x_72_ = lean_usize_dec_lt(v_i_69_, v_sz_68_);
if (v___x_72_ == 0)
{
lean_object* v___x_73_; 
v___x_73_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_73_, 0, v_b_70_);
return v___x_73_;
}
else
{
lean_object* v_snd_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_91_; 
v_snd_74_ = lean_ctor_get(v_b_70_, 1);
v_isSharedCheck_91_ = !lean_is_exclusive(v_b_70_);
if (v_isSharedCheck_91_ == 0)
{
lean_object* v_unused_92_; 
v_unused_92_ = lean_ctor_get(v_b_70_, 0);
lean_dec(v_unused_92_);
v___x_76_ = v_b_70_;
v_isShared_77_ = v_isSharedCheck_91_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_snd_74_);
lean_dec(v_b_70_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_91_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_78_; lean_object* v_a_80_; lean_object* v_a_87_; 
v___x_78_ = lean_box(0);
v_a_87_ = lean_array_uget_borrowed(v_as_67_, v_i_69_);
if (lean_obj_tag(v_a_87_) == 0)
{
v_a_80_ = v_snd_74_;
goto v___jp_79_;
}
else
{
lean_object* v_val_88_; uint8_t v___x_89_; 
v_val_88_ = lean_ctor_get(v_a_87_, 0);
v___x_89_ = lp_aesop_Aesop_isForwardImplDetailHyp(v_val_88_);
if (v___x_89_ == 0)
{
v_a_80_ = v_snd_74_;
goto v___jp_79_;
}
else
{
lean_object* v___x_90_; 
lean_inc(v_val_88_);
v___x_90_ = lean_array_push(v_snd_74_, v_val_88_);
v_a_80_ = v___x_90_;
goto v___jp_79_;
}
}
v___jp_79_:
{
lean_object* v___x_82_; 
if (v_isShared_77_ == 0)
{
lean_ctor_set(v___x_76_, 1, v_a_80_);
lean_ctor_set(v___x_76_, 0, v___x_78_);
v___x_82_ = v___x_76_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v___x_78_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v_a_80_);
v___x_82_ = v_reuseFailAlloc_86_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
size_t v___x_83_; size_t v___x_84_; 
v___x_83_ = ((size_t)1ULL);
v___x_84_ = lean_usize_add(v_i_69_, v___x_83_);
v_i_69_ = v___x_84_;
v_b_70_ = v___x_82_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_as_93_, lean_object* v_sz_94_, lean_object* v_i_95_, lean_object* v_b_96_, lean_object* v___y_97_){
_start:
{
size_t v_sz_boxed_98_; size_t v_i_boxed_99_; lean_object* v_res_100_; 
v_sz_boxed_98_ = lean_unbox_usize(v_sz_94_);
lean_dec(v_sz_94_);
v_i_boxed_99_ = lean_unbox_usize(v_i_95_);
lean_dec(v_i_95_);
v_res_100_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_93_, v_sz_boxed_98_, v_i_boxed_99_, v_b_96_);
lean_dec_ref(v_as_93_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1(lean_object* v_as_101_, size_t v_sz_102_, size_t v_i_103_, lean_object* v_b_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
uint8_t v___x_110_; 
v___x_110_ = lean_usize_dec_lt(v_i_103_, v_sz_102_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; 
v___x_111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_111_, 0, v_b_104_);
return v___x_111_;
}
else
{
lean_object* v_snd_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_129_; 
v_snd_112_ = lean_ctor_get(v_b_104_, 1);
v_isSharedCheck_129_ = !lean_is_exclusive(v_b_104_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v_b_104_, 0);
lean_dec(v_unused_130_);
v___x_114_ = v_b_104_;
v_isShared_115_ = v_isSharedCheck_129_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_snd_112_);
lean_dec(v_b_104_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_129_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_116_; lean_object* v_a_118_; lean_object* v_a_125_; 
v___x_116_ = lean_box(0);
v_a_125_ = lean_array_uget_borrowed(v_as_101_, v_i_103_);
if (lean_obj_tag(v_a_125_) == 0)
{
v_a_118_ = v_snd_112_;
goto v___jp_117_;
}
else
{
lean_object* v_val_126_; uint8_t v___x_127_; 
v_val_126_ = lean_ctor_get(v_a_125_, 0);
v___x_127_ = lp_aesop_Aesop_isForwardImplDetailHyp(v_val_126_);
if (v___x_127_ == 0)
{
v_a_118_ = v_snd_112_;
goto v___jp_117_;
}
else
{
lean_object* v___x_128_; 
lean_inc(v_val_126_);
v___x_128_ = lean_array_push(v_snd_112_, v_val_126_);
v_a_118_ = v___x_128_;
goto v___jp_117_;
}
}
v___jp_117_:
{
lean_object* v___x_120_; 
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 1, v_a_118_);
lean_ctor_set(v___x_114_, 0, v___x_116_);
v___x_120_ = v___x_114_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_116_);
lean_ctor_set(v_reuseFailAlloc_124_, 1, v_a_118_);
v___x_120_ = v_reuseFailAlloc_124_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
size_t v___x_121_; size_t v___x_122_; lean_object* v___x_123_; 
v___x_121_ = ((size_t)1ULL);
v___x_122_ = lean_usize_add(v_i_103_, v___x_121_);
v___x_123_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_101_, v_sz_102_, v___x_122_, v___x_120_);
return v___x_123_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1___boxed(lean_object* v_as_131_, lean_object* v_sz_132_, lean_object* v_i_133_, lean_object* v_b_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
size_t v_sz_boxed_140_; size_t v_i_boxed_141_; lean_object* v_res_142_; 
v_sz_boxed_140_ = lean_unbox_usize(v_sz_132_);
lean_dec(v_sz_132_);
v_i_boxed_141_ = lean_unbox_usize(v_i_133_);
lean_dec(v_i_133_);
v_res_142_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1(v_as_131_, v_sz_boxed_140_, v_i_boxed_141_, v_b_134_, v___y_135_, v___y_136_, v___y_137_, v___y_138_);
lean_dec(v___y_138_);
lean_dec_ref(v___y_137_);
lean_dec(v___y_136_);
lean_dec_ref(v___y_135_);
lean_dec_ref(v_as_131_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_as_143_, size_t v_sz_144_, size_t v_i_145_, lean_object* v_b_146_){
_start:
{
uint8_t v___x_148_; 
v___x_148_ = lean_usize_dec_lt(v_i_145_, v_sz_144_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; 
v___x_149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_149_, 0, v_b_146_);
return v___x_149_;
}
else
{
lean_object* v_snd_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_167_; 
v_snd_150_ = lean_ctor_get(v_b_146_, 1);
v_isSharedCheck_167_ = !lean_is_exclusive(v_b_146_);
if (v_isSharedCheck_167_ == 0)
{
lean_object* v_unused_168_; 
v_unused_168_ = lean_ctor_get(v_b_146_, 0);
lean_dec(v_unused_168_);
v___x_152_ = v_b_146_;
v_isShared_153_ = v_isSharedCheck_167_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_snd_150_);
lean_dec(v_b_146_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_167_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_154_; lean_object* v_a_156_; lean_object* v_a_163_; 
v___x_154_ = lean_box(0);
v_a_163_ = lean_array_uget_borrowed(v_as_143_, v_i_145_);
if (lean_obj_tag(v_a_163_) == 0)
{
v_a_156_ = v_snd_150_;
goto v___jp_155_;
}
else
{
lean_object* v_val_164_; uint8_t v___x_165_; 
v_val_164_ = lean_ctor_get(v_a_163_, 0);
v___x_165_ = lp_aesop_Aesop_isForwardImplDetailHyp(v_val_164_);
if (v___x_165_ == 0)
{
v_a_156_ = v_snd_150_;
goto v___jp_155_;
}
else
{
lean_object* v___x_166_; 
lean_inc(v_val_164_);
v___x_166_ = lean_array_push(v_snd_150_, v_val_164_);
v_a_156_ = v___x_166_;
goto v___jp_155_;
}
}
v___jp_155_:
{
lean_object* v___x_158_; 
if (v_isShared_153_ == 0)
{
lean_ctor_set(v___x_152_, 1, v_a_156_);
lean_ctor_set(v___x_152_, 0, v___x_154_);
v___x_158_ = v___x_152_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_154_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v_a_156_);
v___x_158_ = v_reuseFailAlloc_162_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
size_t v___x_159_; size_t v___x_160_; 
v___x_159_ = ((size_t)1ULL);
v___x_160_ = lean_usize_add(v_i_145_, v___x_159_);
v_i_145_ = v___x_160_;
v_b_146_ = v___x_158_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object* v_as_169_, lean_object* v_sz_170_, lean_object* v_i_171_, lean_object* v_b_172_, lean_object* v___y_173_){
_start:
{
size_t v_sz_boxed_174_; size_t v_i_boxed_175_; lean_object* v_res_176_; 
v_sz_boxed_174_ = lean_unbox_usize(v_sz_170_);
lean_dec(v_sz_170_);
v_i_boxed_175_ = lean_unbox_usize(v_i_171_);
lean_dec(v_i_171_);
v_res_176_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg(v_as_169_, v_sz_boxed_174_, v_i_boxed_175_, v_b_172_);
lean_dec_ref(v_as_169_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2(lean_object* v_as_177_, size_t v_sz_178_, size_t v_i_179_, lean_object* v_b_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
uint8_t v___x_186_; 
v___x_186_ = lean_usize_dec_lt(v_i_179_, v_sz_178_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; 
v___x_187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_187_, 0, v_b_180_);
return v___x_187_;
}
else
{
lean_object* v_snd_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_205_; 
v_snd_188_ = lean_ctor_get(v_b_180_, 1);
v_isSharedCheck_205_ = !lean_is_exclusive(v_b_180_);
if (v_isSharedCheck_205_ == 0)
{
lean_object* v_unused_206_; 
v_unused_206_ = lean_ctor_get(v_b_180_, 0);
lean_dec(v_unused_206_);
v___x_190_ = v_b_180_;
v_isShared_191_ = v_isSharedCheck_205_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_snd_188_);
lean_dec(v_b_180_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_205_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
lean_object* v___x_192_; lean_object* v_a_194_; lean_object* v_a_201_; 
v___x_192_ = lean_box(0);
v_a_201_ = lean_array_uget_borrowed(v_as_177_, v_i_179_);
if (lean_obj_tag(v_a_201_) == 0)
{
v_a_194_ = v_snd_188_;
goto v___jp_193_;
}
else
{
lean_object* v_val_202_; uint8_t v___x_203_; 
v_val_202_ = lean_ctor_get(v_a_201_, 0);
v___x_203_ = lp_aesop_Aesop_isForwardImplDetailHyp(v_val_202_);
if (v___x_203_ == 0)
{
v_a_194_ = v_snd_188_;
goto v___jp_193_;
}
else
{
lean_object* v___x_204_; 
lean_inc(v_val_202_);
v___x_204_ = lean_array_push(v_snd_188_, v_val_202_);
v_a_194_ = v___x_204_;
goto v___jp_193_;
}
}
v___jp_193_:
{
lean_object* v___x_196_; 
if (v_isShared_191_ == 0)
{
lean_ctor_set(v___x_190_, 1, v_a_194_);
lean_ctor_set(v___x_190_, 0, v___x_192_);
v___x_196_ = v___x_190_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v___x_192_);
lean_ctor_set(v_reuseFailAlloc_200_, 1, v_a_194_);
v___x_196_ = v_reuseFailAlloc_200_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
size_t v___x_197_; size_t v___x_198_; lean_object* v___x_199_; 
v___x_197_ = ((size_t)1ULL);
v___x_198_ = lean_usize_add(v_i_179_, v___x_197_);
v___x_199_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg(v_as_177_, v_sz_178_, v___x_198_, v___x_196_);
return v___x_199_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2___boxed(lean_object* v_as_207_, lean_object* v_sz_208_, lean_object* v_i_209_, lean_object* v_b_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
size_t v_sz_boxed_216_; size_t v_i_boxed_217_; lean_object* v_res_218_; 
v_sz_boxed_216_ = lean_unbox_usize(v_sz_208_);
lean_dec(v_sz_208_);
v_i_boxed_217_ = lean_unbox_usize(v_i_209_);
lean_dec(v_i_209_);
v_res_218_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2(v_as_207_, v_sz_boxed_216_, v_i_boxed_217_, v_b_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
lean_dec_ref(v_as_207_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0(lean_object* v_init_219_, lean_object* v_n_220_, lean_object* v_b_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
if (lean_obj_tag(v_n_220_) == 0)
{
lean_object* v_cs_227_; lean_object* v___x_228_; lean_object* v___x_229_; size_t v_sz_230_; size_t v___x_231_; lean_object* v___x_232_; 
v_cs_227_ = lean_ctor_get(v_n_220_, 0);
v___x_228_ = lean_box(0);
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v_b_221_);
v_sz_230_ = lean_array_size(v_cs_227_);
v___x_231_ = ((size_t)0ULL);
v___x_232_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1(v_init_219_, v_cs_227_, v_sz_230_, v___x_231_, v___x_229_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v_a_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_247_; 
v_a_233_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_247_ == 0)
{
v___x_235_ = v___x_232_;
v_isShared_236_ = v_isSharedCheck_247_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_a_233_);
lean_dec(v___x_232_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_247_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v_fst_237_; 
v_fst_237_ = lean_ctor_get(v_a_233_, 0);
if (lean_obj_tag(v_fst_237_) == 0)
{
lean_object* v_snd_238_; lean_object* v___x_239_; lean_object* v___x_241_; 
v_snd_238_ = lean_ctor_get(v_a_233_, 1);
lean_inc(v_snd_238_);
lean_dec(v_a_233_);
v___x_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_239_, 0, v_snd_238_);
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 0, v___x_239_);
v___x_241_ = v___x_235_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
else
{
lean_object* v_val_243_; lean_object* v___x_245_; 
lean_inc_ref(v_fst_237_);
lean_dec(v_a_233_);
v_val_243_ = lean_ctor_get(v_fst_237_, 0);
lean_inc(v_val_243_);
lean_dec_ref_known(v_fst_237_, 1);
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 0, v_val_243_);
v___x_245_ = v___x_235_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_val_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
else
{
lean_object* v_a_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
v_a_248_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_255_ == 0)
{
v___x_250_ = v___x_232_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_a_248_);
lean_dec(v___x_232_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
else
{
lean_object* v_vs_256_; lean_object* v___x_257_; lean_object* v___x_258_; size_t v_sz_259_; size_t v___x_260_; lean_object* v___x_261_; 
v_vs_256_ = lean_ctor_get(v_n_220_, 0);
v___x_257_ = lean_box(0);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_b_221_);
v_sz_259_ = lean_array_size(v_vs_256_);
v___x_260_ = ((size_t)0ULL);
v___x_261_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2(v_vs_256_, v_sz_259_, v___x_260_, v___x_258_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
if (lean_obj_tag(v___x_261_) == 0)
{
lean_object* v_a_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_276_; 
v_a_262_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_276_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_276_ == 0)
{
v___x_264_ = v___x_261_;
v_isShared_265_ = v_isSharedCheck_276_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_a_262_);
lean_dec(v___x_261_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_276_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v_fst_266_; 
v_fst_266_ = lean_ctor_get(v_a_262_, 0);
if (lean_obj_tag(v_fst_266_) == 0)
{
lean_object* v_snd_267_; lean_object* v___x_268_; lean_object* v___x_270_; 
v_snd_267_ = lean_ctor_get(v_a_262_, 1);
lean_inc(v_snd_267_);
lean_dec(v_a_262_);
v___x_268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_268_, 0, v_snd_267_);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v___x_268_);
v___x_270_ = v___x_264_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v___x_268_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
else
{
lean_object* v_val_272_; lean_object* v___x_274_; 
lean_inc_ref(v_fst_266_);
lean_dec(v_a_262_);
v_val_272_ = lean_ctor_get(v_fst_266_, 0);
lean_inc(v_val_272_);
lean_dec_ref_known(v_fst_266_, 1);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v_val_272_);
v___x_274_ = v___x_264_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_val_272_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
}
else
{
lean_object* v_a_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_284_; 
v_a_277_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_284_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_284_ == 0)
{
v___x_279_ = v___x_261_;
v_isShared_280_ = v_isSharedCheck_284_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_a_277_);
lean_dec(v___x_261_);
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
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1(lean_object* v_init_285_, lean_object* v_as_286_, size_t v_sz_287_, size_t v_i_288_, lean_object* v_b_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
uint8_t v___x_295_; 
v___x_295_ = lean_usize_dec_lt(v_i_288_, v_sz_287_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; 
v___x_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_296_, 0, v_b_289_);
return v___x_296_;
}
else
{
lean_object* v_snd_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_331_; 
v_snd_297_ = lean_ctor_get(v_b_289_, 1);
v_isSharedCheck_331_ = !lean_is_exclusive(v_b_289_);
if (v_isSharedCheck_331_ == 0)
{
lean_object* v_unused_332_; 
v_unused_332_ = lean_ctor_get(v_b_289_, 0);
lean_dec(v_unused_332_);
v___x_299_ = v_b_289_;
v_isShared_300_ = v_isSharedCheck_331_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_snd_297_);
lean_dec(v_b_289_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_331_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
lean_object* v_a_301_; lean_object* v___x_302_; 
v_a_301_ = lean_array_uget_borrowed(v_as_286_, v_i_288_);
lean_inc(v_snd_297_);
v___x_302_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0(v_init_285_, v_a_301_, v_snd_297_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_322_; 
v_a_303_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_322_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_322_ == 0)
{
v___x_305_ = v___x_302_;
v_isShared_306_ = v_isSharedCheck_322_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_302_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_322_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
if (lean_obj_tag(v_a_303_) == 0)
{
lean_object* v___x_307_; lean_object* v___x_309_; 
v___x_307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_307_, 0, v_a_303_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v___x_307_);
v___x_309_ = v___x_299_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_307_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v_snd_297_);
v___x_309_ = v_reuseFailAlloc_313_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
lean_object* v___x_311_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 0, v___x_309_);
v___x_311_ = v___x_305_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_309_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
else
{
lean_object* v_a_314_; lean_object* v___x_315_; lean_object* v___x_317_; 
lean_del_object(v___x_305_);
lean_dec(v_snd_297_);
v_a_314_ = lean_ctor_get(v_a_303_, 0);
lean_inc(v_a_314_);
lean_dec_ref_known(v_a_303_, 1);
v___x_315_ = lean_box(0);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 1, v_a_314_);
lean_ctor_set(v___x_299_, 0, v___x_315_);
v___x_317_ = v___x_299_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___x_315_);
lean_ctor_set(v_reuseFailAlloc_321_, 1, v_a_314_);
v___x_317_ = v_reuseFailAlloc_321_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
size_t v___x_318_; size_t v___x_319_; 
v___x_318_ = ((size_t)1ULL);
v___x_319_ = lean_usize_add(v_i_288_, v___x_318_);
v_i_288_ = v___x_319_;
v_b_289_ = v___x_317_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_del_object(v___x_299_);
lean_dec(v_snd_297_);
v_a_323_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_302_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_302_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1___boxed(lean_object* v_init_333_, lean_object* v_as_334_, lean_object* v_sz_335_, lean_object* v_i_336_, lean_object* v_b_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_){
_start:
{
size_t v_sz_boxed_343_; size_t v_i_boxed_344_; lean_object* v_res_345_; 
v_sz_boxed_343_ = lean_unbox_usize(v_sz_335_);
lean_dec(v_sz_335_);
v_i_boxed_344_ = lean_unbox_usize(v_i_336_);
lean_dec(v_i_336_);
v_res_345_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__1(v_init_333_, v_as_334_, v_sz_boxed_343_, v_i_boxed_344_, v_b_337_, v___y_338_, v___y_339_, v___y_340_, v___y_341_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
lean_dec(v___y_339_);
lean_dec_ref(v___y_338_);
lean_dec_ref(v_as_334_);
lean_dec_ref(v_init_333_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0___boxed(lean_object* v_init_346_, lean_object* v_n_347_, lean_object* v_b_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0(v_init_346_, v_n_347_, v_b_348_, v___y_349_, v___y_350_, v___y_351_, v___y_352_);
lean_dec(v___y_352_);
lean_dec_ref(v___y_351_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
lean_dec_ref(v_n_347_);
lean_dec_ref(v_init_346_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0(lean_object* v_t_355_, lean_object* v_init_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v_root_362_; lean_object* v_tail_363_; lean_object* v___x_364_; 
v_root_362_ = lean_ctor_get(v_t_355_, 0);
v_tail_363_ = lean_ctor_get(v_t_355_, 1);
lean_inc_ref(v_init_356_);
v___x_364_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0(v_init_356_, v_root_362_, v_init_356_, v___y_357_, v___y_358_, v___y_359_, v___y_360_);
lean_dec_ref(v_init_356_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_401_; 
v_a_365_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_401_ == 0)
{
v___x_367_ = v___x_364_;
v_isShared_368_ = v_isSharedCheck_401_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_364_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_401_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
if (lean_obj_tag(v_a_365_) == 0)
{
lean_object* v_a_369_; lean_object* v___x_371_; 
v_a_369_ = lean_ctor_get(v_a_365_, 0);
lean_inc(v_a_369_);
lean_dec_ref_known(v_a_365_, 1);
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 0, v_a_369_);
v___x_371_ = v___x_367_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v_a_369_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
else
{
lean_object* v_a_373_; lean_object* v___x_374_; lean_object* v___x_375_; size_t v_sz_376_; size_t v___x_377_; lean_object* v___x_378_; 
lean_del_object(v___x_367_);
v_a_373_ = lean_ctor_get(v_a_365_, 0);
lean_inc(v_a_373_);
lean_dec_ref_known(v_a_365_, 1);
v___x_374_ = lean_box(0);
v___x_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_a_373_);
v_sz_376_ = lean_array_size(v_tail_363_);
v___x_377_ = ((size_t)0ULL);
v___x_378_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1(v_tail_363_, v_sz_376_, v___x_377_, v___x_375_, v___y_357_, v___y_358_, v___y_359_, v___y_360_);
if (lean_obj_tag(v___x_378_) == 0)
{
lean_object* v_a_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_392_; 
v_a_379_ = lean_ctor_get(v___x_378_, 0);
v_isSharedCheck_392_ = !lean_is_exclusive(v___x_378_);
if (v_isSharedCheck_392_ == 0)
{
v___x_381_ = v___x_378_;
v_isShared_382_ = v_isSharedCheck_392_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_a_379_);
lean_dec(v___x_378_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_392_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v_fst_383_; 
v_fst_383_ = lean_ctor_get(v_a_379_, 0);
if (lean_obj_tag(v_fst_383_) == 0)
{
lean_object* v_snd_384_; lean_object* v___x_386_; 
v_snd_384_ = lean_ctor_get(v_a_379_, 1);
lean_inc(v_snd_384_);
lean_dec(v_a_379_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 0, v_snd_384_);
v___x_386_ = v___x_381_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_snd_384_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
else
{
lean_object* v_val_388_; lean_object* v___x_390_; 
lean_inc_ref(v_fst_383_);
lean_dec(v_a_379_);
v_val_388_ = lean_ctor_get(v_fst_383_, 0);
lean_inc(v_val_388_);
lean_dec_ref_known(v_fst_383_, 1);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 0, v_val_388_);
v___x_390_ = v___x_381_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_val_388_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
}
}
else
{
lean_object* v_a_393_; lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_400_; 
v_a_393_ = lean_ctor_get(v___x_378_, 0);
v_isSharedCheck_400_ = !lean_is_exclusive(v___x_378_);
if (v_isSharedCheck_400_ == 0)
{
v___x_395_ = v___x_378_;
v_isShared_396_ = v_isSharedCheck_400_;
goto v_resetjp_394_;
}
else
{
lean_inc(v_a_393_);
lean_dec(v___x_378_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_400_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
lean_object* v___x_398_; 
if (v_isShared_396_ == 0)
{
v___x_398_ = v___x_395_;
goto v_reusejp_397_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v_a_393_);
v___x_398_ = v_reuseFailAlloc_399_;
goto v_reusejp_397_;
}
v_reusejp_397_:
{
return v___x_398_;
}
}
}
}
}
}
else
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_409_; 
v_a_402_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_409_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_409_ == 0)
{
v___x_404_ = v___x_364_;
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_364_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v_a_402_);
v___x_407_ = v_reuseFailAlloc_408_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
return v___x_407_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0___boxed(lean_object* v_t_410_, lean_object* v_init_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0(v_t_410_, v_init_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
lean_dec(v___y_413_);
lean_dec_ref(v___y_412_);
lean_dec_ref(v_t_410_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardImplDetailHyps(lean_object* v_a_420_, lean_object* v_a_421_, lean_object* v_a_422_, lean_object* v_a_423_){
_start:
{
lean_object* v_lctx_425_; lean_object* v_decls_426_; lean_object* v_result_427_; lean_object* v___x_428_; 
v_lctx_425_ = lean_ctor_get(v_a_420_, 2);
v_decls_426_ = lean_ctor_get(v_lctx_425_, 1);
v_result_427_ = ((lean_object*)(lp_aesop_Aesop_getForwardImplDetailHyps___closed__0));
v___x_428_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0(v_decls_426_, v_result_427_, v_a_420_, v_a_421_, v_a_422_, v_a_423_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardImplDetailHyps___boxed(lean_object* v_a_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_aesop_Aesop_getForwardImplDetailHyps(v_a_429_, v_a_430_, v_a_431_, v_a_432_);
lean_dec(v_a_432_);
lean_dec_ref(v_a_431_);
lean_dec(v_a_430_);
lean_dec_ref(v_a_429_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4(lean_object* v_as_435_, size_t v_sz_436_, size_t v_i_437_, lean_object* v_b_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_435_, v_sz_436_, v_i_437_, v_b_438_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4___boxed(lean_object* v_as_445_, lean_object* v_sz_446_, lean_object* v_i_447_, lean_object* v_b_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
size_t v_sz_boxed_454_; size_t v_i_boxed_455_; lean_object* v_res_456_; 
v_sz_boxed_454_ = lean_unbox_usize(v_sz_446_);
lean_dec(v_sz_446_);
v_i_boxed_455_ = lean_unbox_usize(v_i_447_);
lean_dec(v_i_447_);
v_res_456_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__1_spec__4(v_as_445_, v_sz_boxed_454_, v_i_boxed_455_, v_b_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
lean_dec(v___y_452_);
lean_dec_ref(v___y_451_);
lean_dec(v___y_450_);
lean_dec_ref(v___y_449_);
lean_dec_ref(v_as_445_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3(lean_object* v_as_457_, size_t v_sz_458_, size_t v_i_459_, lean_object* v_b_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___redArg(v_as_457_, v_sz_458_, v_i_459_, v_b_460_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_as_467_, lean_object* v_sz_468_, lean_object* v_i_469_, lean_object* v_b_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
size_t v_sz_boxed_476_; size_t v_i_boxed_477_; lean_object* v_res_478_; 
v_sz_boxed_476_ = lean_unbox_usize(v_sz_468_);
lean_dec(v_sz_468_);
v_i_boxed_477_ = lean_unbox_usize(v_i_469_);
lean_dec(v_i_469_);
v_res_478_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_getForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__3(v_as_467_, v_sz_boxed_476_, v_i_boxed_477_, v_b_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
lean_dec_ref(v_as_467_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(lean_object* v_mvarId_479_, lean_object* v_x_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_479_, v_x_480_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
if (lean_obj_tag(v___x_486_) == 0)
{
lean_object* v_a_487_; lean_object* v___x_489_; uint8_t v_isShared_490_; uint8_t v_isSharedCheck_494_; 
v_a_487_ = lean_ctor_get(v___x_486_, 0);
v_isSharedCheck_494_ = !lean_is_exclusive(v___x_486_);
if (v_isSharedCheck_494_ == 0)
{
v___x_489_ = v___x_486_;
v_isShared_490_ = v_isSharedCheck_494_;
goto v_resetjp_488_;
}
else
{
lean_inc(v_a_487_);
lean_dec(v___x_486_);
v___x_489_ = lean_box(0);
v_isShared_490_ = v_isSharedCheck_494_;
goto v_resetjp_488_;
}
v_resetjp_488_:
{
lean_object* v___x_492_; 
if (v_isShared_490_ == 0)
{
v___x_492_ = v___x_489_;
goto v_reusejp_491_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v_a_487_);
v___x_492_ = v_reuseFailAlloc_493_;
goto v_reusejp_491_;
}
v_reusejp_491_:
{
return v___x_492_;
}
}
}
else
{
lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_502_; 
v_a_495_ = lean_ctor_get(v___x_486_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_486_);
if (v_isSharedCheck_502_ == 0)
{
v___x_497_ = v___x_486_;
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_486_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_500_; 
if (v_isShared_498_ == 0)
{
v___x_500_ = v___x_497_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_a_495_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg___boxed(lean_object* v_mvarId_503_, lean_object* v_x_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(v_mvarId_503_, v_x_504_, v___y_505_, v___y_506_, v___y_507_, v___y_508_);
lean_dec(v___y_508_);
lean_dec_ref(v___y_507_);
lean_dec(v___y_506_);
lean_dec_ref(v___y_505_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1(lean_object* v_00_u03b1_511_, lean_object* v_mvarId_512_, lean_object* v_x_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(v_mvarId_512_, v_x_513_, v___y_514_, v___y_515_, v___y_516_, v___y_517_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___boxed(lean_object* v_00_u03b1_520_, lean_object* v_mvarId_521_, lean_object* v_x_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1(v_00_u03b1_520_, v_mvarId_521_, v_x_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec_ref(v___y_523_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0(size_t v_sz_529_, size_t v_i_530_, lean_object* v_bs_531_){
_start:
{
uint8_t v___x_532_; 
v___x_532_ = lean_usize_dec_lt(v_i_530_, v_sz_529_);
if (v___x_532_ == 0)
{
return v_bs_531_;
}
else
{
lean_object* v_v_533_; lean_object* v___x_534_; lean_object* v_bs_x27_535_; lean_object* v___x_536_; size_t v___x_537_; size_t v___x_538_; lean_object* v___x_539_; 
v_v_533_ = lean_array_uget(v_bs_531_, v_i_530_);
v___x_534_ = lean_unsigned_to_nat(0u);
v_bs_x27_535_ = lean_array_uset(v_bs_531_, v_i_530_, v___x_534_);
v___x_536_ = l_Lean_LocalDecl_fvarId(v_v_533_);
lean_dec(v_v_533_);
v___x_537_ = ((size_t)1ULL);
v___x_538_ = lean_usize_add(v_i_530_, v___x_537_);
v___x_539_ = lean_array_uset(v_bs_x27_535_, v_i_530_, v___x_536_);
v_i_530_ = v___x_538_;
v_bs_531_ = v___x_539_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0___boxed(lean_object* v_sz_541_, lean_object* v_i_542_, lean_object* v_bs_543_){
_start:
{
size_t v_sz_boxed_544_; size_t v_i_boxed_545_; lean_object* v_res_546_; 
v_sz_boxed_544_ = lean_unbox_usize(v_sz_541_);
lean_dec(v_sz_541_);
v_i_boxed_545_ = lean_unbox_usize(v_i_542_);
lean_dec(v_i_542_);
v_res_546_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0(v_sz_boxed_544_, v_i_boxed_545_, v_bs_543_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0(lean_object* v_goal_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_aesop_Aesop_getForwardImplDetailHyps(v___y_548_, v___y_549_, v___y_550_, v___y_551_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; size_t v_sz_555_; size_t v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_a_554_);
lean_dec_ref_known(v___x_553_, 1);
v_sz_555_ = lean_array_size(v_a_554_);
v___x_556_ = ((size_t)0ULL);
v___x_557_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_clearForwardImplDetailHyps_spec__0(v_sz_555_, v___x_556_, v_a_554_);
v___x_558_ = l_Lean_MVarId_tryClearMany(v_goal_547_, v___x_557_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec_ref(v___x_557_);
return v___x_558_;
}
else
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
lean_dec(v_goal_547_);
v_a_559_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v___x_553_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_553_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_562_ == 0)
{
v___x_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0___boxed(lean_object* v_goal_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_){
_start:
{
lean_object* v_res_573_; 
v_res_573_ = lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0(v_goal_567_, v___y_568_, v___y_569_, v___y_570_, v___y_571_);
lean_dec(v___y_571_);
lean_dec_ref(v___y_570_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
return v_res_573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps(lean_object* v_goal_574_, lean_object* v_a_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_){
_start:
{
lean_object* v___f_580_; lean_object* v___x_581_; 
lean_inc(v_goal_574_);
v___f_580_ = lean_alloc_closure((void*)(lp_aesop_Aesop_clearForwardImplDetailHyps___lam__0___boxed), 6, 1);
lean_closure_set(v___f_580_, 0, v_goal_574_);
v___x_581_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(v_goal_574_, v___f_580_, v_a_575_, v_a_576_, v_a_577_, v_a_578_);
return v___x_581_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps___boxed(lean_object* v_goal_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_aesop_Aesop_clearForwardImplDetailHyps(v_goal_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_);
lean_dec(v_a_586_);
lean_dec_ref(v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_a_583_);
return v_res_588_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0(void){
_start:
{
lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_589_ = lean_box(0);
v___x_590_ = lean_unsigned_to_nat(16u);
v___x_591_ = lean_mk_array(v___x_590_, v___x_589_);
return v___x_591_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1(void){
_start:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_592_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0, &lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__0);
v___x_593_ = lean_unsigned_to_nat(0u);
v___x_594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
lean_ctor_set(v___x_594_, 1, v___x_592_);
return v___x_594_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardHypData_default(void){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1);
return v___x_595_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardHypData(void){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = lp_aesop_Aesop_instInhabitedForwardHypData_default;
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2___redArg(lean_object* v_a_597_, lean_object* v_b_598_, lean_object* v_x_599_){
_start:
{
if (lean_obj_tag(v_x_599_) == 0)
{
lean_dec(v_b_598_);
lean_dec(v_a_597_);
return v_x_599_;
}
else
{
lean_object* v_key_600_; lean_object* v_value_601_; lean_object* v_tail_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_614_; 
v_key_600_ = lean_ctor_get(v_x_599_, 0);
v_value_601_ = lean_ctor_get(v_x_599_, 1);
v_tail_602_ = lean_ctor_get(v_x_599_, 2);
v_isSharedCheck_614_ = !lean_is_exclusive(v_x_599_);
if (v_isSharedCheck_614_ == 0)
{
v___x_604_ = v_x_599_;
v_isShared_605_ = v_isSharedCheck_614_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_tail_602_);
lean_inc(v_value_601_);
lean_inc(v_key_600_);
lean_dec(v_x_599_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_614_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
uint8_t v___x_606_; 
v___x_606_ = l_Lean_instBEqFVarId_beq(v_key_600_, v_a_597_);
if (v___x_606_ == 0)
{
lean_object* v___x_607_; lean_object* v___x_609_; 
v___x_607_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2___redArg(v_a_597_, v_b_598_, v_tail_602_);
if (v_isShared_605_ == 0)
{
lean_ctor_set(v___x_604_, 2, v___x_607_);
v___x_609_ = v___x_604_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v_key_600_);
lean_ctor_set(v_reuseFailAlloc_610_, 1, v_value_601_);
lean_ctor_set(v_reuseFailAlloc_610_, 2, v___x_607_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
}
}
else
{
lean_object* v___x_612_; 
lean_dec(v_value_601_);
lean_dec(v_key_600_);
if (v_isShared_605_ == 0)
{
lean_ctor_set(v___x_604_, 1, v_b_598_);
lean_ctor_set(v___x_604_, 0, v_a_597_);
v___x_612_ = v___x_604_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_a_597_);
lean_ctor_set(v_reuseFailAlloc_613_, 1, v_b_598_);
lean_ctor_set(v_reuseFailAlloc_613_, 2, v_tail_602_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg(lean_object* v_a_615_, lean_object* v_x_616_){
_start:
{
if (lean_obj_tag(v_x_616_) == 0)
{
uint8_t v___x_617_; 
v___x_617_ = 0;
return v___x_617_;
}
else
{
lean_object* v_key_618_; lean_object* v_tail_619_; uint8_t v___x_620_; 
v_key_618_ = lean_ctor_get(v_x_616_, 0);
v_tail_619_ = lean_ctor_get(v_x_616_, 2);
v___x_620_ = l_Lean_instBEqFVarId_beq(v_key_618_, v_a_615_);
if (v___x_620_ == 0)
{
v_x_616_ = v_tail_619_;
goto _start;
}
else
{
return v___x_620_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg___boxed(lean_object* v_a_622_, lean_object* v_x_623_){
_start:
{
uint8_t v_res_624_; lean_object* v_r_625_; 
v_res_624_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg(v_a_622_, v_x_623_);
lean_dec(v_x_623_);
lean_dec(v_a_622_);
v_r_625_ = lean_box(v_res_624_);
return v_r_625_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_626_, lean_object* v_x_627_){
_start:
{
if (lean_obj_tag(v_x_627_) == 0)
{
return v_x_626_;
}
else
{
lean_object* v_key_628_; lean_object* v_value_629_; lean_object* v_tail_630_; lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_653_; 
v_key_628_ = lean_ctor_get(v_x_627_, 0);
v_value_629_ = lean_ctor_get(v_x_627_, 1);
v_tail_630_ = lean_ctor_get(v_x_627_, 2);
v_isSharedCheck_653_ = !lean_is_exclusive(v_x_627_);
if (v_isSharedCheck_653_ == 0)
{
v___x_632_ = v_x_627_;
v_isShared_633_ = v_isSharedCheck_653_;
goto v_resetjp_631_;
}
else
{
lean_inc(v_tail_630_);
lean_inc(v_value_629_);
lean_inc(v_key_628_);
lean_dec(v_x_627_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_653_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
lean_object* v___x_634_; uint64_t v___x_635_; uint64_t v___x_636_; uint64_t v___x_637_; uint64_t v_fold_638_; uint64_t v___x_639_; uint64_t v___x_640_; uint64_t v___x_641_; size_t v___x_642_; size_t v___x_643_; size_t v___x_644_; size_t v___x_645_; size_t v___x_646_; lean_object* v___x_647_; lean_object* v___x_649_; 
v___x_634_ = lean_array_get_size(v_x_626_);
v___x_635_ = l_Lean_instHashableFVarId_hash(v_key_628_);
v___x_636_ = 32ULL;
v___x_637_ = lean_uint64_shift_right(v___x_635_, v___x_636_);
v_fold_638_ = lean_uint64_xor(v___x_635_, v___x_637_);
v___x_639_ = 16ULL;
v___x_640_ = lean_uint64_shift_right(v_fold_638_, v___x_639_);
v___x_641_ = lean_uint64_xor(v_fold_638_, v___x_640_);
v___x_642_ = lean_uint64_to_usize(v___x_641_);
v___x_643_ = lean_usize_of_nat(v___x_634_);
v___x_644_ = ((size_t)1ULL);
v___x_645_ = lean_usize_sub(v___x_643_, v___x_644_);
v___x_646_ = lean_usize_land(v___x_642_, v___x_645_);
v___x_647_ = lean_array_uget_borrowed(v_x_626_, v___x_646_);
lean_inc(v___x_647_);
if (v_isShared_633_ == 0)
{
lean_ctor_set(v___x_632_, 2, v___x_647_);
v___x_649_ = v___x_632_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_652_; 
v_reuseFailAlloc_652_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_652_, 0, v_key_628_);
lean_ctor_set(v_reuseFailAlloc_652_, 1, v_value_629_);
lean_ctor_set(v_reuseFailAlloc_652_, 2, v___x_647_);
v___x_649_ = v_reuseFailAlloc_652_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
lean_object* v___x_650_; 
v___x_650_ = lean_array_uset(v_x_626_, v___x_646_, v___x_649_);
v_x_626_ = v___x_650_;
v_x_627_ = v_tail_630_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2___redArg(lean_object* v_i_654_, lean_object* v_source_655_, lean_object* v_target_656_){
_start:
{
lean_object* v___x_657_; uint8_t v___x_658_; 
v___x_657_ = lean_array_get_size(v_source_655_);
v___x_658_ = lean_nat_dec_lt(v_i_654_, v___x_657_);
if (v___x_658_ == 0)
{
lean_dec_ref(v_source_655_);
lean_dec(v_i_654_);
return v_target_656_;
}
else
{
lean_object* v_es_659_; lean_object* v___x_660_; lean_object* v_source_661_; lean_object* v_target_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v_es_659_ = lean_array_fget(v_source_655_, v_i_654_);
v___x_660_ = lean_box(0);
v_source_661_ = lean_array_fset(v_source_655_, v_i_654_, v___x_660_);
v_target_662_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4___redArg(v_target_656_, v_es_659_);
v___x_663_ = lean_unsigned_to_nat(1u);
v___x_664_ = lean_nat_add(v_i_654_, v___x_663_);
lean_dec(v_i_654_);
v_i_654_ = v___x_664_;
v_source_655_ = v_source_661_;
v_target_656_ = v_target_662_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1___redArg(lean_object* v_data_666_){
_start:
{
lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v_nbuckets_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_667_ = lean_array_get_size(v_data_666_);
v___x_668_ = lean_unsigned_to_nat(2u);
v_nbuckets_669_ = lean_nat_mul(v___x_667_, v___x_668_);
v___x_670_ = lean_unsigned_to_nat(0u);
v___x_671_ = lean_box(0);
v___x_672_ = lean_mk_array(v_nbuckets_669_, v___x_671_);
v___x_673_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2___redArg(v___x_670_, v_data_666_, v___x_672_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0___redArg(lean_object* v_m_674_, lean_object* v_a_675_, lean_object* v_b_676_){
_start:
{
lean_object* v_size_677_; lean_object* v_buckets_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_721_; 
v_size_677_ = lean_ctor_get(v_m_674_, 0);
v_buckets_678_ = lean_ctor_get(v_m_674_, 1);
v_isSharedCheck_721_ = !lean_is_exclusive(v_m_674_);
if (v_isSharedCheck_721_ == 0)
{
v___x_680_ = v_m_674_;
v_isShared_681_ = v_isSharedCheck_721_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_buckets_678_);
lean_inc(v_size_677_);
lean_dec(v_m_674_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_721_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_682_; uint64_t v___x_683_; uint64_t v___x_684_; uint64_t v___x_685_; uint64_t v_fold_686_; uint64_t v___x_687_; uint64_t v___x_688_; uint64_t v___x_689_; size_t v___x_690_; size_t v___x_691_; size_t v___x_692_; size_t v___x_693_; size_t v___x_694_; lean_object* v_bkt_695_; uint8_t v___x_696_; 
v___x_682_ = lean_array_get_size(v_buckets_678_);
v___x_683_ = l_Lean_instHashableFVarId_hash(v_a_675_);
v___x_684_ = 32ULL;
v___x_685_ = lean_uint64_shift_right(v___x_683_, v___x_684_);
v_fold_686_ = lean_uint64_xor(v___x_683_, v___x_685_);
v___x_687_ = 16ULL;
v___x_688_ = lean_uint64_shift_right(v_fold_686_, v___x_687_);
v___x_689_ = lean_uint64_xor(v_fold_686_, v___x_688_);
v___x_690_ = lean_uint64_to_usize(v___x_689_);
v___x_691_ = lean_usize_of_nat(v___x_682_);
v___x_692_ = ((size_t)1ULL);
v___x_693_ = lean_usize_sub(v___x_691_, v___x_692_);
v___x_694_ = lean_usize_land(v___x_690_, v___x_693_);
v_bkt_695_ = lean_array_uget_borrowed(v_buckets_678_, v___x_694_);
v___x_696_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg(v_a_675_, v_bkt_695_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; lean_object* v_size_x27_698_; lean_object* v___x_699_; lean_object* v_buckets_x27_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; uint8_t v___x_706_; 
v___x_697_ = lean_unsigned_to_nat(1u);
v_size_x27_698_ = lean_nat_add(v_size_677_, v___x_697_);
lean_dec(v_size_677_);
lean_inc(v_bkt_695_);
v___x_699_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_699_, 0, v_a_675_);
lean_ctor_set(v___x_699_, 1, v_b_676_);
lean_ctor_set(v___x_699_, 2, v_bkt_695_);
v_buckets_x27_700_ = lean_array_uset(v_buckets_678_, v___x_694_, v___x_699_);
v___x_701_ = lean_unsigned_to_nat(4u);
v___x_702_ = lean_nat_mul(v_size_x27_698_, v___x_701_);
v___x_703_ = lean_unsigned_to_nat(3u);
v___x_704_ = lean_nat_div(v___x_702_, v___x_703_);
lean_dec(v___x_702_);
v___x_705_ = lean_array_get_size(v_buckets_x27_700_);
v___x_706_ = lean_nat_dec_le(v___x_704_, v___x_705_);
lean_dec(v___x_704_);
if (v___x_706_ == 0)
{
lean_object* v_val_707_; lean_object* v___x_709_; 
v_val_707_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1___redArg(v_buckets_x27_700_);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v_val_707_);
lean_ctor_set(v___x_680_, 0, v_size_x27_698_);
v___x_709_ = v___x_680_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v_size_x27_698_);
lean_ctor_set(v_reuseFailAlloc_710_, 1, v_val_707_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
else
{
lean_object* v___x_712_; 
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v_buckets_x27_700_);
lean_ctor_set(v___x_680_, 0, v_size_x27_698_);
v___x_712_ = v___x_680_;
goto v_reusejp_711_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v_size_x27_698_);
lean_ctor_set(v_reuseFailAlloc_713_, 1, v_buckets_x27_700_);
v___x_712_ = v_reuseFailAlloc_713_;
goto v_reusejp_711_;
}
v_reusejp_711_:
{
return v___x_712_;
}
}
}
else
{
lean_object* v___x_714_; lean_object* v_buckets_x27_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_719_; 
lean_inc(v_bkt_695_);
v___x_714_ = lean_box(0);
v_buckets_x27_715_ = lean_array_uset(v_buckets_678_, v___x_694_, v___x_714_);
v___x_716_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2___redArg(v_a_675_, v_b_676_, v_bkt_695_);
v___x_717_ = lean_array_uset(v_buckets_x27_715_, v___x_694_, v___x_716_);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v___x_717_);
v___x_719_ = v___x_680_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v_size_677_);
lean_ctor_set(v_reuseFailAlloc_720_, 1, v___x_717_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg(lean_object* v_as_722_, size_t v_sz_723_, size_t v_i_724_, lean_object* v_b_725_, lean_object* v___y_726_){
_start:
{
lean_object* v_a_729_; uint8_t v___x_733_; 
v___x_733_ = lean_usize_dec_lt(v_i_724_, v_sz_723_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; 
v___x_734_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_734_, 0, v_b_725_);
return v___x_734_;
}
else
{
lean_object* v_a_735_; lean_object* v___x_736_; lean_object* v___x_737_; 
v_a_735_ = lean_array_uget_borrowed(v_as_722_, v_i_724_);
v___x_736_ = l_Lean_LocalDecl_userName(v_a_735_);
v___x_737_ = lp_aesop_Aesop_matchForwardImplDetailHypName(v___x_736_);
if (lean_obj_tag(v___x_737_) == 1)
{
lean_object* v_val_738_; lean_object* v_fst_739_; lean_object* v_snd_740_; lean_object* v_lctx_741_; lean_object* v___x_742_; 
v_val_738_ = lean_ctor_get(v___x_737_, 0);
lean_inc(v_val_738_);
lean_dec_ref_known(v___x_737_, 1);
v_fst_739_ = lean_ctor_get(v_val_738_, 0);
lean_inc(v_fst_739_);
v_snd_740_ = lean_ctor_get(v_val_738_, 1);
lean_inc(v_snd_740_);
lean_dec(v_val_738_);
v_lctx_741_ = lean_ctor_get(v___y_726_, 2);
v___x_742_ = l_Lean_LocalContext_findFromUserName_x3f(v_lctx_741_, v_snd_740_);
lean_dec(v_snd_740_);
if (lean_obj_tag(v___x_742_) == 1)
{
lean_object* v_val_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v_val_743_ = lean_ctor_get(v___x_742_, 0);
lean_inc(v_val_743_);
lean_dec_ref_known(v___x_742_, 1);
v___x_744_ = l_Lean_LocalDecl_fvarId(v_val_743_);
lean_dec(v_val_743_);
v___x_745_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0___redArg(v_b_725_, v___x_744_, v_fst_739_);
v_a_729_ = v___x_745_;
goto v___jp_728_;
}
else
{
lean_dec(v___x_742_);
lean_dec(v_fst_739_);
v_a_729_ = v_b_725_;
goto v___jp_728_;
}
}
else
{
lean_dec(v___x_737_);
v_a_729_ = v_b_725_;
goto v___jp_728_;
}
}
v___jp_728_:
{
size_t v___x_730_; size_t v___x_731_; 
v___x_730_ = ((size_t)1ULL);
v___x_731_ = lean_usize_add(v_i_724_, v___x_730_);
v_i_724_ = v___x_731_;
v_b_725_ = v_a_729_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg___boxed(lean_object* v_as_746_, lean_object* v_sz_747_, lean_object* v_i_748_, lean_object* v_b_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
size_t v_sz_boxed_752_; size_t v_i_boxed_753_; lean_object* v_res_754_; 
v_sz_boxed_752_ = lean_unbox_usize(v_sz_747_);
lean_dec(v_sz_747_);
v_i_boxed_753_ = lean_unbox_usize(v_i_748_);
lean_dec(v_i_748_);
v_res_754_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg(v_as_746_, v_sz_boxed_752_, v_i_boxed_753_, v_b_749_, v___y_750_);
lean_dec_ref(v___y_750_);
lean_dec_ref(v_as_746_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardHypData(lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_, lean_object* v_a_758_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_aesop_Aesop_getForwardImplDetailHyps(v_a_755_, v_a_756_, v_a_757_, v_a_758_);
if (lean_obj_tag(v___x_760_) == 0)
{
lean_object* v_a_761_; lean_object* v___x_762_; size_t v_sz_763_; size_t v___x_764_; lean_object* v___x_765_; 
v_a_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_a_761_);
lean_dec_ref_known(v___x_760_, 1);
v___x_762_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardHypData_default___closed__1);
v_sz_763_ = lean_array_size(v_a_761_);
v___x_764_ = ((size_t)0ULL);
v___x_765_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg(v_a_761_, v_sz_763_, v___x_764_, v___x_762_, v_a_755_);
lean_dec(v_a_761_);
if (lean_obj_tag(v___x_765_) == 0)
{
lean_object* v_a_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_773_; 
v_a_766_ = lean_ctor_get(v___x_765_, 0);
v_isSharedCheck_773_ = !lean_is_exclusive(v___x_765_);
if (v_isSharedCheck_773_ == 0)
{
v___x_768_ = v___x_765_;
v_isShared_769_ = v_isSharedCheck_773_;
goto v_resetjp_767_;
}
else
{
lean_inc(v_a_766_);
lean_dec(v___x_765_);
v___x_768_ = lean_box(0);
v_isShared_769_ = v_isSharedCheck_773_;
goto v_resetjp_767_;
}
v_resetjp_767_:
{
lean_object* v___x_771_; 
if (v_isShared_769_ == 0)
{
v___x_771_ = v___x_768_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v_a_766_);
v___x_771_ = v_reuseFailAlloc_772_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
return v___x_771_;
}
}
}
else
{
lean_object* v_a_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_781_; 
v_a_774_ = lean_ctor_get(v___x_765_, 0);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_765_);
if (v_isSharedCheck_781_ == 0)
{
v___x_776_ = v___x_765_;
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_a_774_);
lean_dec(v___x_765_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_a_774_);
v___x_779_ = v_reuseFailAlloc_780_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
return v___x_779_;
}
}
}
}
else
{
lean_object* v_a_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_789_; 
v_a_782_ = lean_ctor_get(v___x_760_, 0);
v_isSharedCheck_789_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_789_ == 0)
{
v___x_784_ = v___x_760_;
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_a_782_);
lean_dec(v___x_760_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_787_; 
if (v_isShared_785_ == 0)
{
v___x_787_ = v___x_784_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v_a_782_);
v___x_787_ = v_reuseFailAlloc_788_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
return v___x_787_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getForwardHypData___boxed(lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_){
_start:
{
lean_object* v_res_795_; 
v_res_795_ = lp_aesop_Aesop_getForwardHypData(v_a_790_, v_a_791_, v_a_792_, v_a_793_);
lean_dec(v_a_793_);
lean_dec_ref(v_a_792_);
lean_dec(v_a_791_);
lean_dec_ref(v_a_790_);
return v_res_795_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0(lean_object* v_00_u03b2_796_, lean_object* v_m_797_, lean_object* v_a_798_, lean_object* v_b_799_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0___redArg(v_m_797_, v_a_798_, v_b_799_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1(lean_object* v_as_801_, size_t v_sz_802_, size_t v_i_803_, lean_object* v_b_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
lean_object* v___x_810_; 
v___x_810_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___redArg(v_as_801_, v_sz_802_, v_i_803_, v_b_804_, v___y_805_);
return v___x_810_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1___boxed(lean_object* v_as_811_, lean_object* v_sz_812_, lean_object* v_i_813_, lean_object* v_b_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_){
_start:
{
size_t v_sz_boxed_820_; size_t v_i_boxed_821_; lean_object* v_res_822_; 
v_sz_boxed_820_ = lean_unbox_usize(v_sz_812_);
lean_dec(v_sz_812_);
v_i_boxed_821_ = lean_unbox_usize(v_i_813_);
lean_dec(v_i_813_);
v_res_822_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getForwardHypData_spec__1(v_as_811_, v_sz_boxed_820_, v_i_boxed_821_, v_b_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec_ref(v_as_811_);
return v_res_822_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0(lean_object* v_00_u03b2_823_, lean_object* v_a_824_, lean_object* v_x_825_){
_start:
{
uint8_t v___x_826_; 
v___x_826_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___redArg(v_a_824_, v_x_825_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0___boxed(lean_object* v_00_u03b2_827_, lean_object* v_a_828_, lean_object* v_x_829_){
_start:
{
uint8_t v_res_830_; lean_object* v_r_831_; 
v_res_830_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__0(v_00_u03b2_827_, v_a_828_, v_x_829_);
lean_dec(v_x_829_);
lean_dec(v_a_828_);
v_r_831_ = lean_box(v_res_830_);
return v_r_831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1(lean_object* v_00_u03b2_832_, lean_object* v_data_833_){
_start:
{
lean_object* v___x_834_; 
v___x_834_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1___redArg(v_data_833_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2(lean_object* v_00_u03b2_835_, lean_object* v_a_836_, lean_object* v_b_837_, lean_object* v_x_838_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__2___redArg(v_a_836_, v_b_837_, v_x_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_840_, lean_object* v_i_841_, lean_object* v_source_842_, lean_object* v_target_843_){
_start:
{
lean_object* v___x_844_; 
v___x_844_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2___redArg(v_i_841_, v_source_842_, v_target_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_845_, lean_object* v_x_846_, lean_object* v_x_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_getForwardHypData_spec__0_spec__1_spec__2_spec__4___redArg(v_x_846_, v_x_847_);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(lean_object* v_as_849_, size_t v_sz_850_, size_t v_i_851_, lean_object* v_b_852_){
_start:
{
uint8_t v___x_854_; 
v___x_854_ = lean_usize_dec_lt(v_i_851_, v_sz_850_);
if (v___x_854_ == 0)
{
lean_object* v___x_855_; 
v___x_855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_855_, 0, v_b_852_);
return v___x_855_;
}
else
{
lean_object* v_snd_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_901_; 
v_snd_856_ = lean_ctor_get(v_b_852_, 1);
v_isSharedCheck_901_ = !lean_is_exclusive(v_b_852_);
if (v_isSharedCheck_901_ == 0)
{
lean_object* v_unused_902_; 
v_unused_902_ = lean_ctor_get(v_b_852_, 0);
lean_dec(v_unused_902_);
v___x_858_ = v_b_852_;
v_isShared_859_ = v_isSharedCheck_901_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_snd_856_);
lean_dec(v_b_852_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_901_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v___x_860_; lean_object* v_a_862_; lean_object* v_a_869_; 
v___x_860_ = lean_box(0);
v_a_869_ = lean_array_uget_borrowed(v_as_849_, v_i_851_);
if (lean_obj_tag(v_a_869_) == 0)
{
v_a_862_ = v_snd_856_;
goto v___jp_861_;
}
else
{
lean_object* v_snd_870_; lean_object* v_val_871_; lean_object* v_fst_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_899_; 
v_snd_870_ = lean_ctor_get(v_snd_856_, 1);
lean_inc(v_snd_870_);
v_val_871_ = lean_ctor_get(v_a_869_, 0);
v_fst_872_ = lean_ctor_get(v_snd_856_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v_snd_856_);
if (v_isSharedCheck_899_ == 0)
{
lean_object* v_unused_900_; 
v_unused_900_ = lean_ctor_get(v_snd_856_, 1);
lean_dec(v_unused_900_);
v___x_874_ = v_snd_856_;
v_isShared_875_ = v_isSharedCheck_899_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_fst_872_);
lean_dec(v_snd_856_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_899_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v_fst_876_; lean_object* v_snd_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_898_; 
v_fst_876_ = lean_ctor_get(v_snd_870_, 0);
v_snd_877_ = lean_ctor_get(v_snd_870_, 1);
v_isSharedCheck_898_ = !lean_is_exclusive(v_snd_870_);
if (v_isSharedCheck_898_ == 0)
{
v___x_879_ = v_snd_870_;
v_isShared_880_ = v_isSharedCheck_898_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_snd_877_);
lean_inc(v_fst_876_);
lean_dec(v_snd_870_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_898_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
uint8_t v___x_888_; 
v___x_888_ = l_Lean_LocalDecl_isImplementationDetail(v_val_871_);
if (v___x_888_ == 0)
{
lean_object* v___x_889_; uint8_t v___x_890_; 
v___x_889_ = l_Lean_LocalDecl_userName(v_val_871_);
v___x_890_ = lp_aesop_Aesop_isForwardImplDetailHypName(v___x_889_);
lean_dec(v___x_889_);
if (v___x_890_ == 0)
{
goto v___jp_881_;
}
else
{
lean_object* v___x_891_; uint8_t v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; 
lean_del_object(v___x_879_);
lean_dec(v_snd_877_);
lean_del_object(v___x_874_);
v___x_891_ = l_Lean_LocalDecl_fvarId(v_val_871_);
v___x_892_ = 1;
lean_inc(v___x_891_);
v___x_893_ = l_Lean_LocalContext_setKind(v_fst_872_, v___x_891_, v___x_892_);
v___x_894_ = l_Lean_LocalInstances_erase(v_fst_876_, v___x_891_);
v___x_895_ = lean_box(v___x_854_);
v___x_896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_896_, 0, v___x_894_);
lean_ctor_set(v___x_896_, 1, v___x_895_);
v___x_897_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_897_, 0, v___x_893_);
lean_ctor_set(v___x_897_, 1, v___x_896_);
v_a_862_ = v___x_897_;
goto v___jp_861_;
}
}
else
{
goto v___jp_881_;
}
v___jp_881_:
{
lean_object* v___x_883_; 
if (v_isShared_880_ == 0)
{
v___x_883_ = v___x_879_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v_fst_876_);
lean_ctor_set(v_reuseFailAlloc_887_, 1, v_snd_877_);
v___x_883_ = v_reuseFailAlloc_887_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
lean_object* v___x_885_; 
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 1, v___x_883_);
v___x_885_ = v___x_874_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v_fst_872_);
lean_ctor_set(v_reuseFailAlloc_886_, 1, v___x_883_);
v___x_885_ = v_reuseFailAlloc_886_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
v_a_862_ = v___x_885_;
goto v___jp_861_;
}
}
}
}
}
}
v___jp_861_:
{
lean_object* v___x_864_; 
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 1, v_a_862_);
lean_ctor_set(v___x_858_, 0, v___x_860_);
v___x_864_ = v___x_858_;
goto v_reusejp_863_;
}
else
{
lean_object* v_reuseFailAlloc_868_; 
v_reuseFailAlloc_868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_868_, 0, v___x_860_);
lean_ctor_set(v_reuseFailAlloc_868_, 1, v_a_862_);
v___x_864_ = v_reuseFailAlloc_868_;
goto v_reusejp_863_;
}
v_reusejp_863_:
{
size_t v___x_865_; size_t v___x_866_; 
v___x_865_ = ((size_t)1ULL);
v___x_866_ = lean_usize_add(v_i_851_, v___x_865_);
v_i_851_ = v___x_866_;
v_b_852_ = v___x_864_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_as_903_, lean_object* v_sz_904_, lean_object* v_i_905_, lean_object* v_b_906_, lean_object* v___y_907_){
_start:
{
size_t v_sz_boxed_908_; size_t v_i_boxed_909_; lean_object* v_res_910_; 
v_sz_boxed_908_ = lean_unbox_usize(v_sz_904_);
lean_dec(v_sz_904_);
v_i_boxed_909_ = lean_unbox_usize(v_i_905_);
lean_dec(v_i_905_);
v_res_910_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_903_, v_sz_boxed_908_, v_i_boxed_909_, v_b_906_);
lean_dec_ref(v_as_903_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1(lean_object* v_as_911_, size_t v_sz_912_, size_t v_i_913_, lean_object* v_b_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
uint8_t v___x_920_; 
v___x_920_ = lean_usize_dec_lt(v_i_913_, v_sz_912_);
if (v___x_920_ == 0)
{
lean_object* v___x_921_; 
v___x_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_921_, 0, v_b_914_);
return v___x_921_;
}
else
{
lean_object* v_snd_922_; lean_object* v___x_924_; uint8_t v_isShared_925_; uint8_t v_isSharedCheck_967_; 
v_snd_922_ = lean_ctor_get(v_b_914_, 1);
v_isSharedCheck_967_ = !lean_is_exclusive(v_b_914_);
if (v_isSharedCheck_967_ == 0)
{
lean_object* v_unused_968_; 
v_unused_968_ = lean_ctor_get(v_b_914_, 0);
lean_dec(v_unused_968_);
v___x_924_ = v_b_914_;
v_isShared_925_ = v_isSharedCheck_967_;
goto v_resetjp_923_;
}
else
{
lean_inc(v_snd_922_);
lean_dec(v_b_914_);
v___x_924_ = lean_box(0);
v_isShared_925_ = v_isSharedCheck_967_;
goto v_resetjp_923_;
}
v_resetjp_923_:
{
lean_object* v___x_926_; lean_object* v_a_928_; lean_object* v_a_935_; 
v___x_926_ = lean_box(0);
v_a_935_ = lean_array_uget_borrowed(v_as_911_, v_i_913_);
if (lean_obj_tag(v_a_935_) == 0)
{
v_a_928_ = v_snd_922_;
goto v___jp_927_;
}
else
{
lean_object* v_snd_936_; lean_object* v_val_937_; lean_object* v_fst_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_965_; 
v_snd_936_ = lean_ctor_get(v_snd_922_, 1);
lean_inc(v_snd_936_);
v_val_937_ = lean_ctor_get(v_a_935_, 0);
v_fst_938_ = lean_ctor_get(v_snd_922_, 0);
v_isSharedCheck_965_ = !lean_is_exclusive(v_snd_922_);
if (v_isSharedCheck_965_ == 0)
{
lean_object* v_unused_966_; 
v_unused_966_ = lean_ctor_get(v_snd_922_, 1);
lean_dec(v_unused_966_);
v___x_940_ = v_snd_922_;
v_isShared_941_ = v_isSharedCheck_965_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_fst_938_);
lean_dec(v_snd_922_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_965_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v_fst_942_; lean_object* v_snd_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_964_; 
v_fst_942_ = lean_ctor_get(v_snd_936_, 0);
v_snd_943_ = lean_ctor_get(v_snd_936_, 1);
v_isSharedCheck_964_ = !lean_is_exclusive(v_snd_936_);
if (v_isSharedCheck_964_ == 0)
{
v___x_945_ = v_snd_936_;
v_isShared_946_ = v_isSharedCheck_964_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_snd_943_);
lean_inc(v_fst_942_);
lean_dec(v_snd_936_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_964_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
uint8_t v___x_954_; 
v___x_954_ = l_Lean_LocalDecl_isImplementationDetail(v_val_937_);
if (v___x_954_ == 0)
{
lean_object* v___x_955_; uint8_t v___x_956_; 
v___x_955_ = l_Lean_LocalDecl_userName(v_val_937_);
v___x_956_ = lp_aesop_Aesop_isForwardImplDetailHypName(v___x_955_);
lean_dec(v___x_955_);
if (v___x_956_ == 0)
{
goto v___jp_947_;
}
else
{
lean_object* v___x_957_; uint8_t v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; 
lean_del_object(v___x_945_);
lean_dec(v_snd_943_);
lean_del_object(v___x_940_);
v___x_957_ = l_Lean_LocalDecl_fvarId(v_val_937_);
v___x_958_ = 1;
lean_inc(v___x_957_);
v___x_959_ = l_Lean_LocalContext_setKind(v_fst_938_, v___x_957_, v___x_958_);
v___x_960_ = l_Lean_LocalInstances_erase(v_fst_942_, v___x_957_);
v___x_961_ = lean_box(v___x_920_);
v___x_962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_962_, 0, v___x_960_);
lean_ctor_set(v___x_962_, 1, v___x_961_);
v___x_963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_963_, 0, v___x_959_);
lean_ctor_set(v___x_963_, 1, v___x_962_);
v_a_928_ = v___x_963_;
goto v___jp_927_;
}
}
else
{
goto v___jp_947_;
}
v___jp_947_:
{
lean_object* v___x_949_; 
if (v_isShared_946_ == 0)
{
v___x_949_ = v___x_945_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v_fst_942_);
lean_ctor_set(v_reuseFailAlloc_953_, 1, v_snd_943_);
v___x_949_ = v_reuseFailAlloc_953_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
lean_object* v___x_951_; 
if (v_isShared_941_ == 0)
{
lean_ctor_set(v___x_940_, 1, v___x_949_);
v___x_951_ = v___x_940_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v_fst_938_);
lean_ctor_set(v_reuseFailAlloc_952_, 1, v___x_949_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
v_a_928_ = v___x_951_;
goto v___jp_927_;
}
}
}
}
}
}
v___jp_927_:
{
lean_object* v___x_930_; 
if (v_isShared_925_ == 0)
{
lean_ctor_set(v___x_924_, 1, v_a_928_);
lean_ctor_set(v___x_924_, 0, v___x_926_);
v___x_930_ = v___x_924_;
goto v_reusejp_929_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v___x_926_);
lean_ctor_set(v_reuseFailAlloc_934_, 1, v_a_928_);
v___x_930_ = v_reuseFailAlloc_934_;
goto v_reusejp_929_;
}
v_reusejp_929_:
{
size_t v___x_931_; size_t v___x_932_; lean_object* v___x_933_; 
v___x_931_ = ((size_t)1ULL);
v___x_932_ = lean_usize_add(v_i_913_, v___x_931_);
v___x_933_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_911_, v_sz_912_, v___x_932_, v___x_930_);
return v___x_933_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1___boxed(lean_object* v_as_969_, lean_object* v_sz_970_, lean_object* v_i_971_, lean_object* v_b_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
size_t v_sz_boxed_978_; size_t v_i_boxed_979_; lean_object* v_res_980_; 
v_sz_boxed_978_ = lean_unbox_usize(v_sz_970_);
lean_dec(v_sz_970_);
v_i_boxed_979_ = lean_unbox_usize(v_i_971_);
lean_dec(v_i_971_);
v_res_980_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1(v_as_969_, v_sz_boxed_978_, v_i_boxed_979_, v_b_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
lean_dec(v___y_974_);
lean_dec_ref(v___y_973_);
lean_dec_ref(v_as_969_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg(lean_object* v_as_981_, size_t v_sz_982_, size_t v_i_983_, lean_object* v_b_984_){
_start:
{
uint8_t v___x_986_; 
v___x_986_ = lean_usize_dec_lt(v_i_983_, v_sz_982_);
if (v___x_986_ == 0)
{
lean_object* v___x_987_; 
v___x_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_987_, 0, v_b_984_);
return v___x_987_;
}
else
{
lean_object* v_snd_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1033_; 
v_snd_988_ = lean_ctor_get(v_b_984_, 1);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_b_984_);
if (v_isSharedCheck_1033_ == 0)
{
lean_object* v_unused_1034_; 
v_unused_1034_ = lean_ctor_get(v_b_984_, 0);
lean_dec(v_unused_1034_);
v___x_990_ = v_b_984_;
v_isShared_991_ = v_isSharedCheck_1033_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_snd_988_);
lean_dec(v_b_984_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1033_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___x_992_; lean_object* v_a_994_; lean_object* v_a_1001_; 
v___x_992_ = lean_box(0);
v_a_1001_ = lean_array_uget_borrowed(v_as_981_, v_i_983_);
if (lean_obj_tag(v_a_1001_) == 0)
{
v_a_994_ = v_snd_988_;
goto v___jp_993_;
}
else
{
lean_object* v_snd_1002_; lean_object* v_val_1003_; lean_object* v_fst_1004_; lean_object* v___x_1006_; uint8_t v_isShared_1007_; uint8_t v_isSharedCheck_1031_; 
v_snd_1002_ = lean_ctor_get(v_snd_988_, 1);
lean_inc(v_snd_1002_);
v_val_1003_ = lean_ctor_get(v_a_1001_, 0);
v_fst_1004_ = lean_ctor_get(v_snd_988_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v_snd_988_);
if (v_isSharedCheck_1031_ == 0)
{
lean_object* v_unused_1032_; 
v_unused_1032_ = lean_ctor_get(v_snd_988_, 1);
lean_dec(v_unused_1032_);
v___x_1006_ = v_snd_988_;
v_isShared_1007_ = v_isSharedCheck_1031_;
goto v_resetjp_1005_;
}
else
{
lean_inc(v_fst_1004_);
lean_dec(v_snd_988_);
v___x_1006_ = lean_box(0);
v_isShared_1007_ = v_isSharedCheck_1031_;
goto v_resetjp_1005_;
}
v_resetjp_1005_:
{
lean_object* v_fst_1008_; lean_object* v_snd_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1030_; 
v_fst_1008_ = lean_ctor_get(v_snd_1002_, 0);
v_snd_1009_ = lean_ctor_get(v_snd_1002_, 1);
v_isSharedCheck_1030_ = !lean_is_exclusive(v_snd_1002_);
if (v_isSharedCheck_1030_ == 0)
{
v___x_1011_ = v_snd_1002_;
v_isShared_1012_ = v_isSharedCheck_1030_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_snd_1009_);
lean_inc(v_fst_1008_);
lean_dec(v_snd_1002_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1030_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
uint8_t v___x_1020_; 
v___x_1020_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1003_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; uint8_t v___x_1022_; 
v___x_1021_ = l_Lean_LocalDecl_userName(v_val_1003_);
v___x_1022_ = lp_aesop_Aesop_isForwardImplDetailHypName(v___x_1021_);
lean_dec(v___x_1021_);
if (v___x_1022_ == 0)
{
goto v___jp_1013_;
}
else
{
lean_object* v___x_1023_; uint8_t v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; 
lean_del_object(v___x_1011_);
lean_dec(v_snd_1009_);
lean_del_object(v___x_1006_);
v___x_1023_ = l_Lean_LocalDecl_fvarId(v_val_1003_);
v___x_1024_ = 1;
lean_inc(v___x_1023_);
v___x_1025_ = l_Lean_LocalContext_setKind(v_fst_1004_, v___x_1023_, v___x_1024_);
v___x_1026_ = l_Lean_LocalInstances_erase(v_fst_1008_, v___x_1023_);
v___x_1027_ = lean_box(v___x_986_);
v___x_1028_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1026_);
lean_ctor_set(v___x_1028_, 1, v___x_1027_);
v___x_1029_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1029_, 0, v___x_1025_);
lean_ctor_set(v___x_1029_, 1, v___x_1028_);
v_a_994_ = v___x_1029_;
goto v___jp_993_;
}
}
else
{
goto v___jp_1013_;
}
v___jp_1013_:
{
lean_object* v___x_1015_; 
if (v_isShared_1012_ == 0)
{
v___x_1015_ = v___x_1011_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1019_; 
v_reuseFailAlloc_1019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1019_, 0, v_fst_1008_);
lean_ctor_set(v_reuseFailAlloc_1019_, 1, v_snd_1009_);
v___x_1015_ = v_reuseFailAlloc_1019_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
lean_object* v___x_1017_; 
if (v_isShared_1007_ == 0)
{
lean_ctor_set(v___x_1006_, 1, v___x_1015_);
v___x_1017_ = v___x_1006_;
goto v_reusejp_1016_;
}
else
{
lean_object* v_reuseFailAlloc_1018_; 
v_reuseFailAlloc_1018_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1018_, 0, v_fst_1004_);
lean_ctor_set(v_reuseFailAlloc_1018_, 1, v___x_1015_);
v___x_1017_ = v_reuseFailAlloc_1018_;
goto v_reusejp_1016_;
}
v_reusejp_1016_:
{
v_a_994_ = v___x_1017_;
goto v___jp_993_;
}
}
}
}
}
}
v___jp_993_:
{
lean_object* v___x_996_; 
if (v_isShared_991_ == 0)
{
lean_ctor_set(v___x_990_, 1, v_a_994_);
lean_ctor_set(v___x_990_, 0, v___x_992_);
v___x_996_ = v___x_990_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v___x_992_);
lean_ctor_set(v_reuseFailAlloc_1000_, 1, v_a_994_);
v___x_996_ = v_reuseFailAlloc_1000_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
size_t v___x_997_; size_t v___x_998_; 
v___x_997_ = ((size_t)1ULL);
v___x_998_ = lean_usize_add(v_i_983_, v___x_997_);
v_i_983_ = v___x_998_;
v_b_984_ = v___x_996_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object* v_as_1035_, lean_object* v_sz_1036_, lean_object* v_i_1037_, lean_object* v_b_1038_, lean_object* v___y_1039_){
_start:
{
size_t v_sz_boxed_1040_; size_t v_i_boxed_1041_; lean_object* v_res_1042_; 
v_sz_boxed_1040_ = lean_unbox_usize(v_sz_1036_);
lean_dec(v_sz_1036_);
v_i_boxed_1041_ = lean_unbox_usize(v_i_1037_);
lean_dec(v_i_1037_);
v_res_1042_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg(v_as_1035_, v_sz_boxed_1040_, v_i_boxed_1041_, v_b_1038_);
lean_dec_ref(v_as_1035_);
return v_res_1042_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2(lean_object* v_as_1043_, size_t v_sz_1044_, size_t v_i_1045_, lean_object* v_b_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
uint8_t v___x_1052_; 
v___x_1052_ = lean_usize_dec_lt(v_i_1045_, v_sz_1044_);
if (v___x_1052_ == 0)
{
lean_object* v___x_1053_; 
v___x_1053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1053_, 0, v_b_1046_);
return v___x_1053_;
}
else
{
lean_object* v_snd_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1099_; 
v_snd_1054_ = lean_ctor_get(v_b_1046_, 1);
v_isSharedCheck_1099_ = !lean_is_exclusive(v_b_1046_);
if (v_isSharedCheck_1099_ == 0)
{
lean_object* v_unused_1100_; 
v_unused_1100_ = lean_ctor_get(v_b_1046_, 0);
lean_dec(v_unused_1100_);
v___x_1056_ = v_b_1046_;
v_isShared_1057_ = v_isSharedCheck_1099_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_snd_1054_);
lean_dec(v_b_1046_);
v___x_1056_ = lean_box(0);
v_isShared_1057_ = v_isSharedCheck_1099_;
goto v_resetjp_1055_;
}
v_resetjp_1055_:
{
lean_object* v___x_1058_; lean_object* v_a_1060_; lean_object* v_a_1067_; 
v___x_1058_ = lean_box(0);
v_a_1067_ = lean_array_uget_borrowed(v_as_1043_, v_i_1045_);
if (lean_obj_tag(v_a_1067_) == 0)
{
v_a_1060_ = v_snd_1054_;
goto v___jp_1059_;
}
else
{
lean_object* v_snd_1068_; lean_object* v_val_1069_; lean_object* v_fst_1070_; lean_object* v___x_1072_; uint8_t v_isShared_1073_; uint8_t v_isSharedCheck_1097_; 
v_snd_1068_ = lean_ctor_get(v_snd_1054_, 1);
lean_inc(v_snd_1068_);
v_val_1069_ = lean_ctor_get(v_a_1067_, 0);
v_fst_1070_ = lean_ctor_get(v_snd_1054_, 0);
v_isSharedCheck_1097_ = !lean_is_exclusive(v_snd_1054_);
if (v_isSharedCheck_1097_ == 0)
{
lean_object* v_unused_1098_; 
v_unused_1098_ = lean_ctor_get(v_snd_1054_, 1);
lean_dec(v_unused_1098_);
v___x_1072_ = v_snd_1054_;
v_isShared_1073_ = v_isSharedCheck_1097_;
goto v_resetjp_1071_;
}
else
{
lean_inc(v_fst_1070_);
lean_dec(v_snd_1054_);
v___x_1072_ = lean_box(0);
v_isShared_1073_ = v_isSharedCheck_1097_;
goto v_resetjp_1071_;
}
v_resetjp_1071_:
{
lean_object* v_fst_1074_; lean_object* v_snd_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1096_; 
v_fst_1074_ = lean_ctor_get(v_snd_1068_, 0);
v_snd_1075_ = lean_ctor_get(v_snd_1068_, 1);
v_isSharedCheck_1096_ = !lean_is_exclusive(v_snd_1068_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1077_ = v_snd_1068_;
v_isShared_1078_ = v_isSharedCheck_1096_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_snd_1075_);
lean_inc(v_fst_1074_);
lean_dec(v_snd_1068_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1096_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
uint8_t v___x_1086_; 
v___x_1086_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1069_);
if (v___x_1086_ == 0)
{
lean_object* v___x_1087_; uint8_t v___x_1088_; 
v___x_1087_ = l_Lean_LocalDecl_userName(v_val_1069_);
v___x_1088_ = lp_aesop_Aesop_isForwardImplDetailHypName(v___x_1087_);
lean_dec(v___x_1087_);
if (v___x_1088_ == 0)
{
goto v___jp_1079_;
}
else
{
lean_object* v___x_1089_; uint8_t v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
lean_del_object(v___x_1077_);
lean_dec(v_snd_1075_);
lean_del_object(v___x_1072_);
v___x_1089_ = l_Lean_LocalDecl_fvarId(v_val_1069_);
v___x_1090_ = 1;
lean_inc(v___x_1089_);
v___x_1091_ = l_Lean_LocalContext_setKind(v_fst_1070_, v___x_1089_, v___x_1090_);
v___x_1092_ = l_Lean_LocalInstances_erase(v_fst_1074_, v___x_1089_);
v___x_1093_ = lean_box(v___x_1052_);
v___x_1094_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1094_, 0, v___x_1092_);
lean_ctor_set(v___x_1094_, 1, v___x_1093_);
v___x_1095_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1095_, 0, v___x_1091_);
lean_ctor_set(v___x_1095_, 1, v___x_1094_);
v_a_1060_ = v___x_1095_;
goto v___jp_1059_;
}
}
else
{
goto v___jp_1079_;
}
v___jp_1079_:
{
lean_object* v___x_1081_; 
if (v_isShared_1078_ == 0)
{
v___x_1081_ = v___x_1077_;
goto v_reusejp_1080_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v_fst_1074_);
lean_ctor_set(v_reuseFailAlloc_1085_, 1, v_snd_1075_);
v___x_1081_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1080_;
}
v_reusejp_1080_:
{
lean_object* v___x_1083_; 
if (v_isShared_1073_ == 0)
{
lean_ctor_set(v___x_1072_, 1, v___x_1081_);
v___x_1083_ = v___x_1072_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_fst_1070_);
lean_ctor_set(v_reuseFailAlloc_1084_, 1, v___x_1081_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
v_a_1060_ = v___x_1083_;
goto v___jp_1059_;
}
}
}
}
}
}
v___jp_1059_:
{
lean_object* v___x_1062_; 
if (v_isShared_1057_ == 0)
{
lean_ctor_set(v___x_1056_, 1, v_a_1060_);
lean_ctor_set(v___x_1056_, 0, v___x_1058_);
v___x_1062_ = v___x_1056_;
goto v_reusejp_1061_;
}
else
{
lean_object* v_reuseFailAlloc_1066_; 
v_reuseFailAlloc_1066_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1066_, 0, v___x_1058_);
lean_ctor_set(v_reuseFailAlloc_1066_, 1, v_a_1060_);
v___x_1062_ = v_reuseFailAlloc_1066_;
goto v_reusejp_1061_;
}
v_reusejp_1061_:
{
size_t v___x_1063_; size_t v___x_1064_; lean_object* v___x_1065_; 
v___x_1063_ = ((size_t)1ULL);
v___x_1064_ = lean_usize_add(v_i_1045_, v___x_1063_);
v___x_1065_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg(v_as_1043_, v_sz_1044_, v___x_1064_, v___x_1062_);
return v___x_1065_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2___boxed(lean_object* v_as_1101_, lean_object* v_sz_1102_, lean_object* v_i_1103_, lean_object* v_b_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
size_t v_sz_boxed_1110_; size_t v_i_boxed_1111_; lean_object* v_res_1112_; 
v_sz_boxed_1110_ = lean_unbox_usize(v_sz_1102_);
lean_dec(v_sz_1102_);
v_i_boxed_1111_ = lean_unbox_usize(v_i_1103_);
lean_dec(v_i_1103_);
v_res_1112_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2(v_as_1101_, v_sz_boxed_1110_, v_i_boxed_1111_, v_b_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_);
lean_dec(v___y_1108_);
lean_dec_ref(v___y_1107_);
lean_dec(v___y_1106_);
lean_dec_ref(v___y_1105_);
lean_dec_ref(v_as_1101_);
return v_res_1112_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0(lean_object* v_init_1113_, lean_object* v_n_1114_, lean_object* v_b_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_){
_start:
{
if (lean_obj_tag(v_n_1114_) == 0)
{
lean_object* v_cs_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; size_t v_sz_1124_; size_t v___x_1125_; lean_object* v___x_1126_; 
v_cs_1121_ = lean_ctor_get(v_n_1114_, 0);
v___x_1122_ = lean_box(0);
v___x_1123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1123_, 0, v___x_1122_);
lean_ctor_set(v___x_1123_, 1, v_b_1115_);
v_sz_1124_ = lean_array_size(v_cs_1121_);
v___x_1125_ = ((size_t)0ULL);
v___x_1126_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1(v_init_1113_, v_cs_1121_, v_sz_1124_, v___x_1125_, v___x_1123_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_);
if (lean_obj_tag(v___x_1126_) == 0)
{
lean_object* v_a_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1141_; 
v_a_1127_ = lean_ctor_get(v___x_1126_, 0);
v_isSharedCheck_1141_ = !lean_is_exclusive(v___x_1126_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1129_ = v___x_1126_;
v_isShared_1130_ = v_isSharedCheck_1141_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_a_1127_);
lean_dec(v___x_1126_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1141_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v_fst_1131_; 
v_fst_1131_ = lean_ctor_get(v_a_1127_, 0);
if (lean_obj_tag(v_fst_1131_) == 0)
{
lean_object* v_snd_1132_; lean_object* v___x_1133_; lean_object* v___x_1135_; 
v_snd_1132_ = lean_ctor_get(v_a_1127_, 1);
lean_inc(v_snd_1132_);
lean_dec(v_a_1127_);
v___x_1133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1133_, 0, v_snd_1132_);
if (v_isShared_1130_ == 0)
{
lean_ctor_set(v___x_1129_, 0, v___x_1133_);
v___x_1135_ = v___x_1129_;
goto v_reusejp_1134_;
}
else
{
lean_object* v_reuseFailAlloc_1136_; 
v_reuseFailAlloc_1136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1136_, 0, v___x_1133_);
v___x_1135_ = v_reuseFailAlloc_1136_;
goto v_reusejp_1134_;
}
v_reusejp_1134_:
{
return v___x_1135_;
}
}
else
{
lean_object* v_val_1137_; lean_object* v___x_1139_; 
lean_inc_ref(v_fst_1131_);
lean_dec(v_a_1127_);
v_val_1137_ = lean_ctor_get(v_fst_1131_, 0);
lean_inc(v_val_1137_);
lean_dec_ref_known(v_fst_1131_, 1);
if (v_isShared_1130_ == 0)
{
lean_ctor_set(v___x_1129_, 0, v_val_1137_);
v___x_1139_ = v___x_1129_;
goto v_reusejp_1138_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v_val_1137_);
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
else
{
lean_object* v_a_1142_; lean_object* v___x_1144_; uint8_t v_isShared_1145_; uint8_t v_isSharedCheck_1149_; 
v_a_1142_ = lean_ctor_get(v___x_1126_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v___x_1126_);
if (v_isSharedCheck_1149_ == 0)
{
v___x_1144_ = v___x_1126_;
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
else
{
lean_inc(v_a_1142_);
lean_dec(v___x_1126_);
v___x_1144_ = lean_box(0);
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
v_resetjp_1143_:
{
lean_object* v___x_1147_; 
if (v_isShared_1145_ == 0)
{
v___x_1147_ = v___x_1144_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1148_; 
v_reuseFailAlloc_1148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1148_, 0, v_a_1142_);
v___x_1147_ = v_reuseFailAlloc_1148_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
return v___x_1147_;
}
}
}
}
else
{
lean_object* v_vs_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; size_t v_sz_1153_; size_t v___x_1154_; lean_object* v___x_1155_; 
v_vs_1150_ = lean_ctor_get(v_n_1114_, 0);
v___x_1151_ = lean_box(0);
v___x_1152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1152_, 0, v___x_1151_);
lean_ctor_set(v___x_1152_, 1, v_b_1115_);
v_sz_1153_ = lean_array_size(v_vs_1150_);
v___x_1154_ = ((size_t)0ULL);
v___x_1155_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2(v_vs_1150_, v_sz_1153_, v___x_1154_, v___x_1152_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_);
if (lean_obj_tag(v___x_1155_) == 0)
{
lean_object* v_a_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1170_; 
v_a_1156_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1170_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1170_ == 0)
{
v___x_1158_ = v___x_1155_;
v_isShared_1159_ = v_isSharedCheck_1170_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_a_1156_);
lean_dec(v___x_1155_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1170_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v_fst_1160_; 
v_fst_1160_ = lean_ctor_get(v_a_1156_, 0);
if (lean_obj_tag(v_fst_1160_) == 0)
{
lean_object* v_snd_1161_; lean_object* v___x_1162_; lean_object* v___x_1164_; 
v_snd_1161_ = lean_ctor_get(v_a_1156_, 1);
lean_inc(v_snd_1161_);
lean_dec(v_a_1156_);
v___x_1162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1162_, 0, v_snd_1161_);
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 0, v___x_1162_);
v___x_1164_ = v___x_1158_;
goto v_reusejp_1163_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v___x_1162_);
v___x_1164_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1163_;
}
v_reusejp_1163_:
{
return v___x_1164_;
}
}
else
{
lean_object* v_val_1166_; lean_object* v___x_1168_; 
lean_inc_ref(v_fst_1160_);
lean_dec(v_a_1156_);
v_val_1166_ = lean_ctor_get(v_fst_1160_, 0);
lean_inc(v_val_1166_);
lean_dec_ref_known(v_fst_1160_, 1);
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 0, v_val_1166_);
v___x_1168_ = v___x_1158_;
goto v_reusejp_1167_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v_val_1166_);
v___x_1168_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1167_;
}
v_reusejp_1167_:
{
return v___x_1168_;
}
}
}
}
else
{
lean_object* v_a_1171_; lean_object* v___x_1173_; uint8_t v_isShared_1174_; uint8_t v_isSharedCheck_1178_; 
v_a_1171_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1178_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1178_ == 0)
{
v___x_1173_ = v___x_1155_;
v_isShared_1174_ = v_isSharedCheck_1178_;
goto v_resetjp_1172_;
}
else
{
lean_inc(v_a_1171_);
lean_dec(v___x_1155_);
v___x_1173_ = lean_box(0);
v_isShared_1174_ = v_isSharedCheck_1178_;
goto v_resetjp_1172_;
}
v_resetjp_1172_:
{
lean_object* v___x_1176_; 
if (v_isShared_1174_ == 0)
{
v___x_1176_ = v___x_1173_;
goto v_reusejp_1175_;
}
else
{
lean_object* v_reuseFailAlloc_1177_; 
v_reuseFailAlloc_1177_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1177_, 0, v_a_1171_);
v___x_1176_ = v_reuseFailAlloc_1177_;
goto v_reusejp_1175_;
}
v_reusejp_1175_:
{
return v___x_1176_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1(lean_object* v_init_1179_, lean_object* v_as_1180_, size_t v_sz_1181_, size_t v_i_1182_, lean_object* v_b_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_){
_start:
{
uint8_t v___x_1189_; 
v___x_1189_ = lean_usize_dec_lt(v_i_1182_, v_sz_1181_);
if (v___x_1189_ == 0)
{
lean_object* v___x_1190_; 
v___x_1190_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1190_, 0, v_b_1183_);
return v___x_1190_;
}
else
{
lean_object* v_snd_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1225_; 
v_snd_1191_ = lean_ctor_get(v_b_1183_, 1);
v_isSharedCheck_1225_ = !lean_is_exclusive(v_b_1183_);
if (v_isSharedCheck_1225_ == 0)
{
lean_object* v_unused_1226_; 
v_unused_1226_ = lean_ctor_get(v_b_1183_, 0);
lean_dec(v_unused_1226_);
v___x_1193_ = v_b_1183_;
v_isShared_1194_ = v_isSharedCheck_1225_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_snd_1191_);
lean_dec(v_b_1183_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1225_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v_a_1195_; lean_object* v___x_1196_; 
v_a_1195_ = lean_array_uget_borrowed(v_as_1180_, v_i_1182_);
lean_inc(v_snd_1191_);
v___x_1196_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0(v_init_1179_, v_a_1195_, v_snd_1191_, v___y_1184_, v___y_1185_, v___y_1186_, v___y_1187_);
if (lean_obj_tag(v___x_1196_) == 0)
{
lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1216_; 
v_a_1197_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1199_ = v___x_1196_;
v_isShared_1200_ = v_isSharedCheck_1216_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_dec(v___x_1196_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1216_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
if (lean_obj_tag(v_a_1197_) == 0)
{
lean_object* v___x_1201_; lean_object* v___x_1203_; 
v___x_1201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1201_, 0, v_a_1197_);
if (v_isShared_1194_ == 0)
{
lean_ctor_set(v___x_1193_, 0, v___x_1201_);
v___x_1203_ = v___x_1193_;
goto v_reusejp_1202_;
}
else
{
lean_object* v_reuseFailAlloc_1207_; 
v_reuseFailAlloc_1207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1207_, 0, v___x_1201_);
lean_ctor_set(v_reuseFailAlloc_1207_, 1, v_snd_1191_);
v___x_1203_ = v_reuseFailAlloc_1207_;
goto v_reusejp_1202_;
}
v_reusejp_1202_:
{
lean_object* v___x_1205_; 
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 0, v___x_1203_);
v___x_1205_ = v___x_1199_;
goto v_reusejp_1204_;
}
else
{
lean_object* v_reuseFailAlloc_1206_; 
v_reuseFailAlloc_1206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1206_, 0, v___x_1203_);
v___x_1205_ = v_reuseFailAlloc_1206_;
goto v_reusejp_1204_;
}
v_reusejp_1204_:
{
return v___x_1205_;
}
}
}
else
{
lean_object* v_a_1208_; lean_object* v___x_1209_; lean_object* v___x_1211_; 
lean_del_object(v___x_1199_);
lean_dec(v_snd_1191_);
v_a_1208_ = lean_ctor_get(v_a_1197_, 0);
lean_inc(v_a_1208_);
lean_dec_ref_known(v_a_1197_, 1);
v___x_1209_ = lean_box(0);
if (v_isShared_1194_ == 0)
{
lean_ctor_set(v___x_1193_, 1, v_a_1208_);
lean_ctor_set(v___x_1193_, 0, v___x_1209_);
v___x_1211_ = v___x_1193_;
goto v_reusejp_1210_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v___x_1209_);
lean_ctor_set(v_reuseFailAlloc_1215_, 1, v_a_1208_);
v___x_1211_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1210_;
}
v_reusejp_1210_:
{
size_t v___x_1212_; size_t v___x_1213_; 
v___x_1212_ = ((size_t)1ULL);
v___x_1213_ = lean_usize_add(v_i_1182_, v___x_1212_);
v_i_1182_ = v___x_1213_;
v_b_1183_ = v___x_1211_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
lean_del_object(v___x_1193_);
lean_dec(v_snd_1191_);
v_a_1217_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1196_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1196_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1___boxed(lean_object* v_init_1227_, lean_object* v_as_1228_, lean_object* v_sz_1229_, lean_object* v_i_1230_, lean_object* v_b_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_){
_start:
{
size_t v_sz_boxed_1237_; size_t v_i_boxed_1238_; lean_object* v_res_1239_; 
v_sz_boxed_1237_ = lean_unbox_usize(v_sz_1229_);
lean_dec(v_sz_1229_);
v_i_boxed_1238_ = lean_unbox_usize(v_i_1230_);
lean_dec(v_i_1230_);
v_res_1239_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__1(v_init_1227_, v_as_1228_, v_sz_boxed_1237_, v_i_boxed_1238_, v_b_1231_, v___y_1232_, v___y_1233_, v___y_1234_, v___y_1235_);
lean_dec(v___y_1235_);
lean_dec_ref(v___y_1234_);
lean_dec(v___y_1233_);
lean_dec_ref(v___y_1232_);
lean_dec_ref(v_as_1228_);
lean_dec_ref(v_init_1227_);
return v_res_1239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0___boxed(lean_object* v_init_1240_, lean_object* v_n_1241_, lean_object* v_b_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_){
_start:
{
lean_object* v_res_1248_; 
v_res_1248_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0(v_init_1240_, v_n_1241_, v_b_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec_ref(v_n_1241_);
lean_dec_ref(v_init_1240_);
return v_res_1248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0(lean_object* v_t_1249_, lean_object* v_init_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_){
_start:
{
lean_object* v_root_1256_; lean_object* v_tail_1257_; lean_object* v___x_1258_; 
v_root_1256_ = lean_ctor_get(v_t_1249_, 0);
v_tail_1257_ = lean_ctor_get(v_t_1249_, 1);
lean_inc_ref(v_init_1250_);
v___x_1258_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0(v_init_1250_, v_root_1256_, v_init_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_);
lean_dec_ref(v_init_1250_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; lean_object* v___x_1261_; uint8_t v_isShared_1262_; uint8_t v_isSharedCheck_1295_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1261_ = v___x_1258_;
v_isShared_1262_ = v_isSharedCheck_1295_;
goto v_resetjp_1260_;
}
else
{
lean_inc(v_a_1259_);
lean_dec(v___x_1258_);
v___x_1261_ = lean_box(0);
v_isShared_1262_ = v_isSharedCheck_1295_;
goto v_resetjp_1260_;
}
v_resetjp_1260_:
{
if (lean_obj_tag(v_a_1259_) == 0)
{
lean_object* v_a_1263_; lean_object* v___x_1265_; 
v_a_1263_ = lean_ctor_get(v_a_1259_, 0);
lean_inc(v_a_1263_);
lean_dec_ref_known(v_a_1259_, 1);
if (v_isShared_1262_ == 0)
{
lean_ctor_set(v___x_1261_, 0, v_a_1263_);
v___x_1265_ = v___x_1261_;
goto v_reusejp_1264_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v_a_1263_);
v___x_1265_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1264_;
}
v_reusejp_1264_:
{
return v___x_1265_;
}
}
else
{
lean_object* v_a_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; size_t v_sz_1270_; size_t v___x_1271_; lean_object* v___x_1272_; 
lean_del_object(v___x_1261_);
v_a_1267_ = lean_ctor_get(v_a_1259_, 0);
lean_inc(v_a_1267_);
lean_dec_ref_known(v_a_1259_, 1);
v___x_1268_ = lean_box(0);
v___x_1269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1269_, 0, v___x_1268_);
lean_ctor_set(v___x_1269_, 1, v_a_1267_);
v_sz_1270_ = lean_array_size(v_tail_1257_);
v___x_1271_ = ((size_t)0ULL);
v___x_1272_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1(v_tail_1257_, v_sz_1270_, v___x_1271_, v___x_1269_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_);
if (lean_obj_tag(v___x_1272_) == 0)
{
lean_object* v_a_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1286_; 
v_a_1273_ = lean_ctor_get(v___x_1272_, 0);
v_isSharedCheck_1286_ = !lean_is_exclusive(v___x_1272_);
if (v_isSharedCheck_1286_ == 0)
{
v___x_1275_ = v___x_1272_;
v_isShared_1276_ = v_isSharedCheck_1286_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_a_1273_);
lean_dec(v___x_1272_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1286_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
lean_object* v_fst_1277_; 
v_fst_1277_ = lean_ctor_get(v_a_1273_, 0);
if (lean_obj_tag(v_fst_1277_) == 0)
{
lean_object* v_snd_1278_; lean_object* v___x_1280_; 
v_snd_1278_ = lean_ctor_get(v_a_1273_, 1);
lean_inc(v_snd_1278_);
lean_dec(v_a_1273_);
if (v_isShared_1276_ == 0)
{
lean_ctor_set(v___x_1275_, 0, v_snd_1278_);
v___x_1280_ = v___x_1275_;
goto v_reusejp_1279_;
}
else
{
lean_object* v_reuseFailAlloc_1281_; 
v_reuseFailAlloc_1281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1281_, 0, v_snd_1278_);
v___x_1280_ = v_reuseFailAlloc_1281_;
goto v_reusejp_1279_;
}
v_reusejp_1279_:
{
return v___x_1280_;
}
}
else
{
lean_object* v_val_1282_; lean_object* v___x_1284_; 
lean_inc_ref(v_fst_1277_);
lean_dec(v_a_1273_);
v_val_1282_ = lean_ctor_get(v_fst_1277_, 0);
lean_inc(v_val_1282_);
lean_dec_ref_known(v_fst_1277_, 1);
if (v_isShared_1276_ == 0)
{
lean_ctor_set(v___x_1275_, 0, v_val_1282_);
v___x_1284_ = v___x_1275_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1285_; 
v_reuseFailAlloc_1285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1285_, 0, v_val_1282_);
v___x_1284_ = v_reuseFailAlloc_1285_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
return v___x_1284_;
}
}
}
}
else
{
lean_object* v_a_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1294_; 
v_a_1287_ = lean_ctor_get(v___x_1272_, 0);
v_isSharedCheck_1294_ = !lean_is_exclusive(v___x_1272_);
if (v_isSharedCheck_1294_ == 0)
{
v___x_1289_ = v___x_1272_;
v_isShared_1290_ = v_isSharedCheck_1294_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_a_1287_);
lean_dec(v___x_1272_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1294_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
lean_object* v___x_1292_; 
if (v_isShared_1290_ == 0)
{
v___x_1292_ = v___x_1289_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v_a_1287_);
v___x_1292_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
return v___x_1292_;
}
}
}
}
}
}
else
{
lean_object* v_a_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1303_; 
v_a_1296_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1303_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1298_ = v___x_1258_;
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_a_1296_);
lean_dec(v___x_1258_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1303_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
lean_object* v___x_1301_; 
if (v_isShared_1299_ == 0)
{
v___x_1301_ = v___x_1298_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v_a_1296_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0___boxed(lean_object* v_t_1304_, lean_object* v_init_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_){
_start:
{
lean_object* v_res_1311_; 
v_res_1311_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0(v_t_1304_, v_init_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_);
lean_dec(v___y_1309_);
lean_dec_ref(v___y_1308_);
lean_dec(v___y_1307_);
lean_dec_ref(v___y_1306_);
lean_dec_ref(v_t_1304_);
return v_res_1311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10___redArg(lean_object* v_x_1312_, lean_object* v_x_1313_, lean_object* v_x_1314_, lean_object* v_x_1315_){
_start:
{
lean_object* v_ks_1316_; lean_object* v_vs_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1341_; 
v_ks_1316_ = lean_ctor_get(v_x_1312_, 0);
v_vs_1317_ = lean_ctor_get(v_x_1312_, 1);
v_isSharedCheck_1341_ = !lean_is_exclusive(v_x_1312_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1319_ = v_x_1312_;
v_isShared_1320_ = v_isSharedCheck_1341_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_vs_1317_);
lean_inc(v_ks_1316_);
lean_dec(v_x_1312_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1341_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1321_; uint8_t v___x_1322_; 
v___x_1321_ = lean_array_get_size(v_ks_1316_);
v___x_1322_ = lean_nat_dec_lt(v_x_1313_, v___x_1321_);
if (v___x_1322_ == 0)
{
lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1326_; 
lean_dec(v_x_1313_);
v___x_1323_ = lean_array_push(v_ks_1316_, v_x_1314_);
v___x_1324_ = lean_array_push(v_vs_1317_, v_x_1315_);
if (v_isShared_1320_ == 0)
{
lean_ctor_set(v___x_1319_, 1, v___x_1324_);
lean_ctor_set(v___x_1319_, 0, v___x_1323_);
v___x_1326_ = v___x_1319_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v___x_1323_);
lean_ctor_set(v_reuseFailAlloc_1327_, 1, v___x_1324_);
v___x_1326_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
return v___x_1326_;
}
}
else
{
lean_object* v_k_x27_1328_; uint8_t v___x_1329_; 
v_k_x27_1328_ = lean_array_fget_borrowed(v_ks_1316_, v_x_1313_);
v___x_1329_ = l_Lean_instBEqMVarId_beq(v_x_1314_, v_k_x27_1328_);
if (v___x_1329_ == 0)
{
lean_object* v___x_1331_; 
if (v_isShared_1320_ == 0)
{
v___x_1331_ = v___x_1319_;
goto v_reusejp_1330_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v_ks_1316_);
lean_ctor_set(v_reuseFailAlloc_1335_, 1, v_vs_1317_);
v___x_1331_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1330_;
}
v_reusejp_1330_:
{
lean_object* v___x_1332_; lean_object* v___x_1333_; 
v___x_1332_ = lean_unsigned_to_nat(1u);
v___x_1333_ = lean_nat_add(v_x_1313_, v___x_1332_);
lean_dec(v_x_1313_);
v_x_1312_ = v___x_1331_;
v_x_1313_ = v___x_1333_;
goto _start;
}
}
else
{
lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1339_; 
v___x_1336_ = lean_array_fset(v_ks_1316_, v_x_1313_, v_x_1314_);
v___x_1337_ = lean_array_fset(v_vs_1317_, v_x_1313_, v_x_1315_);
lean_dec(v_x_1313_);
if (v_isShared_1320_ == 0)
{
lean_ctor_set(v___x_1319_, 1, v___x_1337_);
lean_ctor_set(v___x_1319_, 0, v___x_1336_);
v___x_1339_ = v___x_1319_;
goto v_reusejp_1338_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v___x_1336_);
lean_ctor_set(v_reuseFailAlloc_1340_, 1, v___x_1337_);
v___x_1339_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1338_;
}
v_reusejp_1338_:
{
return v___x_1339_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9___redArg(lean_object* v_n_1342_, lean_object* v_k_1343_, lean_object* v_v_1344_){
_start:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; 
v___x_1345_ = lean_unsigned_to_nat(0u);
v___x_1346_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10___redArg(v_n_1342_, v___x_1345_, v_k_1343_, v_v_1344_);
return v___x_1346_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_1347_; 
v___x_1347_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1347_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(lean_object* v_x_1348_, size_t v_x_1349_, size_t v_x_1350_, lean_object* v_x_1351_, lean_object* v_x_1352_){
_start:
{
if (lean_obj_tag(v_x_1348_) == 0)
{
lean_object* v_es_1353_; size_t v___x_1354_; size_t v___x_1355_; lean_object* v_j_1356_; lean_object* v___x_1357_; uint8_t v___x_1358_; 
v_es_1353_ = lean_ctor_get(v_x_1348_, 0);
v___x_1354_ = ((size_t)31ULL);
v___x_1355_ = lean_usize_land(v_x_1349_, v___x_1354_);
v_j_1356_ = lean_usize_to_nat(v___x_1355_);
v___x_1357_ = lean_array_get_size(v_es_1353_);
v___x_1358_ = lean_nat_dec_lt(v_j_1356_, v___x_1357_);
if (v___x_1358_ == 0)
{
lean_dec(v_j_1356_);
lean_dec(v_x_1352_);
lean_dec(v_x_1351_);
return v_x_1348_;
}
else
{
lean_object* v___x_1360_; uint8_t v_isShared_1361_; uint8_t v_isSharedCheck_1397_; 
lean_inc_ref(v_es_1353_);
v_isSharedCheck_1397_ = !lean_is_exclusive(v_x_1348_);
if (v_isSharedCheck_1397_ == 0)
{
lean_object* v_unused_1398_; 
v_unused_1398_ = lean_ctor_get(v_x_1348_, 0);
lean_dec(v_unused_1398_);
v___x_1360_ = v_x_1348_;
v_isShared_1361_ = v_isSharedCheck_1397_;
goto v_resetjp_1359_;
}
else
{
lean_dec(v_x_1348_);
v___x_1360_ = lean_box(0);
v_isShared_1361_ = v_isSharedCheck_1397_;
goto v_resetjp_1359_;
}
v_resetjp_1359_:
{
lean_object* v_v_1362_; lean_object* v___x_1363_; lean_object* v_xs_x27_1364_; lean_object* v___y_1366_; 
v_v_1362_ = lean_array_fget(v_es_1353_, v_j_1356_);
v___x_1363_ = lean_box(0);
v_xs_x27_1364_ = lean_array_fset(v_es_1353_, v_j_1356_, v___x_1363_);
switch(lean_obj_tag(v_v_1362_))
{
case 0:
{
lean_object* v_key_1371_; lean_object* v_val_1372_; lean_object* v___x_1374_; uint8_t v_isShared_1375_; uint8_t v_isSharedCheck_1382_; 
v_key_1371_ = lean_ctor_get(v_v_1362_, 0);
v_val_1372_ = lean_ctor_get(v_v_1362_, 1);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_v_1362_);
if (v_isSharedCheck_1382_ == 0)
{
v___x_1374_ = v_v_1362_;
v_isShared_1375_ = v_isSharedCheck_1382_;
goto v_resetjp_1373_;
}
else
{
lean_inc(v_val_1372_);
lean_inc(v_key_1371_);
lean_dec(v_v_1362_);
v___x_1374_ = lean_box(0);
v_isShared_1375_ = v_isSharedCheck_1382_;
goto v_resetjp_1373_;
}
v_resetjp_1373_:
{
uint8_t v___x_1376_; 
v___x_1376_ = l_Lean_instBEqMVarId_beq(v_x_1351_, v_key_1371_);
if (v___x_1376_ == 0)
{
lean_object* v___x_1377_; lean_object* v___x_1378_; 
lean_del_object(v___x_1374_);
v___x_1377_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1371_, v_val_1372_, v_x_1351_, v_x_1352_);
v___x_1378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
v___y_1366_ = v___x_1378_;
goto v___jp_1365_;
}
else
{
lean_object* v___x_1380_; 
lean_dec(v_val_1372_);
lean_dec(v_key_1371_);
if (v_isShared_1375_ == 0)
{
lean_ctor_set(v___x_1374_, 1, v_x_1352_);
lean_ctor_set(v___x_1374_, 0, v_x_1351_);
v___x_1380_ = v___x_1374_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v_x_1351_);
lean_ctor_set(v_reuseFailAlloc_1381_, 1, v_x_1352_);
v___x_1380_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
v___y_1366_ = v___x_1380_;
goto v___jp_1365_;
}
}
}
}
case 1:
{
lean_object* v_node_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1395_; 
v_node_1383_ = lean_ctor_get(v_v_1362_, 0);
v_isSharedCheck_1395_ = !lean_is_exclusive(v_v_1362_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1385_ = v_v_1362_;
v_isShared_1386_ = v_isSharedCheck_1395_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_node_1383_);
lean_dec(v_v_1362_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1395_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
size_t v___x_1387_; size_t v___x_1388_; size_t v___x_1389_; size_t v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1393_; 
v___x_1387_ = ((size_t)5ULL);
v___x_1388_ = lean_usize_shift_right(v_x_1349_, v___x_1387_);
v___x_1389_ = ((size_t)1ULL);
v___x_1390_ = lean_usize_add(v_x_1350_, v___x_1389_);
v___x_1391_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(v_node_1383_, v___x_1388_, v___x_1390_, v_x_1351_, v_x_1352_);
if (v_isShared_1386_ == 0)
{
lean_ctor_set(v___x_1385_, 0, v___x_1391_);
v___x_1393_ = v___x_1385_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v___x_1391_);
v___x_1393_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
v___y_1366_ = v___x_1393_;
goto v___jp_1365_;
}
}
}
default: 
{
lean_object* v___x_1396_; 
v___x_1396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1396_, 0, v_x_1351_);
lean_ctor_set(v___x_1396_, 1, v_x_1352_);
v___y_1366_ = v___x_1396_;
goto v___jp_1365_;
}
}
v___jp_1365_:
{
lean_object* v___x_1367_; lean_object* v___x_1369_; 
v___x_1367_ = lean_array_fset(v_xs_x27_1364_, v_j_1356_, v___y_1366_);
lean_dec(v_j_1356_);
if (v_isShared_1361_ == 0)
{
lean_ctor_set(v___x_1360_, 0, v___x_1367_);
v___x_1369_ = v___x_1360_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v___x_1367_);
v___x_1369_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
return v___x_1369_;
}
}
}
}
}
else
{
lean_object* v_ks_1399_; lean_object* v_vs_1400_; lean_object* v___x_1402_; uint8_t v_isShared_1403_; uint8_t v_isSharedCheck_1420_; 
v_ks_1399_ = lean_ctor_get(v_x_1348_, 0);
v_vs_1400_ = lean_ctor_get(v_x_1348_, 1);
v_isSharedCheck_1420_ = !lean_is_exclusive(v_x_1348_);
if (v_isSharedCheck_1420_ == 0)
{
v___x_1402_ = v_x_1348_;
v_isShared_1403_ = v_isSharedCheck_1420_;
goto v_resetjp_1401_;
}
else
{
lean_inc(v_vs_1400_);
lean_inc(v_ks_1399_);
lean_dec(v_x_1348_);
v___x_1402_ = lean_box(0);
v_isShared_1403_ = v_isSharedCheck_1420_;
goto v_resetjp_1401_;
}
v_resetjp_1401_:
{
lean_object* v___x_1405_; 
if (v_isShared_1403_ == 0)
{
v___x_1405_ = v___x_1402_;
goto v_reusejp_1404_;
}
else
{
lean_object* v_reuseFailAlloc_1419_; 
v_reuseFailAlloc_1419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1419_, 0, v_ks_1399_);
lean_ctor_set(v_reuseFailAlloc_1419_, 1, v_vs_1400_);
v___x_1405_ = v_reuseFailAlloc_1419_;
goto v_reusejp_1404_;
}
v_reusejp_1404_:
{
lean_object* v_newNode_1406_; uint8_t v___y_1408_; size_t v___x_1414_; uint8_t v___x_1415_; 
v_newNode_1406_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9___redArg(v___x_1405_, v_x_1351_, v_x_1352_);
v___x_1414_ = ((size_t)7ULL);
v___x_1415_ = lean_usize_dec_le(v___x_1414_, v_x_1350_);
if (v___x_1415_ == 0)
{
lean_object* v___x_1416_; lean_object* v___x_1417_; uint8_t v___x_1418_; 
v___x_1416_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1406_);
v___x_1417_ = lean_unsigned_to_nat(4u);
v___x_1418_ = lean_nat_dec_lt(v___x_1416_, v___x_1417_);
lean_dec(v___x_1416_);
v___y_1408_ = v___x_1418_;
goto v___jp_1407_;
}
else
{
v___y_1408_ = v___x_1415_;
goto v___jp_1407_;
}
v___jp_1407_:
{
if (v___y_1408_ == 0)
{
lean_object* v_ks_1409_; lean_object* v_vs_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; 
v_ks_1409_ = lean_ctor_get(v_newNode_1406_, 0);
lean_inc_ref(v_ks_1409_);
v_vs_1410_ = lean_ctor_get(v_newNode_1406_, 1);
lean_inc_ref(v_vs_1410_);
lean_dec_ref(v_newNode_1406_);
v___x_1411_ = lean_unsigned_to_nat(0u);
v___x_1412_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___closed__0);
v___x_1413_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg(v_x_1350_, v_ks_1409_, v_vs_1410_, v___x_1411_, v___x_1412_);
lean_dec_ref(v_vs_1410_);
lean_dec_ref(v_ks_1409_);
return v___x_1413_;
}
else
{
return v_newNode_1406_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg(size_t v_depth_1421_, lean_object* v_keys_1422_, lean_object* v_vals_1423_, lean_object* v_i_1424_, lean_object* v_entries_1425_){
_start:
{
lean_object* v___x_1426_; uint8_t v___x_1427_; 
v___x_1426_ = lean_array_get_size(v_keys_1422_);
v___x_1427_ = lean_nat_dec_lt(v_i_1424_, v___x_1426_);
if (v___x_1427_ == 0)
{
lean_dec(v_i_1424_);
return v_entries_1425_;
}
else
{
lean_object* v_k_1428_; lean_object* v_v_1429_; uint64_t v___x_1430_; size_t v_h_1431_; size_t v___x_1432_; lean_object* v___x_1433_; size_t v___x_1434_; size_t v___x_1435_; size_t v___x_1436_; size_t v_h_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; 
v_k_1428_ = lean_array_fget_borrowed(v_keys_1422_, v_i_1424_);
v_v_1429_ = lean_array_fget_borrowed(v_vals_1423_, v_i_1424_);
v___x_1430_ = l_Lean_instHashableMVarId_hash(v_k_1428_);
v_h_1431_ = lean_uint64_to_usize(v___x_1430_);
v___x_1432_ = ((size_t)5ULL);
v___x_1433_ = lean_unsigned_to_nat(1u);
v___x_1434_ = ((size_t)1ULL);
v___x_1435_ = lean_usize_sub(v_depth_1421_, v___x_1434_);
v___x_1436_ = lean_usize_mul(v___x_1432_, v___x_1435_);
v_h_1437_ = lean_usize_shift_right(v_h_1431_, v___x_1436_);
v___x_1438_ = lean_nat_add(v_i_1424_, v___x_1433_);
lean_dec(v_i_1424_);
lean_inc(v_v_1429_);
lean_inc(v_k_1428_);
v___x_1439_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(v_entries_1425_, v_h_1437_, v_depth_1421_, v_k_1428_, v_v_1429_);
v_i_1424_ = v___x_1438_;
v_entries_1425_ = v___x_1439_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg___boxed(lean_object* v_depth_1441_, lean_object* v_keys_1442_, lean_object* v_vals_1443_, lean_object* v_i_1444_, lean_object* v_entries_1445_){
_start:
{
size_t v_depth_boxed_1446_; lean_object* v_res_1447_; 
v_depth_boxed_1446_ = lean_unbox_usize(v_depth_1441_);
lean_dec(v_depth_1441_);
v_res_1447_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg(v_depth_boxed_1446_, v_keys_1442_, v_vals_1443_, v_i_1444_, v_entries_1445_);
lean_dec_ref(v_vals_1443_);
lean_dec_ref(v_keys_1442_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg___boxed(lean_object* v_x_1448_, lean_object* v_x_1449_, lean_object* v_x_1450_, lean_object* v_x_1451_, lean_object* v_x_1452_){
_start:
{
size_t v_x_6014__boxed_1453_; size_t v_x_6015__boxed_1454_; lean_object* v_res_1455_; 
v_x_6014__boxed_1453_ = lean_unbox_usize(v_x_1449_);
lean_dec(v_x_1449_);
v_x_6015__boxed_1454_ = lean_unbox_usize(v_x_1450_);
lean_dec(v_x_1450_);
v_res_1455_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(v_x_1448_, v_x_6014__boxed_1453_, v_x_6015__boxed_1454_, v_x_1451_, v_x_1452_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3___redArg(lean_object* v_x_1456_, lean_object* v_x_1457_, lean_object* v_x_1458_){
_start:
{
uint64_t v___x_1459_; size_t v___x_1460_; size_t v___x_1461_; lean_object* v___x_1462_; 
v___x_1459_ = l_Lean_instHashableMVarId_hash(v_x_1457_);
v___x_1460_ = lean_uint64_to_usize(v___x_1459_);
v___x_1461_ = ((size_t)1ULL);
v___x_1462_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(v_x_1456_, v___x_1460_, v___x_1461_, v_x_1457_, v_x_1458_);
return v___x_1462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg(lean_object* v_mvarId_1463_, lean_object* v_val_1464_, lean_object* v___y_1465_){
_start:
{
lean_object* v___x_1467_; lean_object* v_mctx_1468_; lean_object* v_cache_1469_; lean_object* v_zetaDeltaFVarIds_1470_; lean_object* v_postponed_1471_; lean_object* v_diag_1472_; lean_object* v___x_1474_; uint8_t v_isShared_1475_; uint8_t v_isSharedCheck_1500_; 
v___x_1467_ = lean_st_ref_take(v___y_1465_);
v_mctx_1468_ = lean_ctor_get(v___x_1467_, 0);
v_cache_1469_ = lean_ctor_get(v___x_1467_, 1);
v_zetaDeltaFVarIds_1470_ = lean_ctor_get(v___x_1467_, 2);
v_postponed_1471_ = lean_ctor_get(v___x_1467_, 3);
v_diag_1472_ = lean_ctor_get(v___x_1467_, 4);
v_isSharedCheck_1500_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1500_ == 0)
{
v___x_1474_ = v___x_1467_;
v_isShared_1475_ = v_isSharedCheck_1500_;
goto v_resetjp_1473_;
}
else
{
lean_inc(v_diag_1472_);
lean_inc(v_postponed_1471_);
lean_inc(v_zetaDeltaFVarIds_1470_);
lean_inc(v_cache_1469_);
lean_inc(v_mctx_1468_);
lean_dec(v___x_1467_);
v___x_1474_ = lean_box(0);
v_isShared_1475_ = v_isSharedCheck_1500_;
goto v_resetjp_1473_;
}
v_resetjp_1473_:
{
lean_object* v_depth_1476_; lean_object* v_levelAssignDepth_1477_; lean_object* v_lmvarCounter_1478_; lean_object* v_mvarCounter_1479_; lean_object* v_lDecls_1480_; lean_object* v_decls_1481_; lean_object* v_userNames_1482_; lean_object* v_lAssignment_1483_; lean_object* v_eAssignment_1484_; lean_object* v_dAssignment_1485_; lean_object* v___x_1487_; uint8_t v_isShared_1488_; uint8_t v_isSharedCheck_1499_; 
v_depth_1476_ = lean_ctor_get(v_mctx_1468_, 0);
v_levelAssignDepth_1477_ = lean_ctor_get(v_mctx_1468_, 1);
v_lmvarCounter_1478_ = lean_ctor_get(v_mctx_1468_, 2);
v_mvarCounter_1479_ = lean_ctor_get(v_mctx_1468_, 3);
v_lDecls_1480_ = lean_ctor_get(v_mctx_1468_, 4);
v_decls_1481_ = lean_ctor_get(v_mctx_1468_, 5);
v_userNames_1482_ = lean_ctor_get(v_mctx_1468_, 6);
v_lAssignment_1483_ = lean_ctor_get(v_mctx_1468_, 7);
v_eAssignment_1484_ = lean_ctor_get(v_mctx_1468_, 8);
v_dAssignment_1485_ = lean_ctor_get(v_mctx_1468_, 9);
v_isSharedCheck_1499_ = !lean_is_exclusive(v_mctx_1468_);
if (v_isSharedCheck_1499_ == 0)
{
v___x_1487_ = v_mctx_1468_;
v_isShared_1488_ = v_isSharedCheck_1499_;
goto v_resetjp_1486_;
}
else
{
lean_inc(v_dAssignment_1485_);
lean_inc(v_eAssignment_1484_);
lean_inc(v_lAssignment_1483_);
lean_inc(v_userNames_1482_);
lean_inc(v_decls_1481_);
lean_inc(v_lDecls_1480_);
lean_inc(v_mvarCounter_1479_);
lean_inc(v_lmvarCounter_1478_);
lean_inc(v_levelAssignDepth_1477_);
lean_inc(v_depth_1476_);
lean_dec(v_mctx_1468_);
v___x_1487_ = lean_box(0);
v_isShared_1488_ = v_isSharedCheck_1499_;
goto v_resetjp_1486_;
}
v_resetjp_1486_:
{
lean_object* v___x_1489_; lean_object* v___x_1491_; 
v___x_1489_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3___redArg(v_eAssignment_1484_, v_mvarId_1463_, v_val_1464_);
if (v_isShared_1488_ == 0)
{
lean_ctor_set(v___x_1487_, 8, v___x_1489_);
v___x_1491_ = v___x_1487_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1498_; 
v_reuseFailAlloc_1498_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1498_, 0, v_depth_1476_);
lean_ctor_set(v_reuseFailAlloc_1498_, 1, v_levelAssignDepth_1477_);
lean_ctor_set(v_reuseFailAlloc_1498_, 2, v_lmvarCounter_1478_);
lean_ctor_set(v_reuseFailAlloc_1498_, 3, v_mvarCounter_1479_);
lean_ctor_set(v_reuseFailAlloc_1498_, 4, v_lDecls_1480_);
lean_ctor_set(v_reuseFailAlloc_1498_, 5, v_decls_1481_);
lean_ctor_set(v_reuseFailAlloc_1498_, 6, v_userNames_1482_);
lean_ctor_set(v_reuseFailAlloc_1498_, 7, v_lAssignment_1483_);
lean_ctor_set(v_reuseFailAlloc_1498_, 8, v___x_1489_);
lean_ctor_set(v_reuseFailAlloc_1498_, 9, v_dAssignment_1485_);
v___x_1491_ = v_reuseFailAlloc_1498_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
lean_object* v___x_1493_; 
if (v_isShared_1475_ == 0)
{
lean_ctor_set(v___x_1474_, 0, v___x_1491_);
v___x_1493_ = v___x_1474_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1497_; 
v_reuseFailAlloc_1497_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1497_, 0, v___x_1491_);
lean_ctor_set(v_reuseFailAlloc_1497_, 1, v_cache_1469_);
lean_ctor_set(v_reuseFailAlloc_1497_, 2, v_zetaDeltaFVarIds_1470_);
lean_ctor_set(v_reuseFailAlloc_1497_, 3, v_postponed_1471_);
lean_ctor_set(v_reuseFailAlloc_1497_, 4, v_diag_1472_);
v___x_1493_ = v_reuseFailAlloc_1497_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; 
v___x_1494_ = lean_st_ref_set(v___y_1465_, v___x_1493_);
v___x_1495_ = lean_box(0);
v___x_1496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1496_, 0, v___x_1495_);
return v___x_1496_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg___boxed(lean_object* v_mvarId_1501_, lean_object* v_val_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_){
_start:
{
lean_object* v_res_1505_; 
v_res_1505_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg(v_mvarId_1501_, v_val_1502_, v___y_1503_);
lean_dec(v___y_1503_);
return v_res_1505_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0(lean_object* v_goal_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_){
_start:
{
lean_object* v_lctx_1512_; lean_object* v_localInstances_1513_; lean_object* v_decls_1514_; uint8_t v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v_lctx_1512_ = lean_ctor_get(v___y_1507_, 2);
v_localInstances_1513_ = lean_ctor_get(v___y_1507_, 3);
v_decls_1514_ = lean_ctor_get(v_lctx_1512_, 1);
v___x_1515_ = 0;
v___x_1516_ = lean_box(v___x_1515_);
lean_inc_ref(v_localInstances_1513_);
v___x_1517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1517_, 0, v_localInstances_1513_);
lean_ctor_set(v___x_1517_, 1, v___x_1516_);
lean_inc_ref(v_lctx_1512_);
v___x_1518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1518_, 0, v_lctx_1512_);
lean_ctor_set(v___x_1518_, 1, v___x_1517_);
v___x_1519_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0(v_decls_1514_, v___x_1518_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
if (lean_obj_tag(v___x_1519_) == 0)
{
lean_object* v_a_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1565_; 
v_a_1520_ = lean_ctor_get(v___x_1519_, 0);
v_isSharedCheck_1565_ = !lean_is_exclusive(v___x_1519_);
if (v_isSharedCheck_1565_ == 0)
{
v___x_1522_ = v___x_1519_;
v_isShared_1523_ = v_isSharedCheck_1565_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_a_1520_);
lean_dec(v___x_1519_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1565_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v_snd_1524_; lean_object* v_snd_1525_; uint8_t v___x_1526_; 
v_snd_1524_ = lean_ctor_get(v_a_1520_, 1);
lean_inc(v_snd_1524_);
v_snd_1525_ = lean_ctor_get(v_snd_1524_, 1);
v___x_1526_ = lean_unbox(v_snd_1525_);
if (v___x_1526_ == 0)
{
lean_object* v___x_1528_; 
lean_dec(v_snd_1524_);
lean_dec(v_a_1520_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 0, v_goal_1506_);
v___x_1528_ = v___x_1522_;
goto v_reusejp_1527_;
}
else
{
lean_object* v_reuseFailAlloc_1529_; 
v_reuseFailAlloc_1529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1529_, 0, v_goal_1506_);
v___x_1528_ = v_reuseFailAlloc_1529_;
goto v_reusejp_1527_;
}
v_reusejp_1527_:
{
return v___x_1528_;
}
}
else
{
lean_object* v_fst_1530_; lean_object* v_fst_1531_; lean_object* v___x_1532_; 
lean_del_object(v___x_1522_);
v_fst_1530_ = lean_ctor_get(v_a_1520_, 0);
lean_inc(v_fst_1530_);
lean_dec(v_a_1520_);
v_fst_1531_ = lean_ctor_get(v_snd_1524_, 0);
lean_inc(v_fst_1531_);
lean_dec(v_snd_1524_);
lean_inc(v_goal_1506_);
v___x_1532_ = l_Lean_MVarId_getType(v_goal_1506_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_object* v_a_1533_; uint8_t v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; 
v_a_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_a_1533_);
lean_dec_ref_known(v___x_1532_, 1);
v___x_1534_ = 0;
v___x_1535_ = lean_box(0);
v___x_1536_ = lean_unsigned_to_nat(0u);
v___x_1537_ = l_Lean_Meta_mkFreshExprMVarAt(v_fst_1530_, v_fst_1531_, v_a_1533_, v___x_1534_, v___x_1535_, v___x_1536_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
if (lean_obj_tag(v___x_1537_) == 0)
{
lean_object* v_a_1538_; lean_object* v___x_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1547_; 
v_a_1538_ = lean_ctor_get(v___x_1537_, 0);
lean_inc_n(v_a_1538_, 2);
lean_dec_ref_known(v___x_1537_, 1);
v___x_1539_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg(v_goal_1506_, v_a_1538_, v___y_1508_);
v_isSharedCheck_1547_ = !lean_is_exclusive(v___x_1539_);
if (v_isSharedCheck_1547_ == 0)
{
lean_object* v_unused_1548_; 
v_unused_1548_ = lean_ctor_get(v___x_1539_, 0);
lean_dec(v_unused_1548_);
v___x_1541_ = v___x_1539_;
v_isShared_1542_ = v_isSharedCheck_1547_;
goto v_resetjp_1540_;
}
else
{
lean_dec(v___x_1539_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1547_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1543_; lean_object* v___x_1545_; 
v___x_1543_ = l_Lean_Expr_mvarId_x21(v_a_1538_);
lean_dec(v_a_1538_);
if (v_isShared_1542_ == 0)
{
lean_ctor_set(v___x_1541_, 0, v___x_1543_);
v___x_1545_ = v___x_1541_;
goto v_reusejp_1544_;
}
else
{
lean_object* v_reuseFailAlloc_1546_; 
v_reuseFailAlloc_1546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1546_, 0, v___x_1543_);
v___x_1545_ = v_reuseFailAlloc_1546_;
goto v_reusejp_1544_;
}
v_reusejp_1544_:
{
return v___x_1545_;
}
}
}
else
{
lean_object* v_a_1549_; lean_object* v___x_1551_; uint8_t v_isShared_1552_; uint8_t v_isSharedCheck_1556_; 
lean_dec(v_goal_1506_);
v_a_1549_ = lean_ctor_get(v___x_1537_, 0);
v_isSharedCheck_1556_ = !lean_is_exclusive(v___x_1537_);
if (v_isSharedCheck_1556_ == 0)
{
v___x_1551_ = v___x_1537_;
v_isShared_1552_ = v_isSharedCheck_1556_;
goto v_resetjp_1550_;
}
else
{
lean_inc(v_a_1549_);
lean_dec(v___x_1537_);
v___x_1551_ = lean_box(0);
v_isShared_1552_ = v_isSharedCheck_1556_;
goto v_resetjp_1550_;
}
v_resetjp_1550_:
{
lean_object* v___x_1554_; 
if (v_isShared_1552_ == 0)
{
v___x_1554_ = v___x_1551_;
goto v_reusejp_1553_;
}
else
{
lean_object* v_reuseFailAlloc_1555_; 
v_reuseFailAlloc_1555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1555_, 0, v_a_1549_);
v___x_1554_ = v_reuseFailAlloc_1555_;
goto v_reusejp_1553_;
}
v_reusejp_1553_:
{
return v___x_1554_;
}
}
}
}
else
{
lean_object* v_a_1557_; lean_object* v___x_1559_; uint8_t v_isShared_1560_; uint8_t v_isSharedCheck_1564_; 
lean_dec(v_fst_1531_);
lean_dec(v_fst_1530_);
lean_dec(v_goal_1506_);
v_a_1557_ = lean_ctor_get(v___x_1532_, 0);
v_isSharedCheck_1564_ = !lean_is_exclusive(v___x_1532_);
if (v_isSharedCheck_1564_ == 0)
{
v___x_1559_ = v___x_1532_;
v_isShared_1560_ = v_isSharedCheck_1564_;
goto v_resetjp_1558_;
}
else
{
lean_inc(v_a_1557_);
lean_dec(v___x_1532_);
v___x_1559_ = lean_box(0);
v_isShared_1560_ = v_isSharedCheck_1564_;
goto v_resetjp_1558_;
}
v_resetjp_1558_:
{
lean_object* v___x_1562_; 
if (v_isShared_1560_ == 0)
{
v___x_1562_ = v___x_1559_;
goto v_reusejp_1561_;
}
else
{
lean_object* v_reuseFailAlloc_1563_; 
v_reuseFailAlloc_1563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1563_, 0, v_a_1557_);
v___x_1562_ = v_reuseFailAlloc_1563_;
goto v_reusejp_1561_;
}
v_reusejp_1561_:
{
return v___x_1562_;
}
}
}
}
}
}
else
{
lean_object* v_a_1566_; lean_object* v___x_1568_; uint8_t v_isShared_1569_; uint8_t v_isSharedCheck_1573_; 
lean_dec(v_goal_1506_);
v_a_1566_ = lean_ctor_get(v___x_1519_, 0);
v_isSharedCheck_1573_ = !lean_is_exclusive(v___x_1519_);
if (v_isSharedCheck_1573_ == 0)
{
v___x_1568_ = v___x_1519_;
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
else
{
lean_inc(v_a_1566_);
lean_dec(v___x_1519_);
v___x_1568_ = lean_box(0);
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
v_resetjp_1567_:
{
lean_object* v___x_1571_; 
if (v_isShared_1569_ == 0)
{
v___x_1571_ = v___x_1568_;
goto v_reusejp_1570_;
}
else
{
lean_object* v_reuseFailAlloc_1572_; 
v_reuseFailAlloc_1572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1572_, 0, v_a_1566_);
v___x_1571_ = v_reuseFailAlloc_1572_;
goto v_reusejp_1570_;
}
v_reusejp_1570_:
{
return v___x_1571_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0___boxed(lean_object* v_goal_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_){
_start:
{
lean_object* v_res_1580_; 
v_res_1580_ = lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0(v_goal_1574_, v___y_1575_, v___y_1576_, v___y_1577_, v___y_1578_);
lean_dec(v___y_1578_);
lean_dec_ref(v___y_1577_);
lean_dec(v___y_1576_);
lean_dec_ref(v___y_1575_);
return v_res_1580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps(lean_object* v_goal_1581_, lean_object* v_a_1582_, lean_object* v_a_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_){
_start:
{
lean_object* v___f_1587_; lean_object* v___x_1588_; 
lean_inc(v_goal_1581_);
v___f_1587_ = lean_alloc_closure((void*)(lp_aesop_Aesop_hideForwardImplDetailHyps___lam__0___boxed), 6, 1);
lean_closure_set(v___f_1587_, 0, v_goal_1581_);
v___x_1588_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_clearForwardImplDetailHyps_spec__1___redArg(v_goal_1581_, v___f_1587_, v_a_1582_, v_a_1583_, v_a_1584_, v_a_1585_);
return v___x_1588_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hideForwardImplDetailHyps___boxed(lean_object* v_goal_1589_, lean_object* v_a_1590_, lean_object* v_a_1591_, lean_object* v_a_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_){
_start:
{
lean_object* v_res_1595_; 
v_res_1595_ = lp_aesop_Aesop_hideForwardImplDetailHyps(v_goal_1589_, v_a_1590_, v_a_1591_, v_a_1592_, v_a_1593_);
lean_dec(v_a_1593_);
lean_dec_ref(v_a_1592_);
lean_dec(v_a_1591_);
lean_dec_ref(v_a_1590_);
return v_res_1595_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1(lean_object* v_mvarId_1596_, lean_object* v_val_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
lean_object* v___x_1603_; 
v___x_1603_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___redArg(v_mvarId_1596_, v_val_1597_, v___y_1599_);
return v___x_1603_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1___boxed(lean_object* v_mvarId_1604_, lean_object* v_val_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_){
_start:
{
lean_object* v_res_1611_; 
v_res_1611_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1(v_mvarId_1604_, v_val_1605_, v___y_1606_, v___y_1607_, v___y_1608_, v___y_1609_);
lean_dec(v___y_1609_);
lean_dec_ref(v___y_1608_);
lean_dec(v___y_1607_);
lean_dec_ref(v___y_1606_);
return v_res_1611_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3(lean_object* v_00_u03b2_1612_, lean_object* v_x_1613_, lean_object* v_x_1614_, lean_object* v_x_1615_){
_start:
{
lean_object* v___x_1616_; 
v___x_1616_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3___redArg(v_x_1613_, v_x_1614_, v_x_1615_);
return v___x_1616_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4(lean_object* v_as_1617_, size_t v_sz_1618_, size_t v_i_1619_, lean_object* v_b_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_){
_start:
{
lean_object* v___x_1626_; 
v___x_1626_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___redArg(v_as_1617_, v_sz_1618_, v_i_1619_, v_b_1620_);
return v___x_1626_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4___boxed(lean_object* v_as_1627_, lean_object* v_sz_1628_, lean_object* v_i_1629_, lean_object* v_b_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
size_t v_sz_boxed_1636_; size_t v_i_boxed_1637_; lean_object* v_res_1638_; 
v_sz_boxed_1636_ = lean_unbox_usize(v_sz_1628_);
lean_dec(v_sz_1628_);
v_i_boxed_1637_ = lean_unbox_usize(v_i_1629_);
lean_dec(v_i_1629_);
v_res_1638_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__1_spec__4(v_as_1627_, v_sz_boxed_1636_, v_i_boxed_1637_, v_b_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_);
lean_dec(v___y_1634_);
lean_dec_ref(v___y_1633_);
lean_dec(v___y_1632_);
lean_dec_ref(v___y_1631_);
lean_dec_ref(v_as_1627_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7(lean_object* v_00_u03b2_1639_, lean_object* v_x_1640_, size_t v_x_1641_, size_t v_x_1642_, lean_object* v_x_1643_, lean_object* v_x_1644_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___redArg(v_x_1640_, v_x_1641_, v_x_1642_, v_x_1643_, v_x_1644_);
return v___x_1645_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7___boxed(lean_object* v_00_u03b2_1646_, lean_object* v_x_1647_, lean_object* v_x_1648_, lean_object* v_x_1649_, lean_object* v_x_1650_, lean_object* v_x_1651_){
_start:
{
size_t v_x_6404__boxed_1652_; size_t v_x_6405__boxed_1653_; lean_object* v_res_1654_; 
v_x_6404__boxed_1652_ = lean_unbox_usize(v_x_1648_);
lean_dec(v_x_1648_);
v_x_6405__boxed_1653_ = lean_unbox_usize(v_x_1649_);
lean_dec(v_x_1649_);
v_res_1654_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7(v_00_u03b2_1646_, v_x_1647_, v_x_6404__boxed_1652_, v_x_6405__boxed_1653_, v_x_1650_, v_x_1651_);
return v_res_1654_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4(lean_object* v_as_1655_, size_t v_sz_1656_, size_t v_i_1657_, lean_object* v_b_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_){
_start:
{
lean_object* v___x_1664_; 
v___x_1664_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___redArg(v_as_1655_, v_sz_1656_, v_i_1657_, v_b_1658_);
return v___x_1664_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_as_1665_, lean_object* v_sz_1666_, lean_object* v_i_1667_, lean_object* v_b_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_){
_start:
{
size_t v_sz_boxed_1674_; size_t v_i_boxed_1675_; lean_object* v_res_1676_; 
v_sz_boxed_1674_ = lean_unbox_usize(v_sz_1666_);
lean_dec(v_sz_1666_);
v_i_boxed_1675_ = lean_unbox_usize(v_i_1667_);
lean_dec(v_i_1667_);
v_res_1676_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_hideForwardImplDetailHyps_spec__0_spec__0_spec__2_spec__4(v_as_1665_, v_sz_boxed_1674_, v_i_boxed_1675_, v_b_1668_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_);
lean_dec(v___y_1672_);
lean_dec_ref(v___y_1671_);
lean_dec(v___y_1670_);
lean_dec_ref(v___y_1669_);
lean_dec_ref(v_as_1665_);
return v_res_1676_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9(lean_object* v_00_u03b2_1677_, lean_object* v_n_1678_, lean_object* v_k_1679_, lean_object* v_v_1680_){
_start:
{
lean_object* v___x_1681_; 
v___x_1681_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9___redArg(v_n_1678_, v_k_1679_, v_v_1680_);
return v___x_1681_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10(lean_object* v_00_u03b2_1682_, size_t v_depth_1683_, lean_object* v_keys_1684_, lean_object* v_vals_1685_, lean_object* v_heq_1686_, lean_object* v_i_1687_, lean_object* v_entries_1688_){
_start:
{
lean_object* v___x_1689_; 
v___x_1689_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___redArg(v_depth_1683_, v_keys_1684_, v_vals_1685_, v_i_1687_, v_entries_1688_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10___boxed(lean_object* v_00_u03b2_1690_, lean_object* v_depth_1691_, lean_object* v_keys_1692_, lean_object* v_vals_1693_, lean_object* v_heq_1694_, lean_object* v_i_1695_, lean_object* v_entries_1696_){
_start:
{
size_t v_depth_boxed_1697_; lean_object* v_res_1698_; 
v_depth_boxed_1697_ = lean_unbox_usize(v_depth_1691_);
lean_dec(v_depth_1691_);
v_res_1698_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__10(v_00_u03b2_1690_, v_depth_boxed_1697_, v_keys_1692_, v_vals_1693_, v_heq_1694_, v_i_1695_, v_entries_1696_);
lean_dec_ref(v_vals_1693_);
lean_dec_ref(v_keys_1692_);
return v_res_1698_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10(lean_object* v_00_u03b2_1699_, lean_object* v_x_1700_, lean_object* v_x_1701_, lean_object* v_x_1702_, lean_object* v_x_1703_){
_start:
{
lean_object* v___x_1704_; 
v___x_1704_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_hideForwardImplDetailHyps_spec__1_spec__3_spec__7_spec__9_spec__10___redArg(v_x_1700_, v_x_1701_, v_x_1702_, v_x_1703_);
return v___x_1704_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedForwardHypData_default = _init_lp_aesop_Aesop_instInhabitedForwardHypData_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardHypData_default);
lp_aesop_Aesop_instInhabitedForwardHypData = _init_lp_aesop_Aesop_instInhabitedForwardHypData();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardHypData);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
