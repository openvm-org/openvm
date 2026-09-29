// Lean compiler output
// Module: Batteries.Lean.Meta.Inaccessible
// Imports: public import Init public meta import Init public import Lean.Meta.Basic
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
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_local_ctx_num_indices(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_LocalContext_getUnusedName(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_LocalContext_setUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVarAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0;
static lean_once_cell_t lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_inaccessibleFVars(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Lean_MVarId_renameInaccessibleFVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars___closed__0 = (const lean_object*)&lp_batteries_Lean_MVarId_renameInaccessibleFVars___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(lean_object* v_a_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 0;
return v___x_3_;
}
else
{
lean_object* v_key_4_; lean_object* v_tail_5_; uint8_t v___x_6_; 
v_key_4_ = lean_ctor_get(v_x_2_, 0);
v_tail_5_ = lean_ctor_get(v_x_2_, 2);
v___x_6_ = lean_name_eq(v_key_4_, v_a_1_);
if (v___x_6_ == 0)
{
v_x_2_ = v_tail_5_;
goto _start;
}
else
{
return v___x_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg___boxed(lean_object* v_a_8_, lean_object* v_x_9_){
_start:
{
uint8_t v_res_10_; lean_object* v_r_11_; 
v_res_10_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(v_a_8_, v_x_9_);
lean_dec(v_x_9_);
lean_dec(v_a_8_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5___redArg(lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
if (lean_obj_tag(v_x_13_) == 0)
{
return v_x_12_;
}
else
{
lean_object* v_key_14_; lean_object* v_value_15_; lean_object* v_tail_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_42_; 
v_key_14_ = lean_ctor_get(v_x_13_, 0);
v_value_15_ = lean_ctor_get(v_x_13_, 1);
v_tail_16_ = lean_ctor_get(v_x_13_, 2);
v_isSharedCheck_42_ = !lean_is_exclusive(v_x_13_);
if (v_isSharedCheck_42_ == 0)
{
v___x_18_ = v_x_13_;
v_isShared_19_ = v_isSharedCheck_42_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_tail_16_);
lean_inc(v_value_15_);
lean_inc(v_key_14_);
lean_dec(v_x_13_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_42_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_20_; uint64_t v___y_22_; 
v___x_20_ = lean_array_get_size(v_x_12_);
if (lean_obj_tag(v_key_14_) == 0)
{
uint64_t v___x_40_; 
v___x_40_ = 1723ULL;
v___y_22_ = v___x_40_;
goto v___jp_21_;
}
else
{
uint64_t v_hash_41_; 
v_hash_41_ = lean_ctor_get_uint64(v_key_14_, sizeof(void*)*2);
v___y_22_ = v_hash_41_;
goto v___jp_21_;
}
v___jp_21_:
{
uint64_t v___x_23_; uint64_t v___x_24_; uint64_t v_fold_25_; uint64_t v___x_26_; uint64_t v___x_27_; uint64_t v___x_28_; size_t v___x_29_; size_t v___x_30_; size_t v___x_31_; size_t v___x_32_; size_t v___x_33_; lean_object* v___x_34_; lean_object* v___x_36_; 
v___x_23_ = 32ULL;
v___x_24_ = lean_uint64_shift_right(v___y_22_, v___x_23_);
v_fold_25_ = lean_uint64_xor(v___y_22_, v___x_24_);
v___x_26_ = 16ULL;
v___x_27_ = lean_uint64_shift_right(v_fold_25_, v___x_26_);
v___x_28_ = lean_uint64_xor(v_fold_25_, v___x_27_);
v___x_29_ = lean_uint64_to_usize(v___x_28_);
v___x_30_ = lean_usize_of_nat(v___x_20_);
v___x_31_ = ((size_t)1ULL);
v___x_32_ = lean_usize_sub(v___x_30_, v___x_31_);
v___x_33_ = lean_usize_land(v___x_29_, v___x_32_);
v___x_34_ = lean_array_uget_borrowed(v_x_12_, v___x_33_);
lean_inc(v___x_34_);
if (v_isShared_19_ == 0)
{
lean_ctor_set(v___x_18_, 2, v___x_34_);
v___x_36_ = v___x_18_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v_key_14_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v_value_15_);
lean_ctor_set(v_reuseFailAlloc_39_, 2, v___x_34_);
v___x_36_ = v_reuseFailAlloc_39_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
lean_object* v___x_37_; 
v___x_37_ = lean_array_uset(v_x_12_, v___x_33_, v___x_36_);
v_x_12_ = v___x_37_;
v_x_13_ = v_tail_16_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2___redArg(lean_object* v_i_43_, lean_object* v_source_44_, lean_object* v_target_45_){
_start:
{
lean_object* v___x_46_; uint8_t v___x_47_; 
v___x_46_ = lean_array_get_size(v_source_44_);
v___x_47_ = lean_nat_dec_lt(v_i_43_, v___x_46_);
if (v___x_47_ == 0)
{
lean_dec_ref(v_source_44_);
lean_dec(v_i_43_);
return v_target_45_;
}
else
{
lean_object* v_es_48_; lean_object* v___x_49_; lean_object* v_source_50_; lean_object* v_target_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v_es_48_ = lean_array_fget(v_source_44_, v_i_43_);
v___x_49_ = lean_box(0);
v_source_50_ = lean_array_fset(v_source_44_, v_i_43_, v___x_49_);
v_target_51_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5___redArg(v_target_45_, v_es_48_);
v___x_52_ = lean_unsigned_to_nat(1u);
v___x_53_ = lean_nat_add(v_i_43_, v___x_52_);
lean_dec(v_i_43_);
v_i_43_ = v___x_53_;
v_source_44_ = v_source_50_;
v_target_45_ = v_target_51_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1___redArg(lean_object* v_data_55_){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v_nbuckets_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_56_ = lean_array_get_size(v_data_55_);
v___x_57_ = lean_unsigned_to_nat(2u);
v_nbuckets_58_ = lean_nat_mul(v___x_56_, v___x_57_);
v___x_59_ = lean_unsigned_to_nat(0u);
v___x_60_ = lean_box(0);
v___x_61_ = lean_mk_array(v_nbuckets_58_, v___x_60_);
v___x_62_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2___redArg(v___x_59_, v_data_55_, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0___redArg(lean_object* v_m_63_, lean_object* v_a_64_, lean_object* v_b_65_){
_start:
{
lean_object* v_size_66_; lean_object* v_buckets_67_; lean_object* v___x_68_; uint64_t v___y_70_; 
v_size_66_ = lean_ctor_get(v_m_63_, 0);
v_buckets_67_ = lean_ctor_get(v_m_63_, 1);
v___x_68_ = lean_array_get_size(v_buckets_67_);
if (lean_obj_tag(v_a_64_) == 0)
{
uint64_t v___x_107_; 
v___x_107_ = 1723ULL;
v___y_70_ = v___x_107_;
goto v___jp_69_;
}
else
{
uint64_t v_hash_108_; 
v_hash_108_ = lean_ctor_get_uint64(v_a_64_, sizeof(void*)*2);
v___y_70_ = v_hash_108_;
goto v___jp_69_;
}
v___jp_69_:
{
uint64_t v___x_71_; uint64_t v___x_72_; uint64_t v_fold_73_; uint64_t v___x_74_; uint64_t v___x_75_; uint64_t v___x_76_; size_t v___x_77_; size_t v___x_78_; size_t v___x_79_; size_t v___x_80_; size_t v___x_81_; lean_object* v_bkt_82_; uint8_t v___x_83_; 
v___x_71_ = 32ULL;
v___x_72_ = lean_uint64_shift_right(v___y_70_, v___x_71_);
v_fold_73_ = lean_uint64_xor(v___y_70_, v___x_72_);
v___x_74_ = 16ULL;
v___x_75_ = lean_uint64_shift_right(v_fold_73_, v___x_74_);
v___x_76_ = lean_uint64_xor(v_fold_73_, v___x_75_);
v___x_77_ = lean_uint64_to_usize(v___x_76_);
v___x_78_ = lean_usize_of_nat(v___x_68_);
v___x_79_ = ((size_t)1ULL);
v___x_80_ = lean_usize_sub(v___x_78_, v___x_79_);
v___x_81_ = lean_usize_land(v___x_77_, v___x_80_);
v_bkt_82_ = lean_array_uget_borrowed(v_buckets_67_, v___x_81_);
v___x_83_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(v_a_64_, v_bkt_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_85_; uint8_t v_isShared_86_; uint8_t v_isSharedCheck_104_; 
lean_inc_ref(v_buckets_67_);
lean_inc(v_size_66_);
v_isSharedCheck_104_ = !lean_is_exclusive(v_m_63_);
if (v_isSharedCheck_104_ == 0)
{
lean_object* v_unused_105_; lean_object* v_unused_106_; 
v_unused_105_ = lean_ctor_get(v_m_63_, 1);
lean_dec(v_unused_105_);
v_unused_106_ = lean_ctor_get(v_m_63_, 0);
lean_dec(v_unused_106_);
v___x_85_ = v_m_63_;
v_isShared_86_ = v_isSharedCheck_104_;
goto v_resetjp_84_;
}
else
{
lean_dec(v_m_63_);
v___x_85_ = lean_box(0);
v_isShared_86_ = v_isSharedCheck_104_;
goto v_resetjp_84_;
}
v_resetjp_84_:
{
lean_object* v___x_87_; lean_object* v_size_x27_88_; lean_object* v___x_89_; lean_object* v_buckets_x27_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_87_ = lean_unsigned_to_nat(1u);
v_size_x27_88_ = lean_nat_add(v_size_66_, v___x_87_);
lean_dec(v_size_66_);
lean_inc(v_bkt_82_);
v___x_89_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_89_, 0, v_a_64_);
lean_ctor_set(v___x_89_, 1, v_b_65_);
lean_ctor_set(v___x_89_, 2, v_bkt_82_);
v_buckets_x27_90_ = lean_array_uset(v_buckets_67_, v___x_81_, v___x_89_);
v___x_91_ = lean_unsigned_to_nat(4u);
v___x_92_ = lean_nat_mul(v_size_x27_88_, v___x_91_);
v___x_93_ = lean_unsigned_to_nat(3u);
v___x_94_ = lean_nat_div(v___x_92_, v___x_93_);
lean_dec(v___x_92_);
v___x_95_ = lean_array_get_size(v_buckets_x27_90_);
v___x_96_ = lean_nat_dec_le(v___x_94_, v___x_95_);
lean_dec(v___x_94_);
if (v___x_96_ == 0)
{
lean_object* v_val_97_; lean_object* v___x_99_; 
v_val_97_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1___redArg(v_buckets_x27_90_);
if (v_isShared_86_ == 0)
{
lean_ctor_set(v___x_85_, 1, v_val_97_);
lean_ctor_set(v___x_85_, 0, v_size_x27_88_);
v___x_99_ = v___x_85_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v_size_x27_88_);
lean_ctor_set(v_reuseFailAlloc_100_, 1, v_val_97_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
else
{
lean_object* v___x_102_; 
if (v_isShared_86_ == 0)
{
lean_ctor_set(v___x_85_, 1, v_buckets_x27_90_);
lean_ctor_set(v___x_85_, 0, v_size_x27_88_);
v___x_102_ = v___x_85_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v_size_x27_88_);
lean_ctor_set(v_reuseFailAlloc_103_, 1, v_buckets_x27_90_);
v___x_102_ = v_reuseFailAlloc_103_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
return v___x_102_;
}
}
}
}
else
{
lean_dec(v_b_65_);
lean_dec(v_a_64_);
return v_m_63_;
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg(lean_object* v_m_109_, lean_object* v_a_110_){
_start:
{
lean_object* v_buckets_111_; lean_object* v___x_112_; uint64_t v___y_114_; 
v_buckets_111_ = lean_ctor_get(v_m_109_, 1);
v___x_112_ = lean_array_get_size(v_buckets_111_);
if (lean_obj_tag(v_a_110_) == 0)
{
uint64_t v___x_128_; 
v___x_128_ = 1723ULL;
v___y_114_ = v___x_128_;
goto v___jp_113_;
}
else
{
uint64_t v_hash_129_; 
v_hash_129_ = lean_ctor_get_uint64(v_a_110_, sizeof(void*)*2);
v___y_114_ = v_hash_129_;
goto v___jp_113_;
}
v___jp_113_:
{
uint64_t v___x_115_; uint64_t v___x_116_; uint64_t v_fold_117_; uint64_t v___x_118_; uint64_t v___x_119_; uint64_t v___x_120_; size_t v___x_121_; size_t v___x_122_; size_t v___x_123_; size_t v___x_124_; size_t v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_115_ = 32ULL;
v___x_116_ = lean_uint64_shift_right(v___y_114_, v___x_115_);
v_fold_117_ = lean_uint64_xor(v___y_114_, v___x_116_);
v___x_118_ = 16ULL;
v___x_119_ = lean_uint64_shift_right(v_fold_117_, v___x_118_);
v___x_120_ = lean_uint64_xor(v_fold_117_, v___x_119_);
v___x_121_ = lean_uint64_to_usize(v___x_120_);
v___x_122_ = lean_usize_of_nat(v___x_112_);
v___x_123_ = ((size_t)1ULL);
v___x_124_ = lean_usize_sub(v___x_122_, v___x_123_);
v___x_125_ = lean_usize_land(v___x_121_, v___x_124_);
v___x_126_ = lean_array_uget_borrowed(v_buckets_111_, v___x_125_);
v___x_127_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(v_a_110_, v___x_126_);
return v___x_127_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg___boxed(lean_object* v_m_130_, lean_object* v_a_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg(v_m_130_, v_a_131_);
lean_dec(v_a_131_);
lean_dec_ref(v_m_130_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7(lean_object* v_as_134_, size_t v_i_135_, size_t v_stop_136_, lean_object* v_b_137_){
_start:
{
uint8_t v___x_138_; 
v___x_138_ = lean_usize_dec_eq(v_i_135_, v_stop_136_);
if (v___x_138_ == 0)
{
size_t v___x_139_; size_t v___x_140_; lean_object* v___x_141_; 
v___x_139_ = ((size_t)1ULL);
v___x_140_ = lean_usize_sub(v_i_135_, v___x_139_);
v___x_141_ = lean_array_uget_borrowed(v_as_134_, v___x_140_);
if (lean_obj_tag(v___x_141_) == 0)
{
v_i_135_ = v___x_140_;
goto _start;
}
else
{
lean_object* v_val_143_; lean_object* v_fst_144_; lean_object* v_snd_145_; uint8_t v___x_146_; 
v_val_143_ = lean_ctor_get(v___x_141_, 0);
v_fst_144_ = lean_ctor_get(v_b_137_, 0);
v_snd_145_ = lean_ctor_get(v_b_137_, 1);
v___x_146_ = l_Lean_LocalDecl_isImplementationDetail(v_val_143_);
if (v___x_146_ == 0)
{
lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_164_; 
lean_inc(v_snd_145_);
lean_inc(v_fst_144_);
v_isSharedCheck_164_ = !lean_is_exclusive(v_b_137_);
if (v_isSharedCheck_164_ == 0)
{
lean_object* v_unused_165_; lean_object* v_unused_166_; 
v_unused_165_ = lean_ctor_get(v_b_137_, 1);
lean_dec(v_unused_165_);
v_unused_166_ = lean_ctor_get(v_b_137_, 0);
lean_dec(v_unused_166_);
v___x_148_ = v_b_137_;
v_isShared_149_ = v_isSharedCheck_164_;
goto v_resetjp_147_;
}
else
{
lean_dec(v_b_137_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_164_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_150_; lean_object* v___y_152_; uint8_t v___y_160_; uint8_t v___x_162_; 
v___x_150_ = l_Lean_LocalDecl_userName(v_val_143_);
v___x_162_ = l_Lean_Name_hasMacroScopes(v___x_150_);
if (v___x_162_ == 0)
{
uint8_t v___x_163_; 
v___x_163_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg(v_snd_145_, v___x_150_);
v___y_160_ = v___x_163_;
goto v___jp_159_;
}
else
{
v___y_160_ = v___x_162_;
goto v___jp_159_;
}
v___jp_151_:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_156_; 
v___x_153_ = lean_box(0);
v___x_154_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0___redArg(v_snd_145_, v___x_150_, v___x_153_);
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 1, v___x_154_);
lean_ctor_set(v___x_148_, 0, v___y_152_);
v___x_156_ = v___x_148_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v___y_152_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v___x_154_);
v___x_156_ = v_reuseFailAlloc_158_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
v_i_135_ = v___x_140_;
v_b_137_ = v___x_156_;
goto _start;
}
}
v___jp_159_:
{
if (v___y_160_ == 0)
{
v___y_152_ = v_fst_144_;
goto v___jp_151_;
}
else
{
lean_object* v___x_161_; 
lean_inc(v_val_143_);
v___x_161_ = lean_array_push(v_fst_144_, v_val_143_);
v___y_152_ = v___x_161_;
goto v___jp_151_;
}
}
}
}
else
{
v_i_135_ = v___x_140_;
goto _start;
}
}
}
else
{
return v_b_137_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7___boxed(lean_object* v_as_168_, lean_object* v_i_169_, lean_object* v_stop_170_, lean_object* v_b_171_){
_start:
{
size_t v_i_boxed_172_; size_t v_stop_boxed_173_; lean_object* v_res_174_; 
v_i_boxed_172_ = lean_unbox_usize(v_i_169_);
lean_dec(v_i_169_);
v_stop_boxed_173_ = lean_unbox_usize(v_stop_170_);
lean_dec(v_stop_170_);
v_res_174_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7(v_as_168_, v_i_boxed_172_, v_stop_boxed_173_, v_b_171_);
lean_dec_ref(v_as_168_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(lean_object* v_x_175_, lean_object* v_x_176_){
_start:
{
if (lean_obj_tag(v_x_175_) == 0)
{
lean_object* v_cs_177_; lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v_cs_177_ = lean_ctor_get(v_x_175_, 0);
v___x_178_ = lean_array_get_size(v_cs_177_);
v___x_179_ = lean_unsigned_to_nat(0u);
v___x_180_ = lean_nat_dec_lt(v___x_179_, v___x_178_);
if (v___x_180_ == 0)
{
return v_x_176_;
}
else
{
size_t v___x_181_; size_t v___x_182_; lean_object* v___x_183_; 
v___x_181_ = lean_usize_of_nat(v___x_178_);
v___x_182_ = ((size_t)0ULL);
v___x_183_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8(v_cs_177_, v___x_181_, v___x_182_, v_x_176_);
return v___x_183_;
}
}
else
{
lean_object* v_vs_184_; lean_object* v___x_185_; lean_object* v___x_186_; uint8_t v___x_187_; 
v_vs_184_ = lean_ctor_get(v_x_175_, 0);
v___x_185_ = lean_array_get_size(v_vs_184_);
v___x_186_ = lean_unsigned_to_nat(0u);
v___x_187_ = lean_nat_dec_lt(v___x_186_, v___x_185_);
if (v___x_187_ == 0)
{
return v_x_176_;
}
else
{
size_t v___x_188_; size_t v___x_189_; lean_object* v___x_190_; 
v___x_188_ = lean_usize_of_nat(v___x_185_);
v___x_189_ = ((size_t)0ULL);
v___x_190_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7(v_vs_184_, v___x_188_, v___x_189_, v_x_176_);
return v___x_190_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8(lean_object* v_as_191_, size_t v_i_192_, size_t v_stop_193_, lean_object* v_b_194_){
_start:
{
uint8_t v___x_195_; 
v___x_195_ = lean_usize_dec_eq(v_i_192_, v_stop_193_);
if (v___x_195_ == 0)
{
size_t v___x_196_; size_t v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_196_ = ((size_t)1ULL);
v___x_197_ = lean_usize_sub(v_i_192_, v___x_196_);
v___x_198_ = lean_array_uget_borrowed(v_as_191_, v___x_197_);
v___x_199_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(v___x_198_, v_b_194_);
v_i_192_ = v___x_197_;
v_b_194_ = v___x_199_;
goto _start;
}
else
{
return v_b_194_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8___boxed(lean_object* v_as_201_, lean_object* v_i_202_, lean_object* v_stop_203_, lean_object* v_b_204_){
_start:
{
size_t v_i_boxed_205_; size_t v_stop_boxed_206_; lean_object* v_res_207_; 
v_i_boxed_205_ = lean_unbox_usize(v_i_202_);
lean_dec(v_i_202_);
v_stop_boxed_206_ = lean_unbox_usize(v_stop_203_);
lean_dec(v_stop_203_);
v_res_207_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6_spec__8(v_as_201_, v_i_boxed_205_, v_stop_boxed_206_, v_b_204_);
lean_dec_ref(v_as_201_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6___boxed(lean_object* v_x_208_, lean_object* v_x_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(v_x_208_, v_x_209_);
lean_dec_ref(v_x_208_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4(lean_object* v_t_211_, lean_object* v_init_212_){
_start:
{
lean_object* v_root_213_; lean_object* v_tail_214_; lean_object* v___x_215_; lean_object* v___x_216_; uint8_t v___x_217_; 
v_root_213_ = lean_ctor_get(v_t_211_, 0);
v_tail_214_ = lean_ctor_get(v_t_211_, 1);
v___x_215_ = lean_array_get_size(v_tail_214_);
v___x_216_ = lean_unsigned_to_nat(0u);
v___x_217_ = lean_nat_dec_lt(v___x_216_, v___x_215_);
if (v___x_217_ == 0)
{
lean_object* v___x_218_; 
v___x_218_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(v_root_213_, v_init_212_);
return v___x_218_;
}
else
{
size_t v___x_219_; size_t v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_219_ = lean_usize_of_nat(v___x_215_);
v___x_220_ = ((size_t)0ULL);
v___x_221_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__7(v_tail_214_, v___x_219_, v___x_220_, v_init_212_);
v___x_222_ = lp_batteries___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldrMAux___at___00Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4_spec__6(v_root_213_, v___x_221_);
return v___x_222_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4___boxed(lean_object* v_t_223_, lean_object* v_init_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4(v_t_223_, v_init_224_);
lean_dec_ref(v_t_223_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2(lean_object* v_lctx_226_, lean_object* v_init_227_){
_start:
{
lean_object* v_decls_228_; lean_object* v___x_229_; 
v_decls_228_ = lean_ctor_get(v_lctx_226_, 1);
v___x_229_ = lp_batteries_Lean_PersistentArray_foldrM___at___00Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2_spec__4(v_decls_228_, v_init_227_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2___boxed(lean_object* v_lctx_230_, lean_object* v_init_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2(v_lctx_230_, v_init_231_);
lean_dec_ref(v_lctx_230_);
return v_res_232_;
}
}
static lean_object* _init_lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0(void){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_233_ = lean_box(0);
v___x_234_ = lean_unsigned_to_nat(16u);
v___x_235_ = lean_mk_array(v___x_234_, v___x_233_);
return v___x_235_;
}
}
static lean_object* _init_lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_236_ = lean_obj_once(&lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0, &lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0_once, _init_lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__0);
v___x_237_ = lean_unsigned_to_nat(0u);
v___x_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
lean_ctor_set(v___x_238_, 1, v___x_236_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_LocalContext_inaccessibleFVars(lean_object* v_lctx_239_){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v_fst_245_; lean_object* v___x_246_; 
lean_inc_ref(v_lctx_239_);
v___x_240_ = lean_local_ctx_num_indices(v_lctx_239_);
v___x_241_ = lean_mk_empty_array_with_capacity(v___x_240_);
lean_dec(v___x_240_);
v___x_242_ = lean_obj_once(&lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1, &lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1_once, _init_lp_batteries_Lean_LocalContext_inaccessibleFVars___closed__1);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
v___x_244_ = lp_batteries_Lean_LocalContext_foldrM___at___00Lean_LocalContext_inaccessibleFVars_spec__2(v_lctx_239_, v___x_243_);
lean_dec_ref(v_lctx_239_);
v_fst_245_ = lean_ctor_get(v___x_244_, 0);
lean_inc(v_fst_245_);
lean_dec_ref(v___x_244_);
v___x_246_ = l_Array_reverse___redArg(v_fst_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0(lean_object* v_00_u03b2_247_, lean_object* v_m_248_, lean_object* v_a_249_, lean_object* v_b_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0___redArg(v_m_248_, v_a_249_, v_b_250_);
return v___x_251_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1(lean_object* v_00_u03b2_252_, lean_object* v_m_253_, lean_object* v_a_254_){
_start:
{
uint8_t v___x_255_; 
v___x_255_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___redArg(v_m_253_, v_a_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1___boxed(lean_object* v_00_u03b2_256_, lean_object* v_m_257_, lean_object* v_a_258_){
_start:
{
uint8_t v_res_259_; lean_object* v_r_260_; 
v_res_259_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_LocalContext_inaccessibleFVars_spec__1(v_00_u03b2_256_, v_m_257_, v_a_258_);
lean_dec(v_a_258_);
lean_dec_ref(v_m_257_);
v_r_260_ = lean_box(v_res_259_);
return v_r_260_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0(lean_object* v_00_u03b2_261_, lean_object* v_a_262_, lean_object* v_x_263_){
_start:
{
uint8_t v___x_264_; 
v___x_264_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___redArg(v_a_262_, v_x_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0___boxed(lean_object* v_00_u03b2_265_, lean_object* v_a_266_, lean_object* v_x_267_){
_start:
{
uint8_t v_res_268_; lean_object* v_r_269_; 
v_res_268_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__0(v_00_u03b2_265_, v_a_266_, v_x_267_);
lean_dec(v_x_267_);
lean_dec(v_a_266_);
v_r_269_ = lean_box(v_res_268_);
return v_r_269_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1(lean_object* v_00_u03b2_270_, lean_object* v_data_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1___redArg(v_data_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_273_, lean_object* v_i_274_, lean_object* v_source_275_, lean_object* v_target_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2___redArg(v_i_274_, v_source_275_, v_target_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_278_, lean_object* v_x_279_, lean_object* v_x_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_LocalContext_inaccessibleFVars_spec__0_spec__1_spec__2_spec__5___redArg(v_x_279_, v_x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars___redArg___lam__0(lean_object* v_toPure_282_, lean_object* v_____do__lift_283_){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_284_ = lp_batteries_Lean_LocalContext_inaccessibleFVars(v_____do__lift_283_);
v___x_285_ = lean_apply_2(v_toPure_282_, lean_box(0), v___x_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars___redArg(lean_object* v_inst_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v_toApplicative_288_; lean_object* v_toBind_289_; lean_object* v_toPure_290_; lean_object* v___f_291_; lean_object* v___x_292_; 
v_toApplicative_288_ = lean_ctor_get(v_inst_286_, 0);
lean_inc_ref(v_toApplicative_288_);
v_toBind_289_ = lean_ctor_get(v_inst_286_, 1);
lean_inc(v_toBind_289_);
lean_dec_ref(v_inst_286_);
v_toPure_290_ = lean_ctor_get(v_toApplicative_288_, 1);
lean_inc(v_toPure_290_);
lean_dec_ref(v_toApplicative_288_);
v___f_291_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_getInaccessibleFVars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_291_, 0, v_toPure_290_);
v___x_292_ = lean_apply_4(v_toBind_289_, lean_box(0), lean_box(0), v_inst_287_, v___f_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getInaccessibleFVars(lean_object* v_m_293_, lean_object* v_inst_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_batteries_Lean_Meta_getInaccessibleFVars___redArg(v_inst_294_, v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object* v_x_297_, lean_object* v_x_298_, lean_object* v_x_299_, lean_object* v_x_300_){
_start:
{
lean_object* v_ks_301_; lean_object* v_vs_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_326_; 
v_ks_301_ = lean_ctor_get(v_x_297_, 0);
v_vs_302_ = lean_ctor_get(v_x_297_, 1);
v_isSharedCheck_326_ = !lean_is_exclusive(v_x_297_);
if (v_isSharedCheck_326_ == 0)
{
v___x_304_ = v_x_297_;
v_isShared_305_ = v_isSharedCheck_326_;
goto v_resetjp_303_;
}
else
{
lean_inc(v_vs_302_);
lean_inc(v_ks_301_);
lean_dec(v_x_297_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_326_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v___x_306_; uint8_t v___x_307_; 
v___x_306_ = lean_array_get_size(v_ks_301_);
v___x_307_ = lean_nat_dec_lt(v_x_298_, v___x_306_);
if (v___x_307_ == 0)
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_311_; 
lean_dec(v_x_298_);
v___x_308_ = lean_array_push(v_ks_301_, v_x_299_);
v___x_309_ = lean_array_push(v_vs_302_, v_x_300_);
if (v_isShared_305_ == 0)
{
lean_ctor_set(v___x_304_, 1, v___x_309_);
lean_ctor_set(v___x_304_, 0, v___x_308_);
v___x_311_ = v___x_304_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_308_);
lean_ctor_set(v_reuseFailAlloc_312_, 1, v___x_309_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
else
{
lean_object* v_k_x27_313_; uint8_t v___x_314_; 
v_k_x27_313_ = lean_array_fget_borrowed(v_ks_301_, v_x_298_);
v___x_314_ = l_Lean_instBEqMVarId_beq(v_x_299_, v_k_x27_313_);
if (v___x_314_ == 0)
{
lean_object* v___x_316_; 
if (v_isShared_305_ == 0)
{
v___x_316_ = v___x_304_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v_ks_301_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v_vs_302_);
v___x_316_ = v_reuseFailAlloc_320_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = lean_unsigned_to_nat(1u);
v___x_318_ = lean_nat_add(v_x_298_, v___x_317_);
lean_dec(v_x_298_);
v_x_297_ = v___x_316_;
v_x_298_ = v___x_318_;
goto _start;
}
}
else
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_324_; 
v___x_321_ = lean_array_fset(v_ks_301_, v_x_298_, v_x_299_);
v___x_322_ = lean_array_fset(v_vs_302_, v_x_298_, v_x_300_);
lean_dec(v_x_298_);
if (v_isShared_305_ == 0)
{
lean_ctor_set(v___x_304_, 1, v___x_322_);
lean_ctor_set(v___x_304_, 0, v___x_321_);
v___x_324_ = v___x_304_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_321_);
lean_ctor_set(v_reuseFailAlloc_325_, 1, v___x_322_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_n_327_, lean_object* v_k_328_, lean_object* v_v_329_){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = lean_unsigned_to_nat(0u);
v___x_331_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_n_327_, v___x_330_, v_k_328_, v_v_329_);
return v___x_331_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(lean_object* v_x_333_, size_t v_x_334_, size_t v_x_335_, lean_object* v_x_336_, lean_object* v_x_337_){
_start:
{
if (lean_obj_tag(v_x_333_) == 0)
{
lean_object* v_es_338_; size_t v___x_339_; size_t v___x_340_; lean_object* v_j_341_; lean_object* v___x_342_; uint8_t v___x_343_; 
v_es_338_ = lean_ctor_get(v_x_333_, 0);
v___x_339_ = ((size_t)31ULL);
v___x_340_ = lean_usize_land(v_x_334_, v___x_339_);
v_j_341_ = lean_usize_to_nat(v___x_340_);
v___x_342_ = lean_array_get_size(v_es_338_);
v___x_343_ = lean_nat_dec_lt(v_j_341_, v___x_342_);
if (v___x_343_ == 0)
{
lean_dec(v_j_341_);
lean_dec(v_x_337_);
lean_dec(v_x_336_);
return v_x_333_;
}
else
{
lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_382_; 
lean_inc_ref(v_es_338_);
v_isSharedCheck_382_ = !lean_is_exclusive(v_x_333_);
if (v_isSharedCheck_382_ == 0)
{
lean_object* v_unused_383_; 
v_unused_383_ = lean_ctor_get(v_x_333_, 0);
lean_dec(v_unused_383_);
v___x_345_ = v_x_333_;
v_isShared_346_ = v_isSharedCheck_382_;
goto v_resetjp_344_;
}
else
{
lean_dec(v_x_333_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_382_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v_v_347_; lean_object* v___x_348_; lean_object* v_xs_x27_349_; lean_object* v___y_351_; 
v_v_347_ = lean_array_fget(v_es_338_, v_j_341_);
v___x_348_ = lean_box(0);
v_xs_x27_349_ = lean_array_fset(v_es_338_, v_j_341_, v___x_348_);
switch(lean_obj_tag(v_v_347_))
{
case 0:
{
lean_object* v_key_356_; lean_object* v_val_357_; lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_367_; 
v_key_356_ = lean_ctor_get(v_v_347_, 0);
v_val_357_ = lean_ctor_get(v_v_347_, 1);
v_isSharedCheck_367_ = !lean_is_exclusive(v_v_347_);
if (v_isSharedCheck_367_ == 0)
{
v___x_359_ = v_v_347_;
v_isShared_360_ = v_isSharedCheck_367_;
goto v_resetjp_358_;
}
else
{
lean_inc(v_val_357_);
lean_inc(v_key_356_);
lean_dec(v_v_347_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_367_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
uint8_t v___x_361_; 
v___x_361_ = l_Lean_instBEqMVarId_beq(v_x_336_, v_key_356_);
if (v___x_361_ == 0)
{
lean_object* v___x_362_; lean_object* v___x_363_; 
lean_del_object(v___x_359_);
v___x_362_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_356_, v_val_357_, v_x_336_, v_x_337_);
v___x_363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
v___y_351_ = v___x_363_;
goto v___jp_350_;
}
else
{
lean_object* v___x_365_; 
lean_dec(v_val_357_);
lean_dec(v_key_356_);
if (v_isShared_360_ == 0)
{
lean_ctor_set(v___x_359_, 1, v_x_337_);
lean_ctor_set(v___x_359_, 0, v_x_336_);
v___x_365_ = v___x_359_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_x_336_);
lean_ctor_set(v_reuseFailAlloc_366_, 1, v_x_337_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
v___y_351_ = v___x_365_;
goto v___jp_350_;
}
}
}
}
case 1:
{
lean_object* v_node_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_380_; 
v_node_368_ = lean_ctor_get(v_v_347_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v_v_347_);
if (v_isSharedCheck_380_ == 0)
{
v___x_370_ = v_v_347_;
v_isShared_371_ = v_isSharedCheck_380_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_node_368_);
lean_dec(v_v_347_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_380_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
size_t v___x_372_; size_t v___x_373_; size_t v___x_374_; size_t v___x_375_; lean_object* v___x_376_; lean_object* v___x_378_; 
v___x_372_ = ((size_t)5ULL);
v___x_373_ = lean_usize_shift_right(v_x_334_, v___x_372_);
v___x_374_ = ((size_t)1ULL);
v___x_375_ = lean_usize_add(v_x_335_, v___x_374_);
v___x_376_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(v_node_368_, v___x_373_, v___x_375_, v_x_336_, v_x_337_);
if (v_isShared_371_ == 0)
{
lean_ctor_set(v___x_370_, 0, v___x_376_);
v___x_378_ = v___x_370_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v___x_376_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
v___y_351_ = v___x_378_;
goto v___jp_350_;
}
}
}
default: 
{
lean_object* v___x_381_; 
v___x_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_381_, 0, v_x_336_);
lean_ctor_set(v___x_381_, 1, v_x_337_);
v___y_351_ = v___x_381_;
goto v___jp_350_;
}
}
v___jp_350_:
{
lean_object* v___x_352_; lean_object* v___x_354_; 
v___x_352_ = lean_array_fset(v_xs_x27_349_, v_j_341_, v___y_351_);
lean_dec(v_j_341_);
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 0, v___x_352_);
v___x_354_ = v___x_345_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_352_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
}
else
{
lean_object* v_ks_384_; lean_object* v_vs_385_; lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_405_; 
v_ks_384_ = lean_ctor_get(v_x_333_, 0);
v_vs_385_ = lean_ctor_get(v_x_333_, 1);
v_isSharedCheck_405_ = !lean_is_exclusive(v_x_333_);
if (v_isSharedCheck_405_ == 0)
{
v___x_387_ = v_x_333_;
v_isShared_388_ = v_isSharedCheck_405_;
goto v_resetjp_386_;
}
else
{
lean_inc(v_vs_385_);
lean_inc(v_ks_384_);
lean_dec(v_x_333_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_405_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
lean_object* v___x_390_; 
if (v_isShared_388_ == 0)
{
v___x_390_ = v___x_387_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v_ks_384_);
lean_ctor_set(v_reuseFailAlloc_404_, 1, v_vs_385_);
v___x_390_ = v_reuseFailAlloc_404_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
lean_object* v_newNode_391_; uint8_t v___y_393_; size_t v___x_399_; uint8_t v___x_400_; 
v_newNode_391_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3___redArg(v___x_390_, v_x_336_, v_x_337_);
v___x_399_ = ((size_t)7ULL);
v___x_400_ = lean_usize_dec_le(v___x_399_, v_x_335_);
if (v___x_400_ == 0)
{
lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_401_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_391_);
v___x_402_ = lean_unsigned_to_nat(4u);
v___x_403_ = lean_nat_dec_lt(v___x_401_, v___x_402_);
lean_dec(v___x_401_);
v___y_393_ = v___x_403_;
goto v___jp_392_;
}
else
{
v___y_393_ = v___x_400_;
goto v___jp_392_;
}
v___jp_392_:
{
if (v___y_393_ == 0)
{
lean_object* v_ks_394_; lean_object* v_vs_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
v_ks_394_ = lean_ctor_get(v_newNode_391_, 0);
lean_inc_ref(v_ks_394_);
v_vs_395_ = lean_ctor_get(v_newNode_391_, 1);
lean_inc_ref(v_vs_395_);
lean_dec_ref(v_newNode_391_);
v___x_396_ = lean_unsigned_to_nat(0u);
v___x_397_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___closed__0);
v___x_398_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg(v_x_335_, v_ks_394_, v_vs_395_, v___x_396_, v___x_397_);
lean_dec_ref(v_vs_395_);
lean_dec_ref(v_ks_394_);
return v___x_398_;
}
else
{
return v_newNode_391_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg(size_t v_depth_406_, lean_object* v_keys_407_, lean_object* v_vals_408_, lean_object* v_i_409_, lean_object* v_entries_410_){
_start:
{
lean_object* v___x_411_; uint8_t v___x_412_; 
v___x_411_ = lean_array_get_size(v_keys_407_);
v___x_412_ = lean_nat_dec_lt(v_i_409_, v___x_411_);
if (v___x_412_ == 0)
{
lean_dec(v_i_409_);
return v_entries_410_;
}
else
{
lean_object* v_k_413_; lean_object* v_v_414_; uint64_t v___x_415_; size_t v_h_416_; size_t v___x_417_; lean_object* v___x_418_; size_t v___x_419_; size_t v___x_420_; size_t v___x_421_; size_t v_h_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v_k_413_ = lean_array_fget_borrowed(v_keys_407_, v_i_409_);
v_v_414_ = lean_array_fget_borrowed(v_vals_408_, v_i_409_);
v___x_415_ = l_Lean_instHashableMVarId_hash(v_k_413_);
v_h_416_ = lean_uint64_to_usize(v___x_415_);
v___x_417_ = ((size_t)5ULL);
v___x_418_ = lean_unsigned_to_nat(1u);
v___x_419_ = ((size_t)1ULL);
v___x_420_ = lean_usize_sub(v_depth_406_, v___x_419_);
v___x_421_ = lean_usize_mul(v___x_417_, v___x_420_);
v_h_422_ = lean_usize_shift_right(v_h_416_, v___x_421_);
v___x_423_ = lean_nat_add(v_i_409_, v___x_418_);
lean_dec(v_i_409_);
lean_inc(v_v_414_);
lean_inc(v_k_413_);
v___x_424_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(v_entries_410_, v_h_422_, v_depth_406_, v_k_413_, v_v_414_);
v_i_409_ = v___x_423_;
v_entries_410_ = v___x_424_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_depth_426_, lean_object* v_keys_427_, lean_object* v_vals_428_, lean_object* v_i_429_, lean_object* v_entries_430_){
_start:
{
size_t v_depth_boxed_431_; lean_object* v_res_432_; 
v_depth_boxed_431_ = lean_unbox_usize(v_depth_426_);
lean_dec(v_depth_426_);
v_res_432_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_boxed_431_, v_keys_427_, v_vals_428_, v_i_429_, v_entries_430_);
lean_dec_ref(v_vals_428_);
lean_dec_ref(v_keys_427_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_433_, lean_object* v_x_434_, lean_object* v_x_435_, lean_object* v_x_436_, lean_object* v_x_437_){
_start:
{
size_t v_x_1639__boxed_438_; size_t v_x_1640__boxed_439_; lean_object* v_res_440_; 
v_x_1639__boxed_438_ = lean_unbox_usize(v_x_434_);
lean_dec(v_x_434_);
v_x_1640__boxed_439_ = lean_unbox_usize(v_x_435_);
lean_dec(v_x_435_);
v_res_440_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(v_x_433_, v_x_1639__boxed_438_, v_x_1640__boxed_439_, v_x_436_, v_x_437_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1___redArg(lean_object* v_x_441_, lean_object* v_x_442_, lean_object* v_x_443_){
_start:
{
uint64_t v___x_444_; size_t v___x_445_; size_t v___x_446_; lean_object* v___x_447_; 
v___x_444_ = l_Lean_instHashableMVarId_hash(v_x_442_);
v___x_445_ = lean_uint64_to_usize(v___x_444_);
v___x_446_ = ((size_t)1ULL);
v___x_447_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(v_x_441_, v___x_445_, v___x_446_, v_x_442_, v_x_443_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg(lean_object* v_mvarId_448_, lean_object* v_val_449_, lean_object* v___y_450_){
_start:
{
lean_object* v___x_452_; lean_object* v_mctx_453_; lean_object* v_cache_454_; lean_object* v_zetaDeltaFVarIds_455_; lean_object* v_postponed_456_; lean_object* v_diag_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_485_; 
v___x_452_ = lean_st_ref_take(v___y_450_);
v_mctx_453_ = lean_ctor_get(v___x_452_, 0);
v_cache_454_ = lean_ctor_get(v___x_452_, 1);
v_zetaDeltaFVarIds_455_ = lean_ctor_get(v___x_452_, 2);
v_postponed_456_ = lean_ctor_get(v___x_452_, 3);
v_diag_457_ = lean_ctor_get(v___x_452_, 4);
v_isSharedCheck_485_ = !lean_is_exclusive(v___x_452_);
if (v_isSharedCheck_485_ == 0)
{
v___x_459_ = v___x_452_;
v_isShared_460_ = v_isSharedCheck_485_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_diag_457_);
lean_inc(v_postponed_456_);
lean_inc(v_zetaDeltaFVarIds_455_);
lean_inc(v_cache_454_);
lean_inc(v_mctx_453_);
lean_dec(v___x_452_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_485_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v_depth_461_; lean_object* v_levelAssignDepth_462_; lean_object* v_lmvarCounter_463_; lean_object* v_mvarCounter_464_; lean_object* v_lDecls_465_; lean_object* v_decls_466_; lean_object* v_userNames_467_; lean_object* v_lAssignment_468_; lean_object* v_eAssignment_469_; lean_object* v_dAssignment_470_; lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_484_; 
v_depth_461_ = lean_ctor_get(v_mctx_453_, 0);
v_levelAssignDepth_462_ = lean_ctor_get(v_mctx_453_, 1);
v_lmvarCounter_463_ = lean_ctor_get(v_mctx_453_, 2);
v_mvarCounter_464_ = lean_ctor_get(v_mctx_453_, 3);
v_lDecls_465_ = lean_ctor_get(v_mctx_453_, 4);
v_decls_466_ = lean_ctor_get(v_mctx_453_, 5);
v_userNames_467_ = lean_ctor_get(v_mctx_453_, 6);
v_lAssignment_468_ = lean_ctor_get(v_mctx_453_, 7);
v_eAssignment_469_ = lean_ctor_get(v_mctx_453_, 8);
v_dAssignment_470_ = lean_ctor_get(v_mctx_453_, 9);
v_isSharedCheck_484_ = !lean_is_exclusive(v_mctx_453_);
if (v_isSharedCheck_484_ == 0)
{
v___x_472_ = v_mctx_453_;
v_isShared_473_ = v_isSharedCheck_484_;
goto v_resetjp_471_;
}
else
{
lean_inc(v_dAssignment_470_);
lean_inc(v_eAssignment_469_);
lean_inc(v_lAssignment_468_);
lean_inc(v_userNames_467_);
lean_inc(v_decls_466_);
lean_inc(v_lDecls_465_);
lean_inc(v_mvarCounter_464_);
lean_inc(v_lmvarCounter_463_);
lean_inc(v_levelAssignDepth_462_);
lean_inc(v_depth_461_);
lean_dec(v_mctx_453_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_484_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
lean_object* v___x_474_; lean_object* v___x_476_; 
v___x_474_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1___redArg(v_eAssignment_469_, v_mvarId_448_, v_val_449_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 8, v___x_474_);
v___x_476_ = v___x_472_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v_depth_461_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v_levelAssignDepth_462_);
lean_ctor_set(v_reuseFailAlloc_483_, 2, v_lmvarCounter_463_);
lean_ctor_set(v_reuseFailAlloc_483_, 3, v_mvarCounter_464_);
lean_ctor_set(v_reuseFailAlloc_483_, 4, v_lDecls_465_);
lean_ctor_set(v_reuseFailAlloc_483_, 5, v_decls_466_);
lean_ctor_set(v_reuseFailAlloc_483_, 6, v_userNames_467_);
lean_ctor_set(v_reuseFailAlloc_483_, 7, v_lAssignment_468_);
lean_ctor_set(v_reuseFailAlloc_483_, 8, v___x_474_);
lean_ctor_set(v_reuseFailAlloc_483_, 9, v_dAssignment_470_);
v___x_476_ = v_reuseFailAlloc_483_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
lean_object* v___x_478_; 
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 0, v___x_476_);
v___x_478_ = v___x_459_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v___x_476_);
lean_ctor_set(v_reuseFailAlloc_482_, 1, v_cache_454_);
lean_ctor_set(v_reuseFailAlloc_482_, 2, v_zetaDeltaFVarIds_455_);
lean_ctor_set(v_reuseFailAlloc_482_, 3, v_postponed_456_);
lean_ctor_set(v_reuseFailAlloc_482_, 4, v_diag_457_);
v___x_478_ = v_reuseFailAlloc_482_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_479_ = lean_st_ref_set(v___y_450_, v___x_478_);
v___x_480_ = lean_box(0);
v___x_481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_481_, 0, v___x_480_);
return v___x_481_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg___boxed(lean_object* v_mvarId_486_, lean_object* v_val_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg(v_mvarId_486_, v_val_487_, v___y_488_);
lean_dec(v___y_488_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg(lean_object* v_as_491_, size_t v_sz_492_, size_t v_i_493_, lean_object* v_b_494_){
_start:
{
uint8_t v___x_496_; 
v___x_496_ = lean_usize_dec_lt(v_i_493_, v_sz_492_);
if (v___x_496_ == 0)
{
lean_object* v___x_497_; 
v___x_497_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_497_, 0, v_b_494_);
return v___x_497_;
}
else
{
lean_object* v_fst_498_; lean_object* v_snd_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_515_; 
v_fst_498_ = lean_ctor_get(v_b_494_, 0);
v_snd_499_ = lean_ctor_get(v_b_494_, 1);
v_isSharedCheck_515_ = !lean_is_exclusive(v_b_494_);
if (v_isSharedCheck_515_ == 0)
{
v___x_501_ = v_b_494_;
v_isShared_502_ = v_isSharedCheck_515_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_snd_499_);
lean_inc(v_fst_498_);
lean_dec(v_b_494_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_515_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v_a_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_510_; 
v_a_503_ = lean_array_uget_borrowed(v_as_491_, v_i_493_);
v___x_504_ = l_Lean_LocalDecl_userName(v_a_503_);
v___x_505_ = l_Lean_LocalContext_getUnusedName(v_fst_498_, v___x_504_);
lean_dec(v___x_504_);
v___x_506_ = l_Lean_LocalDecl_fvarId(v_a_503_);
lean_inc(v___x_506_);
v___x_507_ = l_Lean_LocalContext_setUserName(v_fst_498_, v___x_506_, v___x_505_);
v___x_508_ = lean_array_push(v_snd_499_, v___x_506_);
if (v_isShared_502_ == 0)
{
lean_ctor_set(v___x_501_, 1, v___x_508_);
lean_ctor_set(v___x_501_, 0, v___x_507_);
v___x_510_ = v___x_501_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_507_);
lean_ctor_set(v_reuseFailAlloc_514_, 1, v___x_508_);
v___x_510_ = v_reuseFailAlloc_514_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
size_t v___x_511_; size_t v___x_512_; 
v___x_511_ = ((size_t)1ULL);
v___x_512_ = lean_usize_add(v_i_493_, v___x_511_);
v_i_493_ = v___x_512_;
v_b_494_ = v___x_510_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg___boxed(lean_object* v_as_516_, lean_object* v_sz_517_, lean_object* v_i_518_, lean_object* v_b_519_, lean_object* v___y_520_){
_start:
{
size_t v_sz_boxed_521_; size_t v_i_boxed_522_; lean_object* v_res_523_; 
v_sz_boxed_521_ = lean_unbox_usize(v_sz_517_);
lean_dec(v_sz_517_);
v_i_boxed_522_ = lean_unbox_usize(v_i_518_);
lean_dec(v_i_518_);
v_res_523_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg(v_as_516_, v_sz_boxed_521_, v_i_boxed_522_, v_b_519_);
lean_dec_ref(v_as_516_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars(lean_object* v_mvarId_526_, lean_object* v_a_527_, lean_object* v_a_528_, lean_object* v_a_529_, lean_object* v_a_530_){
_start:
{
lean_object* v___x_532_; 
lean_inc(v_mvarId_526_);
v___x_532_ = l_Lean_MVarId_getDecl(v_mvarId_526_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_596_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_596_ == 0)
{
v___x_535_ = v___x_532_;
v_isShared_536_ = v_isSharedCheck_596_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_532_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_596_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v_lctx_537_; lean_object* v_type_538_; lean_object* v_localInstances_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; uint8_t v___x_543_; 
v_lctx_537_ = lean_ctor_get(v_a_533_, 1);
lean_inc_ref_n(v_lctx_537_, 2);
v_type_538_ = lean_ctor_get(v_a_533_, 2);
lean_inc_ref(v_type_538_);
v_localInstances_539_ = lean_ctor_get(v_a_533_, 4);
lean_inc_ref(v_localInstances_539_);
lean_dec(v_a_533_);
v___x_540_ = lp_batteries_Lean_LocalContext_inaccessibleFVars(v_lctx_537_);
v___x_541_ = lean_array_get_size(v___x_540_);
v___x_542_ = lean_unsigned_to_nat(0u);
v___x_543_ = lean_nat_dec_eq(v___x_541_, v___x_542_);
if (v___x_543_ == 0)
{
lean_object* v_decls_544_; lean_object* v_size_545_; lean_object* v___x_546_; lean_object* v___x_547_; size_t v_sz_548_; size_t v___x_549_; lean_object* v___x_550_; 
lean_del_object(v___x_535_);
v_decls_544_ = lean_ctor_get(v_lctx_537_, 1);
v_size_545_ = lean_ctor_get(v_decls_544_, 2);
v___x_546_ = lean_mk_empty_array_with_capacity(v_size_545_);
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v_lctx_537_);
lean_ctor_set(v___x_547_, 1, v___x_546_);
v_sz_548_ = lean_array_size(v___x_540_);
v___x_549_ = ((size_t)0ULL);
v___x_550_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg(v___x_540_, v_sz_548_, v___x_549_, v___x_547_);
lean_dec_ref(v___x_540_);
if (lean_obj_tag(v___x_550_) == 0)
{
lean_object* v_a_551_; lean_object* v_fst_552_; lean_object* v_snd_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_582_; 
v_a_551_ = lean_ctor_get(v___x_550_, 0);
lean_inc(v_a_551_);
lean_dec_ref_known(v___x_550_, 1);
v_fst_552_ = lean_ctor_get(v_a_551_, 0);
v_snd_553_ = lean_ctor_get(v_a_551_, 1);
v_isSharedCheck_582_ = !lean_is_exclusive(v_a_551_);
if (v_isSharedCheck_582_ == 0)
{
v___x_555_ = v_a_551_;
v_isShared_556_ = v_isSharedCheck_582_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_snd_553_);
lean_inc(v_fst_552_);
lean_dec(v_a_551_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_582_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
uint8_t v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_557_ = 0;
v___x_558_ = lean_box(0);
v___x_559_ = l_Lean_Meta_mkFreshExprMVarAt(v_fst_552_, v_localInstances_539_, v_type_538_, v___x_557_, v___x_558_, v___x_542_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_559_) == 0)
{
lean_object* v_a_560_; lean_object* v___x_561_; lean_object* v___x_563_; uint8_t v_isShared_564_; uint8_t v_isSharedCheck_572_; 
v_a_560_ = lean_ctor_get(v___x_559_, 0);
lean_inc_n(v_a_560_, 2);
lean_dec_ref_known(v___x_559_, 1);
v___x_561_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg(v_mvarId_526_, v_a_560_, v_a_528_);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_561_);
if (v_isSharedCheck_572_ == 0)
{
lean_object* v_unused_573_; 
v_unused_573_ = lean_ctor_get(v___x_561_, 0);
lean_dec(v_unused_573_);
v___x_563_ = v___x_561_;
v_isShared_564_ = v_isSharedCheck_572_;
goto v_resetjp_562_;
}
else
{
lean_dec(v___x_561_);
v___x_563_ = lean_box(0);
v_isShared_564_ = v_isSharedCheck_572_;
goto v_resetjp_562_;
}
v_resetjp_562_:
{
lean_object* v___x_565_; lean_object* v___x_567_; 
v___x_565_ = l_Lean_Expr_mvarId_x21(v_a_560_);
lean_dec(v_a_560_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_565_);
v___x_567_ = v___x_555_;
goto v_reusejp_566_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v___x_565_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v_snd_553_);
v___x_567_ = v_reuseFailAlloc_571_;
goto v_reusejp_566_;
}
v_reusejp_566_:
{
lean_object* v___x_569_; 
if (v_isShared_564_ == 0)
{
lean_ctor_set(v___x_563_, 0, v___x_567_);
v___x_569_ = v___x_563_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_567_);
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
else
{
lean_object* v_a_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_581_; 
lean_del_object(v___x_555_);
lean_dec(v_snd_553_);
lean_dec(v_mvarId_526_);
v_a_574_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_581_ == 0)
{
v___x_576_ = v___x_559_;
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_a_574_);
lean_dec(v___x_559_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v___x_579_; 
if (v_isShared_577_ == 0)
{
v___x_579_ = v___x_576_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v_a_574_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
}
}
else
{
lean_object* v_a_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_590_; 
lean_dec_ref(v_localInstances_539_);
lean_dec_ref(v_type_538_);
lean_dec(v_mvarId_526_);
v_a_583_ = lean_ctor_get(v___x_550_, 0);
v_isSharedCheck_590_ = !lean_is_exclusive(v___x_550_);
if (v_isSharedCheck_590_ == 0)
{
v___x_585_ = v___x_550_;
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_a_583_);
lean_dec(v___x_550_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_588_; 
if (v_isShared_586_ == 0)
{
v___x_588_ = v___x_585_;
goto v_reusejp_587_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v_a_583_);
v___x_588_ = v_reuseFailAlloc_589_;
goto v_reusejp_587_;
}
v_reusejp_587_:
{
return v___x_588_;
}
}
}
}
else
{
lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_594_; 
lean_dec_ref(v___x_540_);
lean_dec_ref(v_localInstances_539_);
lean_dec_ref(v_type_538_);
lean_dec_ref(v_lctx_537_);
v___x_591_ = ((lean_object*)(lp_batteries_Lean_MVarId_renameInaccessibleFVars___closed__0));
v___x_592_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_592_, 0, v_mvarId_526_);
lean_ctor_set(v___x_592_, 1, v___x_591_);
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 0, v___x_592_);
v___x_594_ = v___x_535_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v___x_592_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
else
{
lean_object* v_a_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_604_; 
lean_dec(v_mvarId_526_);
v_a_597_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_604_ == 0)
{
v___x_599_ = v___x_532_;
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_a_597_);
lean_dec(v___x_532_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_602_; 
if (v_isShared_600_ == 0)
{
v___x_602_ = v___x_599_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v_a_597_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars___boxed(lean_object* v_mvarId_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_, lean_object* v_a_609_, lean_object* v_a_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_batteries_Lean_MVarId_renameInaccessibleFVars(v_mvarId_605_, v_a_606_, v_a_607_, v_a_608_, v_a_609_);
lean_dec(v_a_609_);
lean_dec_ref(v_a_608_);
lean_dec(v_a_607_);
lean_dec_ref(v_a_606_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0(lean_object* v_as_612_, size_t v_sz_613_, size_t v_i_614_, lean_object* v_b_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v___x_621_; 
v___x_621_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___redArg(v_as_612_, v_sz_613_, v_i_614_, v_b_615_);
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0___boxed(lean_object* v_as_622_, lean_object* v_sz_623_, lean_object* v_i_624_, lean_object* v_b_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
size_t v_sz_boxed_631_; size_t v_i_boxed_632_; lean_object* v_res_633_; 
v_sz_boxed_631_ = lean_unbox_usize(v_sz_623_);
lean_dec(v_sz_623_);
v_i_boxed_632_ = lean_unbox_usize(v_i_624_);
lean_dec(v_i_624_);
v_res_633_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MVarId_renameInaccessibleFVars_spec__0(v_as_622_, v_sz_boxed_631_, v_i_boxed_632_, v_b_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec_ref(v_as_622_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1(lean_object* v_mvarId_634_, lean_object* v_val_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___redArg(v_mvarId_634_, v_val_635_, v___y_637_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1___boxed(lean_object* v_mvarId_642_, lean_object* v_val_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1(v_mvarId_642_, v_val_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1(lean_object* v_00_u03b2_650_, lean_object* v_x_651_, lean_object* v_x_652_, lean_object* v_x_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1___redArg(v_x_651_, v_x_652_, v_x_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_655_, lean_object* v_x_656_, size_t v_x_657_, size_t v_x_658_, lean_object* v_x_659_, lean_object* v_x_660_){
_start:
{
lean_object* v___x_661_; 
v___x_661_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___redArg(v_x_656_, v_x_657_, v_x_658_, v_x_659_, v_x_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_662_, lean_object* v_x_663_, lean_object* v_x_664_, lean_object* v_x_665_, lean_object* v_x_666_, lean_object* v_x_667_){
_start:
{
size_t v_x_2073__boxed_668_; size_t v_x_2074__boxed_669_; lean_object* v_res_670_; 
v_x_2073__boxed_668_ = lean_unbox_usize(v_x_664_);
lean_dec(v_x_664_);
v_x_2074__boxed_669_ = lean_unbox_usize(v_x_665_);
lean_dec(v_x_665_);
v_res_670_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2(v_00_u03b2_662_, v_x_663_, v_x_2073__boxed_668_, v_x_2074__boxed_669_, v_x_666_, v_x_667_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_671_, lean_object* v_n_672_, lean_object* v_k_673_, lean_object* v_v_674_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3___redArg(v_n_672_, v_k_673_, v_v_674_);
return v___x_675_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_676_, size_t v_depth_677_, lean_object* v_keys_678_, lean_object* v_vals_679_, lean_object* v_heq_680_, lean_object* v_i_681_, lean_object* v_entries_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_677_, v_keys_678_, v_vals_679_, v_i_681_, v_entries_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b2_684_, lean_object* v_depth_685_, lean_object* v_keys_686_, lean_object* v_vals_687_, lean_object* v_heq_688_, lean_object* v_i_689_, lean_object* v_entries_690_){
_start:
{
size_t v_depth_boxed_691_; lean_object* v_res_692_; 
v_depth_boxed_691_ = lean_unbox_usize(v_depth_685_);
lean_dec(v_depth_685_);
v_res_692_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__4(v_00_u03b2_684_, v_depth_boxed_691_, v_keys_686_, v_vals_687_, v_heq_688_, v_i_689_, v_entries_690_);
lean_dec_ref(v_vals_687_);
lean_dec_ref(v_keys_686_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_693_, lean_object* v_x_694_, lean_object* v_x_695_, lean_object* v_x_696_, lean_object* v_x_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_renameInaccessibleFVars_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_x_694_, v_x_695_, v_x_696_, v_x_697_);
return v___x_698_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
}
#ifdef __cplusplus
}
#endif
