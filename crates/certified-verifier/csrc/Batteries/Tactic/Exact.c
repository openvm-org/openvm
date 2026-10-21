// Lean compiler output
// Module: Batteries.Tactic.Exact
// Imports: public import Init public meta import Init public meta import Batteries.Tactic.Alias
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_MVarId_assignIfDefEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "assignIfDefEq"};
static const lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___closed__0 = (const lean_object*)&lp_batteries_Lean_MVarId_assignIfDefEq___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_MVarId_assignIfDefEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_MVarId_assignIfDefEq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 200, 89, 151, 239, 49, 149, 195)}};
static const lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___closed__1 = (const lean_object*)&lp_batteries_Lean_MVarId_assignIfDefEq___closed__1_value;
static const lean_string_object lp_batteries_Lean_MVarId_assignIfDefEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___closed__2 = (const lean_object*)&lp_batteries_Lean_MVarId_assignIfDefEq___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_MVarId_assignIfDefEq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_ks_5_; lean_object* v_vs_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_30_; 
v_ks_5_ = lean_ctor_get(v_x_1_, 0);
v_vs_6_ = lean_ctor_get(v_x_1_, 1);
v_isSharedCheck_30_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_30_ == 0)
{
v___x_8_ = v_x_1_;
v_isShared_9_ = v_isSharedCheck_30_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_vs_6_);
lean_inc(v_ks_5_);
lean_dec(v_x_1_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_30_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_10_; uint8_t v___x_11_; 
v___x_10_ = lean_array_get_size(v_ks_5_);
v___x_11_ = lean_nat_dec_lt(v_x_2_, v___x_10_);
if (v___x_11_ == 0)
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_15_; 
lean_dec(v_x_2_);
v___x_12_ = lean_array_push(v_ks_5_, v_x_3_);
v___x_13_ = lean_array_push(v_vs_6_, v_x_4_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___x_13_);
lean_ctor_set(v___x_8_, 0, v___x_12_);
v___x_15_ = v___x_8_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_12_);
lean_ctor_set(v_reuseFailAlloc_16_, 1, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
else
{
lean_object* v_k_x27_17_; uint8_t v___x_18_; 
v_k_x27_17_ = lean_array_fget_borrowed(v_ks_5_, v_x_2_);
v___x_18_ = l_Lean_instBEqMVarId_beq(v_x_3_, v_k_x27_17_);
if (v___x_18_ == 0)
{
lean_object* v___x_20_; 
if (v_isShared_9_ == 0)
{
v___x_20_ = v___x_8_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_ks_5_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_vs_6_);
v___x_20_ = v_reuseFailAlloc_24_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_unsigned_to_nat(1u);
v___x_22_ = lean_nat_add(v_x_2_, v___x_21_);
lean_dec(v_x_2_);
v_x_1_ = v___x_20_;
v_x_2_ = v___x_22_;
goto _start;
}
}
else
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_28_; 
v___x_25_ = lean_array_fset(v_ks_5_, v_x_2_, v_x_3_);
v___x_26_ = lean_array_fset(v_vs_6_, v_x_2_, v_x_4_);
lean_dec(v_x_2_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___x_26_);
lean_ctor_set(v___x_8_, 0, v___x_25_);
v___x_28_ = v___x_8_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___x_25_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___x_26_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_n_31_, lean_object* v_k_32_, lean_object* v_v_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(v_n_31_, v___x_34_, v_k_32_, v_v_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(lean_object* v_x_37_, size_t v_x_38_, size_t v_x_39_, lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
if (lean_obj_tag(v_x_37_) == 0)
{
lean_object* v_es_42_; size_t v___x_43_; size_t v___x_44_; lean_object* v_j_45_; lean_object* v___x_46_; uint8_t v___x_47_; 
v_es_42_ = lean_ctor_get(v_x_37_, 0);
v___x_43_ = ((size_t)31ULL);
v___x_44_ = lean_usize_land(v_x_38_, v___x_43_);
v_j_45_ = lean_usize_to_nat(v___x_44_);
v___x_46_ = lean_array_get_size(v_es_42_);
v___x_47_ = lean_nat_dec_lt(v_j_45_, v___x_46_);
if (v___x_47_ == 0)
{
lean_dec(v_j_45_);
lean_dec(v_x_41_);
lean_dec(v_x_40_);
return v_x_37_;
}
else
{
lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_86_; 
lean_inc_ref(v_es_42_);
v_isSharedCheck_86_ = !lean_is_exclusive(v_x_37_);
if (v_isSharedCheck_86_ == 0)
{
lean_object* v_unused_87_; 
v_unused_87_ = lean_ctor_get(v_x_37_, 0);
lean_dec(v_unused_87_);
v___x_49_ = v_x_37_;
v_isShared_50_ = v_isSharedCheck_86_;
goto v_resetjp_48_;
}
else
{
lean_dec(v_x_37_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_86_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v_v_51_; lean_object* v___x_52_; lean_object* v_xs_x27_53_; lean_object* v___y_55_; 
v_v_51_ = lean_array_fget(v_es_42_, v_j_45_);
v___x_52_ = lean_box(0);
v_xs_x27_53_ = lean_array_fset(v_es_42_, v_j_45_, v___x_52_);
switch(lean_obj_tag(v_v_51_))
{
case 0:
{
lean_object* v_key_60_; lean_object* v_val_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_71_; 
v_key_60_ = lean_ctor_get(v_v_51_, 0);
v_val_61_ = lean_ctor_get(v_v_51_, 1);
v_isSharedCheck_71_ = !lean_is_exclusive(v_v_51_);
if (v_isSharedCheck_71_ == 0)
{
v___x_63_ = v_v_51_;
v_isShared_64_ = v_isSharedCheck_71_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_val_61_);
lean_inc(v_key_60_);
lean_dec(v_v_51_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_71_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
uint8_t v___x_65_; 
v___x_65_ = l_Lean_instBEqMVarId_beq(v_x_40_, v_key_60_);
if (v___x_65_ == 0)
{
lean_object* v___x_66_; lean_object* v___x_67_; 
lean_del_object(v___x_63_);
v___x_66_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_60_, v_val_61_, v_x_40_, v_x_41_);
v___x_67_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
v___y_55_ = v___x_67_;
goto v___jp_54_;
}
else
{
lean_object* v___x_69_; 
lean_dec(v_val_61_);
lean_dec(v_key_60_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 1, v_x_41_);
lean_ctor_set(v___x_63_, 0, v_x_40_);
v___x_69_ = v___x_63_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v_x_40_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_x_41_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
v___y_55_ = v___x_69_;
goto v___jp_54_;
}
}
}
}
case 1:
{
lean_object* v_node_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_84_; 
v_node_72_ = lean_ctor_get(v_v_51_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v_v_51_);
if (v_isSharedCheck_84_ == 0)
{
v___x_74_ = v_v_51_;
v_isShared_75_ = v_isSharedCheck_84_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_node_72_);
lean_dec(v_v_51_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_84_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
size_t v___x_76_; size_t v___x_77_; size_t v___x_78_; size_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_82_; 
v___x_76_ = ((size_t)5ULL);
v___x_77_ = lean_usize_shift_right(v_x_38_, v___x_76_);
v___x_78_ = ((size_t)1ULL);
v___x_79_ = lean_usize_add(v_x_39_, v___x_78_);
v___x_80_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(v_node_72_, v___x_77_, v___x_79_, v_x_40_, v_x_41_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_80_);
v___x_82_ = v___x_74_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_80_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
v___y_55_ = v___x_82_;
goto v___jp_54_;
}
}
}
default: 
{
lean_object* v___x_85_; 
v___x_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_85_, 0, v_x_40_);
lean_ctor_set(v___x_85_, 1, v_x_41_);
v___y_55_ = v___x_85_;
goto v___jp_54_;
}
}
v___jp_54_:
{
lean_object* v___x_56_; lean_object* v___x_58_; 
v___x_56_ = lean_array_fset(v_xs_x27_53_, v_j_45_, v___y_55_);
lean_dec(v_j_45_);
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 0, v___x_56_);
v___x_58_ = v___x_49_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v___x_56_);
v___x_58_ = v_reuseFailAlloc_59_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
return v___x_58_;
}
}
}
}
}
else
{
lean_object* v_ks_88_; lean_object* v_vs_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_109_; 
v_ks_88_ = lean_ctor_get(v_x_37_, 0);
v_vs_89_ = lean_ctor_get(v_x_37_, 1);
v_isSharedCheck_109_ = !lean_is_exclusive(v_x_37_);
if (v_isSharedCheck_109_ == 0)
{
v___x_91_ = v_x_37_;
v_isShared_92_ = v_isSharedCheck_109_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_vs_89_);
lean_inc(v_ks_88_);
lean_dec(v_x_37_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_109_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v_ks_88_);
lean_ctor_set(v_reuseFailAlloc_108_, 1, v_vs_89_);
v___x_94_ = v_reuseFailAlloc_108_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
lean_object* v_newNode_95_; uint8_t v___y_97_; size_t v___x_103_; uint8_t v___x_104_; 
v_newNode_95_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4___redArg(v___x_94_, v_x_40_, v_x_41_);
v___x_103_ = ((size_t)7ULL);
v___x_104_ = lean_usize_dec_le(v___x_103_, v_x_39_);
if (v___x_104_ == 0)
{
lean_object* v___x_105_; lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_105_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_95_);
v___x_106_ = lean_unsigned_to_nat(4u);
v___x_107_ = lean_nat_dec_lt(v___x_105_, v___x_106_);
lean_dec(v___x_105_);
v___y_97_ = v___x_107_;
goto v___jp_96_;
}
else
{
v___y_97_ = v___x_104_;
goto v___jp_96_;
}
v___jp_96_:
{
if (v___y_97_ == 0)
{
lean_object* v_ks_98_; lean_object* v_vs_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_ks_98_ = lean_ctor_get(v_newNode_95_, 0);
lean_inc_ref(v_ks_98_);
v_vs_99_ = lean_ctor_get(v_newNode_95_, 1);
lean_inc_ref(v_vs_99_);
lean_dec_ref(v_newNode_95_);
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___closed__0);
v___x_102_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg(v_x_39_, v_ks_98_, v_vs_99_, v___x_100_, v___x_101_);
lean_dec_ref(v_vs_99_);
lean_dec_ref(v_ks_98_);
return v___x_102_;
}
else
{
return v_newNode_95_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg(size_t v_depth_110_, lean_object* v_keys_111_, lean_object* v_vals_112_, lean_object* v_i_113_, lean_object* v_entries_114_){
_start:
{
lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_115_ = lean_array_get_size(v_keys_111_);
v___x_116_ = lean_nat_dec_lt(v_i_113_, v___x_115_);
if (v___x_116_ == 0)
{
lean_dec(v_i_113_);
return v_entries_114_;
}
else
{
lean_object* v_k_117_; lean_object* v_v_118_; uint64_t v___x_119_; size_t v_h_120_; size_t v___x_121_; lean_object* v___x_122_; size_t v___x_123_; size_t v___x_124_; size_t v___x_125_; size_t v_h_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v_k_117_ = lean_array_fget_borrowed(v_keys_111_, v_i_113_);
v_v_118_ = lean_array_fget_borrowed(v_vals_112_, v_i_113_);
v___x_119_ = l_Lean_instHashableMVarId_hash(v_k_117_);
v_h_120_ = lean_uint64_to_usize(v___x_119_);
v___x_121_ = ((size_t)5ULL);
v___x_122_ = lean_unsigned_to_nat(1u);
v___x_123_ = ((size_t)1ULL);
v___x_124_ = lean_usize_sub(v_depth_110_, v___x_123_);
v___x_125_ = lean_usize_mul(v___x_121_, v___x_124_);
v_h_126_ = lean_usize_shift_right(v_h_120_, v___x_125_);
v___x_127_ = lean_nat_add(v_i_113_, v___x_122_);
lean_dec(v_i_113_);
lean_inc(v_v_118_);
lean_inc(v_k_117_);
v___x_128_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(v_entries_114_, v_h_126_, v_depth_110_, v_k_117_, v_v_118_);
v_i_113_ = v___x_127_;
v_entries_114_ = v___x_128_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_depth_130_, lean_object* v_keys_131_, lean_object* v_vals_132_, lean_object* v_i_133_, lean_object* v_entries_134_){
_start:
{
size_t v_depth_boxed_135_; lean_object* v_res_136_; 
v_depth_boxed_135_ = lean_unbox_usize(v_depth_130_);
lean_dec(v_depth_130_);
v_res_136_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg(v_depth_boxed_135_, v_keys_131_, v_vals_132_, v_i_133_, v_entries_134_);
lean_dec_ref(v_vals_132_);
lean_dec_ref(v_keys_131_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_137_, lean_object* v_x_138_, lean_object* v_x_139_, lean_object* v_x_140_, lean_object* v_x_141_){
_start:
{
size_t v_x_2350__boxed_142_; size_t v_x_2351__boxed_143_; lean_object* v_res_144_; 
v_x_2350__boxed_142_ = lean_unbox_usize(v_x_138_);
lean_dec(v_x_138_);
v_x_2351__boxed_143_ = lean_unbox_usize(v_x_139_);
lean_dec(v_x_139_);
v_res_144_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(v_x_137_, v_x_2350__boxed_142_, v_x_2351__boxed_143_, v_x_140_, v_x_141_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0___redArg(lean_object* v_x_145_, lean_object* v_x_146_, lean_object* v_x_147_){
_start:
{
uint64_t v___x_148_; size_t v___x_149_; size_t v___x_150_; lean_object* v___x_151_; 
v___x_148_ = l_Lean_instHashableMVarId_hash(v_x_146_);
v___x_149_ = lean_uint64_to_usize(v___x_148_);
v___x_150_ = ((size_t)1ULL);
v___x_151_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(v_x_145_, v___x_149_, v___x_150_, v_x_146_, v_x_147_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg(lean_object* v_mvarId_152_, lean_object* v_val_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_156_; lean_object* v_mctx_157_; lean_object* v_cache_158_; lean_object* v_zetaDeltaFVarIds_159_; lean_object* v_postponed_160_; lean_object* v_diag_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_189_; 
v___x_156_ = lean_st_ref_take(v___y_154_);
v_mctx_157_ = lean_ctor_get(v___x_156_, 0);
v_cache_158_ = lean_ctor_get(v___x_156_, 1);
v_zetaDeltaFVarIds_159_ = lean_ctor_get(v___x_156_, 2);
v_postponed_160_ = lean_ctor_get(v___x_156_, 3);
v_diag_161_ = lean_ctor_get(v___x_156_, 4);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_189_ == 0)
{
v___x_163_ = v___x_156_;
v_isShared_164_ = v_isSharedCheck_189_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_diag_161_);
lean_inc(v_postponed_160_);
lean_inc(v_zetaDeltaFVarIds_159_);
lean_inc(v_cache_158_);
lean_inc(v_mctx_157_);
lean_dec(v___x_156_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_189_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v_depth_165_; lean_object* v_levelAssignDepth_166_; lean_object* v_lmvarCounter_167_; lean_object* v_mvarCounter_168_; lean_object* v_lDecls_169_; lean_object* v_decls_170_; lean_object* v_userNames_171_; lean_object* v_lAssignment_172_; lean_object* v_eAssignment_173_; lean_object* v_dAssignment_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_188_; 
v_depth_165_ = lean_ctor_get(v_mctx_157_, 0);
v_levelAssignDepth_166_ = lean_ctor_get(v_mctx_157_, 1);
v_lmvarCounter_167_ = lean_ctor_get(v_mctx_157_, 2);
v_mvarCounter_168_ = lean_ctor_get(v_mctx_157_, 3);
v_lDecls_169_ = lean_ctor_get(v_mctx_157_, 4);
v_decls_170_ = lean_ctor_get(v_mctx_157_, 5);
v_userNames_171_ = lean_ctor_get(v_mctx_157_, 6);
v_lAssignment_172_ = lean_ctor_get(v_mctx_157_, 7);
v_eAssignment_173_ = lean_ctor_get(v_mctx_157_, 8);
v_dAssignment_174_ = lean_ctor_get(v_mctx_157_, 9);
v_isSharedCheck_188_ = !lean_is_exclusive(v_mctx_157_);
if (v_isSharedCheck_188_ == 0)
{
v___x_176_ = v_mctx_157_;
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_dAssignment_174_);
lean_inc(v_eAssignment_173_);
lean_inc(v_lAssignment_172_);
lean_inc(v_userNames_171_);
lean_inc(v_decls_170_);
lean_inc(v_lDecls_169_);
lean_inc(v_mvarCounter_168_);
lean_inc(v_lmvarCounter_167_);
lean_inc(v_levelAssignDepth_166_);
lean_inc(v_depth_165_);
lean_dec(v_mctx_157_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_178_; lean_object* v___x_180_; 
v___x_178_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0___redArg(v_eAssignment_173_, v_mvarId_152_, v_val_153_);
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 8, v___x_178_);
v___x_180_ = v___x_176_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_depth_165_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_levelAssignDepth_166_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_lmvarCounter_167_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_mvarCounter_168_);
lean_ctor_set(v_reuseFailAlloc_187_, 4, v_lDecls_169_);
lean_ctor_set(v_reuseFailAlloc_187_, 5, v_decls_170_);
lean_ctor_set(v_reuseFailAlloc_187_, 6, v_userNames_171_);
lean_ctor_set(v_reuseFailAlloc_187_, 7, v_lAssignment_172_);
lean_ctor_set(v_reuseFailAlloc_187_, 8, v___x_178_);
lean_ctor_set(v_reuseFailAlloc_187_, 9, v_dAssignment_174_);
v___x_180_ = v_reuseFailAlloc_187_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
lean_object* v___x_182_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v___x_180_);
v___x_182_ = v___x_163_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_cache_158_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_zetaDeltaFVarIds_159_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_postponed_160_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_diag_161_);
v___x_182_ = v_reuseFailAlloc_186_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_183_ = lean_st_ref_set(v___y_154_, v___x_182_);
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
return v___x_185_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg___boxed(lean_object* v_mvarId_190_, lean_object* v_val_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg(v_mvarId_190_, v_val_191_, v___y_192_);
lean_dec(v___y_192_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2(lean_object* v_msgData_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_){
_start:
{
lean_object* v___x_201_; lean_object* v_env_202_; lean_object* v___x_203_; lean_object* v_mctx_204_; lean_object* v_lctx_205_; lean_object* v_options_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_201_ = lean_st_ref_get(v___y_199_);
v_env_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc_ref(v_env_202_);
lean_dec(v___x_201_);
v___x_203_ = lean_st_ref_get(v___y_197_);
v_mctx_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc_ref(v_mctx_204_);
lean_dec(v___x_203_);
v_lctx_205_ = lean_ctor_get(v___y_196_, 2);
v_options_206_ = lean_ctor_get(v___y_198_, 2);
lean_inc_ref(v_options_206_);
lean_inc_ref(v_lctx_205_);
v___x_207_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_207_, 0, v_env_202_);
lean_ctor_set(v___x_207_, 1, v_mctx_204_);
lean_ctor_set(v___x_207_, 2, v_lctx_205_);
lean_ctor_set(v___x_207_, 3, v_options_206_);
v___x_208_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_msgData_195_);
v___x_209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2___boxed(lean_object* v_msgData_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2(v_msgData_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg(lean_object* v_msg_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v_ref_223_; lean_object* v___x_224_; lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_233_; 
v_ref_223_ = lean_ctor_get(v___y_220_, 5);
v___x_224_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1_spec__2(v_msg_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
v_a_225_ = lean_ctor_get(v___x_224_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_233_ == 0)
{
v___x_227_ = v___x_224_;
v_isShared_228_ = v_isSharedCheck_233_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_224_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_233_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_231_; 
lean_inc(v_ref_223_);
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v_ref_223_);
lean_ctor_set(v___x_229_, 1, v_a_225_);
if (v_isShared_228_ == 0)
{
lean_ctor_set_tag(v___x_227_, 1);
lean_ctor_set(v___x_227_, 0, v___x_229_);
v___x_231_ = v___x_227_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v___x_229_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg___boxed(lean_object* v_msg_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg(v_msg_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
return v_res_240_;
}
}
static lean_object* _init_lp_batteries_Lean_MVarId_assignIfDefEq___closed__3(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = ((lean_object*)(lp_batteries_Lean_MVarId_assignIfDefEq___closed__2));
v___x_246_ = l_Lean_stringToMessageData(v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object* v_g_247_, lean_object* v_e_248_, lean_object* v_a_249_, lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_){
_start:
{
lean_object* v___x_258_; 
lean_inc(v_g_247_);
v___x_258_ = l_Lean_MVarId_getType(v_g_247_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
if (lean_obj_tag(v___x_258_) == 0)
{
lean_object* v_a_259_; lean_object* v___x_260_; 
v_a_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_a_259_);
lean_dec_ref_known(v___x_258_, 1);
lean_inc(v_a_252_);
lean_inc_ref(v_a_251_);
lean_inc(v_a_250_);
lean_inc_ref(v_a_249_);
lean_inc_ref(v_e_248_);
v___x_260_ = lean_infer_type(v_e_248_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v_a_261_; lean_object* v___x_262_; 
v_a_261_ = lean_ctor_get(v___x_260_, 0);
lean_inc(v_a_261_);
lean_dec_ref_known(v___x_260_, 1);
v___x_262_ = l_Lean_Meta_isExprDefEq(v_a_259_, v_a_261_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v_a_263_; uint8_t v___x_264_; 
v_a_263_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v___x_262_, 1);
v___x_264_ = lean_unbox(v_a_263_);
lean_dec(v_a_263_);
if (v___x_264_ == 0)
{
lean_object* v___x_265_; lean_object* v___x_266_; 
lean_dec_ref(v_e_248_);
lean_dec(v_g_247_);
v___x_265_ = lean_obj_once(&lp_batteries_Lean_MVarId_assignIfDefEq___closed__3, &lp_batteries_Lean_MVarId_assignIfDefEq___closed__3_once, _init_lp_batteries_Lean_MVarId_assignIfDefEq___closed__3);
v___x_266_ = lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg(v___x_265_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
return v___x_266_;
}
else
{
goto v___jp_254_;
}
}
else
{
lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_274_; 
lean_dec_ref(v_e_248_);
lean_dec(v_g_247_);
v_a_267_ = lean_ctor_get(v___x_262_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_262_);
if (v_isSharedCheck_274_ == 0)
{
v___x_269_ = v___x_262_;
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_dec(v___x_262_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_272_; 
if (v_isShared_270_ == 0)
{
v___x_272_ = v___x_269_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_a_267_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
else
{
lean_object* v_a_275_; lean_object* v___x_277_; uint8_t v_isShared_278_; uint8_t v_isSharedCheck_282_; 
lean_dec(v_a_259_);
lean_dec_ref(v_e_248_);
lean_dec(v_g_247_);
v_a_275_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_282_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_282_ == 0)
{
v___x_277_ = v___x_260_;
v_isShared_278_ = v_isSharedCheck_282_;
goto v_resetjp_276_;
}
else
{
lean_inc(v_a_275_);
lean_dec(v___x_260_);
v___x_277_ = lean_box(0);
v_isShared_278_ = v_isSharedCheck_282_;
goto v_resetjp_276_;
}
v_resetjp_276_:
{
lean_object* v___x_280_; 
if (v_isShared_278_ == 0)
{
v___x_280_ = v___x_277_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v_a_275_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
}
}
else
{
lean_object* v_a_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_290_; 
lean_dec_ref(v_e_248_);
lean_dec(v_g_247_);
v_a_283_ = lean_ctor_get(v___x_258_, 0);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_258_);
if (v_isSharedCheck_290_ == 0)
{
v___x_285_ = v___x_258_;
v_isShared_286_ = v_isSharedCheck_290_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_a_283_);
lean_dec(v___x_258_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_290_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___x_288_; 
if (v_isShared_286_ == 0)
{
v___x_288_ = v___x_285_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v_a_283_);
v___x_288_ = v_reuseFailAlloc_289_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
return v___x_288_;
}
}
}
v___jp_254_:
{
lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_255_ = ((lean_object*)(lp_batteries_Lean_MVarId_assignIfDefEq___closed__1));
lean_inc(v_g_247_);
v___x_256_ = l_Lean_MVarId_checkNotAssigned(v_g_247_, v___x_255_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
if (lean_obj_tag(v___x_256_) == 0)
{
lean_object* v___x_257_; 
lean_dec_ref_known(v___x_256_, 1);
v___x_257_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg(v_g_247_, v_e_248_, v_a_250_);
return v___x_257_;
}
else
{
lean_dec_ref(v_e_248_);
lean_dec(v_g_247_);
return v___x_256_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assignIfDefEq___boxed(lean_object* v_g_291_, lean_object* v_e_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_batteries_Lean_MVarId_assignIfDefEq(v_g_291_, v_e_292_, v_a_293_, v_a_294_, v_a_295_, v_a_296_);
lean_dec(v_a_296_);
lean_dec_ref(v_a_295_);
lean_dec(v_a_294_);
lean_dec_ref(v_a_293_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0(lean_object* v_mvarId_299_, lean_object* v_val_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___redArg(v_mvarId_299_, v_val_300_, v___y_302_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0___boxed(lean_object* v_mvarId_307_, lean_object* v_val_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0(v_mvarId_307_, v_val_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1(lean_object* v_00_u03b1_315_, lean_object* v_msg_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___redArg(v_msg_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1___boxed(lean_object* v_00_u03b1_323_, lean_object* v_msg_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_batteries_Lean_throwError___at___00Lean_MVarId_assignIfDefEq_spec__1(v_00_u03b1_323_, v_msg_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0(lean_object* v_00_u03b2_331_, lean_object* v_x_332_, lean_object* v_x_333_, lean_object* v_x_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0___redArg(v_x_332_, v_x_333_, v_x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_336_, lean_object* v_x_337_, size_t v_x_338_, size_t v_x_339_, lean_object* v_x_340_, lean_object* v_x_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___redArg(v_x_337_, v_x_338_, v_x_339_, v_x_340_, v_x_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_343_, lean_object* v_x_344_, lean_object* v_x_345_, lean_object* v_x_346_, lean_object* v_x_347_, lean_object* v_x_348_){
_start:
{
size_t v_x_2766__boxed_349_; size_t v_x_2767__boxed_350_; lean_object* v_res_351_; 
v_x_2766__boxed_349_ = lean_unbox_usize(v_x_345_);
lean_dec(v_x_345_);
v_x_2767__boxed_350_ = lean_unbox_usize(v_x_346_);
lean_dec(v_x_346_);
v_res_351_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1(v_00_u03b2_343_, v_x_344_, v_x_2766__boxed_349_, v_x_2767__boxed_350_, v_x_347_, v_x_348_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4(lean_object* v_00_u03b2_352_, lean_object* v_n_353_, lean_object* v_k_354_, lean_object* v_v_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4___redArg(v_n_353_, v_k_354_, v_v_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5(lean_object* v_00_u03b2_357_, size_t v_depth_358_, lean_object* v_keys_359_, lean_object* v_vals_360_, lean_object* v_heq_361_, lean_object* v_i_362_, lean_object* v_entries_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___redArg(v_depth_358_, v_keys_359_, v_vals_360_, v_i_362_, v_entries_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_00_u03b2_365_, lean_object* v_depth_366_, lean_object* v_keys_367_, lean_object* v_vals_368_, lean_object* v_heq_369_, lean_object* v_i_370_, lean_object* v_entries_371_){
_start:
{
size_t v_depth_boxed_372_; lean_object* v_res_373_; 
v_depth_boxed_372_ = lean_unbox_usize(v_depth_366_);
lean_dec(v_depth_366_);
v_res_373_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__5(v_00_u03b2_365_, v_depth_boxed_372_, v_keys_367_, v_vals_368_, v_heq_369_, v_i_370_, v_entries_371_);
lean_dec_ref(v_vals_368_);
lean_dec_ref(v_keys_367_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object* v_00_u03b2_374_, lean_object* v_x_375_, lean_object* v_x_376_, lean_object* v_x_377_, lean_object* v_x_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_assignIfDefEq_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(v_x_375_, v_x_376_, v_x_377_, v_x_378_);
return v___x_379_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Exact(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Exact(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Exact(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Exact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Exact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Exact(builtin);
}
#ifdef __cplusplus
}
#endif
