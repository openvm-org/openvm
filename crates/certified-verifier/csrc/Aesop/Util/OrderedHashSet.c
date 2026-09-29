// Lean compiler output
// Module: Aesop.Util.OrderedHashSet
// Imports: public import Init public meta import Init public import Std.Data.HashSet.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Nat_nextPowerOfTwo(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Array_instForIn_x27InferInstanceMembershipOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instForInOfForIn_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__0 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__0_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__2_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_instForIn_x27InferInstanceMembershipOfMonad___redArg___lam__0, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9_value)} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instForInOfForIn_x27___redArg___lam__1, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__10_value)} };
static const lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__11_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_contains___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_contains(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_contains___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instMembership(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instMembership___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_instDecidableMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instDecidableMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lean_box(0);
v___x_4_ = lean_unsigned_to_nat(16u);
v___x_5_ = lean_mk_array(v___x_4_, v___x_3_);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1, &lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__1);
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2, &lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2);
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__0));
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3, &lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__3);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet_default___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_aesop_Aesop_instInhabitedOrderedHashSet_default(v_00_u03b1_16_, v_inst_17_, v_inst_18_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___redArg(lean_object* v_a_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_aesop_Aesop_instInhabitedOrderedHashSet_default(lean_box(0), v_a_20_, v_a_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___redArg___boxed(lean_object* v_a_23_, lean_object* v_a_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_instInhabitedOrderedHashSet___redArg(v_a_23_, v_a_24_);
lean_dec_ref(v_a_24_);
lean_dec_ref(v_a_23_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet(lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_aesop_Aesop_instInhabitedOrderedHashSet_default(lean_box(0), v_a_27_, v_a_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedOrderedHashSet___boxed(lean_object* v_a_30_, lean_object* v_a_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_aesop_Aesop_instInhabitedOrderedHashSet(v_a_30_, v_a_31_, v_a_32_);
lean_dec_ref(v_a_32_);
lean_dec_ref(v_a_31_);
return v_res_33_;
}
}
static lean_object* _init_lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2, &lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedOrderedHashSet_default___closed__2);
v___x_37_ = ((lean_object*)(lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__0));
v___x_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
lean_ctor_set(v___x_38_, 1, v___x_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_obj_once(&lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1, &lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1_once, _init_lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___boxed(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop_Aesop_OrderedHashSet_instEmptyCollection(v_00_u03b1_43_, v_inst_44_, v_inst_45_);
lean_dec_ref(v_inst_45_);
lean_dec_ref(v_inst_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg(lean_object* v_n_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_48_ = lean_mk_empty_array_with_capacity(v_n_47_);
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_unsigned_to_nat(4u);
v___x_51_ = lean_nat_mul(v_n_47_, v___x_50_);
v___x_52_ = lean_unsigned_to_nat(3u);
v___x_53_ = lean_nat_div(v___x_51_, v___x_52_);
lean_dec(v___x_51_);
v___x_54_ = l_Nat_nextPowerOfTwo(v___x_53_);
lean_dec(v___x_53_);
v___x_55_ = lean_box(0);
v___x_56_ = lean_mk_array(v___x_54_, v___x_55_);
v___x_57_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_49_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
v___x_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_48_);
lean_ctor_set(v___x_58_, 1, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg___boxed(lean_object* v_n_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg(v_n_59_);
lean_dec(v_n_59_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_n_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___redArg(v_n_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity___boxed(lean_object* v_00_u03b1_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_n_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_aesop_Aesop_OrderedHashSet_emptyWithCapacity(v_00_u03b1_66_, v_inst_67_, v_inst_68_, v_n_69_);
lean_dec(v_n_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insert___redArg(lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_x_73_, lean_object* v_s_74_){
_start:
{
lean_object* v_toArray_75_; lean_object* v_toHashSet_76_; uint8_t v___x_77_; 
v_toArray_75_ = lean_ctor_get(v_s_74_, 0);
v_toHashSet_76_ = lean_ctor_get(v_s_74_, 1);
lean_inc(v_x_73_);
lean_inc_ref(v_inst_72_);
lean_inc_ref(v_inst_71_);
v___x_77_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_71_, v_inst_72_, v_toHashSet_76_, v_x_73_);
if (v___x_77_ == 0)
{
lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_87_; 
lean_inc_ref(v_toHashSet_76_);
lean_inc_ref(v_toArray_75_);
v_isSharedCheck_87_ = !lean_is_exclusive(v_s_74_);
if (v_isSharedCheck_87_ == 0)
{
lean_object* v_unused_88_; lean_object* v_unused_89_; 
v_unused_88_ = lean_ctor_get(v_s_74_, 1);
lean_dec(v_unused_88_);
v_unused_89_ = lean_ctor_get(v_s_74_, 0);
lean_dec(v_unused_89_);
v___x_79_ = v_s_74_;
v_isShared_80_ = v_isSharedCheck_87_;
goto v_resetjp_78_;
}
else
{
lean_dec(v_s_74_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_87_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_85_; 
lean_inc(v_x_73_);
v___x_81_ = lean_array_push(v_toArray_75_, v_x_73_);
v___x_82_ = lean_box(0);
v___x_83_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v_inst_71_, v_inst_72_, v_toHashSet_76_, v_x_73_, v___x_82_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 1, v___x_83_);
lean_ctor_set(v___x_79_, 0, v___x_81_);
v___x_85_ = v___x_79_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v___x_83_);
v___x_85_ = v_reuseFailAlloc_86_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
return v___x_85_;
}
}
}
else
{
lean_dec(v_x_73_);
lean_dec_ref(v_inst_72_);
lean_dec_ref(v_inst_71_);
return v_s_74_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insert(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_x_93_, lean_object* v_s_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_aesop_Aesop_OrderedHashSet_insert___redArg(v_inst_91_, v_inst_92_, v_x_93_, v_s_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany___redArg___lam__0(lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_x_98_, lean_object* v_____s_99_){
_start:
{
lean_object* v_result_100_; lean_object* v___x_101_; 
v_result_100_ = lp_aesop_Aesop_OrderedHashSet_insert___redArg(v_inst_96_, v_inst_97_, v_x_98_, v_____s_99_);
v___x_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_101_, 0, v_result_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_xs_105_, lean_object* v_s_106_){
_start:
{
lean_object* v___f_107_; lean_object* v___x_108_; 
v___f_107_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_insertMany___redArg___lam__0), 4, 2);
lean_closure_set(v___f_107_, 0, v_inst_102_);
lean_closure_set(v___f_107_, 1, v_inst_103_);
v___x_108_ = lean_apply_4(v_inst_104_, lean_box(0), v_xs_105_, v_s_106_, v___f_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_insertMany(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_00_u03c1_112_, lean_object* v_inst_113_, lean_object* v_xs_114_, lean_object* v_s_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_aesop_Aesop_OrderedHashSet_insertMany___redArg(v_inst_110_, v_inst_111_, v_inst_113_, v_xs_114_, v_s_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray___redArg(lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_xs_142_){
_start:
{
lean_object* v___f_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___f_143_ = ((lean_object*)(lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__11));
v___x_144_ = lean_obj_once(&lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1, &lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1_once, _init_lp_aesop_Aesop_OrderedHashSet_instEmptyCollection___closed__1);
v___x_145_ = lp_aesop_Aesop_OrderedHashSet_insertMany___redArg(v_inst_140_, v_inst_141_, v___f_143_, v_xs_142_, v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_ofArray(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_xs_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_aesop_Aesop_OrderedHashSet_ofArray___redArg(v_inst_147_, v_inst_148_, v_xs_149_);
return v___x_150_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_contains___redArg(lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_x_153_, lean_object* v_s_154_){
_start:
{
lean_object* v_toHashSet_155_; uint8_t v___x_156_; 
v_toHashSet_155_ = lean_ctor_get(v_s_154_, 1);
v___x_156_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_151_, v_inst_152_, v_toHashSet_155_, v_x_153_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_contains___redArg___boxed(lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_x_159_, lean_object* v_s_160_){
_start:
{
uint8_t v_res_161_; lean_object* v_r_162_; 
v_res_161_ = lp_aesop_Aesop_OrderedHashSet_contains___redArg(v_inst_157_, v_inst_158_, v_x_159_, v_s_160_);
lean_dec_ref(v_s_160_);
v_r_162_ = lean_box(v_res_161_);
return v_r_162_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_contains(lean_object* v_00_u03b1_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_x_166_, lean_object* v_s_167_){
_start:
{
uint8_t v___x_168_; 
v___x_168_ = lp_aesop_Aesop_OrderedHashSet_contains___redArg(v_inst_164_, v_inst_165_, v_x_166_, v_s_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_contains___boxed(lean_object* v_00_u03b1_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_x_172_, lean_object* v_s_173_){
_start:
{
uint8_t v_res_174_; lean_object* v_r_175_; 
v_res_174_ = lp_aesop_Aesop_OrderedHashSet_contains(v_00_u03b1_169_, v_inst_170_, v_inst_171_, v_x_172_, v_s_173_);
lean_dec_ref(v_s_173_);
v_r_175_ = lean_box(v_res_174_);
return v_r_175_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instMembership(lean_object* v_00_u03b1_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_box(0);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instMembership___boxed(lean_object* v_00_u03b1_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_aesop_Aesop_OrderedHashSet_instMembership(v_00_u03b1_180_, v_inst_181_, v_inst_182_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_181_);
return v_res_183_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg(lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_x_186_, lean_object* v_s_187_){
_start:
{
lean_object* v_toHashSet_188_; uint8_t v___x_189_; 
v_toHashSet_188_ = lean_ctor_get(v_s_187_, 1);
v___x_189_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_184_, v_inst_185_, v_toHashSet_188_, v_x_186_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg___boxed(lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_x_192_, lean_object* v_s_193_){
_start:
{
uint8_t v_res_194_; lean_object* v_r_195_; 
v_res_194_ = lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg(v_inst_190_, v_inst_191_, v_x_192_, v_s_193_);
lean_dec_ref(v_s_193_);
v_r_195_ = lean_box(v_res_194_);
return v_r_195_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_OrderedHashSet_instDecidableMem(lean_object* v_00_u03b1_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_x_199_, lean_object* v_s_200_){
_start:
{
uint8_t v___x_201_; 
v___x_201_ = lp_aesop_Aesop_OrderedHashSet_instDecidableMem___redArg(v_inst_197_, v_inst_198_, v_x_199_, v_s_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instDecidableMem___boxed(lean_object* v_00_u03b1_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_x_205_, lean_object* v_s_206_){
_start:
{
uint8_t v_res_207_; lean_object* v_r_208_; 
v_res_207_ = lp_aesop_Aesop_OrderedHashSet_instDecidableMem(v_00_u03b1_202_, v_inst_203_, v_inst_204_, v_x_205_, v_s_206_);
lean_dec_ref(v_s_206_);
v_r_208_ = lean_box(v_res_207_);
return v_r_208_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM___redArg(lean_object* v_inst_209_, lean_object* v_f_210_, lean_object* v_init_211_, lean_object* v_s_212_){
_start:
{
lean_object* v_toArray_213_; lean_object* v___x_214_; lean_object* v___x_215_; uint8_t v___x_216_; 
v_toArray_213_ = lean_ctor_get(v_s_212_, 0);
lean_inc_ref(v_toArray_213_);
lean_dec_ref(v_s_212_);
v___x_214_ = lean_unsigned_to_nat(0u);
v___x_215_ = lean_array_get_size(v_toArray_213_);
v___x_216_ = lean_nat_dec_lt(v___x_214_, v___x_215_);
if (v___x_216_ == 0)
{
lean_object* v_toApplicative_217_; lean_object* v_toPure_218_; lean_object* v___x_219_; 
lean_dec_ref(v_toArray_213_);
lean_dec(v_f_210_);
v_toApplicative_217_ = lean_ctor_get(v_inst_209_, 0);
lean_inc_ref(v_toApplicative_217_);
lean_dec_ref(v_inst_209_);
v_toPure_218_ = lean_ctor_get(v_toApplicative_217_, 1);
lean_inc(v_toPure_218_);
lean_dec_ref(v_toApplicative_217_);
v___x_219_ = lean_apply_2(v_toPure_218_, lean_box(0), v_init_211_);
return v___x_219_;
}
else
{
uint8_t v___x_220_; 
v___x_220_ = lean_nat_dec_le(v___x_215_, v___x_215_);
if (v___x_220_ == 0)
{
if (v___x_216_ == 0)
{
lean_object* v_toApplicative_221_; lean_object* v_toPure_222_; lean_object* v___x_223_; 
lean_dec_ref(v_toArray_213_);
lean_dec(v_f_210_);
v_toApplicative_221_ = lean_ctor_get(v_inst_209_, 0);
lean_inc_ref(v_toApplicative_221_);
lean_dec_ref(v_inst_209_);
v_toPure_222_ = lean_ctor_get(v_toApplicative_221_, 1);
lean_inc(v_toPure_222_);
lean_dec_ref(v_toApplicative_221_);
v___x_223_ = lean_apply_2(v_toPure_222_, lean_box(0), v_init_211_);
return v___x_223_;
}
else
{
size_t v___x_224_; size_t v___x_225_; lean_object* v___x_226_; 
v___x_224_ = ((size_t)0ULL);
v___x_225_ = lean_usize_of_nat(v___x_215_);
v___x_226_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_209_, v_f_210_, v_toArray_213_, v___x_224_, v___x_225_, v_init_211_);
return v___x_226_;
}
}
else
{
size_t v___x_227_; size_t v___x_228_; lean_object* v___x_229_; 
v___x_227_ = ((size_t)0ULL);
v___x_228_ = lean_usize_of_nat(v___x_215_);
v___x_229_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_209_, v_f_210_, v_toArray_213_, v___x_227_, v___x_228_, v_init_211_);
return v___x_229_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM(lean_object* v_00_u03b1_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_m_233_, lean_object* v_00_u03b2_234_, lean_object* v_inst_235_, lean_object* v_f_236_, lean_object* v_init_237_, lean_object* v_s_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_aesop_Aesop_OrderedHashSet_foldlM___redArg(v_inst_235_, v_f_236_, v_init_237_, v_s_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldlM___boxed(lean_object* v_00_u03b1_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_m_243_, lean_object* v_00_u03b2_244_, lean_object* v_inst_245_, lean_object* v_f_246_, lean_object* v_init_247_, lean_object* v_s_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_aesop_Aesop_OrderedHashSet_foldlM(v_00_u03b1_240_, v_inst_241_, v_inst_242_, v_m_243_, v_00_u03b2_244_, v_inst_245_, v_f_246_, v_init_247_, v_s_248_);
lean_dec_ref(v_inst_242_);
lean_dec_ref(v_inst_241_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___redArg___lam__0(lean_object* v_f_250_, lean_object* v_x1_251_, lean_object* v_x2_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lean_apply_2(v_f_250_, v_x1_251_, v_x2_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___redArg(lean_object* v_f_254_, lean_object* v_init_255_, lean_object* v_s_256_){
_start:
{
lean_object* v_toArray_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; uint8_t v___x_261_; 
v_toArray_257_ = lean_ctor_get(v_s_256_, 0);
lean_inc_ref(v_toArray_257_);
lean_dec_ref(v_s_256_);
v___x_258_ = lean_unsigned_to_nat(0u);
v___x_259_ = lean_array_get_size(v_toArray_257_);
v___x_260_ = ((lean_object*)(lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9));
v___x_261_ = lean_nat_dec_lt(v___x_258_, v___x_259_);
if (v___x_261_ == 0)
{
lean_dec_ref(v_toArray_257_);
lean_dec(v_f_254_);
return v_init_255_;
}
else
{
lean_object* v___f_262_; uint8_t v___x_263_; 
v___f_262_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_foldl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_262_, 0, v_f_254_);
v___x_263_ = lean_nat_dec_le(v___x_259_, v___x_259_);
if (v___x_263_ == 0)
{
if (v___x_261_ == 0)
{
lean_dec_ref(v___f_262_);
lean_dec_ref(v_toArray_257_);
return v_init_255_;
}
else
{
size_t v___x_264_; size_t v___x_265_; lean_object* v___x_266_; 
v___x_264_ = ((size_t)0ULL);
v___x_265_ = lean_usize_of_nat(v___x_259_);
v___x_266_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_260_, v___f_262_, v_toArray_257_, v___x_264_, v___x_265_, v_init_255_);
return v___x_266_;
}
}
else
{
size_t v___x_267_; size_t v___x_268_; lean_object* v___x_269_; 
v___x_267_ = ((size_t)0ULL);
v___x_268_ = lean_usize_of_nat(v___x_259_);
v___x_269_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_260_, v___f_262_, v_toArray_257_, v___x_267_, v___x_268_, v_init_255_);
return v___x_269_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl(lean_object* v_00_u03b1_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_00_u03b2_273_, lean_object* v_f_274_, lean_object* v_init_275_, lean_object* v_s_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_aesop_Aesop_OrderedHashSet_foldl___redArg(v_f_274_, v_init_275_, v_s_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldl___boxed(lean_object* v_00_u03b1_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_00_u03b2_281_, lean_object* v_f_282_, lean_object* v_init_283_, lean_object* v_s_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_aesop_Aesop_OrderedHashSet_foldl(v_00_u03b1_278_, v_inst_279_, v_inst_280_, v_00_u03b2_281_, v_f_282_, v_init_283_, v_s_284_);
lean_dec_ref(v_inst_280_);
lean_dec_ref(v_inst_279_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM___redArg(lean_object* v_inst_286_, lean_object* v_f_287_, lean_object* v_init_288_, lean_object* v_s_289_){
_start:
{
lean_object* v_toArray_290_; lean_object* v___x_291_; lean_object* v___x_292_; uint8_t v___x_293_; 
v_toArray_290_ = lean_ctor_get(v_s_289_, 0);
lean_inc_ref(v_toArray_290_);
lean_dec_ref(v_s_289_);
v___x_291_ = lean_array_get_size(v_toArray_290_);
v___x_292_ = lean_unsigned_to_nat(0u);
v___x_293_ = lean_nat_dec_lt(v___x_292_, v___x_291_);
if (v___x_293_ == 0)
{
lean_object* v_toApplicative_294_; lean_object* v_toPure_295_; lean_object* v___x_296_; 
lean_dec_ref(v_toArray_290_);
lean_dec(v_f_287_);
v_toApplicative_294_ = lean_ctor_get(v_inst_286_, 0);
lean_inc_ref(v_toApplicative_294_);
lean_dec_ref(v_inst_286_);
v_toPure_295_ = lean_ctor_get(v_toApplicative_294_, 1);
lean_inc(v_toPure_295_);
lean_dec_ref(v_toApplicative_294_);
v___x_296_ = lean_apply_2(v_toPure_295_, lean_box(0), v_init_288_);
return v___x_296_;
}
else
{
size_t v___x_297_; size_t v___x_298_; lean_object* v___x_299_; 
v___x_297_ = lean_usize_of_nat(v___x_291_);
v___x_298_ = ((size_t)0ULL);
v___x_299_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_286_, v_f_287_, v_toArray_290_, v___x_297_, v___x_298_, v_init_288_);
return v___x_299_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM(lean_object* v_00_u03b1_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_m_303_, lean_object* v_00_u03b2_304_, lean_object* v_inst_305_, lean_object* v_f_306_, lean_object* v_init_307_, lean_object* v_s_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lp_aesop_Aesop_OrderedHashSet_foldrM___redArg(v_inst_305_, v_f_306_, v_init_307_, v_s_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldrM___boxed(lean_object* v_00_u03b1_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_m_313_, lean_object* v_00_u03b2_314_, lean_object* v_inst_315_, lean_object* v_f_316_, lean_object* v_init_317_, lean_object* v_s_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_aesop_Aesop_OrderedHashSet_foldrM(v_00_u03b1_310_, v_inst_311_, v_inst_312_, v_m_313_, v_00_u03b2_314_, v_inst_315_, v_f_316_, v_init_317_, v_s_318_);
lean_dec_ref(v_inst_312_);
lean_dec_ref(v_inst_311_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr___redArg(lean_object* v_f_320_, lean_object* v_init_321_, lean_object* v_s_322_){
_start:
{
lean_object* v_toArray_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; uint8_t v___x_327_; 
v_toArray_323_ = lean_ctor_get(v_s_322_, 0);
lean_inc_ref(v_toArray_323_);
lean_dec_ref(v_s_322_);
v___x_324_ = lean_array_get_size(v_toArray_323_);
v___x_325_ = lean_unsigned_to_nat(0u);
v___x_326_ = ((lean_object*)(lp_aesop_Aesop_OrderedHashSet_ofArray___redArg___closed__9));
v___x_327_ = lean_nat_dec_lt(v___x_325_, v___x_324_);
if (v___x_327_ == 0)
{
lean_dec_ref(v_toArray_323_);
lean_dec(v_f_320_);
return v_init_321_;
}
else
{
lean_object* v___f_328_; size_t v___x_329_; size_t v___x_330_; lean_object* v___x_331_; 
v___f_328_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_foldl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_328_, 0, v_f_320_);
v___x_329_ = lean_usize_of_nat(v___x_324_);
v___x_330_ = ((size_t)0ULL);
v___x_331_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_326_, v___f_328_, v_toArray_323_, v___x_329_, v___x_330_, v_init_321_);
return v___x_331_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr(lean_object* v_00_u03b1_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_00_u03b2_335_, lean_object* v_f_336_, lean_object* v_init_337_, lean_object* v_s_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_aesop_Aesop_OrderedHashSet_foldr___redArg(v_f_336_, v_init_337_, v_s_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_foldr___boxed(lean_object* v_00_u03b1_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_00_u03b2_343_, lean_object* v_f_344_, lean_object* v_init_345_, lean_object* v_s_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_aesop_Aesop_OrderedHashSet_foldr(v_00_u03b1_340_, v_inst_341_, v_inst_342_, v_00_u03b2_343_, v_f_344_, v_init_345_, v_s_346_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__0(lean_object* v_f_348_, lean_object* v_a_349_, lean_object* v_x_350_, lean_object* v___y_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lean_apply_2(v_f_348_, v_a_349_, v___y_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__1(lean_object* v_inst_353_, lean_object* v_00_u03b2_354_, lean_object* v_s_355_, lean_object* v_b_356_, lean_object* v_f_357_){
_start:
{
lean_object* v_toArray_358_; lean_object* v___f_359_; size_t v_sz_360_; size_t v___x_361_; lean_object* v___x_362_; 
v_toArray_358_ = lean_ctor_get(v_s_355_, 0);
lean_inc_ref(v_toArray_358_);
lean_dec_ref(v_s_355_);
v___f_359_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__0), 4, 1);
lean_closure_set(v___f_359_, 0, v_f_357_);
v_sz_360_ = lean_array_size(v_toArray_358_);
v___x_361_ = ((size_t)0ULL);
v___x_362_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_353_, v_toArray_358_, v___f_359_, v_sz_360_, v___x_361_, v_b_356_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg(lean_object* v_inst_363_){
_start:
{
lean_object* v___f_364_; 
v___f_364_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_364_, 0, v_inst_363_);
return v___f_364_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad(lean_object* v_00_u03b1_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_m_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___f_370_; 
v___f_370_ = lean_alloc_closure((void*)(lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_370_, 0, v_inst_369_);
return v___f_370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_OrderedHashSet_instForInOfMonad___boxed(lean_object* v_00_u03b1_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_m_374_, lean_object* v_inst_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_aesop_Aesop_OrderedHashSet_instForInOfMonad(v_00_u03b1_371_, v_inst_372_, v_inst_373_, v_m_374_, v_inst_375_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
return v_res_376_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashSet_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_OrderedHashSet(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_OrderedHashSet(uint8_t builtin) {
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
lean_object* initialize_Std_Data_HashSet_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_OrderedHashSet(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_OrderedHashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_OrderedHashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_OrderedHashSet(builtin);
}
#ifdef __cplusplus
}
#endif
