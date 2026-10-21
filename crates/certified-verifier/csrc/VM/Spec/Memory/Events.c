// Lean compiler output
// Module: VM.Spec.Memory.Events
// Imports: public import Init public meta import Init public import VM.Spec.Execution.Field
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
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lp_mathlib_ZMod_decidableEq(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_decidableEq___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Array_instDecidableEqImpl___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent;
static const lean_closure_object lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ZMod_decidableEq___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___closed__0_value;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(2013265921u);
v___x_2_ = lp_mathlib_ZMod_commRing(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default(void){
_start:
{
lean_object* v___x_3_; lean_object* v_toSemiring_4_; lean_object* v___x_5_; lean_object* v_toZero_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_3_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0);
v_toSemiring_4_ = lean_ctor_get(v___x_3_, 0);
lean_inc_ref(v_toSemiring_4_);
v___x_5_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_4_);
v_toZero_6_ = lean_ctor_get(v___x_5_, 1);
lean_inc_n(v_toZero_6_, 8);
lean_dec_ref(v___x_5_);
v___x_7_ = lean_unsigned_to_nat(8u);
v___x_8_ = lean_mk_array(v___x_7_, v_toZero_6_);
lean_inc_ref_n(v___x_8_, 2);
v___x_9_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_9_, 0, v_toZero_6_);
lean_ctor_set(v___x_9_, 1, v_toZero_6_);
lean_ctor_set(v___x_9_, 2, v_toZero_6_);
lean_ctor_set(v___x_9_, 3, v_toZero_6_);
lean_ctor_set(v___x_9_, 4, v_toZero_6_);
lean_ctor_set(v___x_9_, 5, v___x_8_);
lean_ctor_set(v___x_9_, 6, v___x_8_);
lean_ctor_set(v___x_9_, 7, v___x_8_);
lean_ctor_set(v___x_9_, 8, v_toZero_6_);
lean_ctor_set(v___x_9_, 9, v_toZero_6_);
return v___x_9_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default;
return v___x_10_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq(lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
lean_object* v_direction_15_; lean_object* v_height_16_; lean_object* v_height__section_17_; lean_object* v_parent__as__label_18_; lean_object* v_parent__address__label_19_; lean_object* v_parent__hash_20_; lean_object* v_left__child__hash_21_; lean_object* v_right__child__hash_22_; lean_object* v_left__direction__different_23_; lean_object* v_right__direction__different_24_; lean_object* v_direction_25_; lean_object* v_height_26_; lean_object* v_height__section_27_; lean_object* v_parent__as__label_28_; lean_object* v_parent__address__label_29_; lean_object* v_parent__hash_30_; lean_object* v_left__child__hash_31_; lean_object* v_right__child__hash_32_; lean_object* v_left__direction__different_33_; lean_object* v_right__direction__different_34_; lean_object* v___x_35_; uint8_t v___x_36_; 
v_direction_15_ = lean_ctor_get(v_x_13_, 0);
v_height_16_ = lean_ctor_get(v_x_13_, 1);
v_height__section_17_ = lean_ctor_get(v_x_13_, 2);
v_parent__as__label_18_ = lean_ctor_get(v_x_13_, 3);
v_parent__address__label_19_ = lean_ctor_get(v_x_13_, 4);
v_parent__hash_20_ = lean_ctor_get(v_x_13_, 5);
v_left__child__hash_21_ = lean_ctor_get(v_x_13_, 6);
v_right__child__hash_22_ = lean_ctor_get(v_x_13_, 7);
v_left__direction__different_23_ = lean_ctor_get(v_x_13_, 8);
v_right__direction__different_24_ = lean_ctor_get(v_x_13_, 9);
v_direction_25_ = lean_ctor_get(v_x_14_, 0);
v_height_26_ = lean_ctor_get(v_x_14_, 1);
v_height__section_27_ = lean_ctor_get(v_x_14_, 2);
v_parent__as__label_28_ = lean_ctor_get(v_x_14_, 3);
v_parent__address__label_29_ = lean_ctor_get(v_x_14_, 4);
v_parent__hash_30_ = lean_ctor_get(v_x_14_, 5);
v_left__child__hash_31_ = lean_ctor_get(v_x_14_, 6);
v_right__child__hash_32_ = lean_ctor_get(v_x_14_, 7);
v_left__direction__different_33_ = lean_ctor_get(v_x_14_, 8);
v_right__direction__different_34_ = lean_ctor_get(v_x_14_, 9);
v___x_35_ = lean_unsigned_to_nat(2013265921u);
v___x_36_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_direction_15_, v_direction_25_);
if (v___x_36_ == 0)
{
return v___x_36_;
}
else
{
uint8_t v___x_37_; 
v___x_37_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_height_16_, v_height_26_);
if (v___x_37_ == 0)
{
return v___x_37_;
}
else
{
uint8_t v___x_38_; 
v___x_38_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_height__section_17_, v_height__section_27_);
if (v___x_38_ == 0)
{
return v___x_38_;
}
else
{
uint8_t v___x_39_; 
v___x_39_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_parent__as__label_18_, v_parent__as__label_28_);
if (v___x_39_ == 0)
{
return v___x_39_;
}
else
{
uint8_t v___x_40_; 
v___x_40_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_parent__address__label_19_, v_parent__address__label_29_);
if (v___x_40_ == 0)
{
return v___x_40_;
}
else
{
lean_object* v___x_41_; uint8_t v___x_42_; 
v___x_41_ = ((lean_object*)(lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___closed__0));
v___x_42_ = l_Array_instDecidableEqImpl___redArg(v___x_41_, v_parent__hash_20_, v_parent__hash_30_);
if (v___x_42_ == 0)
{
return v___x_42_;
}
else
{
uint8_t v___x_43_; 
v___x_43_ = l_Array_instDecidableEqImpl___redArg(v___x_41_, v_left__child__hash_21_, v_left__child__hash_31_);
if (v___x_43_ == 0)
{
return v___x_43_;
}
else
{
uint8_t v___x_44_; 
v___x_44_ = l_Array_instDecidableEqImpl___redArg(v___x_41_, v_right__child__hash_22_, v_right__child__hash_32_);
if (v___x_44_ == 0)
{
return v___x_44_;
}
else
{
uint8_t v___x_45_; 
v___x_45_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_left__direction__different_23_, v_left__direction__different_33_);
if (v___x_45_ == 0)
{
return v___x_45_;
}
else
{
uint8_t v___x_46_; 
v___x_46_ = lp_mathlib_ZMod_decidableEq(v___x_35_, v_right__direction__different_24_, v_right__direction__different_34_);
return v___x_46_;
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___boxed(lean_object* v_x_47_, lean_object* v_x_48_){
_start:
{
uint8_t v_res_49_; lean_object* v_r_50_; 
v_res_49_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq(v_x_47_, v_x_48_);
lean_dec_ref(v_x_48_);
lean_dec_ref(v_x_47_);
v_r_50_ = lean_box(v_res_49_);
return v_r_50_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent(lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq(v_x_51_, v_x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent___boxed(lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
uint8_t v_res_56_; lean_object* v_r_57_; 
v_res_56_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent(v_x_54_, v_x_55_);
lean_dec_ref(v_x_55_);
lean_dec_ref(v_x_54_);
v_r_57_ = lean_box(v_res_56_);
return v_r_57_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default(void){
_start:
{
lean_object* v___x_58_; lean_object* v_toSemiring_59_; lean_object* v___x_60_; lean_object* v_toZero_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_58_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default___closed__0);
v_toSemiring_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc_ref(v_toSemiring_59_);
v___x_60_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_59_);
v_toZero_61_ = lean_ctor_get(v___x_60_, 1);
lean_inc_n(v_toZero_61_, 5);
lean_dec_ref(v___x_60_);
v___x_62_ = lean_unsigned_to_nat(8u);
v___x_63_ = lean_mk_array(v___x_62_, v_toZero_61_);
v___x_64_ = lean_unsigned_to_nat(2u);
v___x_65_ = lean_mk_array(v___x_64_, v_toZero_61_);
lean_inc_ref(v___x_63_);
v___x_66_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_66_, 0, v_toZero_61_);
lean_ctor_set(v___x_66_, 1, v_toZero_61_);
lean_ctor_set(v___x_66_, 2, v_toZero_61_);
lean_ctor_set(v___x_66_, 3, v___x_63_);
lean_ctor_set(v___x_66_, 4, v___x_63_);
lean_ctor_set(v___x_66_, 5, v___x_65_);
return v___x_66_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default;
return v___x_67_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq(lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
lean_object* v_direction_70_; lean_object* v_address__space_71_; lean_object* v_leaf__label_72_; lean_object* v_values_73_; lean_object* v_hash_74_; lean_object* v_timestamps_75_; lean_object* v_direction_76_; lean_object* v_address__space_77_; lean_object* v_leaf__label_78_; lean_object* v_values_79_; lean_object* v_hash_80_; lean_object* v_timestamps_81_; lean_object* v___x_82_; uint8_t v___x_83_; 
v_direction_70_ = lean_ctor_get(v_x_68_, 0);
v_address__space_71_ = lean_ctor_get(v_x_68_, 1);
v_leaf__label_72_ = lean_ctor_get(v_x_68_, 2);
v_values_73_ = lean_ctor_get(v_x_68_, 3);
v_hash_74_ = lean_ctor_get(v_x_68_, 4);
v_timestamps_75_ = lean_ctor_get(v_x_68_, 5);
v_direction_76_ = lean_ctor_get(v_x_69_, 0);
v_address__space_77_ = lean_ctor_get(v_x_69_, 1);
v_leaf__label_78_ = lean_ctor_get(v_x_69_, 2);
v_values_79_ = lean_ctor_get(v_x_69_, 3);
v_hash_80_ = lean_ctor_get(v_x_69_, 4);
v_timestamps_81_ = lean_ctor_get(v_x_69_, 5);
v___x_82_ = lean_unsigned_to_nat(2013265921u);
v___x_83_ = lp_mathlib_ZMod_decidableEq(v___x_82_, v_direction_70_, v_direction_76_);
if (v___x_83_ == 0)
{
return v___x_83_;
}
else
{
uint8_t v___x_84_; 
v___x_84_ = lp_mathlib_ZMod_decidableEq(v___x_82_, v_address__space_71_, v_address__space_77_);
if (v___x_84_ == 0)
{
return v___x_84_;
}
else
{
uint8_t v___x_85_; 
v___x_85_ = lp_mathlib_ZMod_decidableEq(v___x_82_, v_leaf__label_72_, v_leaf__label_78_);
if (v___x_85_ == 0)
{
return v___x_85_;
}
else
{
lean_object* v___x_86_; uint8_t v___x_87_; 
v___x_86_ = ((lean_object*)(lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMemoryMerkleExpansionEvent_decEq___closed__0));
v___x_87_ = l_Array_instDecidableEqImpl___redArg(v___x_86_, v_values_73_, v_values_79_);
if (v___x_87_ == 0)
{
return v___x_87_;
}
else
{
uint8_t v___x_88_; 
v___x_88_ = l_Array_instDecidableEqImpl___redArg(v___x_86_, v_hash_74_, v_hash_80_);
if (v___x_88_ == 0)
{
return v___x_88_;
}
else
{
uint8_t v___x_89_; 
v___x_89_ = l_Array_instDecidableEqImpl___redArg(v___x_86_, v_timestamps_75_, v_timestamps_81_);
return v___x_89_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq___boxed(lean_object* v_x_90_, lean_object* v_x_91_){
_start:
{
uint8_t v_res_92_; lean_object* v_r_93_; 
v_res_92_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq(v_x_90_, v_x_91_);
lean_dec_ref(v_x_91_);
lean_dec_ref(v_x_90_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent(lean_object* v_x_94_, lean_object* v_x_95_){
_start:
{
uint8_t v___x_96_; 
v___x_96_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent_decEq(v_x_94_, v_x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent___boxed(lean_object* v_x_97_, lean_object* v_x_98_){
_start:
{
uint8_t v_res_99_; lean_object* v_r_100_; 
v_res_99_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqPersistentBoundaryEvent(v_x_97_, v_x_98_);
lean_dec_ref(v_x_98_);
lean_dec_ref(v_x_97_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Execution_Field(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Events(uint8_t builtin) {
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
res = initialize_openvm_x2dfv_VM_Spec_Execution_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent_default);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMemoryMerkleExpansionEvent);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent_default);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedPersistentBoundaryEvent);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
