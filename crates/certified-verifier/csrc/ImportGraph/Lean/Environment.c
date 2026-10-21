// Lean compiler output
// Module: ImportGraph.Lean.Environment
// Imports: public import Init public meta import Init public import Lean.Environment
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
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Environment_constants(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_getModuleFor_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_getModuleFor_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_1_, lean_object* v_i_2_, lean_object* v_k_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
v___x_4_ = lean_array_get_size(v_keys_1_);
v___x_5_ = lean_nat_dec_lt(v_i_2_, v___x_4_);
if (v___x_5_ == 0)
{
lean_dec(v_i_2_);
return v___x_5_;
}
else
{
lean_object* v_k_x27_6_; uint8_t v___x_7_; 
v_k_x27_6_ = lean_array_fget_borrowed(v_keys_1_, v_i_2_);
v___x_7_ = lean_name_eq(v_k_3_, v_k_x27_6_);
if (v___x_7_ == 0)
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lean_unsigned_to_nat(1u);
v___x_9_ = lean_nat_add(v_i_2_, v___x_8_);
lean_dec(v_i_2_);
v_i_2_ = v___x_9_;
goto _start;
}
else
{
lean_dec(v_i_2_);
return v___x_7_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_11_, lean_object* v_i_12_, lean_object* v_k_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg(v_keys_11_, v_i_12_, v_k_13_);
lean_dec(v_k_13_);
lean_dec_ref(v_keys_11_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg(lean_object* v_x_16_, size_t v_x_17_, lean_object* v_x_18_){
_start:
{
if (lean_obj_tag(v_x_16_) == 0)
{
lean_object* v_es_19_; lean_object* v___x_20_; size_t v___x_21_; size_t v___x_22_; lean_object* v_j_23_; lean_object* v___x_24_; 
v_es_19_ = lean_ctor_get(v_x_16_, 0);
v___x_20_ = lean_box(2);
v___x_21_ = ((size_t)31ULL);
v___x_22_ = lean_usize_land(v_x_17_, v___x_21_);
v_j_23_ = lean_usize_to_nat(v___x_22_);
v___x_24_ = lean_array_get_borrowed(v___x_20_, v_es_19_, v_j_23_);
lean_dec(v_j_23_);
switch(lean_obj_tag(v___x_24_))
{
case 0:
{
lean_object* v_key_25_; uint8_t v___x_26_; 
v_key_25_ = lean_ctor_get(v___x_24_, 0);
v___x_26_ = lean_name_eq(v_x_18_, v_key_25_);
return v___x_26_;
}
case 1:
{
lean_object* v_node_27_; size_t v___x_28_; size_t v___x_29_; 
v_node_27_ = lean_ctor_get(v___x_24_, 0);
v___x_28_ = ((size_t)5ULL);
v___x_29_ = lean_usize_shift_right(v_x_17_, v___x_28_);
v_x_16_ = v_node_27_;
v_x_17_ = v___x_29_;
goto _start;
}
default: 
{
uint8_t v___x_31_; 
v___x_31_ = 0;
return v___x_31_;
}
}
}
else
{
lean_object* v_ks_32_; lean_object* v___x_33_; uint8_t v___x_34_; 
v_ks_32_ = lean_ctor_get(v_x_16_, 0);
v___x_33_ = lean_unsigned_to_nat(0u);
v___x_34_ = lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg(v_ks_32_, v___x_33_, v_x_18_);
return v___x_34_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_x_35_, lean_object* v_x_36_, lean_object* v_x_37_){
_start:
{
size_t v_x_202__boxed_38_; uint8_t v_res_39_; lean_object* v_r_40_; 
v_x_202__boxed_38_ = lean_unbox_usize(v_x_36_);
lean_dec(v_x_36_);
v_res_39_ = lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg(v_x_35_, v_x_202__boxed_38_, v_x_37_);
lean_dec(v_x_37_);
lean_dec_ref(v_x_35_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg(lean_object* v_x_41_, lean_object* v_x_42_){
_start:
{
uint64_t v___y_44_; 
if (lean_obj_tag(v_x_42_) == 0)
{
uint64_t v___x_47_; 
v___x_47_ = 1723ULL;
v___y_44_ = v___x_47_;
goto v___jp_43_;
}
else
{
uint64_t v_hash_48_; 
v_hash_48_ = lean_ctor_get_uint64(v_x_42_, sizeof(void*)*2);
v___y_44_ = v_hash_48_;
goto v___jp_43_;
}
v___jp_43_:
{
size_t v___x_45_; uint8_t v___x_46_; 
v___x_45_ = lean_uint64_to_usize(v___y_44_);
v___x_46_ = lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg(v_x_41_, v___x_45_, v_x_42_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg___boxed(lean_object* v_x_49_, lean_object* v_x_50_){
_start:
{
uint8_t v_res_51_; lean_object* v_r_52_; 
v_res_51_ = lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg(v_x_49_, v_x_50_);
lean_dec(v_x_50_);
lean_dec_ref(v_x_49_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_getModuleFor_x3f(lean_object* v_env_53_, lean_object* v_declName_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_53_, v_declName_54_);
if (lean_obj_tag(v___x_55_) == 0)
{
lean_object* v___x_56_; lean_object* v_map_u2082_57_; uint8_t v___x_58_; 
lean_inc_ref(v_env_53_);
v___x_56_ = l_Lean_Environment_constants(v_env_53_);
v_map_u2082_57_ = lean_ctor_get(v___x_56_, 1);
lean_inc_ref(v_map_u2082_57_);
lean_dec_ref(v___x_56_);
v___x_58_ = lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg(v_map_u2082_57_, v_declName_54_);
lean_dec_ref(v_map_u2082_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; 
lean_dec_ref(v_env_53_);
v___x_59_ = lean_box(0);
return v___x_59_;
}
else
{
lean_object* v___x_60_; lean_object* v_mainModule_61_; lean_object* v___x_62_; 
v___x_60_ = l_Lean_Environment_header(v_env_53_);
lean_dec_ref(v_env_53_);
v_mainModule_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc(v_mainModule_61_);
lean_dec_ref(v___x_60_);
v___x_62_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_62_, 0, v_mainModule_61_);
return v___x_62_;
}
}
else
{
lean_object* v_val_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_74_; 
v_val_63_ = lean_ctor_get(v___x_55_, 0);
v_isSharedCheck_74_ = !lean_is_exclusive(v___x_55_);
if (v_isSharedCheck_74_ == 0)
{
v___x_65_ = v___x_55_;
v_isShared_66_ = v_isSharedCheck_74_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_val_63_);
lean_dec(v___x_55_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_74_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_72_; 
v___x_67_ = lean_box(0);
v___x_68_ = l_Lean_Environment_header(v_env_53_);
lean_dec_ref(v_env_53_);
v___x_69_ = l_Lean_EnvironmentHeader_moduleNames(v___x_68_);
v___x_70_ = lean_array_get(v___x_67_, v___x_69_, v_val_63_);
lean_dec(v_val_63_);
lean_dec_ref(v___x_69_);
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 0, v___x_70_);
v___x_72_ = v___x_65_;
goto v_reusejp_71_;
}
else
{
lean_object* v_reuseFailAlloc_73_; 
v_reuseFailAlloc_73_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_73_, 0, v___x_70_);
v___x_72_ = v_reuseFailAlloc_73_;
goto v_reusejp_71_;
}
v_reusejp_71_:
{
return v___x_72_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_getModuleFor_x3f___boxed(lean_object* v_env_75_, lean_object* v_declName_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_importGraph_Lean_Environment_getModuleFor_x3f(v_env_75_, v_declName_76_);
lean_dec(v_declName_76_);
return v_res_77_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0(lean_object* v_00_u03b2_78_, lean_object* v_x_79_, lean_object* v_x_80_){
_start:
{
uint8_t v___x_81_; 
v___x_81_ = lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___redArg(v_x_79_, v_x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0___boxed(lean_object* v_00_u03b2_82_, lean_object* v_x_83_, lean_object* v_x_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_importGraph_Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0(v_00_u03b2_82_, v_x_83_, v_x_84_);
lean_dec(v_x_84_);
lean_dec_ref(v_x_83_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0(lean_object* v_00_u03b2_87_, lean_object* v_x_88_, size_t v_x_89_, lean_object* v_x_90_){
_start:
{
uint8_t v___x_91_; 
v___x_91_ = lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___redArg(v_x_88_, v_x_89_, v_x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0___boxed(lean_object* v_00_u03b2_92_, lean_object* v_x_93_, lean_object* v_x_94_, lean_object* v_x_95_){
_start:
{
size_t v_x_303__boxed_96_; uint8_t v_res_97_; lean_object* v_r_98_; 
v_x_303__boxed_96_ = lean_unbox_usize(v_x_94_);
lean_dec(v_x_94_);
v_res_97_ = lp_importGraph_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0(v_00_u03b2_92_, v_x_93_, v_x_303__boxed_96_, v_x_95_);
lean_dec(v_x_95_);
lean_dec_ref(v_x_93_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_99_, lean_object* v_keys_100_, lean_object* v_vals_101_, lean_object* v_heq_102_, lean_object* v_i_103_, lean_object* v_k_104_){
_start:
{
uint8_t v___x_105_; 
v___x_105_ = lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___redArg(v_keys_100_, v_i_103_, v_k_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_106_, lean_object* v_keys_107_, lean_object* v_vals_108_, lean_object* v_heq_109_, lean_object* v_i_110_, lean_object* v_k_111_){
_start:
{
uint8_t v_res_112_; lean_object* v_r_113_; 
v_res_112_ = lp_importGraph_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_Environment_getModuleFor_x3f_spec__0_spec__0_spec__1(v_00_u03b2_106_, v_keys_107_, v_vals_108_, v_heq_109_, v_i_110_, v_k_111_);
lean_dec(v_k_111_);
lean_dec_ref(v_vals_108_);
lean_dec_ref(v_keys_107_);
v_r_113_ = lean_box(v_res_112_);
return v_r_113_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Environment(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Lean_Environment(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Lean_Environment(uint8_t builtin) {
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
lean_object* initialize_Lean_Environment(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Lean_Environment(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Lean_Environment(builtin);
}
#ifdef __cplusplus
}
#endif
