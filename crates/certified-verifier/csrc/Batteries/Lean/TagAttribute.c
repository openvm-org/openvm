// Lean compiler output
// Module: Batteries.Lean.TagAttribute
// Imports: public import Init public meta import Init public import Lean.Attributes
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
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls_core(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_TagAttribute_getDecls___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_TagAttribute_getDecls___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(lean_object* v_init_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v_k_3_; lean_object* v_l_4_; lean_object* v_r_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_k_3_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_k_3_);
v_l_4_ = lean_ctor_get(v_x_2_, 3);
lean_inc(v_l_4_);
v_r_5_ = lean_ctor_get(v_x_2_, 4);
lean_inc(v_r_5_);
lean_dec_ref_known(v_x_2_, 5);
v___x_6_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(v_init_1_, v_l_4_);
v___x_7_ = lean_array_push(v___x_6_, v_k_3_);
v_init_1_ = v___x_7_;
v_x_2_ = v_r_5_;
goto _start;
}
else
{
return v_init_1_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1(lean_object* v_as_9_, size_t v_sz_10_, size_t v_i_11_, lean_object* v_b_12_){
_start:
{
uint8_t v___x_13_; 
v___x_13_ = lean_usize_dec_lt(v_i_11_, v_sz_10_);
if (v___x_13_ == 0)
{
return v_b_12_;
}
else
{
lean_object* v_a_14_; lean_object* v___x_15_; size_t v___x_16_; size_t v___x_17_; 
v_a_14_ = lean_array_uget_borrowed(v_as_9_, v_i_11_);
v___x_15_ = l_Array_append___redArg(v_b_12_, v_a_14_);
v___x_16_ = ((size_t)1ULL);
v___x_17_ = lean_usize_add(v_i_11_, v___x_16_);
v_i_11_ = v___x_17_;
v_b_12_ = v___x_15_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1___boxed(lean_object* v_as_19_, lean_object* v_sz_20_, lean_object* v_i_21_, lean_object* v_b_22_){
_start:
{
size_t v_sz_boxed_23_; size_t v_i_boxed_24_; lean_object* v_res_25_; 
v_sz_boxed_23_ = lean_unbox_usize(v_sz_20_);
lean_dec(v_sz_20_);
v_i_boxed_24_ = lean_unbox_usize(v_i_21_);
lean_dec(v_i_21_);
v_res_25_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1(v_as_19_, v_sz_boxed_23_, v_i_boxed_24_, v_b_22_);
lean_dec_ref(v_as_19_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls_core(lean_object* v_st_26_){
_start:
{
lean_object* v_importedEntries_27_; lean_object* v_state_28_; lean_object* v___y_30_; 
v_importedEntries_27_ = lean_ctor_get(v_st_26_, 0);
lean_inc_ref(v_importedEntries_27_);
v_state_28_ = lean_ctor_get(v_st_26_, 1);
lean_inc(v_state_28_);
lean_dec_ref(v_st_26_);
if (lean_obj_tag(v_state_28_) == 0)
{
lean_object* v_size_36_; 
v_size_36_ = lean_ctor_get(v_state_28_, 0);
lean_inc(v_size_36_);
v___y_30_ = v_size_36_;
goto v___jp_29_;
}
else
{
lean_object* v___x_37_; 
v___x_37_ = lean_unsigned_to_nat(0u);
v___y_30_ = v___x_37_;
goto v___jp_29_;
}
v___jp_29_:
{
lean_object* v___x_31_; lean_object* v___x_32_; size_t v_sz_33_; size_t v___x_34_; lean_object* v___x_35_; 
v___x_31_ = lean_mk_empty_array_with_capacity(v___y_30_);
lean_dec(v___y_30_);
v___x_32_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(v___x_31_, v_state_28_);
v_sz_33_ = lean_array_size(v_importedEntries_27_);
v___x_34_ = ((size_t)0ULL);
v___x_35_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__1(v_importedEntries_27_, v_sz_33_, v___x_34_, v___x_32_);
lean_dec_ref(v_importedEntries_27_);
return v___x_35_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0(lean_object* v_init_38_, lean_object* v_t_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(v_init_38_, v_t_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_batteries_Lean_TagAttribute_getDecls___closed__0(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_41_ = lean_box(1);
v___x_42_ = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(v___x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls(lean_object* v_attr_43_, lean_object* v_env_44_){
_start:
{
lean_object* v_ext_45_; lean_object* v_toEnvExtension_46_; lean_object* v_asyncMode_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v_ext_45_ = lean_ctor_get(v_attr_43_, 1);
v_toEnvExtension_46_ = lean_ctor_get(v_ext_45_, 0);
v_asyncMode_47_ = lean_ctor_get(v_toEnvExtension_46_, 2);
v___x_48_ = lean_obj_once(&lp_batteries_Lean_TagAttribute_getDecls___closed__0, &lp_batteries_Lean_TagAttribute_getDecls___closed__0_once, _init_lp_batteries_Lean_TagAttribute_getDecls___closed__0);
v___x_49_ = lean_box(0);
v___x_50_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_48_, v_toEnvExtension_46_, v_env_44_, v_asyncMode_47_, v___x_49_);
v___x_51_ = lp_batteries_Lean_TagAttribute_getDecls_core(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls___boxed(lean_object* v_attr_52_, lean_object* v_env_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_batteries_Lean_TagAttribute_getDecls(v_attr_52_, v_env_53_);
lean_dec_ref(v_attr_52_);
return v_res_54_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Attributes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin) {
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
lean_object* initialize_Lean_Attributes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_TagAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_TagAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_TagAttribute(builtin);
}
#ifdef __cplusplus
}
#endif
