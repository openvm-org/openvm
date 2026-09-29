// Lean compiler output
// Module: Aesop.Forward.PremiseIndex
// Imports: public import Init public meta import Init
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
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPremiseIndex_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPremiseIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqPremiseIndex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqPremiseIndex___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqPremiseIndex___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqPremiseIndex = (const lean_object*)&lp_aesop_Aesop_instBEqPremiseIndex___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePremiseIndex_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashablePremiseIndex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashablePremiseIndex___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashablePremiseIndex___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashablePremiseIndex = (const lean_object*)&lp_aesop_Aesop_instHashablePremiseIndex___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdPremiseIndex_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdPremiseIndex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdPremiseIndex___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdPremiseIndex___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdPremiseIndex = (const lean_object*)&lp_aesop_Aesop_instOrdPremiseIndex___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTPremiseIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLEPremiseIndex;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLe___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToStringPremiseIndex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_reprFast, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToStringPremiseIndex___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringPremiseIndex___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToStringPremiseIndex = (const lean_object*)&lp_aesop_Aesop_instToStringPremiseIndex___closed__0_value;
static lean_object* _init_lp_aesop_Aesop_instInhabitedPremiseIndex_default(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPremiseIndex(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = lean_nat_dec_eq(v_x_3_, v_x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPremiseIndex_beq___boxed(lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_aesop_Aesop_instBEqPremiseIndex_beq(v_x_6_, v_x_7_);
lean_dec(v_x_7_);
lean_dec(v_x_6_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePremiseIndex_hash(lean_object* v_x_12_){
_start:
{
uint64_t v___x_13_; uint64_t v___x_14_; uint64_t v___x_15_; 
v___x_13_ = 0ULL;
v___x_14_ = lean_uint64_of_nat(v_x_12_);
v___x_15_ = lean_uint64_mix_hash(v___x_13_, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePremiseIndex_hash___boxed(lean_object* v_x_16_){
_start:
{
uint64_t v_res_17_; lean_object* v_r_18_; 
v_res_17_ = lp_aesop_Aesop_instHashablePremiseIndex_hash(v_x_16_);
lean_dec(v_x_16_);
v_r_18_ = lean_box_uint64(v_res_17_);
return v_r_18_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(lean_object* v_x_21_, lean_object* v_x_22_){
_start:
{
uint8_t v___x_23_; 
v___x_23_ = lean_nat_dec_eq(v_x_21_, v_x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq___boxed(lean_object* v_x_24_, lean_object* v_x_25_){
_start:
{
uint8_t v_res_26_; lean_object* v_r_27_; 
v_res_26_ = lp_aesop_Aesop_instDecidableEqPremiseIndex_decEq(v_x_24_, v_x_25_);
lean_dec(v_x_25_);
lean_dec(v_x_24_);
v_r_27_ = lean_box(v_res_26_);
return v_r_27_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqPremiseIndex(lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
uint8_t v___x_30_; 
v___x_30_ = lean_nat_dec_eq(v_x_28_, v_x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqPremiseIndex___boxed(lean_object* v_x_31_, lean_object* v_x_32_){
_start:
{
uint8_t v_res_33_; lean_object* v_r_34_; 
v_res_33_ = lp_aesop_Aesop_instDecidableEqPremiseIndex(v_x_31_, v_x_32_);
lean_dec(v_x_32_);
lean_dec(v_x_31_);
v_r_34_ = lean_box(v_res_33_);
return v_r_34_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdPremiseIndex_ord(lean_object* v_x_35_, lean_object* v_x_36_){
_start:
{
uint8_t v___x_37_; 
v___x_37_ = lean_nat_dec_lt(v_x_35_, v_x_36_);
if (v___x_37_ == 0)
{
uint8_t v___x_38_; 
v___x_38_ = lean_nat_dec_eq(v_x_35_, v_x_36_);
if (v___x_38_ == 0)
{
uint8_t v___x_39_; 
v___x_39_ = 2;
return v___x_39_;
}
else
{
uint8_t v___x_40_; 
v___x_40_ = 1;
return v___x_40_;
}
}
else
{
uint8_t v___x_41_; 
v___x_41_ = 0;
return v___x_41_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdPremiseIndex_ord___boxed(lean_object* v_x_42_, lean_object* v_x_43_){
_start:
{
uint8_t v_res_44_; lean_object* v_r_45_; 
v_res_44_ = lp_aesop_Aesop_instOrdPremiseIndex_ord(v_x_42_, v_x_43_);
lean_dec(v_x_43_);
lean_dec(v_x_42_);
v_r_45_ = lean_box(v_res_44_);
return v_r_45_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTPremiseIndex(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_box(0);
return v___x_48_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLt(lean_object* v_i_49_, lean_object* v_j_50_){
_start:
{
uint8_t v___x_51_; 
v___x_51_ = lean_nat_dec_lt(v_i_49_, v_j_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLt___boxed(lean_object* v_i_52_, lean_object* v_j_53_){
_start:
{
uint8_t v_res_54_; lean_object* v_r_55_; 
v_res_54_ = lp_aesop_Aesop_instDecidableRelPremiseIndexLt(v_i_52_, v_j_53_);
lean_dec(v_j_53_);
lean_dec(v_i_52_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLEPremiseIndex(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_box(0);
return v___x_56_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableRelPremiseIndexLe(lean_object* v_i_57_, lean_object* v_j_58_){
_start:
{
uint8_t v___x_59_; 
v___x_59_ = lean_nat_dec_le(v_i_57_, v_j_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableRelPremiseIndexLe___boxed(lean_object* v_i_60_, lean_object* v_j_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_aesop_Aesop_instDecidableRelPremiseIndexLe(v_i_60_, v_j_61_);
lean_dec(v_j_61_);
lean_dec(v_i_60_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPremiseIndex_default = _init_lp_aesop_Aesop_instInhabitedPremiseIndex_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPremiseIndex_default);
lp_aesop_Aesop_instInhabitedPremiseIndex = _init_lp_aesop_Aesop_instInhabitedPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPremiseIndex);
lp_aesop_Aesop_instLTPremiseIndex = _init_lp_aesop_Aesop_instLTPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instLTPremiseIndex);
lp_aesop_Aesop_instLEPremiseIndex = _init_lp_aesop_Aesop_instLEPremiseIndex();
lean_mark_persistent(lp_aesop_Aesop_instLEPremiseIndex);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_PremiseIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_PremiseIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_PremiseIndex(builtin);
}
#ifdef __cplusplus
}
#endif
