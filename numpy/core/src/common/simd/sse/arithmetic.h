#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_ARITHMETIC_H
#define _NPY_SIMD_SSE_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  _mm_add_epi8
#define npyv_add_s8  _mm_add_epi8
#define npyv_add_u16 _mm_add_epi16
#define npyv_add_s16 _mm_add_epi16
#define npyv_add_u32 _mm_add_epi32
#define npyv_add_s32 _mm_add_epi32
#define npyv_add_u64 _mm_add_epi64
#define npyv_add_s64 _mm_add_epi64
#define npyv_add_f32 _mm_add_ps
#define npyv_add_f64 _mm_add_pd

// saturated
#define npyv_adds_u8  _mm_adds_epu8
#define npyv_adds_s8  _mm_adds_epi8
#define npyv_adds_u16 _mm_adds_epu16
#define npyv_adds_s16 _mm_adds_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  _mm_sub_epi8
#define npyv_sub_s8  _mm_sub_epi8
#define npyv_sub_u16 _mm_sub_epi16
#define npyv_sub_s16 _mm_sub_epi16
#define npyv_sub_u32 _mm_sub_epi32
#define npyv_sub_s32 _mm_sub_epi32
#define npyv_sub_u64 _mm_sub_epi64
#define npyv_sub_s64 _mm_sub_epi64
#define npyv_sub_f32 _mm_sub_ps
#define npyv_sub_f64 _mm_sub_pd

// saturated
#define npyv_subs_u8  _mm_subs_epu8
#define npyv_subs_s8  _mm_subs_epi8
#define npyv_subs_u16 _mm_subs_epu16
#define npyv_subs_s16 _mm_subs_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Multiplication
 ***************************/
// non-saturated
NPY_FINLINE __m128i npyv_mul_u8(__m128i a, __m128i b)
{
    const __m128i mask = _mm_set1_epi32(0xFF00FF00);
    __m128i even = _mm_mullo_epi16(a, b);
    __m128i odd  = _mm_mullo_epi16(_mm_srai_epi16(a, 8), _mm_srai_epi16(b, 8));
            odd  = _mm_slli_epi16(odd, 8);
    return npyv_select_u8(mask, odd, even);
}
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_u16 _mm_mullo_epi16
#define npyv_mul_s16 _mm_mullo_epi16

#ifdef NPY_HAVE_SSE41
    #define npyv_mul_u32 _mm_mullo_epi32
#else
    NPY_FINLINE __m128i npyv_mul_u32(__m128i a, __m128i b)
    {
        __m128i even = _mm_mul_epu32(a, b);
        __m128i odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
        __m128i low  = _mm_unpacklo_epi32(even, odd);
        __m128i high = _mm_unpackhi_epi32(even, odd);
        return _mm_unpacklo_epi64(low, high);
    }
#endif // NPY_HAVE_SSE41
#define npyv_mul_s32 npyv_mul_u32
// TODO: emulate 64-bit*/
#define npyv_mul_f32 _mm_mul_ps
#define npyv_mul_f64 _mm_mul_pd

// saturated
// TODO: after implment Packs intrins

/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm_div_ps
#define npyv_div_f64 _mm_div_pd
// XXX Implement 32 and 64 bit versions
// encapsulate parameters for fast division on vector of 3 16-bit signed integers
NPY_INLINE npyv_s16x3 npyv_divisor_s16(npy_int16 d)
{
    const int d1 = abs(d);
    int sh, m;
    if (d1 > 1) {
       // TODO: implment npy_bit_scan_reverse_u32
        sh = (int)npy_bit_scan_reverse_u32(d1-1); // shift count = ceil(log2(d1))-1 = (bit_scan_reverse(d1-1)+1)-1
        m = ((1 << (16 + sh)) / d1 - ((1 << 16) - 1)); // calculate multiplier
    }
    else {
        m = 1;                                                  // for d1 = 1
        sh = 0;
        if (d == 0) {
            m /= d;  // provoke error here if d = 0
        }
        if (d == 0x8000) { // fix overflow for this special case
            m = 0x8001;
            sh = 14;
        }
    }
    npyv_s16x3 divisor;
    divisor.val[0] = _mm_set1_epi16((npy_int16)m);           // broadcast multiplier
    divisor.val[1] = _mm_setr_epi32(sh, 0, 0, 0);            // shift count
    divisor.val[2] = _mm_set1_epi32(d < 0 ? -1 : 0);         // sign of divisor
    return divisor;
}

NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    __m128i t1 = _mm_mulhi_epi16(a, divisor.val[0]);   // multiply high signed words
    __m128i t2 = _mm_add_epi16(t1, a);                 // + a
    __m128i t3 = _mm_sra_epi16(t2, divisor.val[1]);    // shift right arithmetic
    __m128i t4 = _mm_srai_epi16(a, 15);                // sign of a
    __m128i t5 = _mm_sub_epi16(t4, divisor.val[2]);    // sign of a - sign of d
    __m128i t6 = _mm_sub_epi16(t3, t5);                // + 1 if a < 0, -1 if d < 0
    return _mm_xor_si128(t6, divisor.val[2]);          // change sign if divisor negative
}
/***************************
 * FUSED
 ***************************/
#ifdef NPY_HAVE_FMA3
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_fmadd_ps
    #define npyv_muladd_f64 _mm_fmadd_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_fmsub_ps
    #define npyv_mulsub_f64 _mm_fmsub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_fnmadd_ps
    #define npyv_nmuladd_f64 _mm_fnmadd_pd
    // negate multiply and subtract, -(a*b) - c
    #define npyv_nmulsub_f32 _mm_fnmsub_ps
    #define npyv_nmulsub_f64 _mm_fnmsub_pd
#elif defined(NPY_HAVE_FMA4)
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_macc_ps
    #define npyv_muladd_f64 _mm_macc_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_msub_ps
    #define npyv_mulsub_f64 _mm_msub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_nmacc_ps
    #define npyv_nmuladd_f64 _mm_nmacc_pd
#else
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_add_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_add_f64(npyv_mul_f64(a, b), c); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(npyv_mul_f64(a, b), c); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(c, npyv_mul_f32(a, b)); }
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(c, npyv_mul_f64(a, b)); }
#endif // NPY_HAVE_FMA3
#ifndef NPY_HAVE_FMA3 // for FMA4 and NON-FMA3
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        npyv_f32 neg_a = npyv_xor_f32(a, npyv_setall_f32(-0.0f));
        return npyv_sub_f32(npyv_mul_f32(neg_a, b), c);
    }
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        npyv_f64 neg_a = npyv_xor_f64(a, npyv_setall_f64(-0.0));
        return npyv_sub_f64(npyv_mul_f64(neg_a, b), c);
    }
#endif // !NPY_HAVE_FMA3

// Horizontal add: Calculates the sum of all vector elements.

NPY_FINLINE npy_uint32 npyv_sum_u32(__m128i a)
{
    __m128i t = _mm_add_epi32(a, _mm_srli_si128(a, 8));
    t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
    return (unsigned)_mm_cvtsi128_si32(t);
}

NPY_FINLINE float npyv_sum_f32(__m128 a)
{
#ifdef NPY_HAVE_SSE3
    __m128 sum_halves = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
#else
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4); 
#endif
}

NPY_FINLINE double npyv_sum_f64(__m128d a)
{
#ifdef NPY_HAVE_SSE3
    return _mm_cvtsd_f64(_mm_hadd_pd(a, a));
#else
    return _mm_cvtsd_f64(_mm_add_pd(a, _mm_unpackhi_pd(a, a)));
#endif
}

#endif // _NPY_SIMD_SSE_ARITHMETIC_H


