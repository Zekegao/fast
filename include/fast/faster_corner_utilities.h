#ifndef FAST_CORNER_UTILITIES_H
#define FAST_CORNER_UTILITIES_H

#if __ARM_NEON__
#include <arm_neon.h>
#elif __SSE2__
#include <emmintrin.h>
#endif

namespace fast
{

/// Check if the pointer is aligned to the specified byte granularity
template<int bytes> bool is_aligned(const void* ptr);
template<> inline bool is_aligned<8>(const void* ptr) { return ((reinterpret_cast<std::size_t>(ptr)) & 0x7) == 0; }
template<> inline bool is_aligned<16>(const void* ptr) { return ((reinterpret_cast<std::size_t>(ptr)) & 0xF) == 0; }


struct Less
{
   template <class T1, class T2> static bool eval(const T1 a, const T2 b)
   {
      return a < b;
   }
   static short prep_t(short pixel_val, short barrier)
   {
      return pixel_val - barrier;
   }
};

struct Greater
{
   template <class T1, class T2> static bool eval(const T1 a, const T2 b)
   {
      return a > b;
   }
   static short prep_t(short pixel_val, short barrier)
   {
      return pixel_val + barrier;
   }
};

#if __SSE2__

#define CHECK_BARRIER(lo, hi, other, flags)       \
  {                 \
  __m128i diff = _mm_subs_epu8(lo, other);      \
  __m128i diff2 = _mm_subs_epu8(other, hi);     \
  __m128i z = _mm_setzero_si128();        \
  diff = _mm_cmpeq_epi8(diff, z);         \
  diff2 = _mm_cmpeq_epi8(diff2, z);       \
  flags = ~(_mm_movemask_epi8(diff) | (_mm_movemask_epi8(diff2) << 16)); \
  }
     
  template <bool Aligned> inline __m128i load_si128(const void* addr) { return _mm_loadu_si128((const __m128i*)addr); }
  template <> inline __m128i load_si128<true>(const void* addr) { return _mm_load_si128((const __m128i*)addr); }

#endif

#if __ARM_NEON__


#define CHECK_BARRIER(lo, hi, other, flags)     \
  {   \
    uint8x16_t diff = vqsubq_u8(lo, other);     \
    uint8x16_t diff2 = vqsubq_u8(other, hi);    \
    uint8x16_t z = vdupq_n_u8(0);      \
    diff = vceqq_u8(diff, z);          \
    diff2 = vceqq_u8(diff2, z);        \
    flags = ~(_mm_movemask_epi8_neon(diff) | (_mm_movemask_epi8_neon(diff2) << 16)); \
  }

 int32_t _mm_movemask_epi8_neon(uint8x16_t input)
  {
    const int8_t __attribute__ ((aligned (16))) xr[8] = {-7,-6,-5,-4,-3,-2,-1,0};
    uint8x8_t mask_and = vdup_n_u8(0x80);
    int8x8_t mask_shift = vld1_s8(xr);

    uint8x8_t lo = vget_low_u8(input);
    uint8x8_t hi = vget_high_u8(input);

    lo = vand_u8(lo, mask_and);
    lo = vshl_u8(lo, mask_shift);

    hi = vand_u8(hi, mask_and);
    hi = vshl_u8(hi, mask_shift);

    lo = vpadd_u8(lo,lo);
    lo = vpadd_u8(lo,lo);
    lo = vpadd_u8(lo,lo);

    hi = vpadd_u8(hi,hi);
    hi = vpadd_u8(hi,hi);

    hi = vpadd_u8(hi,hi);

    return ((hi[0] << 8) | (lo[0] & 0xFF));
  }

  template <bool Aligned> inline uint8x16_t neon_load_si128(const void* addr) { return vld1q_u8((const uint8_t*)addr); }
  template <> inline uint8x16_t neon_load_si128<true>(const void* addr) { return vld1q_u8((const uint8_t*)addr); }

#endif


} // namespace fast

#endif
