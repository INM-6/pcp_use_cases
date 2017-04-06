
double squared_simple_cython(const double input)
{
    return  input * input;
}


/*

The MIT License (MIT)

Copyright (c) 2015 Jacques-Henri Jourdan <jourgun@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef SIMD_MATH_PRIMS_H
#define SIMD_MATH_PRIMS_H

#include<math.h>
#include <omp.h>
#include <iostream>


#ifdef __cplusplus
extern "C" {
#endif

/* Workaround a lack of optimization in gcc */
float exp_cst1 = 2139095040.f;
float exp_cst2 = 0.f;

// Relative error bounded by 1e-5 for normalized outputs
//   Returns invalid outputs for nan inputs
//   Continuous error
#pragma omp declare simd //processor(mic-avx512)
inline float expapprox(float val) {

  union { int i; float f; } xu, xu2;
  float val2, val3, val4, b;
  int val4i;
  val2 = 12102203.1615614f*val+1065353216.f;
  val3 = val2 < exp_cst1 ? val2 : exp_cst1;
  val4 = val3 > exp_cst2 ? val3 : exp_cst2;
  val4i = (int) val4;
  xu.i = val4i & 0x7F800000;
  xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
  b = xu2.f;

  // Generated in Sollya with:
  //   > f=remez(1-x*exp(-(x-1)*log(2)),
  //             [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x|],
  //             [1,2], exp(-(x-1)*log(2)));
  //   > plot(exp((x-1)*log(2))/(f+x)-1, [1,2]);
  //   > f+x;

  return
    xu.f * (0.510397365625862338668154f + b *
            (0.310670891004095530771135f + b *
             (0.168143436463395944830000f + b *
              (-2.88093587581985443087955e-3f + b *
               1.3671023382430374383648148e-2f))));
}



inline void expapprox_array_inner(float* input, int entries)
{
   // __assume_aligned(input, 64);
      union { int i; float f; } xu, xu2;
      float val2, val3, val4, b;
      int val4i;

    const register float c1 = 0.510397365625862338668154f;
    const register float c2 =    0.310670891004095530771135f ;
    const register float c3 = 0.168143436463395944830000f;
    const register float c4 = -2.88093587581985443087955e-3f;
    const register float c5 = 1.3671023382430374383648148e-2f;

#pragma simd private(xu, xu2, val2, val3, val4, b, val4i)
    for (int x = 0; x < entries; ++x)
    {

      val2 = 12102203.1615614f*input[x]+1065353216.f;
      val3 = val2 < exp_cst1 ? val2 : exp_cst1;
      val4 = val3 > exp_cst2 ? val3 : exp_cst2;
      val4i = (int) val4;
      xu.i = val4i & 0x7F800000;
      xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
      b = xu2.f;

      input[x] = xu.f * (c1 + b *
                (c2+ b *
                 (c3 + b *
                  (c4 + b *
                   c5))));
    }

}

inline void expopt_array(float* input, int entries, float logK)
{
	float cutoff = -30.0 - logK;  // exp(-30) is basically zero
	for (int idx = 0; idx < entries; ++idx)
	{
		// if the exponent is really small or negative infinity return 0
		// else calculate the exponents
		float local = input[idx];
		input[idx] = local < cutoff ? 0 : exp(local + logK);
	}
}

inline void expapprox_array(
	float* input1,
	int entries)
{

#pragma omp parallel
	{
		int number_of_thread = omp_get_num_threads();
		int thread_idx = omp_get_thread_num();
		int start_item = (entries * thread_idx) / number_of_thread;
		int end_item = (entries * (thread_idx + 1)) / number_of_thread;
		int items_local = end_item - start_item;


		float * local_input1 = input1 + start_item;

		expapprox_array_inner(local_input1, items_local);
	}
}



/* Absolute error bounded by 1e-6 for normalized inputs
   Returns a finite number for +inf input
   Returns -inf for nan and <= 0 inputs.
   Continuous error. */
//
inline float logapprox(float val) {

  union { float f; int i; } valu;
  float exp, addcst, x;
  valu.f = val;
  exp = valu.i >> 23;
  // 89.970756366f = 127 * log(2) - constant term of polynomial
  addcst = val > 0 ? -89.970756366f : -(float)INFINITY;
  if (val == 0)
  {
   return  -(float)INFINITY;
  }
  valu.i = (valu.i & 0x7FFFFF) | 0x3F800000;
  x = valu.f;


  //Generated in Sollya using :
   // > f = remez(log(x)-(x-1)*log(2),
   //        [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x,
   //           (x-1)*(x-2)*x*x*x|], [1,2], 1, 1e-8);
   // > plot(f+(x-1)*log(2)-log(x), [1,2]);
   //> f+(x-1)*log(2)

  return
    x * (3.529304993f + x * (-2.461222105f +
      x * (1.130626167f + x * (-0.288739945f +
        x * 3.110401639e-2f))))
    + (addcst + 0.69314718055995f*exp);

}

inline void logapprox_array(float* input, int entries)
{
    #pragma vector aligned
    for (int x = 0; x < entries; ++x)
    {
        input[x] = logapprox(input[x]);
    }
}


inline void logapprox_multiply_array_outer(
	float* input1,
	float* input2,
	int entries)
{

	//__assume_aligned(input1, 64);
    //__assume_aligned(input2, 64);

    union { float f; int i; } valu;
    float exp, addcst, x;

    #pragma simd private( valu, exp, addcst,x)

    for (int idx = 0; idx < entries; ++idx)
    {

          valu.f = input1[idx];
          exp = valu.i >> 23;
          // 89.970756366f = 127 * log(2) - constant term of polynomial
          addcst = input1[idx] > 0 ? -89.970756366f : -(float)INFINITY;
          valu.i = (valu.i & 0x7FFFFF) | 0x3F800000;
          x = valu.f;


          //Generated in Sollya using :
           // > f = remez(log(x)-(x-1)*log(2),
           //        [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x,
           //           (x-1)*(x-2)*x*x*x|], [1,2], 1, 1e-8);
           // > plot(f+(x-1)*log(2)-log(x), [1,2]);
           //> f+(x-1)*log(2)

          input1[idx] =
            x * (3.529304993f + x * (-2.461222105f +
              x * (1.130626167f + x * (-0.288739945f +
                x * 3.110401639e-2f))))
            + (addcst + 0.69314718055995f*exp) * input2[idx];


    }
}


inline void logapprox_multiply_array(
	float* input1,
	float* input2,
	int entries)
{

#pragma omp parallel
	{
		int number_of_thread = omp_get_num_threads();
		int thread_idx = omp_get_thread_num();
		int start_item = (entries * thread_idx) / number_of_thread;
		int end_item = (entries * (thread_idx + 1)) / number_of_thread;
		int items_local = end_item - start_item;


		float * local_input1 = input1 + start_item;
		float * local_input2 = input2 + start_item;

		logapprox_multiply_array_outer(local_input1, local_input2, items_local);
	}
}

inline void multiply_two_array(float* input1, float* input2,  int entries)
{
    #pragma vector aligned
    for (int x = 0; x < entries; ++x)
    {
        // Might be faster of not writing to the same array?
        input1[x] *= input2[x];
    }
}




/* Correct only in [-pi, pi]
   Absolute error bounded by 5e-5
   Continuous error */
inline float cosapprox(float val) {
  float val2 = val*val;
  return
    0.999959766864776611328125f + val2 *
    (-0.4997930824756622314453125f + val2 *
     (4.1496001183986663818359375e-2f + val2 *
      (-1.33926304988563060760498046875e-3f + val2 *
       1.8791708498611114919185638427734375e-5f)));
}

/* Correct only in [-pi, pi]
   Absolute error bounded by 6e-6
   Continuous error */
inline float sinapprox(float val) {
  float val2 = val*val;
  return
    val * (0.99997937679290771484375f + val2 *
           (-0.166624367237091064453125f + val2 *
            (8.30897875130176544189453125e-3f + val2 *
             (-1.92649182281456887722015380859375e-4f + val2 *
              2.147840177713078446686267852783203125e-6f))));
}



inline void multiplysum_arrays(float* output,
                         float* input1,
                         float* input2,
                         int entries,
                         int sum_step)
{
    // First do the multiply
    #pragma vector aligned
    for (int x = 0; x < entries; ++x)
    {
        input1[x] *= input2[x];
    }

    const int one_sumstep = sum_step;
    const int two_sumstep = 2 *sum_step;
    const int three_sumstep = 3 * sum_step;
    const int four_sumstep = 4 * sum_step;
    const int five_sumstep = 5 * sum_step;

    // Do the sum over the first axis
    #pragma vector aligned
    for (int x =0; x < sum_step; ++x)
    {
        output[x] = input1[x] +
                    input1[x + one_sumstep] +
                    input1[x + two_sumstep] +
                    input1[x + three_sumstep] +
                    input1[x + four_sumstep] +
                    input1[x + five_sumstep ];
    }

}

#ifdef __cplusplus
}
#endif

#endif
