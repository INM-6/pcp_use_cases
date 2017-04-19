
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

#ifndef OPT_EXP
#define OPT_EXP

#include<math.h>

#ifdef __cplusplus
extern "C" {
#endif
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

#ifdef __cplusplus
}
#endif

#endif
