/*
TODO: Licence information for ELEPHANT : ASSET
*/

#ifndef OPT_EXP
#define OPT_EXP

#include<math.h>

#ifdef __cplusplus
extern "C" {
#endif
inline void expopt_array(float* input, int entries, float logK)
{
	float cutoff = -30.0 - logK;  // exp(-30) is already almost zero
	// Use the additional logK to assure that we have maximum significance 


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
