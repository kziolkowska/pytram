/*

    _xtram.h - xTRAM implementation in C (header file)

    author: Antonia Mey <antonia.mey@fu-berlin.de>

*/

#ifndef PYTRAM_XTRAM
#define PYTRAM_XTRAM

#include <math.h>

void _B_i_IJ_equation(
	int T_length, 
	int n_therm_states, 
	int n_markov_states,
	int *T_x, 
	int *M_x,
	int *N,  
	double *f,  
	double *w,
	double *u,
	double *b_i_IJ);


#endif
