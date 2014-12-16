/*

    _xtram.c - xTRAM implementation in C

    author: Antonia Mey <antonia.mey@fu-berlin.de>

*/

#include "_xtram.h"


void _b_i_IJ_equation(
	int T_length, 
	int n_therm_states, 
	int n_markov_states,
	int *T_x, 
	int *M_x,
	int *N,  
	double *f,  
	double *w,
	double *u,
	double *b_i_IJ)
	{
	
	int x, i, I, J, TT;
	double delta, metropolis;
	
	TT=n_therm_states*n_therm_states;
	for(i = 0;i<n_markov_states; i++)
		for(I=0; I<n_therm_states; I++)
			for(J=0; J<n_therm_states; J++)
				b_i_IJ[i*TT+I*n_therm_states+J]=0.0;
			

	for(x=0; x<T_length; x++)
	{
		i=M_x[x];
		I=T_x[x];
		for(J=0;J<n_therm_states; J++)
		{
			if (I==J) continue;
			delta = f[J]-f[I]+u[I*T_length+x]-u[J*T_length+x];
			if(N[J]==N[I])
			{
				if (delta<0.0) metropolis =exp(delta);
				else metropolis=1.0;
			}
			else
			{
				delta+= log((double)N[J]/(double)N[I]);
				if (delta<0.0) metropolis =exp(delta);
				else metropolis=1.0;
			}
			b_i_IJ[i*TT+I*n_therm_states+J]+=w[I]*metropolis;	
		}
	}
	for(i = 0;i<n_markov_states; i++)
	{
		for(I=0; I<n_therm_states; I++)
		{
			delta=0.0;
			for(J=0; J<n_therm_states; J++)
				delta += b_i_IJ[i*TT+I*n_therm_states+J];
			b_i_IJ[i*TT+I*n_therm_states+I]=1-delta;
		}	
	}
}


