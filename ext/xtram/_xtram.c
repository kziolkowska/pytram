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


double _iterate_x(
	int n_entries,
	int pi_length,
	int maxiter,
	double ftol,
	int *C_i,
	int *C_j,
	double *C_ij,
	double *C_ji,
	double *x_row,
	double *c_column,
	double *pi)
	{
		int i, j;
		sparse_x *x = (sparse_x *)malloc(pi_length*sizeof(sparse_x));
		double *temp_pi = (double *)malloc(pi_length*sizeof(double));
		double ferr;
		
		
		for (i=0;i<maxiter; i++)
		{
			update_x(x_row, x, C_i, C_j, C_ij, C_ji, c_column, n_entries);
			update_x_row(n_entries, x, x_row, x, pi_length);
			compute_pi(temp_pi, x_row, pi_length);
			ferr = converged(pi, temp_pi, pi_length);
			for(j=0; j<pi_length;j++)
			{
				pi[j] = temp_pi[j];
			}	
			if(ferr<ftol)break;
		}
		if(ferr>ftol)printf("not converged warning %f" ferr);
		
		free x;
		free temp_pi;
		return ferr;
	}
	
void update_x( double *x_row, sparse_x *x, int *C_i, int *C_j, double *C_ij, double *C_ji, double *c_column, int L)
{
	int t,i,j;
	for(t=0; t<L; t++)
	{
		i = C_i[t];
		j = C_i[t];
		x[t].i = i;
		x[t].j = j;
		x[t].value = (C_ij[t]+C_ji[t])/((c_column[i]/x_row[i])+(c_column[j]/x_row[j]));	
	}
	
}

void update_x_row(int L, sparse_x *x, double *x_row, int x_row_l)
{
	int t,i,j;
	for (i=0; i<x_row_l; i++)
	{
		x_row[i]=0;
	}
	for(t=0; t<L; t++)
	{
		i = x[t].i;
		j = x[t].j;
		if(i==j) x_row[i]+=x[t].value; //check this!!!!
		else
		{
			x_row[i]+=x[t].value;
			x_row[j]+=x[t].value;
		}
	}
}

void compute_pi(double *pi, double *x_row, int l_pi)
{
	int i,sum=0;
	for (i=0; i<l_pi, i++) sum+=x_row[i];
	for (i=0;i<l_pi;i++) pi[i]=x_row[i]/sum;
}

double converged(double *pi_old, double *pi_new, int l_pi)
{
	int i;
	double sqr_norm=0.0;
	double diff;
	
	for (i=0; i<l_pi;i++)
	{
		diff = fabs(pi_old[i]-pi_new[i])
		sqr_norm+=diff*diff;
	}
	return sqrt(sqr_norm);
}
