import numpy as np
cimport numpy as np

cdef extern from "_xtram.h":
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
		double *b_i_IJ)
		
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

def B_i_IJ_equation(
		np.ndarray[int, ndim=1, mode="c"] T_x not None,
		np.ndarray[int, ndim=1, mode="c"] M_x not None,
		np.ndarray[int, ndim=1, mode="c"] N_K not None,
		np.ndarray[double, ndim=1, mode="c"] f_K not None,
		np.ndarray[double, ndim=1, mode="c"] w_K not None,
		np.ndarray[double, ndim=2, mode="c"] u_K_x not None,
		np.ndarray[double, ndim=3, mode="c"] b_i_IJ not None
	):
	_B_i_IJ_equation(
		T_x.shape[0],
		N_K.shape[0],
		b_i_IJ.shape[0],
		<int*> np.PyArray_DATA( T_x ),
		<int*> np.PyArray_DATA( M_x ),
		<int*> np.PyArray_DATA( N_K ),
		<double*> np.PyArray_DATA( f_K ),
		<double*> np.PyArray_DATA( w_K ),
		<double*> np.PyArray_DATA( u_K_x ),
		<double*> np.PyArray_DATA( b_i_IJ )
		)

def iterate_x():
	return _iterate_x()

