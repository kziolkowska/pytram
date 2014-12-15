r"""

======================
xTRAM estimator module
======================

.. moduleauthor:: Antonia Mey <antonia.mey@fu-berlin.de>

"""

import numpy as np
from .. import Estimator, NotConvergedWarning, ExpressionError




########################################################################
#                                                                      #
#   XTRAM ESTIMATOR CLASS                                              #
#                                                                      #
########################################################################


class XTRAM( Estimator ):
    r"""
    I am the xTRAM estimator
    """
    def __init__( self, C_K_ij, u_I_x, T_x, M_x, N_K_i, target = None, verbose = False ):

        r"""
        Initialize the XTRAM object
        
        Parameters
        ----------
        C_K_ij : 3-D numpy array
            Countmatrix for each thermodynamic state K
        u_IK_t : 3-D numpy array
            Biasing tensor
        M_K_t : 2-D numpy array
            Markov state trajectories
        target : Integer 
            target state for which pi_K should be computed
            default : 0
        verbose : Boolean
            Be loud and noisy
        """
        super( XTRAM, self ).__init__( C_K_ij )
        self._citation()
        self.verbose = verbose
        
        self.u_IK_t = u_IK_t
        self.M_x = M_x
        self.N_K_i = N_K_i

        if kT!=None:
            self.kT = kT
        
        self.N_K = self._compute_N_K()
        self.w_K = self._compute_w_K()
        self.f_K = self._compute_f_K()
        self.pi_K_i = self._compute_pi_K_i()
        
        
    def scf_iteration( self , ftol=10e-10, maxiter = 10, verbose = False):
        r"""Main iteration method
        Parameters
        ----------
            ftol : float
                tolerance of the free energy difference of each iteration update
                Default : 10e-10
            maxiter : int
                maximum number of iterations
                Default : 10
            verbose : boolean
                Be loud and noisy
                Default = False
        
        """
        finc = None
        f_old = np.zeros(self.f_K.shape[0])
        if verbose:
            print "# %8s %16s" % ( "[Step]", "[rel. Increment]" )
        for i in xrange(maxiter):
            f_old[:]=self.f_K[:]
            self.b_i_IJ = self._compute_b_i_IJ()
            N_tilde = self._compute_sparse_N()
            self._x_scf_iteration(N_tilde)
            self._update_free_energies()
            print self.f_K
            finc = np.sum(np.abs(f_old-self.f_K))
            if verbose:
                print "  %8d %16.8e" % ( i+1, finc )
            if finc < ftol:
                    break
        if finc > ftol:
                raise NotConvergedWarning( "XTRAM", finc )

    def _x_scf_iteration( self , N_tilde, ftol = 10e-15, maxiter = 10000, verbose = False):
        r""" Inner iteration routine that iterates x_i^I until convergence
        
        Parameters
        ----------
            N_tilde : 2D numpy array
                contains Nx4 entries in a sparse format for the extended count matrix 
                N_tilde[0] = i
                N_tile[1] = j
                N_tile[2] = N_ij
                N_tile[3] = N_ji
            ftol : float
                tolerance of the free energy difference of each iteration update
                Default : 10e-10
            maxiter : int
                maximum number of iterations
                Default : 100000
            verbose : boolean
                be loud and noisy
        
        """
        X_row, N_column = self._initialise_X_and_N(N_tilde)
        pi_curr = X_row/np.sum(X_row)
        pi_old = pi_curr[:]
        perr = None
        loglikelihood=False
        if loglikelihood:
            ll_hist = []
        if verbose:
            print "# %8s %16s" % ( "[Step]", "[rel. Increment]" )
        for i in xrange(maxiter):
            #replace this function with c-function
            pi_old[:] = pi_curr[:]
            X = self._update_x( N_tilde , X_row, N_column)
            X_row = self._update_x_row(X, X_row)
            if loglikelihood:
                ll_hist.append(self._compute_loglikelihood())
            perr = np.linalg.norm(pi_old-X_row/np.sum(X_row))
            pi_curr[:] =X_row/np.sum(X_row)
            if verbose:
                print "  %8d %16.8e %16.8e" % ( i+1, perr, np.sum(X_row) )
            if perr < ftol:
                break
        if perr > ftol:
            raise NotConvergedWarning( "XTRAM", perr )
        self._update_pi_K_i(pi_curr)
            
            
            
    def _initialise_X_and_N( self, N_tilde ):
        r"""
            sets default values for x_i and N_i
        """
        X_row = np.zeros(np.max(N_tilde[:,0])+1)
        N_column = np.zeros(np.max(N_tilde[:,0])+1)
        for i in xrange(len(N_tilde)):
            entry = N_tilde[i]
            if entry[0]==entry[1]:
                X_row[entry[0]]+=(entry[2]+entry[3])*0.5
                N_column[entry[0]]+=entry[2]
            else:
                N_column[entry[0].astype(int)]+=entry[2] #Check that this is the right summation!
                N_column[entry[1].astype(int)]+=entry[3]
                X_row[entry[0]]+=(entry[2]+entry[3])*0.5
                X_row[entry[1]]+=(entry[2]+entry[3])*0.5
        return (X_row, N_column)
    
    
    def _update_x( self, N_tilde, X_row, N_column ):
        r"""
            computes X from the extended coutn matrix and the current X_row and N_column
        """
        _X = np.zeros(shape=(N_tilde.shape[0],3))
        for t in xrange(N_tilde.shape[0]):
            
            i = N_tilde[t,0]
            j = N_tilde[t,1]
            _X[t,0] = i
            _X[t,1] = j
            numerator = (N_tilde[t,2]+N_tilde[t,3])
            denominator = (N_column[i]/X_row[i])+(N_column[j]/X_row[j])
            _X[t,2] = numerator/denominator
        return _X
        
        
    def _update_x_row( self, X, X_row):
        r"""
            updates the x_row according to X
        """
        X_row = X_row*0
        
        for i in xrange(X.shape[0]):
            if X[i,0]==X[i,1]:
                X_row[X[i,0]]+=X[i,2]
            else:
                X_row[X[i,0]]+=X[i,2]
                X_row[X[i,1]]+=X[i,2]
        return X_row
        
    def _update_pi_K_i( self, pi_curr ):
        r"""
        copies the current iteration pi_curr into the pi_K_i variable and normalises it as required
        """
        for K in xrange(self.n_therm_states):
            initial = K*self.n_markov_states
            final =K*self.n_markov_states+self.n_markov_states
            self.pi_K_i[K][:] = pi_curr[initial:final]/np.sum(pi_curr[:])
            
    def _update_free_energies( self ):
        r"""
        computes the free energies based on the current pi_K_i
        """
        for K in xrange( self.f_K.shape[0] ):
            self.f_K[K] = self.f_K[K]- np.log((np.sum(self.N_K).astype(float)/self.N_K[K])*(np.sum(self.pi_K_i[K,:])))

        
    ####################################################################
    #                                                                  #
    # Computes b_i^KI matrix                                           #
    #                                                                  #
    ####################################################################    
    #replace by c-function for speed
    def _compute_b_i_IJ( self ):
        r"""computes the b_i_IJ temperature transition counts
        
        """
        _b_i_IJ = np.zeros(shape=(self.n_markov_states, self.n_therm_states, self.n_therm_states))
        for I in xrange(self.n_therm_states):
            for t in xrange(self.N_K[I]):
                i=self.M_x[I][t] #this is the current Markov state


                therm_sum = 0.0 
                for J in xrange(self.n_therm_states):
                    if I!=J:
                        var = 0
                        exponent = self.f_K[J]-self.f_K[I]+self.u_IK_t[I,I,t]-self.u_IK_t[J,I,t]
                        if exponent > 10:
                            var =self.w_K[I]*1.0
                            #print "yes!"
                        else:
                            factor = float(self.N_K[J])/self.N_K[I]
                            var = self.w_K[I]*min(1.0, factor*np.exp(exponent))
                        therm_sum += var
                        _b_i_IJ[i][I][J] += var
                _b_i_IJ[i][I][I] += 1-therm_sum
        return _b_i_IJ
        
    ####################################################################
    #                                                                  #
    # Computes the extended count matrix                               #
    #                                                                  #
    ####################################################################
        
    def _compute_sparse_N( self , factor=1.0):
        r"""Computes a Nx4 array containing the count matrix in a sparse format
        
        Parameters
        ----------
            factor : float
                multiplication factor default of 1 is fine
        Returns
        -------
            N_tilde : numpy 2d-array
                N-4 numpy array containing the count matrix N-tilde
        """

        N_tilde=[]
        for I in xrange(self.n_therm_states):
            for i in xrange(self.n_markov_states):
                for j in xrange(i,self.n_markov_states):
                    s1=i+I*self.n_markov_states
                    s2=j+I*self.n_markov_states
                    if i==j:
                        n_ij = (self.C_K_ij[I,i,j]*factor+self.b_i_IJ[i,I,I])
                        n_ji = (self.C_K_ij[I,i,j]*factor+self.b_i_IJ[i,I,I])
                        entry=np.zeros(4)
                        entry[0] = s1
                        entry[1] = s2
                        entry[2] = n_ij
                        entry[3] = n_ji
                        
                        N_tilde.append(entry)
                    else:
                        n_ij = self.C_K_ij[I,i,j]*factor
                        n_ji = self.C_K_ij[I,j,i]*factor
                        if n_ij or n_ji !=0: 
                            entry=np.zeros(4)
                            entry[0] = s1
                            entry[1] = s2
                            entry[2] = n_ij
                            entry[3] = n_ji
                            N_tilde.append(entry)
        
        for I in xrange(self.n_therm_states):
            for J in xrange(I,self.n_therm_states):
                for i in xrange(self.n_markov_states):
                    s1=self.n_markov_states*I+i
                    s2=self.n_markov_states*J+i
                    if I!=J:
                        n_ij = self.b_i_IJ[i,I,J]
                        n_ji = self.b_i_IJ[i,J,I]
                        entry=np.zeros(4)
                        entry[0] = s1
                        entry[1] = s2
                        entry[2] = n_ij
                        entry[3] = n_ji
                        N_tilde.append(entry)
        return np.array(N_tilde)

        
    ####################################################################
    #                                                                  #
    # Computes the initial guess of free energies vie bar ratios       #
    #                                                                  #
    ####################################################################
    
    def _compute_f_K( self ):
        _f_K = np.ones(self.n_therm_states)
        for i in xrange (1,self.n_therm_states):
            _f_K[i] = _f_K[i-1] - np.log(self._bar_ratio(i, i-1))
        print _f_K
        return _f_K
        
    ####################################################################
    #                                                                  #
    # Computes BAR ratios                                              #
    #                                                                  #
    ####################################################################
    
    def _bar_ratio( self, I_plus_1, I ):
        r"""computes the bar ratio for the free energy initialisation
        Parameters
        ----------
        I_plus_one : Integer
            thermodynamic state index I+1
        I : Integer
            thermodynamic state index I
        Returns
        -------
        bar_ratio : float
            bar ratio between neighbouring thermodynamic states
        """
        numerator = 0.0
        denominator = 0.0
        for t in xrange (self.N_K[I]):
            exponent = self.u_IK_t[I,I,t]-self.u_IK_t[I_plus_1,I,t]
            if exponent>0:
                numerator+=1
            else:
                numerator += min(1,np.exp(exponent))
        for t in xrange (self.N_K[I_plus_1]):
            exponent = self.u_IK_t[I_plus_1,I_plus_1,t]-self.u_IK_t[I,I_plus_1,t]
            if exponent>0:
                denominator+=1
            else:
                denominator += min(1,np.exp(exponent))
        numerator = numerator*1.0/self.N_K[I]
        denominator = denominator*1.0/self.N_K[I_plus_1]
        return numerator/denominator
    
    ####################################################################
    #                                                                  #
    # Initialises the stationary probabilities                         #
    #                                                                  #
    ####################################################################
    
    def _compute_pi_K_i( self ):
        _pi_K_i = np.ones(self.n_therm_states*self.n_markov_states).reshape(self.n_therm_states,self.n_markov_states)
        
        return _pi_K_i

    ####################################################################
    #                                                                  #
    # Computes the number of thermoydnamic samples                     #
    #                                                                  #
    ####################################################################

    def _compute_N_K( self ):
        return np.sum(self.N_K_i, axis=1).astype(np.int32)

        
    ####################################################################
    #                                                                  #
    # Computes the the weight at each thermoydnamic state              #
    #                                                                  #
    ####################################################################
        
    def _compute_w_K( self ):
        return self.N_K.astype(float)/np.sum(self.N_K) #weight array based on thermodynamics sample counts
        
    
    ####################################################################
    #                                                                  #
    # prints the needed citation                                       #
    #                                                                  #
    ####################################################################

    def _citation( self ):
        r"""Prints citation string"""
        citation_string = (
            "If you use this method for your data analysis please do not forget to cite:\n"
            "xTRAM: Estimating Equilibrium Expectations from Time-Correlated Simulation\n" 
            "Data at Multiple Thermodyncamic States \n"
            "Antonia S. J. S. Mey, Hao Wu and Frank Noe, \n"
            "Phys. Rev. X 4, 041018")
        
        print citation_string

    ####################################################################
    #                                                                  #
    # Getters and setters and checks                                   #
    #                                                                  #
    ####################################################################

    @property
    def u_IK_t( self ):
        return self._u_IK_t
        
    @u_IK_t.setter
    def u_IK_t( self, u_IK_t ):
        self._u_IK_t = None
        if self._check_u_IK_t( u_IK_t ):
            self._u_IK_t = u_IK_t
    
    def _check_u_IK_t( self, u_IK_t ):
        if None == u_IK_t:
            raise ExpressionError( "u_IK_t", "is None" )
        if not isinstance( u_IK_t, (np.ndarray,) ):
            raise ExpressionError( "u_IK_t", "invalid type (%s)" % str( type( u_IK_t ) ) )
        if 3 != u_IK_t.ndim:
            raise ExpressionError( "u_IK_t", "invalid number of dimensions (%d)" % u_IK_t.ndim )
        if u_IK_t.shape[0] != self.n_therm_states:
            raise ExpressionError( "U_IK_t", "unmatching number of thermodynamic states (%d,%d)" % (u_IK_t.shape[0], self.n_therm_states) )
        if u_IK_t.shape[1] != self.n_therm_states:
            raise ExpressionError( "u_IK_t", "unmatching number of thermodynamic states (%d,%d)" % (u_IK_t.shape[1], self.n_therm_states) )
        if np.float64 != u_IK_t.dtype:
            raise ExpressionError( "u_IK_t", "invalid dtype (%s)" % str( u_IK_t.dtype ) )
        return True
    
    @property
    def M_x( self ):
        return self._M_x

    @M_x.setter
    def M_x( self, M_x ):
        self._M_x = None
        if self._check_M_x( M_x ):
            if self.verbose:
                print "M_x check pass"
            self._M_x = M_x

    def _check_M_x( self, M_x ):
        if M_x is None:
            raise ExpressionError( "M_x", "is None" )
        if not isinstance( M_x, (np.ndarray,) ):
            raise ExpressionError( "M_x", "invalid type (%s)" % str( type( M_x ) ) )
        if 2 != M_x.ndim:
            raise ExpressionError( "M_x", "invalid number of dimensions (%d)" % M_x.ndim )
        if M_x.shape[0] != self.n_therm_states:
            raise ExpressionError( "M_x", "unmatching number of thermodynamic states (%d,%d)" % (M_x.shape[0], self.n_therm_states) )
        if M_x.shape[1] != self.u_IK_t.shape[2]:
            raise ExpressionError( "M_x", "unmatching number thermodynamic samples (%d,%d)" % (M_x.shape[1], self.u_IK_t.shape[2]) )
        if np.float64 != M_x.dtype:
            raise ExpressionError( "M_x", "invalid dtype (%s)" % str( M_x.dtype ) )
        return True
        
    @property
    def N_K_i( self ):
        return self._N_K_i
        
    @N_K_i.setter
    def N_K_i( self, N_K_i ):
        self._N_K_i = None
        if self._check_N_K_i( N_K_i ):
            self._N_K_i = N_K_i
    
    def _check_N_K_i( self, N_K_i ):
        if N_K_i is None:
            raise ExpressionError( "N_K_i", "is None" )
        if not isinstance( N_K_i, (np.ndarray,) ):
            raise ExpressionError( "N_K_i", "invalid type (%s)" % str( type( N_K_i ) ) )
        if 2 != N_K_i.ndim:
            raise ExpressionError( "N_K_i", "invalid number of dimensions (%d)" % N_K_i.ndim )
        if N_K_i.shape[0] != self.n_therm_states:
            raise ExpressionError( "N_K_i", "unmatching number of thermodynamic states (%d,%d)" % (N_K_i.shape[0], self.n_therm_states) )
        if N_K_i.shape[1] != self.n_markov_states:
            raise ExpressionError( "N_K_i", "unmatching number of Markov states (%d,%d)" % (N_K_i.shape[1], self.n_markov_states) )
        return True
