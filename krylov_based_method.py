from krylov_subspace import *
import sketching

def matrix_function_approximator_arnoldi(A, Q, H, norm_c, function, i) :
    not_exhausted= arnoldi_one_step(A, Q, H, i)
    if not_exhausted:
        return norm_c * (Q[:, :i] @ function(H[:i, :i])[:, 0]), True
    else:
        return norm_c * (Q[:, :i-1] @ function(H[:i-1, :i-1])[:, 0]), False

def matrix_function_approximator_trunc_arnold(A, Q, H, norm_c, function, i, k_t= 5) :
    not_exhausted= trunc_arnoldi_one_step(A, Q, H, i, k_t)
    if not_exhausted:
        return norm_c * (Q[:, :i] @ function(H[:i, :i])[:, 0]), True
    else:
        return norm_c * (Q[:, :i-1] @ function(H[:i-1, :i-1])[:, 0]), False

def matrix_function_approximator_apply_whitening(sketch_Q, R, H_hat, Q, H, beta_0, function, S, i) :
    apply_whitening_one_step(sketch_Q, R, S @ Q[:, i], H_hat, H[:i+1, :i+1], i)
    return beta_0*(Q[:, :i] @ sclg.solve_triangular(R[:i, :i], function(H_hat[:i, :i])[:,0]))

def matrix_function_approximator_lanczos(A, V, T, W, A_v, norm_c, beta_old, sigma_old, q_old, w_old, function, i) :
    not_exhausted, beta_old, sigma_old, q_old, w_old, A_v= lanczos_one_step(A, V, T, W, A_v, i, beta_old, sigma_old, q_old, w_old)
    if not_exhausted :
        return norm_c * (V[:, :i] @ function(T[:i, :i])[:, 0]), True, beta_old, sigma_old, q_old, w_old, A_v
    else:
        return norm_c * (V[:, :i-1] @ function(T[:i-1, :i-1])[:, 0]), False, beta_old, sigma_old, q_old, w_old, A_v

def matrix_function_approximator_arnoldi_with_reortho(A, Q, H, norm_c, function, i) :
    not_exhausted= arnoldi_reortho_one_step(A, Q, H, i)
    if not_exhausted:
        return norm_c * (Q[:, :i] @ function(H[:i, :i])[:, 0]), True
    else:
        return norm_c * (Q[:, :i-1] @ function(H[:i-1, :i-1])[:, 0]), False
    
def full_approx_arnoldi(A, c, function, max_dim= 500, num_checks=5, tol= 1e-14) :
    n = A.shape[1]
    norm_c = sclg.norm(c, ord=2)

    Q_a = np.zeros(shape=(n, max_dim+1))
    Q_a[:, 0] = c
    H_a= np.zeros(shape= (max_dim + 1, max_dim))
    
    f_old= None
    stop_crit= 1
    
    for k in range(1, max_dim + 1) :
        arnoldi_one_step(A, Q_a, H_a, k)
        if (k-1)%num_checks == 0 :
            f_current= norm_c * (Q_a[:, :k] @ function(H_a[:k, :k])[:, 0])
            if isinstance(f_old, np.ndarray) :
                stop_crit= sclg.norm(f_current - f_old, ord=2)
            f_old= f_current
            if stop_crit < tol :
                print("Arnoldi converged after ", k, " iterations.")
                return f_old
    #print("Arnoldi never converged")
    return f_old

def full_approx_trunc_arnoldi(A, c, function, max_dim= 500, num_checks=5, tol= 1e-14, k_t= 5) :
    n = A.shape[1]
    norm_c = sclg.norm(c, ord=2)

    Q_a = np.zeros(shape=(n, max_dim+1))
    Q_a[:, 0] = c
    H_a= np.zeros(shape= (max_dim + 1, max_dim))
    
    f_old= None
    stop_crit= 1
    
    for k in range(1, max_dim + 1) :
        trunc_arnoldi_one_step(A, Q_a, H_a, k, k_t)
        if (k-1)%num_checks == 0 :
            f_current= norm_c * (Q_a[:, :k] @ function(H_a[:k, :k])[:, 0])
            if isinstance(f_old, np.ndarray) :
                stop_crit= sclg.norm(f_current - f_old, ord=2)
            f_old= f_current
            if stop_crit < tol :
                print("Trunc. Arnoldi converged after ", k, " iterations.")
                return f_old
    #print("Trunc. Arnoldi never converged")
    return f_old

def full_approx_lanczos(A, c, function, max_dim= 500, num_checks=5, tol= 1e-14) :
    n = A.shape[1]
    norm_c = sclg.norm(c, ord=2)

    rng= np.random.default_rng()
    w= rng.normal(size= n)
    w= w/np.dot(c, w)
    V_l= np.zeros(shape= (n, max_dim + 1))
    V_l[:, 0]= c
    W_l= np.zeros(shape= (n, max_dim + 1)) 
    W_l[:, 0]= w
    T_l= np.zeros(shape= (max_dim + 1, max_dim + 1))
    T_l[0, 0]= np.dot(A @ c, w)
    sigma_old= 0
    beta_old= 0
    q_old= np.zeros(shape= n)
    w_old= np.zeros(shape= n)
    A_v= A @ c

    f_old= None
    stop_crit= 1
    
    for k in range(1, max_dim + 1) :
        _, beta_old, sigma_old, q_old, w_old, A_v= lanczos_one_step(A, V_l, T_l, W_l, A_v, k, beta_old, sigma_old, q_old, w_old)
        if (k-1)%num_checks == 0 :
            f_current= norm_c * (V_l[:, :k] @ function(T_l[:k, :k])[:, 0])
            if isinstance(f_old, np.ndarray) :
                stop_crit= sclg.norm(f_current - f_old, ord=2)
            f_old= f_current
            if stop_crit < tol :
                print("Lanczos converged after ", k, " iterations.")
                return f_old
    #print("Lanczos never converged")
    return f_old

def full_approx_trunc_arnoldi_whitened(A, c, function, max_dim= 500, num_checks=5, tol= 1e-14, k_t= 5) :
    n = A.shape[1]

    Q_a = np.zeros(shape=(n, max_dim+1))
    Q_a[:, 0] = c
    H_a= np.zeros(shape= (max_dim + 1, max_dim))

    R= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat= np.zeros(shape= (max_dim + 1, max_dim + 1))
    s= 2*max_dim
    S= sketching.SRDCT(n, s)
    S_c= S @ c
    R[0, 0]= sclg.norm(S_c, ord= 2)
    sketch_Q= np.zeros(shape= (s, max_dim + 1))
    sketch_Q[:, 0]= S_c/R[0, 0]
    beta_0= R[0, 0]
    
    f_old= None
    stop_crit= 1
    
    for k in range(1, max_dim + 1) :
        trunc_arnoldi_one_step(A, Q_a, H_a, k, k_t)
        apply_whitening_one_step(sketch_Q, R, S @ Q_a[:, k], H_hat, H_a[:k+1, :k+1], k)
        if (k-1)%num_checks == 0 :
            f_current= beta_0*(Q_a[:, :k] @ sclg.solve_triangular(R[:k, :k], function(H_hat[:k, :k])[:,0]))
            if isinstance(f_old, np.ndarray) :
                stop_crit= sclg.norm(f_current - f_old, ord=2)
            f_old= f_current
            if stop_crit < tol :
                print("Stabilized Trunc. Arnoldi converged after ", k, " iterations.")
                return f_old
    #print("Stabilized Trunc. Arnoldi never converged")
    return f_old

def full_approx_lanczos_whitened(A, c, function, max_dim= 500, num_checks= 5, tol= 1e-14) :
    n = A.shape[1]

    rng= np.random.default_rng()
    w= rng.normal(size= n)
    w= w/np.dot(c, w)
    V_l= np.zeros(shape= (n, max_dim + 1))
    V_l[:, 0]= c
    W_l= np.zeros(shape= (n, max_dim + 1)) 
    W_l[:, 0]= w
    T_l= np.zeros(shape= (max_dim + 1, max_dim + 1))
    T_l[0, 0]= np.dot(A @ c, w)
    sigma_old= 0
    beta_old= 0
    q_old= np.zeros(shape= n)
    w_old= np.zeros(shape= n)
    A_v= A @ c

    R= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat= np.zeros(shape= (max_dim + 1, max_dim + 1))
    s= 2*max_dim
    S= sketching.SRDCT(n, s)
    S_c= S @ c
    R[0, 0]= sclg.norm(S_c, ord= 2)
    sketch_Q= np.zeros(shape= (s, max_dim + 1))
    sketch_Q[:, 0]= S_c/R[0, 0]
    beta_0= R[0, 0]
    
    f_old= None
    stop_crit= 1
    
    for k in range(1, max_dim + 1) :
        _, beta_old, sigma_old, q_old, w_old, A_v= lanczos_one_step(A, V_l, T_l, W_l, A_v, k, beta_old, sigma_old, q_old, w_old)
        apply_whitening_one_step(sketch_Q, R, S @ V_l[:, k], H_hat, T_l[:k+1, :k+1], k)
        if (k-1)%num_checks == 0 :
            f_current= beta_0*(V_l[:, :k] @ sclg.solve_triangular(R[:k, :k], function(H_hat[:k, :k])[:,0]))
            if isinstance(f_old, np.ndarray) :
                stop_crit= sclg.norm(f_current - f_old, ord=2)
            f_old= f_current
            if stop_crit < tol :
                print("Stabilized Lanczos converged after ", k, " iterations.")
                return f_old
    #print("Stabilized Lanczos never converged")
    return f_old

def gmres(A, b, k : int, initial_guess, tol= 1e-16, method= "arnoldi", k_t= 3, s= 1) :
    m, _ = A.shape
    r_0= b - A @ initial_guess
    beta= sclg.norm(r_0)
    initial_q= r_0/beta
    if method== "arnoldi" :
        Q_m_plus_1, H_k, k_if_break= arnoldi_iterations(A, initial_q, k, tol)
        b_small= np.zeros(shape= k_if_break + 1)
        b_small[0]= beta
        y_k, _, _, _= np.linalg.lstsq(H_k, b_small, rcond= None)
    if method== "truncated_arnoldi" :
        Q_m_plus_1, H_k, k_if_break= truncated_arnoldi_iterations(A, initial_q, k, k_t, tol)
        if seed is not None :
            seed= seed + 1
        S= sketching.GaussianSketch(m, s, seed= seed)
        y_k, _, _, _= np.linalg.lstsq(S @ Q_m_plus_1 @ H_k, S @ r_0, rcond= None)
    return initial_guess + Q_m_plus_1[:, :k_if_break] @ y_k

def restarted_gmres(A, b, cycle_size= None, max_num_cycle= None, tol= 1e-6, method= "arnoldi", k_t= 3, s= 1) :
    _, n= A.shape
    initial_guess= np.zeros(shape= n)
    if cycle_size is None :
        cycle_size= min(20, n)
    if max_num_cycle is None :
        err_mem= 0
        err= 10
        while abs(err - err_mem) >= tol :
            initial_guess= gmres(A, b, cycle_size, initial_guess)
            err_mem= err
            err= sclg.norm(A @ initial_guess - b)
            print("last error : ", err)
    else :
        num_cycle=0
        while num_cycle < max_num_cycle :
            initial_guess= gmres(A, b, cycle_size, initial_guess)
            num_cycle+=1
    return initial_guess