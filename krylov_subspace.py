import numpy as np
import scipy.linalg as sclg

def update_QR(sketch_Q, R, new_col, k, tol= 1e-15) :
    for i in range(k) :
        R[i, k]= np.dot(sketch_Q[:, i], new_col)
        new_col-= R[i,k]*sketch_Q[:, i]
    R[k, k]= sclg.norm(new_col)
    if R[k, k] <= tol :
        print("QR update exhausted")
        return False
    new_col= new_col/R[k, k]
    sketch_Q[:, k]= new_col
    return True

def update_QR_2MGS(sketch_Q, R, new_col, k, tol= 1e-15) :
    for i in range(k) :
        R[i, k]= np.dot(sketch_Q[:, i], new_col)
        new_col-= R[i,k]*sketch_Q[:, i]
    for i in range(k) :
        coef= np.dot(sketch_Q[:, i], new_col)
        R[i, k]+= coef
        new_col-= coef*sketch_Q[:, i]
    R[k, k]= sclg.norm(new_col)
    if R[k, k] <= tol :
        print("QR update exhausted")
        return False
    new_col= new_col/R[k, k]
    sketch_Q[:, k]= new_col
    return True

def apply_whitening_one_step(sketch_Q, R, new_col, B_hat, B_k_plus_1, k_plus_1) :
    not_exhausted= update_QR(sketch_Q, R, new_col, k_plus_1)
    k= k_plus_1 - 1
    if k_plus_1 == 1 :
        B_hat[0, 0]= B_k_plus_1[0, 0] + R[0, 1]* B_k_plus_1[1, 0] / R[0, 0]
    else :
        r_k_plus_1= R[:k_plus_1, k_plus_1]
        rho_k= R[k, k]
        b_k_plus_1= B_k_plus_1[k_plus_1, k]
		#Second term is vector to add to last col of B_k_hat
        second_term= (b_k_plus_1/rho_k) * r_k_plus_1
        
        r_k= R[:k, k]
        rho_k_minus_1= R[k-1, k-1]
        beta_k= B_k_plus_1[k, k]
        b_k= B_k_plus_1[:k, k]
        
        B_hat[:k, k]= (1/rho_k) * (R[:k, :k] @ b_k + beta_k * r_k - B_hat[:k, :k] @ r_k)
        B_hat[k, k]= beta_k - B_k_plus_1[k, k-1]*r_k[-1]/rho_k_minus_1
        B_hat[k, k-1]= B_k_plus_1[k, k-1]*rho_k/rho_k_minus_1
        B_hat[:k_plus_1, k]+= second_term
    
def lanczos_one_step(A, V, T, W, A_v, i, beta_old, sigma_old, q_old, w_old, tol= 1e-15) :
    i-=1
    V[:, i+1]= A_v - T[i, i] * V[:, i] - beta_old * q_old
    W[:, i+1]= A.T @ W[:, i] - T[i, i] * W[:, i] - sigma_old * w_old
    mem= np.dot(V[:, i+1], W[:, i+1])
    sigma_old= np.sqrt(abs(mem))
    T[i+1, i]= sigma_old
    if sigma_old <= tol :
        print("Lanczos exhausted at iteration ", i + 1)
        return False, beta_old, sigma_old, q_old, w_old
    beta_old= mem/sigma_old
    T[i, i+1]= beta_old
    W[:, i+1]= W[:, i+1]/beta_old
    V[:, i+1]= V[:, i+1]/sigma_old
    A_v= A @ V[:, i+1]
    T[i+1, i+1]= np.dot(A_v, W[:, i+1])
    q_old= V[:, i]
    w_old= W[:, i]
    return True, beta_old, sigma_old, q_old, w_old, A_v

def lanczos_biorthogonalization(A, q, k : int, tol=1e-15, seed= None) :
    _, n= A.shape
    k_if_break= k
    rng= np.random.default_rng(seed)
    w= rng.normal(size= n)
    w= w/np.dot(q, w)
    T_k= np.zeros(shape= (k, k))
    V_k= np.zeros(shape= (n, k))
    V_k[:, 0]= q
    W_k= np.zeros(shape= (n, k))
    W_k[:, 0]= w
    T_k[0, 0]= np.dot(A @ q, w)
    A_T= A.T
    w_old= np.zeros(shape= n)
    q_old= np.zeros(shape= n)
    beta_old= 0
    sigma_old= 0
    for i in range(1, k) :
        V_k[:, i]= A @ V_k[:, i-1] - T_k[i-1, i-1] * V_k[:,i-1] - beta_old * q_old
        q_old= V_k[:, i-1]
        W_k[:, i]= A_T @ W_k[:, i-1] - T_k[i-1, i-1] * W_k[:, i-1] - sigma_old * w_old
        w_old= W_k[:, i-1]
        mem= np.dot(V_k[:, i], W_k[:, i])
        sigma_old= np.sqrt(abs(mem))
        T_k[i, i-1]= sigma_old
        if sigma_old <= tol :
            k_if_break= i
            print("Lanczos biorthogonalization exhausted at iteration ", i)
            break
        beta_old= mem/sigma_old
        T_k[i-1, i]= beta_old
        W_k[:, i]= W_k[:, i]/beta_old
        V_k[:, i]= V_k[:, i]/sigma_old
        T_k[i, i]= np.dot(A @ V_k[:, i], W_k[:, i])
    return V_k[:, :k_if_break], W_k[:, :k_if_break], T_k[:k_if_break, :k_if_break], k_if_break

def trunc_arnoldi_one_step(A, Q, H, i, k_t, tol= 1e-15) :
    i-=1
    v_new= A @ Q[:, i]
    start= max(0, i + 1 - k_t)
    for j in range(start, i + 1) :
        H[j, i]= np.dot(Q[:, j], v_new)
        v_new= v_new - H[j, i] * Q[:, j]
    H[i+1, i]= sclg.norm(v_new)
    if H[i+1, i] <= tol :
        print("Truncate arnoldi exhausted at iteration ", i + 1)
        return False
    Q[:, i + 1]= v_new/H[i+1, i]
    return True

def truncated_arnoldi_iterations(A, q, k : int, k_t : int, tol=1e-15) :
    _, n= A.shape
    Q= np.empty(shape= (n, k+1))
    Q[:, 0]= q
    H_k= np.zeros(shape= (k+1, k))
    k_if_break= k
    for i in range(k) :
        v_new= A @ Q[:, i]
        start= max(0, i + 1 - k_t)
        for j in range(start, i + 1) :
            H_k[j, i] = np.dot(Q[:, j], v_new)
            v_new= v_new - H_k[j, i] * Q[:, j]
        H_k[i+1, i]= sclg.norm(v_new)
        if H_k[i+1, i] <= tol :
            k_if_break= i + 1
            print("Truncated arnoldi exhausted at iteration ", i + 1)
            break
        Q[:, i + 1]= v_new/H_k[i+1, i]
    return Q[:, : k_if_break + 1], H_k[:k_if_break + 1 , :k_if_break], k_if_break

def arnoldi_one_step(A, Q, H, i, tol= 1e-15) :
    i-=1
    v_new= A @ Q[:, i]
    for j in range(i + 1) :
        H[j, i]= np.dot(Q[:, j], v_new)
        v_new= v_new - H[j, i] * Q[:, j]
    H[i+1, i]= sclg.norm(v_new)
    if H[i+1, i] <= tol :
        print("Arnoldi exhausted at iteration ", i + 1)
        return False
    Q[:, i + 1]= v_new/H[i+1, i]
    return True

def arnoldi_iterations(A, q, k : int, tol=1e-15) :
    _, n= A.shape
    Q= np.empty(shape= (n, k+1))
    Q[:, 0]= q
    H_k= np.zeros(shape= (k+1, k))
    k_if_break= k
    for i in range(k) :
        v_new= A @ Q[:, i]
        for j in range(i + 1) :
            H_k[j, i]= np.dot(Q[:, j], v_new)
            v_new= v_new - H_k[j, i] * Q[:, j]
        H_k[i+1, i]= sclg.norm(v_new)
        if H_k[i+1, i] <= tol :
            k_if_break= i + 1
            print("Arnoldi exhausted at iteration ", i + 1)
            break
        Q[:, i + 1]= v_new/H_k[i+1, i]
    return Q[:, : k_if_break + 1], H_k[:k_if_break + 1 , :k_if_break], k_if_break

def arnoldi_reortho_one_step(A, Q, H, i, tol= 1e-15) :
    i-=1
    v_new= A @ Q[:, i]
    for j in range(i + 1) :
        H[j, i]= np.dot(Q[:, j], v_new)
        v_new= v_new - H[j, i] * Q[:, j]
    for j in range(i + 1) :
        coef= np.dot(Q[:, j], v_new)
        H[j, i]+= coef
        v_new= v_new - coef * Q[:, j]
    H[i+1, i]= sclg.norm(v_new)
    if H[i+1, i] <= tol :
        print("Arnoldi exhausted at iteration ", i + 1)
        return False
    Q[:, i + 1]= v_new/H[i+1, i]
    return True

def arnoldi_iterations_with_reortho(A, q, k : int, tol=1e-15) :
    _, n= A.shape
    Q= np.empty(shape= (n, k+1))
    Q[:, 0]= q
    H_k= np.zeros(shape= (k+1, k))
    k_if_break= k
    for i in range(k) :
        v_new= A @ Q[:, i]
        for j in range(i + 1) :
            H_k[j, i]= np.dot(Q[:, j], v_new)
            v_new= v_new - H_k[j, i] * Q[:, j]
        for j in range(i + 1) :
            coef= np.dot(Q[:, j], v_new)
            H_k[j, i]+= coef
            v_new= v_new - coef * Q[:, j]
        H_k[i+1, i]= sclg.norm(v_new)
        if H_k[i+1, i] <= tol :
            k_if_break= i + 1
            print("Arnoldi exhausted at iteration ", i + 1)
            break
        Q[:, i + 1]= v_new/H_k[i+1, i]
    return Q[:, : k_if_break + 1], H_k[:k_if_break + 1 , :k_if_break], k_if_break