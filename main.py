#%% Dependencies
import importlib
import numpy as np
import scipy as sp
import scipy.linalg as sclg
import matplotlib.pyplot as plt
import krylov_subspace
import krylov_based_method
import sketching
import time
importlib.reload(krylov_based_method)
importlib.reload(krylov_subspace)
importlib.reload(sketching)

#%% Code for convergence analysis
def comparing_methods(A, c, function, true, max_dim, S_t, S_l, k_t= 5, seed= None) :    
    n= A.shape[1]
    norm_c= sclg.norm(c, ord= 2)

    errors_arnoldi= []
    cond_a= []
    Q_a= np.zeros(shape= (n, max_dim+1))
    Q_a[:,0]= c
    H_a= np.zeros(shape= (max_dim + 1, max_dim))
    not_exhausted_a= True
    
    errors_trunc= []
    cond_ta= []
    Q_ta= np.zeros(shape= (n, max_dim+1))
    Q_ta[:,0]= c
    H_ta= np.zeros(shape= (max_dim + 1, max_dim))
    not_exhausted_ta= True
    
    errors_trunc_whitening= []
    cond_taw= []
    R_t= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat_t= np.zeros(shape= (max_dim + 1, max_dim + 1))
    S_c_t= S_t @ c
    R_t[0, 0]= sclg.norm(S_c_t, ord= 2)
    sketch_Q_t= np.zeros(shape= (s, max_dim + 1))
    sketch_Q_t[:, 0]= S_c_t/R_t[0, 0]
    beta_0_t= R_t[0, 0]

    errors_lanczos= []
    cond_l= []
    rng= np.random.default_rng(seed)
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
    not_exhausted_l= True
    
    errors_lanczos_whitening= []
    cond_lw= []
    R= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat= np.zeros(shape= (max_dim + 1, max_dim + 1))
    S_c= S_l @ c
    R[0, 0]= sclg.norm(S_c, ord= 2)
    sketch_Q= np.zeros(shape= (s, max_dim + 1))
    sketch_Q[:, 0]= (S_c)/R[0, 0]
    beta_0= R[0, 0]
    
    for k in range(1, max_dim + 1) :
        if not_exhausted_a:
            approx_a, not_exhausted_a= krylov_based_method.matrix_function_approximator_arnoldi(A, Q_a, H_a, norm_c, function, k)
            cond_a.append(np.linalg.cond(Q_a[:, :k]))
            errors_arnoldi.append(sclg.norm(true - approx_a))
        else:
            errors_arnoldi.append(errors_arnoldi[-1])
        
        if not_exhausted_ta:
            approx_t, not_exhausted_ta = krylov_based_method.matrix_function_approximator_trunc_arnold(A, Q_ta, H_ta, norm_c, function, k, k_t)
            cond_ta.append(np.linalg.cond(Q_ta[:, :k]))
            try:
                errors_trunc.append(sclg.norm(true - approx_t))
            except Exception as e:
                errors_trunc.append(1e200)
        else:
            errors_trunc.append(errors_trunc[-1])
        
        if not_exhausted_ta:
            approx_t_w = krylov_based_method.matrix_function_approximator_apply_whitening(sketch_Q_t, R_t, H_hat_t, Q_ta, H_ta, beta_0_t, function, S_t, k)
            Q_hat_temp= sclg.solve_triangular(R_t[:k, :k].T, Q_ta[:, :k].T, lower=True).T
            cond_taw.append(np.linalg.cond(Q_hat_temp))
            try:
                errors_trunc_whitening.append(sclg.norm(true - approx_t_w))
            except Exception as e:
                errors_trunc_whitening.append(1e200)
        else:
            errors_trunc_whitening.append(errors_trunc_whitening[-1])

        if not_exhausted_l:
            approx_l, not_exhausted_l, beta_old, sigma_old, q_old, w_old, A_v = krylov_based_method.matrix_function_approximator_lanczos(A, V_l, T_l, W_l, A_v, norm_c, beta_old, sigma_old, q_old, w_old, function, k)
            cond_l.append(np.linalg.cond(V_l[:, :k]))
            try:
                errors_lanczos.append(sclg.norm(true - approx_l))
            except Exception as e:
                errors_lanczos.append(1e200)
        else:
            errors_lanczos.append(errors_lanczos[-1])
            
        if not_exhausted_l:
            approx_l_w= krylov_based_method.matrix_function_approximator_apply_whitening(sketch_Q, R, H_hat, V_l, T_l, beta_0, function, S_l, k)
            Q_hat_temp2= sclg.solve_triangular(R[:k, :k].T, V_l[:, :k].T, lower=True).T
            cond_lw.append(np.linalg.cond(Q_hat_temp2))
            try:
                errors_lanczos_whitening.append(sclg.norm(true - approx_l_w))
            except Exception as e:
                errors_lanczos_whitening.append(1e200)
        else:
            errors_lanczos_whitening.append(errors_lanczos_whitening[-1])

        if k%100 == 0 :
            print(k)
    return errors_arnoldi, errors_trunc, errors_trunc_whitening, errors_lanczos, errors_lanczos_whitening, cond_a, cond_ta, cond_taw, cond_l, cond_lw

def comparing_sketching(A, c, function, true, max_dim, seed= None) :
    n = A.shape[1]
    norm_c = sclg.norm(c, ord=2)

    errors_arnoldi = []
    Q_a = np.zeros(shape=(n, max_dim+1))
    Q_a[:, 0] = c
    H_a= np.zeros(shape= (max_dim + 1, max_dim))
    not_exhausted_a= True
    
    errors_lanczos= []
    rng= np.random.default_rng(seed)
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
    not_exhausted_l= True
    
    S_g= sketching.GaussianSketch(n, 2*max_dim)
    errors_lanczos_whitening= []
    R= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat= np.zeros(shape= (max_dim + 1, max_dim + 1))
    S_c= S_g @ c
    R[0, 0]= sclg.norm(S_c, ord= 2)
    sketch_Q= np.zeros(shape= (2*max_dim, max_dim + 1))
    sketch_Q[:, 0]= (S_c)/R[0, 0]
    beta_0= R[0, 0]
    
    S_cos= sketching.SRDCT(n, 2*max_dim)
    errors_lanczos_whitening2= []
    R_cos= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat_cos= np.zeros(shape= (max_dim + 1, max_dim + 1))
    S_c_cos= S_cos @ c
    R_cos[0, 0]= sclg.norm(S_c_cos, ord= 2)
    sketch_Q_cos= np.zeros(shape= (2*max_dim, max_dim + 1))
    sketch_Q_cos[:, 0]= (S_c_cos)/R[0, 0]
    beta_0_cos= R_cos[0, 0]
    
    S_cos_big= sketching.SRDCT(n, 4*max_dim)
    errors_lanczos_whitening3= []
    R_big= np.zeros(shape= (max_dim + 1, max_dim + 1))
    H_hat_big= np.zeros(shape= (max_dim + 1, max_dim + 1))
    S_c_big= S_cos_big @ c
    R_big[0, 0]= sclg.norm(S_c_big, ord= 2)
    sketch_Q_big= np.zeros(shape= (4*max_dim, max_dim + 1))
    sketch_Q_big[:, 0]= (S_c_big)/R[0, 0]
    beta_0_big= R_big[0, 0]
    
    for k in range(1, max_dim + 1) :
        if not_exhausted_a:
            approx_a, not_exhausted_a= krylov_based_method.matrix_function_approximator_arnoldi(A, Q_a, H_a, norm_c, function, k)
            errors_arnoldi.append(sclg.norm(true - approx_a))
        else:
            errors_arnoldi.append(errors_arnoldi[-1])
            
        if not_exhausted_l:
            approx_l, not_exhausted_l, beta_old, sigma_old, q_old, w_old, A_v = krylov_based_method.matrix_function_approximator_lanczos(A, V_l, T_l, W_l, A_v, norm_c, beta_old, sigma_old, q_old, w_old, function, k)
            try:
                errors_lanczos.append(sclg.norm(true - approx_l))
            except Exception as e:
                errors_lanczos.append(1e200)
        else:
            errors_lanczos.append(errors_lanczos[-1])
            
        if not_exhausted_l:
            approx_l_w= krylov_based_method.matrix_function_approximator_apply_whitening(sketch_Q, R, H_hat, V_l, T_l, beta_0, function, S_g, k)
            try:
                errors_lanczos_whitening.append(sclg.norm(true - approx_l_w))
            except Exception as e:
                errors_lanczos_whitening.append(1e200)
        else:
            errors_lanczos_whitening.append(errors_lanczos_whitening[-1])
            
        if not_exhausted_l:
            approx_l_w = krylov_based_method.matrix_function_approximator_apply_whitening(sketch_Q_cos, R_cos, H_hat_cos, V_l, T_l, beta_0_cos, function, S_cos, k)
            try:
                errors_lanczos_whitening2.append(sclg.norm(true - approx_l_w))
            except Exception as e:
                errors_lanczos_whitening2.append(1e200)
        else:
            errors_lanczos_whitening2.append(errors_lanczos_whitening2[-1])

        if not_exhausted_l:
            approx_l_w = krylov_based_method.matrix_function_approximator_apply_whitening(sketch_Q_big, R_big, H_hat_big, V_l, T_l, beta_0_big, function, S_cos_big, k)
            try:
                errors_lanczos_whitening3.append(sclg.norm(true - approx_l_w))
            except Exception as e:
                errors_lanczos_whitening3.append(1e200)
        else:
            errors_lanczos_whitening3.append(errors_lanczos_whitening3[-1])
        
        if k%100 == 0:
            print(k)
    return errors_arnoldi, errors_lanczos, errors_lanczos_whitening, errors_lanczos_whitening2, errors_lanczos_whitening3

#%% Different matrix function to try
def negative_exp(A) :
    return sclg.expm(-A)

def sqrt(A) :
	return sclg.sqrtm(A)

def p_inv(A) :
    return np.linalg.pinv(A)

def log(A) :
    return sclg.logm(A, disp= False)[0]

def is_positive_definite(A):
    try:
        _ = sclg.cholesky(A, lower=True)
        return True
    except sclg.LinAlgError:
        return False
    
#%% Generating test matrix 
n= 1000

A= np.diag(np.linspace(1, 1000, n, endpoint=True))
rng= np.random.default_rng()
Random_, _= np.linalg.qr(rng.normal(size= (n, n)), mode= "complete")
cond_target_tweeker = 1e1
S = np.diag(np.logspace(0, np.log10(np.sqrt(cond_target_tweeker)), n, endpoint=True))
Random_ = Random_ @ S @ np.linalg.inv(Random_)
A= Random_ @ A @ np.linalg.inv(Random_)
print(f"Condition number of A : {np.linalg.cond(A):.4e}")
eigval= sclg.eigvals(A)
print(f"Max eigen of A : {eigval.max().real:.4e}")
print(f"Min eigen of A : {eigval.min().real:.4e}")
print(f"A symmetric : {np.allclose(A, A.T)}")
print(f"A positive definite : {is_positive_definite(A)}")

#%% Defining which matrix function to approximate and computing the "true" solution
c= np.ones(n)/np.sqrt(n)
function= negative_exp

t1= time.perf_counter()

if sp.sparse.issparse(A) :
	A_dense = A.toarray()
else:
	A_dense= A
true = function(A_dense) @ c

print("Time to compute exact solution : ", time.perf_counter() - t1)

#%% Comparing sketching
max= 200
err_a, err_l, err_l_gaus, err_l_SRDCT, err_l_SRDCT_big= comparing_sketching(A, c, function, true, max, seed= None)

#%% Plots
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig1= plt.figure(figsize= (8,6))
#plt.plot(range(1, max + 1), err_a, label= "Arnoldi", linewidth= 3)
#plt.plot(range(1, max + 1), err_l, label= "Lanczos", color= colors[3], linewidth= 3)
plt.plot(range(1, max + 1), err_l_SRDCT, label= r"SRDCT     ($s= 2k_{max}$)", color= colors[4], linewidth= 3)
plt.plot(range(1, max + 1), err_l_SRDCT_big, label= r"SRDCT     ($s= 4k_{max}$)", color= "darkgreen", linewidth= 3)
plt.plot(range(1, max + 1), err_l_gaus, label= r"Gaussian ($s= 2k_{max}$)", color= "purple", linewidth= 3)
plt.xlabel(r"$k$", fontsize= 20)
plt.ylabel(r"$||f(A)\cdot c - f_k||_2$", fontsize= 20)
plt.yscale("log")
plt.ylim(top= 4e-13, bottom= 3.2e-13)
plt.xlim(175, 200)
plt.grid()
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(loc= "lower left", framealpha= 1, fontsize= 18)
plt.tight_layout()
plt.savefig("Figures/conpa_sketch_zoom.eps", format= "eps")

#%% Running the code for convergence analysis
max= 100
s= 2*max
m= A.shape[0]
S_t= sketching.SRDCT(m, s, seed= None)
S_l= sketching.SRDCT(m, s, seed= None)
errors_arnoldi, errors_trunc, errors_trunc_whitening, errors_lanczos, errors_lanczos_whitening, cond_a, cond_ta, cond_taw, cond_l, cond_lw= comparing_methods(A, c, function, true, max, S_t, S_l)

#%% Plots
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig1= plt.figure(figsize= (8,6))
plt.plot(range(1, max + 1), errors_arnoldi, label= "Arnoldi", linewidth= 3)
plt.plot(range(1, max + 1), errors_lanczos, label= "Lanczos", color= colors[3], linewidth= 3)
plt.plot(range(1, max + 1), errors_lanczos_whitening, label= "Stabilized Lanczos", color= colors[4], linewidth= 3)
plt.xlabel(r"$k$", fontsize= 20)
plt.ylabel(r"$||f(A)\cdot c - f_k||_2$", fontsize= 20)
plt.yscale("log")
plt.ylim(top= 1e1, bottom= 0.5e-13)
plt.grid()
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(loc= "upper right", framealpha= 1, fontsize= 18)
plt.tight_layout()
#plt.savefig("Figures/conv_sqrt_simple_l.eps", format= "eps")

fig2= plt.figure(figsize= (8,6))
plt.plot(range(1, max + 1), errors_arnoldi, label= "Arnoldi", linewidth= 3)
plt.plot(range(1, max + 1), errors_trunc, label= "Trunc. Arnoldi", color= colors[1], linewidth= 3)
plt.plot(range(1, max + 1), errors_trunc_whitening, label= "Stabilized Trunc.", color= colors[2], linewidth= 3)
plt.xlabel(r"$k$", fontsize= 20)
plt.ylabel(r"$||f(A)\cdot c - f_k||_2$", fontsize= 20)
plt.yscale("log")
plt.ylim(top= 1e1, bottom= 0.5e-13)
plt.grid()
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(loc= "upper right", framealpha= 1, fontsize= 18)
plt.tight_layout()
#plt.savefig("Figures/conv_sqrt_simple_t.eps", format= "eps")

fig3= plt.figure(figsize= (8,6))
plt.plot(range(1, max + 1), cond_a, label= "Arnoldi", linewidth= 3)
plt.plot(range(1, max + 1), cond_l, label= "Lanczos", color= colors[3], linewidth= 3)
plt.plot(range(1, max + 1), cond_lw, label= "Stabilized Lanczos", color= colors[4], linewidth= 3)
plt.xlabel(r"$k$", fontsize= 20)
plt.ylabel(r"$\kappa(U_k)$", fontsize= 20)
plt.yscale("log")
plt.grid()
plt.ylim(top= 1e6, bottom= 0.5e0)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(loc= "upper left", framealpha= 1, fontsize= 18)
plt.tight_layout()
#plt.savefig("Figures/cond_sqrt_simple_l.eps", format= "eps")

fig4= plt.figure(figsize= (8,6))
plt.plot(range(1, max + 1), cond_a, label= "Arnoldi", linewidth= 3)
plt.plot(range(1, max + 1), cond_ta, label= "Trunc. Arnoldi", color= colors[1], linewidth= 3)
plt.plot(range(1, max + 1), cond_taw, label= "Stabilized trunc", color= colors[2], linewidth= 3)
plt.xlabel(r"$k$", fontsize= 20)
plt.ylabel(r"$\kappa(U_k)$", fontsize= 20)
plt.yscale("log")
plt.grid()
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(loc= "upper left", framealpha= 1, fontsize= 18)
plt.tight_layout()
#plt.savefig("Figures/cond_sqrt_simple_t.eps", format= "eps")

#%% Loading a sparse matrix from mtx file
folder= "Matrices/"
filename= "cavity26.mtx"

A= sp.io.mmread(folder + filename)
A= A.tocsr()
print(A.shape)
n= A.shape[1]

eigval_max= sp.sparse.linalg.eigs(A, k=2, which='LM', return_eigenvectors=False)
eigval_min= sp.sparse.linalg.eigs(A, k=2, which='SM', return_eigenvectors=False)

print(f"Max eigen of A : {eigval_max.max():.4e}")
print(f"Min eigen of A : {eigval_min.min():.4e}")
is_symmetric = (A != A.T).nnz == 0
print(f"A symmetric: {is_symmetric}")

#%% Computing True sol
c= np.ones(n)/np.sqrt(n)
function= negative_exp
average_param= 1

times= []
for i in range(average_param) :
    t1= time.perf_counter()
    if sp.sparse.issparse(A) :
        A_dense = A.toarray()
    else:
        A_dense= A
    true = function(A_dense) @ c
    times.append(time.perf_counter() - t1)
    print(i)

print("Time to compute exact solution : ", np.mean(times))

#%% Performance analysis
k_max= 450
num_checks= 10
tol= -1
k_t= 5

t1= time.perf_counter()
approx= krylov_based_method.full_approx_arnoldi(A, c, function, k_max, num_checks, tol)
print("Time to compute approx solution : ", time.perf_counter() - t1)
print("Error to true solution : ", sclg.norm(true-approx, ord=2))
print(" ")

t1= time.perf_counter()
approx= krylov_based_method.full_approx_trunc_arnoldi(A, c, function, k_max, num_checks, tol, k_t)
print("Time to compute approx solution : ", time.perf_counter() - t1)
print("Error to true solution : ", sclg.norm(true - approx, ord=2))
print(" ")

t1= time.perf_counter()
approx= krylov_based_method.full_approx_lanczos(A, c, function, k_max, num_checks, tol)
print("Time to compute approx solution : ", time.perf_counter() - t1)
print("Error to true solution : ", sclg.norm(true-approx, ord=2))
print(" ")

t1= time.perf_counter()
approx= krylov_based_method.full_approx_trunc_arnoldi_whitened(A, c, function, k_max, num_checks, tol, k_t)
print("Time to compute approx solution : ", time.perf_counter() - t1)
print("Error to true solution : ", sclg.norm(true-approx, ord=2))
print(" ")

t1= time.perf_counter()
approx= krylov_based_method.full_approx_lanczos_whitened(A, c, function, k_max, num_checks, tol)
print("Time to compute approx solution : ", time.perf_counter() - t1)
print("Error to true solution : ", sclg.norm(true-approx, ord=2))

#%% Scaling analysis
def true_sol(A, c, function) :
    if sp.sparse.issparse(A) :
        A_dense = A.toarray()
    else:
        A_dense= A
    true = function(A_dense) @ c
    return true

function= negative_exp
tol= -1
num_checks= 5
average_param= 5

methods= [krylov_based_method.full_approx_arnoldi, 
          krylov_based_method.full_approx_trunc_arnoldi, 
          krylov_based_method.full_approx_trunc_arnoldi_whitened, 
          krylov_based_method.full_approx_lanczos,
          krylov_based_method.full_approx_lanczos_whitened]
k_maxs= [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
times= []
for method in methods:
    time_method= []
    for k_max in k_maxs :
        times_temp= []
        for i in range(average_param) :
            t1= time.perf_counter()
            x= method(A, c, function, k_max, num_checks, tol)
            times_temp.append(time.perf_counter() - t1)
        time_method.append(float(np.mean(times_temp)))
    times.append(time_method)
    print(times)

#%% Plots
k_maxs_squared= [4.4e-5*num*num for num in k_maxs]
labels= ["Arnoldi", "Trunc. Arnoldi", "Stabilized Trunc.", "Lanczos", "Stabilized Lanczos"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, time_method in enumerate(times) :
    plt.plot(k_maxs, time_method, label= labels[i], linestyle= "-", linewidth= 3, marker= ".", markersize= 18, markeredgewidth= 3)
plt.xlabel(r"$k_{max}$", fontsize= 16)
plt.ylabel(r"$T(k_{max})$ [s]", fontsize= 16)
ax.set_xticks(k_maxs)
ax.tick_params(axis='both', which='major', labelsize=14) 
plt.grid()
plt.legend(loc= "upper left", framealpha= 1, fontsize= 16)
fig.tight_layout()
plt.savefig("Figures/time_scaling_k.eps", bbox_inches='tight', format= 'eps')
