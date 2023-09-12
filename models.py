import numpy as np

class BoundedRankModule():
    
    def __init__(self, k, n, m, train_idxs=None):
        if train_idxs is None:
            train_idxs = np.arange(m)
        self.k = k
        self.train_idxs = train_idxs
        self.A = np.zeros((n,k))
        self.B = np.zeros((k,m))
        self.D = np.zeros((k))
    
    def fit(self, x, y):
        diff = y[:,self.train_idxs] - x[:,self.train_idxs]
        u, s, vh = np.linalg.svd(diff)
        self.A = u[:,:self.k]
        self.B[:,self.train_idxs] = vh[:self.k,:]
        self.D = s[:self.k]

    def interpolate_B(self):
        end_idxs = np.where(np.diff(self.train_idxs)!=0)[0] # Identify blocks left out of training
        end_trains = self.train_idxs[end_idxs]
        start_trains = self.train_idxs[end_idxs+1]

        for i in range(len(end_trains)):
            end_i, start_i = end_trains[i], start_trains[i]
            l = start_i - end_i
            test_block = np.arange(end_i,start_i+1)
            # Linearly interpolate
            self.B[:,test_block] = np.outer(self.B[:,start_i]-self.B[:,end_i],(test_block-end_i)/l) + self.B[:,end_i][:,np.newaxis]

    def predict(self, x, ks, test_idxs = None):
        if test_idxs is None:
            test_idxs = np.arange(self.B.shape[1])
        
        if len(self.train_idxs) != self.B.shape[1]:
            self.interpolate_B()
        
        result = np.zeros((self.A.shape[0],test_idxs.size,ks.size))
        for k in range(len(ks)):
            result[:,:,k] = self.A[:,:ks[k]] @ np.diag(self.D[:ks[k]]) @ self.B[:ks[k],test_idxs] + x[:,test_idxs]
        return result


class BasisSplineModule():
    
    def __init__(self, k, n, m, knots=None, knot_interval=10, deg=3, train_idxs=None):
        if knots is None:
            knots = np.arange(-deg*knot_interval, m+((deg+1)*knot_interval)+1, knot_interval)
        if train_idxs is None:
            train_idxs = np.arange(m)
        self.train_idxs = train_idxs
        self.k = k
        self.A = np.zeros((n,k))
        self.D = np.zeros((k))
        
        self.spline_coef = interpolate.BSpline.design_matrix(np.arange(m),knots,deg).todense()
        s_deg = self.spline_coef.shape[1]
        self.B = np.zeros((k,s_deg))
        assert k <= s_deg, "k higher than spline coef matrix"
    
    def fit(self, x, y):
        diff = y[:,self.train_idxs] - x[:,self.train_idxs]
        sc = self.spline_coef[self.train_idxs,:]
        b_ols = np.dot(np.linalg.inv(np.dot(sc.T,sc)),np.dot(sc.T,diff.T)).T
        diff_hat = np.dot(b_ols,sc.T)
        
        u, s, vh = np.linalg.svd(diff_hat)
        diff_hat_rankk = u[:,:self.k] @ np.diag(s[:self.k]) @ vh[:self.k,:]
        sc = self.spline_coef[self.train_idxs,:]
        spline_inv = np.linalg.pinv(sc)
        assert np.allclose(np.dot(spline_inv,sc), np.eye(sc.shape[1])) # Left inverse
        b_rrr = np.dot(spline_inv,diff_hat_rankk.T).T
        
        u, s, vh = np.linalg.svd(b_rrr)
        self.A = u[:,:self.k]
        self.B = vh[:self.k,:]
        self.D = s[:self.k]
        assert np.allclose(b_rrr, self.A @ np.diag(self.D) @ self.B) # b_rrr is rank <= k
    
    def predict(self, x, ks, test_idxs = None):
        if test_idxs is None:
            test_idxs = np.arange(self.spline_coef.shape[0])
        
        result = np.zeros((self.A.shape[0],test_idxs.size,ks.size))
        for k in range(len(ks)):
            result[:,:,k] = self.A[:,:ks[k]] @ np.diag(self.D[:ks[k]]) @ self.B[:ks[k],:] @ self.spline_coef.T[:,test_idxs] + x[:,test_idxs]
        return result


class FusedLassoModule():
    
    def __init__(self, k, n, m, lmbda=1, train_idxs=None, admm_params={}):
        self.k = k
        self.n = n
        self.m = m
        self.lmbda = lmbda
        if train_idxs is None:
            self.train_idxs = np.arange(m)
        else:
            self.train_idxs = train_idxs # req: sorted
        self.admm_params = {}
        self.admm_params["niter"] = admm_params.get("niter", 5000)
        self.admm_params["noter"] = admm_params.get("noter", 10)
        self.admm_params["rthres"] = admm_params.get("rthres", 1e-4)
        self.admm_params["sthres"] = admm_params.get("sthres", 1e-4)
        self.admm_params["rho0"] = admm_params.get("rho0", self.lmbda)
        self.admm_params["outthres"] = admm_params.get("outthres",1e-3)        
        
        self.A = np.zeros((n,k))
        self.B = np.zeros((k,m))
        self.D = np.zeros((k))
    
    def fit(self, x, y):

    	def sthresh(x, l):
    		return np.where(x > l, x-l, np.where(x < -l, x+l, 0))
        
        diff = y[:,self.train_idxs] - x[:,self.train_idxs]
        n, m = self.n, self.train_idxs.size
        D = np.zeros((m-1,m))
        for i in range(m-1):
            if self.train_idxs[i+1] - self.train_idxs[i] == 1:
                D[i,[i,i+1]] = [1,-1]
        D = D[np.sum(D!=0,axis=1) == 2,:]
        
        
        Dt = D.T
        DtD = np.dot(Dt,D)
        DtDeigL, DtDeigQ = np.linalg.eig(DtD)
        DtDeigQinv = np.linalg.inv(DtDeigQ)
        I = np.eye(m)
        
        niter = self.admm_params["niter"]
        noter = self.admm_params["noter"]

        tmp = diff.copy()

        for k in range(self.k):
            print("Start rank %d"%k)
            v = np.random.random(m)
            v = v/np.linalg.norm(v)

            vs = np.zeros((m,noter+1))
            vs[:,0] = v

            for i in range(noter):
                u = np.dot(tmp,vs[:,i])
                u = u/np.linalg.norm(u)
                u = u*np.sign(u[0])

                b = np.dot(u,tmp)

                xs = np.zeros((m,niter+1))
                zs = np.zeros((D.shape[0],niter+1))
                us = np.zeros((D.shape[0],niter+1))
                rhos = np.zeros((niter+1))
                rs = np.zeros((niter))
                ss = np.zeros((niter))

                xs[:,0] = v
                zs[:,0] = np.dot(D,v)
                rhos[0] = self.admm_params["rho0"]

                for j in range(niter):
                    #inv_mat = DtDeigQ @ ((1/(rhos[j]*DtDeigL+1))[:,np.newaxis] * DtDeigQinv) # slower than np inv
                    xs[:,j+1] = np.linalg.inv(I + rhos[j]*DtD) @ (b + rhos[j]*np.dot(Dt,(zs[:,j] - us[:,j])))
                    
                    Dx = np.dot(D,xs[:,j+1])
                    zs[:,j+1] = sthresh(Dx + us[:,j], self.lmbda/rhos[j])
                    us[:,j+1] = us[:,j] + Dx - zs[:,j+1]

                    r = np.linalg.norm(Dx - zs[:,j+1])
                    s = np.linalg.norm(rhos[j]*np.dot(Dt,zs[:,j+1] - zs[:,j]))
                    rs[j] = r
                    ss[j] = s
                    if s > 0 and r > 0:
                        rhos[j+1] = np.where(r > 100*s, 2*rhos[j], np.where(s > 100*r, 0.5*rhos[j], rhos[j]))
                    else:
                        rhos[j+1] = rhos[j]
                    if r < self.admm_params["rthres"] and s < self.admm_params["sthres"]:
                        print("Converged at inner iter %d"%j)
                        vs[:,i+1] = xs[:,j+1]
                        break
                    elif j == niter-1:
                        vs[:,i+1] = xs[:,-1]
                vs[:,i+1] = vs[:,i+1]/np.linalg.norm(vs[:,i+1])
                print(np.linalg.norm(vs[:,i+1]-vs[:,i]))
                if np.linalg.norm(vs[:,i+1]-vs[:,i]) < self.admm_params["outthres"]:
                    print("Converged at outer iter %d"%i)
                    d = np.dot(np.dot(u,tmp),vs[:,i+1])
                    self.A[:,k] = u
                    self.B[k,self.train_idxs] = vs[:,i+1]
                    self.D[k] = d
                    break
                elif i == noter-1:
                    d = np.dot(np.dot(u,tmp),vs[:,i+1])
                    self.A[:,k] = u
                    self.B[k,self.train_idxs] = vs[:,i+1]
                    self.D[k] = d
            tmp = tmp - self.D[k]*np.outer(self.A[:,k],self.B[k,self.train_idxs])
    
    def pred(self, x):
        return self.A @ np.diag(self.D) @ self.B + x