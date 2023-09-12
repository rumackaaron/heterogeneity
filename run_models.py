all_df = pd.read_csv("state_df.csv",index_col=False,parse_dates=["time_value"])
all_df = all_df.loc[~np.isin(all_df.geo_value,["as","gu","mp","pr","vi"]),:]
grouped_df = all_df.loc.groupby(["geo_value"])
states = np.array(list(grouped_df.groups.keys()))
def st2idx(sts):
    if not isinstance(sts,list):
        sts = [sts]
    return np.array([np.where(states==st)[0][0] for st in sts])
dv_arr = np.stack(all_df.groupby(["geo_value"]).dv.apply(np.array).to_numpy())
cases_arr = np.stack(all_df.groupby(["geo_value"]).cases.apply(np.array).to_numpy())

all2_df = pd.read_csv("fb_google.csv",index_col=False,parse_dates=["time_value"])
fb_arr = np.stack(all2_df.groupby(["geo_value"]).fb.apply(np.array).to_numpy())
google_arr = np.stack(all2_df.groupby(["geo_value"]).google.apply(np.array).to_numpy())

n,m = dv_arr.shape
log_cases_arr = np.log(np.maximum(cases_arr,0)+1)
x_data = np.log(dv_arr+1)
y_data = log_cases_arr

def interpolate_B(model, train_idxs):
    end_idxs = np.where(np.diff(train_idxs)!=0)[0] # Identify blocks left out of training
    end_trains = train_idxs[end_idxs]
    start_trains = train_idxs[end_idxs+1]
    
    for i in range(len(end_trains)):
        end_i, start_i = end_trains[i], start_trains[i]
        l = start_i - end_i
        test_block = np.arange(end_i,start_i+1)
        # Linearly interpolate
        model.B[:,test_block] = np.outer(model.B[:,start_i]-model.B[:,end_i],(test_block-end_i)/l) + model.B[:,end_i][:,np.newaxis]


# CV for Bounded Rank Model
nfolds = 6
K = 51
br_errs = np.zeros((K,m,nfolds))

for l in range(nfolds):
    test_idxs = np.where((np.arange(1,m-1)//10)%nfolds == l)[0]+1
    border_idxs = set().union(*np.array([idx+np.arange(-5,6) for idx in test_idxs]))
    border_idxs.discard(0)
    border_idxs.discard(m-1)
    train_idxs = np.arange(m)[~np.isin(np.arange(m),np.array(list(border_idxs)))]
    x_train = x_data[:,train_idxs]
    x_test = x_data[:,test_idxs]
    y_train = y_data[:,train_idxs]
    y_test = y_data[:,test_idxs]

    model = BoundedRankModule(K,n=x_data.shape[0],m=x_data.shape[1],train_idxs=train_idxs)
    model.fit(x_data,y_data)
    interpolate_B(model, train_idxs)

    for k in range(1,K+1):
        br_errs[k-1,test_idxs,l] = np.mean(np.power((model.A[:,:k] @ np.diag(model.D[:k]) @ model.B[:k,test_idxs]) + x_data[:,test_idxs] - y_data[:,test_idxs],2),axis=0)


# CV for Basis Spline Model
kis = [10,20,30,40,50,60]
bs_mses = np.zeros((len(kis),n,m,n,nfolds))
    
for l in range(nfolds):
    test_idxs = np.where((np.arange(1,m-1)//10)%nfolds == l)[0]+1
    border_idxs = set().union(*np.array([idx+np.arange(-5,6) for idx in test_idxs]))
    border_idxs.discard(0)
    border_idxs.discard(m-1)
    train_idxs = np.arange(m)[~np.isin(np.arange(m),np.array(list(border_idxs)))]
    x_train = x_data[:,train_idxs]
    x_test = x_data[:,test_idxs]
    y_train = y_data[:,train_idxs]
    y_test = y_data[:,test_idxs]
    for ki in range(len(kis)):
        model = BasisSplineModule(1,n=x_data.shape[0],m=x_data.shape[1],train_idxs=train_idxs,knot_interval=kis[ki])
        rank = min(model.spline_coef.shape[1],K)
        model = BasisSplineModule(rank,n=x_data.shape[0],m=x_data.shape[1],train_idxs=train_idxs,knot_interval=kis[ki])
        model.fit(x_data,y_data)
        model.B = model.B @ model.spline_coef.T
        interpolate_B(model, train_idxs)
        
        for k in range(1,rank+1):
            preds = model.A[:,:k] @ np.diag(model.D[:k]) @ model.B[:k,test_idxs]
            errs = np.power(preds - (y_data - x_data)[:,test_idxs],2)
            bs_mses[ki,:,test_idxs,k-1,l] = errs.T


# CV for Fused Lasso Model
fl_mses = np.zeros((len(lmbdas),n,m,n,nfolds))
for k in range(nfolds):
    test_idxs = np.where((np.arange(1,m-1)//10)%nfolds == k)[0]+1
    border_idxs = set().union(*np.array([idx+np.arange(-5,6) for idx in test_idxs]))
    border_idxs.discard(0)
    border_idxs.discard(m-1)
    train_idxs = np.arange(m)[~np.isin(np.arange(m),np.array(list(border_idxs)))]
    
    for l in range(len(lmbdas)):
        model = FusedLassoModule(K,n=x_data.shape[0],m=x_data.shape[1],lmbda=lmbdas[l],train_idxs=train_idxs)
    	model.fit(x_data,y_data)
        model.interpolate_B()

        for i in range(1,n+1):
            preds = model.A[:,:i] @ np.diag(model.D[:i]) @ model.B[:i,test_idxs]
            errs = np.power(preds - (y_data - x_data)[:,test_idxs],2)
            fl_mses[l,:,test_idxs,i-1,k] = errs.T