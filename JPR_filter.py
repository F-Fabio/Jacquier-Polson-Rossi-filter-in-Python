
"""
JPR filter in Python. Fabio Franco
"""

def get_SV(hlast, g, mubar, sigmabar, errors):
    
    ''' JPR: nonlinear simulator which approximates 
    the latent state of a stochastic volatility model.
    
    Inputs:
    
    1. hlast - volatility measure
    2. g - variance of the SV latent process
    3. mubar - mean of log(h0)
    4. sigmabar - variance 
    5. errors - Data '''
    
    T=errors.shape[0]
    hnew = np.empty((T+1,1))
    
    # Initial condition
    
    i=0 # Time period 0
    
    hlead=hlast[i+1]
    ss=sigmabar*g/(g+sigmabar)                                       
    mu=ss*((mubar/sigmabar)+(math.log(hlead)/g))                     
    # Draw from lognormal  using mu and ss
    h=math.exp(mu+((ss**.5)*np.random.normal(0, 1, 1)))
    hnew[i]=h;
    
    # Recursion: time period from 1 to t-1
    for i in range(1,T):
        hlead=hlast[i+1]
        hlag=hnew[i-1]
        yt=errors[i-1]
        
        # Mean and variance of the proposal log normal density
        mu=(np.log(hlead)+np.log(hlag))/2
        ss=g/2
        
        # Candidate draw from lognormal
        htrial=np.exp(mu+(ss**.5)*np.random.normal(0,1,1))
        
        # Acceptance probability 
        lp1=-.5*np.log(htrial)-(yt**2)/(2*htrial)                   
        lp0=-.5*np.log(hlast[i])-(yt**2)/(2*hlast[i])               
        accept=min(np.array([1,math.exp(lp1-lp0)]))                 
        
        u=np.random.rand(1,1)
        if u<=accept:
            h=htrial
        else:
            h=hlast[i]
        
        hnew[i]=h
    
    # Time period T
    i=T
    yt=errors[i-1]
    hlag=hnew[i-1]
    # Mean and variance of the proposal density
    mu=np.log(hlag) 
    ss=g
    # Candidate draw from lognormal
    htrial=math.exp(mu+(ss**.5)*np.random.normal(0,1,1))
    # Acceptance probability
    lp1=-.5*np.log(htrial)-(yt**2)/(2*htrial)
    lp0=-.5*np.log(hlast[i])-(yt**2)/(2*hlast[i])
    accept=min(np.array([1,math.exp(lp1-lp0)]))                     

    u=np.random.rand(1,1)
    if u<=accept:
        h=htrial
    else:
        h=hlast[i]
    
    hnew[i]=h
    
    return hnew