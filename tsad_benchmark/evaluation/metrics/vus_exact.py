import numpy as np
from numba import njit


@njit(cache=True)
def integrate(y,bins,pred_counts,starts,ends,max_window):
    n=len(y);positive=np.sum(y);mean_roc=0.;mean_pr=0.
    for window in range(max_window+1):
        weights=np.zeros(250);hits=np.zeros(250);mass=0.;segments=0;minimum=250
        half=window//2
        for j in range(n):
            value=float(y[j])
            if value==0. and window>0:
                for k in range(len(starts)):
                    if j<starts[k] and starts[k]-j<=half:
                        value+=np.sqrt(1.-(starts[k]-j)/window)
                    if j>ends[k] and j-ends[k]<half:
                        value+=np.sqrt(1.-(j-ends[k])/window)
                value=min(value,1.)
            mass+=value;weights[bins[j]]+=value
            if value>0.:
                minimum=min(minimum,bins[j])
            elif minimum<250:
                hits[minimum]+=1;segments+=1;minimum=250
        if minimum<250:
            hits[minimum]+=1;segments+=1
        pnew=(positive+mass)/2.;nnew=n-pnew
        tp=0.;exist=0.;last_tpr=0.;last_fpr=0.;last_prec=1.;roc=0.;pr=0.
        for k in range(250):
            tp+=weights[k];exist+=hits[k]
            tpr=min(tp/pnew,1.)*exist/segments
            fpr=(pred_counts[k]-tp)/nnew
            prec=tp/pred_counts[k]
            roc+=(fpr-last_fpr)*(tpr+last_tpr)/2.
            pr+=(tpr-last_tpr)*(prec+last_prec)/2.
            last_tpr=tpr;last_fpr=fpr;last_prec=prec
        roc+=(1.-last_fpr)*(1.+last_tpr)/2.
        mean_roc+=roc;mean_pr+=pr
    return mean_roc/(max_window+1),mean_pr/(max_window+1)


def exact_vus(y,s,max_window=None):
    if int(y.sum())==0:return np.nan,0.
    if int(y.sum())==len(y):return np.nan,1.
    diff=np.diff(np.r_[0,y,0]);starts=np.flatnonzero(diff==1);ends=np.flatnonzero(diff==-1)-1
    if max_window is None:max_window=2*max(1,int(np.median(ends-starts+1)))
    thresholds=-np.sort(-s)[np.linspace(0,len(s)-1,250).astype(int)]
    bins=np.searchsorted(-thresholds,-s,side='left')
    counts=np.cumsum(np.bincount(bins,minlength=250))
    return integrate(y,bins,counts,starts,ends,max_window)
