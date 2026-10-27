"""Analytical check of LRPO Eq.5; not an LRPO implementation reproduction.
Run with matplotlib installed. Outputs synthetic examples, never model scores.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
def reward(ordered, shuffled, mu=.8, correct=True):
    delta = ordered - mu * shuffled
    gate = min(1., max(0., (abs(delta) - .05) / .15))
    return gate * max(0., delta) if correct else 0.

def main():
    # Independently calculated counterexamples and correctness assignment check.
    assert abs(reward(.8, .8) - .1173333333333333) < 1e-12
    assert abs(reward(.7, .8) - .004) < 1e-12
    assert reward(.8, .8, correct=False) == 0
    assert reward(.8, .8, mu=1.) == 0
    assert reward(.7, .8, mu=1.) == 0
    values = [i/100 for i in range(101)]
    rows = [{'ordered': a, 'shuffled': b, 'published_equation': reward(a,b),
             'strict_outperformance_gate': reward(a,b) if a>b else 0.,
             'unscaled_difference_diagnostic': reward(a,b,mu=1.)}
            for a,b in [(.5,.5),(.7,.7),(.8,.8),(1.,1.),(.7,.8),(.8,.7)]]
    result = {'kind':'synthetic analytical diagnostic, no inference',
              'source':'https://arxiv.org/html/2607.00654v1',
              'location':'Section 3.4 Eq.5; Appendix C',
              'parameters':{'mu':.8,'epsilon0':.05,'epsilon1':.2},
              'assumption':'Output is correct under ordered input; group accuracy rewards are supplied analytically.',
              'caveat':'An additional ordered>shuffled implementation gate would remove these counterexamples; code not inspected.',
              'examples':rows}
    (ROOT/'evidence/temporal_reward_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    axes[0].plot(values,[reward(a,a) for a in values],label='Published equation')
    axes[0].plot(values,[0.]*len(values),'--',label='Strict outperformance gate')
    axes[0].set(xlabel='Equal ordered and shuffled accuracy reward',ylabel='Temporal reward',title='No order advantage')
    axes[0].legend(fontsize=8)
    matrix=[[reward(a,b) for b in values] for a in values]
    im=axes[1].imshow(matrix,origin='lower',extent=[0,1,0,1],aspect='auto',vmin=0,vmax=1)
    axes[1].plot([0,1],[0,1],'w--',lw=1)
    axes[1].set(xlabel='Shuffled accuracy reward',ylabel='Ordered accuracy reward',title='Published equation; correct output')
    fig.colorbar(im,ax=axes[1],label='Temporal reward')
    for ext in ('pdf','png'):
        fig.savefig(ROOT/f'figures/temporal_reward_audit.{ext}',dpi=180)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
