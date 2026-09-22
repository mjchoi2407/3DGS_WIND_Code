"""GPU Gauss 비교의 국소 끝 속도 차이/풀이+검산 비용을 정적 그림으로 저장."""
import os,json,argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/wind-gauss-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
from matplotlib import font_manager
font_manager.fontManager.addfont(font.get_file())
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
labels={'baseline':'기존 dt','newmark32':'Newmark 32','newmark64':'Newmark 64','newmark128':'Newmark 128','newmark256':'Newmark 256',
        'gauss_cpu8':'CPU Gauss 8','gauss_gpu8_rebuild1_v4':'GPU Gauss 8 · 매번 갱신','gauss_gpu8_reuse64':'GPU Gauss 8 · 재사용','gauss_gpu16_reuse64':'GPU Gauss 16 · 재사용'}
for ax,root,title in zip(axes,[a.root,a.root/'failure_point'],['2.0초 시작 상태','실패 시도 직전 상태']):
    d=json.loads((root/'comparison.json').read_text());rows={r['name']:r for r in d['rows'] if r['status']=='passed'}
    ref_gap=100*rows['newmark256']['difference_to_reference']['velocity_relative_mass']
    ax.axhline(ref_gap,color='#777777',linestyle='--',linewidth=1)
    for name,r in rows.items():
        if name not in labels or 'solve_plus_gpu_audit_s' not in r:continue
        x=r['solve_plus_gpu_audit_s'];y=100*r['difference_to_reference']['velocity_relative_mass']
        color='#16805d' if 'gauss_gpu' in name else ('#dc8b24' if 'cpu' in name else '#3375b5')
        ax.scatter(x,y,s=55,c=color,zorder=3)
        offset=(5,6)
        if name=='newmark64':offset=(5,-14)
        if name=='gauss_gpu8_rebuild1_v4':offset=(5,8)
        ax.annotate(labels[name],(x,y),xytext=offset,textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.15)
    ax.set_title(title);ax.set_xlabel('풀이 + GPU 검산 합산 시간 (초)')
    ax.set_ylabel('Newmark 512 참조 대비 끝 속도 차이 (%)')
    ax.set_xlim(.85,25);ax.set_ylim(.003,85)
fig.suptitle('각 1/3840초 구간의 비용·정확도 비교 — 전체 시뮬레이션 결과 아님',fontsize=13)
fig.text(.5,-.025,'점선: Newmark 256↔512 참조 세분 차이. 참해 오차 상한이 아니며, 시간은 별도 측정 구간의 합입니다.',ha='center',fontsize=9)
fig.savefig(a.root/'cost_accuracy.png',dpi=180,bbox_inches='tight')
fig.savefig(a.root/'cost_accuracy.pdf',bbox_inches='tight')
