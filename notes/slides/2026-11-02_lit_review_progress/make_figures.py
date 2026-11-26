"""Draw the figures for the Literature Review and Progress deck (2 Nov 2026).

All four figures are plain matplotlib drawings: hand-placed shapes and text, no data files.
  orchard.png           sketch of Commons Harvest: 7 agents, apple patches, one stripped patch
  spatial_overseer.png  the same sketch with the overseer checking 2 of the 7 agents
  my_game_round.png     one round of my lake game, as five boxes
  lit_map.png           the papers I read, scored on the five questions that define the gap

The orchard is an illustration, not the real SocialJax map: positions are chosen by hand.

Usage:  python make_figures.py [out_dir]     (default: ./figures)
"""
import sys, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)

BG="#F5F4F2"; NAVY="#062746"; TAN="#B09E88"; INK="#1C191A"; MUTED="#55524E"; APPLE="#3F7A4A"; BARE="#D9D3CA"; GRID="#E6E2DC"
AG="#B5523B"

def orchard(ax, overseer=False):
    W,H=16,11
    for x in range(W+1): ax.plot([x,x],[0,H],color=GRID,lw=0.8,zorder=0)
    for y in range(H+1): ax.plot([0,W],[y,y],color=GRID,lw=0.8,zorder=0)
    patches=[(3,7),(8,8),(12,6),(5,2),(11,2)]
    shapes=[(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]
    for k,(cx,cy) in enumerate(patches):
        stripped = (k==3)
        for dx,dy in shapes:
            if stripped and (dx,dy)!=(0,0) and (dx+dy)%2==0: pass
            ax.add_patch(Rectangle((cx+dx,cy+dy),1,1,fc=BARE if stripped else "#DDE8DA",ec="none",zorder=1))
            if not stripped:
                ax.add_patch(Circle((cx+dx+0.5,cy+dy+0.5),0.28,fc=APPLE,ec="none",zorder=2))
    ax.text(5.5,0.35,"stripped patch: never grows back",ha="center",va="bottom",fontsize=11,color=MUTED,style="italic",zorder=5)
    agents=[(2,8),(6,6),(9,9),(13,7),(4,3),(10,4),(14,3)]
    checked={1,5}
    for i,(x,y) in enumerate(agents):
        c=Circle((x+0.5,y+0.5),0.42,fc=AG,ec="white",lw=1.5,zorder=4)
        ax.add_patch(c)
        ax.text(x+0.5,y+0.5,str(i+1),ha="center",va="center",color="white",fontsize=11,fontweight="bold",zorder=5)
        if overseer and i in checked:
            ax.add_patch(Circle((x+0.5,y+0.5),0.85,fc="none",ec=NAVY,lw=3,zorder=6))
    if overseer:
        x,y=agents[1]
        ax.add_patch(Rectangle((x+0.5-2.5,y+0.5-2.5),5,5,fc="none",ec=TAN,lw=2,ls="--",zorder=3))
    ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect("equal"); ax.axis("off")

def save(fig,name):
    fig.patch.set_facecolor(BG); fig.savefig(f"{out}/{name}",bbox_inches="tight",facecolor=BG,pad_inches=0.12,dpi=200); plt.close(fig)

# A: orchard overview
fig,ax=plt.subplots(figsize=(8,5.8)); orchard(ax,False)
ax.text(8,11.35,"Commons Harvest: 7 agents share an orchard",ha="center",va="bottom",fontsize=14,color=NAVY,fontfamily="Georgia",fontweight="bold")
save(fig,"orchard.png")

# B: spatial overseer
fig,ax=plt.subplots(figsize=(8,6.3)); orchard(ax,True)
ax.text(8,11.35,"The overseer can check only 2 of the 7 agents this step",ha="center",va="bottom",fontsize=14,color=NAVY,fontfamily="Georgia",fontweight="bold")
ax.add_patch(Circle((0.6,-0.9),0.3,fc="none",ec=NAVY,lw=3,clip_on=False)); ax.text(1.1,-0.9,"checked this step",va="center",fontsize=11,color=INK)
ax.add_patch(Rectangle((5.2,-1.2),0.6,0.6,fc="none",ec=TAN,lw=2,ls="--",clip_on=False)); ax.text(6.0,-0.9,"what one agent can see",va="center",fontsize=11,color=INK)
ax.add_patch(Circle((12.4,-0.9),0.25,fc=APPLE,ec="none",clip_on=False)); ax.text(12.8,-0.9,"apple",va="center",fontsize=11,color=INK)
ax.set_ylim(-1.6,11)
save(fig,"spatial_overseer.png")

# C: my game, one round
fig,ax=plt.subplots(figsize=(14,4.4)); ax.axis("off")
steps=[("1. Ask","Each of 6 agents\nasks for a share\nof the lake"),
       ("2. Allowance","The overseer gives\neach agent an\nallowance that keeps\nthe lake healthy"),
       ("3. Fish","Agents fish. A\ncheater takes more\nthan its allowance"),
       ("4. Spot-check","The overseer checks\na few agents at\nrandom, not everyone"),
       ("5. Consequence","A caught cheater\npays a penalty, or\ngets a smaller share\nin later rounds")]
bw,bh,gap=2.75,2.2,0.4
for i,(h,t) in enumerate(steps):
    x=i*(bw+gap); hi = i==3
    ax.add_patch(FancyBboxPatch((x,0),bw,bh,boxstyle="round,pad=0,rounding_size=0.08",fc=NAVY if hi else "white",ec=NAVY if hi else "#CFC9C1",lw=1.2))
    ax.text(x+bw/2,bh-0.32,h,ha="center",va="center",fontsize=14,fontweight="bold",color="white" if hi else NAVY,fontfamily="Georgia")
    ax.text(x+bw/2,(bh-0.6)/2,t,ha="center",va="center",fontsize=11.5,color="white" if hi else INK,linespacing=1.35)
    if i<4: ax.annotate("",xy=(x+bw+gap-0.04,bh/2),xytext=(x+bw+0.04,bh/2),arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=1.6))
end=4*(bw+gap)+bw
yl=-0.45
ax.plot([end-bw/2,end-bw/2,bw/2],[0,yl,yl],color=MUTED,lw=1.6)
ax.annotate("",xy=(bw/2,-0.02),xytext=(bw/2,yl),arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=1.6))
ax.text(end/2,yl-0.28,"the lake regrows, and the next round starts",ha="center",va="center",fontsize=12,color=MUTED,style="italic")
ax.set_xlim(-0.1,end+0.1); ax.set_ylim(yl-0.6,bh+0.1)
save(fig,"my_game_round.png")

# D: the literature map
cols = ["Learning agents\nin a social dilemma", "Has an\nintervener", "Intervener sees\nonly some agents", "Chooses whom\nto check", "Agents learn\nto evade it"]
rows = [
 ("Leibo 2017 · SSDs",                 ["yes","no","–","–","–"]),
 ("SocialJax 2026",                    ["yes","no","–","–","–"]),
 ("OpenSocialJax 2026 · LLM agents",   ["yes","no","–","–","–"]),
 ("Baumann 2020 · learning planner",   ["partly","yes","no","no","no"]),
 ("Sun 2019 · detector + boycott",     ["partly","yes","no","no","no"]),
 ("Gasztowtt 2024 · LLM tax-setter",   ["yes","yes","no","no","no"]),
 ("Casheekar 2026 · random audits",    ["partly","yes","yes","partly","no"]),
 ("My overseer",                       ["yes","yes","yes","yes","yes"]),
]
fill = {"yes":"#062746","partly":"#B09E88","no":"#DAD5CE","–":"#F5F4F2"}
ink  = {"yes":"#FFFFFF","partly":"#1C191A","no":"#55524E","–":"#8F8F8F"}
label= {"yes":"yes","partly":"partly","no":"no","–":"–"}
W, H = 1.8, 0.62
fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=200)
fig.patch.set_facecolor("#F5F4F2"); ax.set_facecolor("#F5F4F2")
x0 = 3.6
for j,c in enumerate(cols):
    ax.text(x0+j*W+W/2, len(rows)*H+0.12, c, ha="center", va="bottom", fontsize=11.5, color="#1C191A", fontfamily="Georgia", fontweight="bold")
for i,(name,vals) in enumerate(rows):
    y = (len(rows)-1-i)*H
    if name=="My overseer":
        y -= 0.18
        ax.plot([0.05, x0+len(cols)*W], [y+H+0.09]*2, color="#131B28", lw=2.2)
    ax.text(x0-0.15, y+H/2, name, ha="right", va="center", fontsize=12,
            color="#062746" if name=="My overseer" else "#1C191A",
            fontweight="bold" if name=="My overseer" else "normal", )
    for j,v in enumerate(vals):
        ax.add_patch(FancyBboxPatch((x0+j*W+0.05, y+0.05), W-0.1, H-0.1, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fill[v], ec="none"))
        ax.text(x0+j*W+W/2, y+H/2, label[v], ha="center", va="center", fontsize=12, color=ink[v],
                fontweight="bold" if v=="yes" else "normal")
ax.set_xlim(0, x0+len(cols)*W+0.05); ax.set_ylim(-0.25, len(rows)*H+1.0)
ax.axis("off")
plt.savefig(f"{out}/lit_map.png", bbox_inches="tight", facecolor="#F5F4F2", pad_inches=0.15)
