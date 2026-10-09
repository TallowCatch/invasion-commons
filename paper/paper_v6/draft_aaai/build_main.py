"""Assemble main.tex (single source file, as AAAI requires) from body.tex, the preamble in preamble.tex and the
TikZ/figure sources. Edit body.tex or the captions below, then run:  python3 build_main.py && pdflatex ... """
pre = open('preamble.tex').read()
body = open('body.tex').read()
tikz = lambda p: open(p).read()
caps = {k: open(f'captions/{k}.tex').read().strip() for k in ('fig1', 'fig2', 'fig3', 'fig4')}
figs = {
 'FIGURE1': r'''\begin{figure*}[t]
  \centering
  \resizebox{0.76\textwidth}{!}{''' + tikz('../figures/fig1_game_step.tex') + r'''}\\[3mm]
  \resizebox{0.66\textwidth}{!}{''' + tikz('../figures/fig_networks.tex') + r'''}
  \caption{''' + caps['fig1'] + r'''}
  \label{fig:step}
\end{figure*}''',
 'FIGURE2': r'''\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig3_audits.pdf}
  \caption{''' + caps['fig2'] + r'''}
  \label{fig:audits}
\end{figure*}''',
 'FIGURE3': r'''\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig4_llm_main.pdf}
  \caption{''' + caps['fig3'] + r'''}
  \label{fig:llm}
\end{figure*}''',
 'FIGURE4': r'''\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig5_spine.pdf}
  \caption{''' + caps['fig4'] + r'''}
  \label{fig:spine}
\end{figure}''',
}
for k, v in figs.items():
    assert body.count(k) == 1, k
    body = body.replace(k, v)
open('main.tex', 'w').write(pre + body + '\n\\end{document}\n')
print('main.tex written')
