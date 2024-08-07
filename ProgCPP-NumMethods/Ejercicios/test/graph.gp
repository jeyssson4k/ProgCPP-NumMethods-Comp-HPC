set term pdf
plot 'data.txt' w lp lw 4 ps2 t 'data'
set xlabel 'N'
set ylabel 'Norma'
set out 'graph.pdf'