import pstats

with open('cprofile.txt', 'w') as f:
    p = pstats.Stats('cprofile.prof', stream=f)
    p.strip_dirs().sort_stats('cumulative').print_stats()