# create dataset of rubics cubes
# take param N for NxNxN cubes
# do some random moves aprox. 10 ~ N using magiccube https://pypi.org/project/magiccube/
# 'label' = solved cube
# 'input' = scrambled cube
# metadata:
#  define vocab: pad + input (pieces) + output (all moves)
#  ignorelable pad (0)
#  sequence length