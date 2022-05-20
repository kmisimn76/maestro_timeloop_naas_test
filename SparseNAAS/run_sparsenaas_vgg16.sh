cd src/SparseNAAS

RUNPYTHON=python
#RUNPYTHON=python -m cProfile -s cumulative

#python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 100000 --df shi --alg genetic #VGG genetic example
$RUNPYTHON main.py --outdir outdir --model vgg16_sparse_group --fitness latency --cstr area --mul 0.5 --epochs 100 --df shi --alg genetic
cd ../../
