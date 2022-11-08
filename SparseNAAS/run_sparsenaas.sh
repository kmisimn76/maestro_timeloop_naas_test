cd src/SparseNAAS

RUNPYTHON=python
#RUNPYTHON=pypy3
#RUNPYTHON=python -m cProfile -s cumulative

#python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 100000 --df shi --alg genetic #VGG genetic example

if [ $1 = "VGG" ]
then
    $RUNPYTHON main.py --outdir outdir/res --model vgg16_sparse_group --fitness latency --cstr area --mul 0.5 --epochs 100 --num_pop 50 --num_parents 25 --df shi --alg genetic
else
    $RUNPYTHON main.py --outdir outdir/res_new_opt --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 21000 --num_pop 100 --num_parents 25 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res5 --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 21000 --num_pop 100 --num_parents 25 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 21000 --num_pop 200 --num_parents 60 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 21000 --num_pop 30 --num_parents 10 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 140000 --num_pop 2000 --num_parents 500 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res --model resnet50_dense --fitness latency --cstr area --mul 0.5  --epochs 7000 --num_pop 100 --num_parents 40 --df shi --alg genetic #dense
fi

cd ../../
