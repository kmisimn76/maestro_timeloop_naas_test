cd src/SparseNAAS

RUNPYTHON=python
#RUNPYTHON=python -m cProfile -s cumulative

#python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 100000 --df shi --alg genetic #VGG genetic example

if [ $1 = "VGG" ]
then
    $RUNPYTHON main.py --outdir outdir/res --model vgg16_sparse_group --fitness latency --cstr area --mul 0.5 --epochs 100 --num_pop 50 --num_parents 25 --df shi --alg genetic
else
    $RUNPYTHON main.py --outdir outdir/res --model resnet50_sparse --fitness latency --cstr area --mul 0.5  --epochs 7000 --num_pop 100 --num_parents 25 --df shi --alg genetic #sparse
    #$RUNPYTHON main.py --outdir outdir/res --model resnet50_dense --fitness latency --cstr area --mul 0.5  --epochs 7000 --num_pop 100 --num_parents 40 --df shi --alg genetic #dense
fi

cd ../../
