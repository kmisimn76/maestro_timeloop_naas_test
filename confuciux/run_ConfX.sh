cd src/ConfX
python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 500 --df shi --alg RL_GA
cd ../../
