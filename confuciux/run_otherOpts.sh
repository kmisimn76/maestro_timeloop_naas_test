cd src/Other_Opts
#python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 100000 --df shi --alg genetic #VGG genetic example
python main.py --outdir outdir --model vgg16 --fitness latency --cstr area --mul 0.5 --epochs 100000 --df shi --alg genetic
cd ../../