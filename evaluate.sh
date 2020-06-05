gpu=0
dataset=cifar10

for divergence in KL JS Jef LogD
do
	python generate_images.py --gpu $gpu --dataset $dataset --divergence $divergence --load_num 12000000 --dir_name fakes-IS-50k --generate_num 50000
	python evaluate.py --gpu $gpu --score IS --path1 ./results/$dataset-$divergence/fakes-IS-50k >> $divergence.txt

	for i in 1 2 3 4 5
	do
		python generate_images.py --gpu $gpu --dataset $dataset --divergence $divergence --load_num 12000000 --dir_name fakes-FID-10k-$i --generate_num 10000 -seed ${i}000
		python evaluate.py --gpu $gpu --score FID --path1 ./results/$dataset-$divergence/fakes-FID-10k-$i --path2 ./datasets/$dataset/reals >> $divergence.txt
	done
done