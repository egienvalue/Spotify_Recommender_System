all: recommend

recommend: 
	python main_recommend.py 52

train:
	python neural_recsys.py

data_process:
	python produce_graph.py

clean:
	#rm ./output/*
	#rm *.png
