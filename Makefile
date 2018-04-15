all: final

final:
	python produce_graph.py
	python neural_recsys.py

clean:
	rm ./output/*
	rm ./Model_saved/*
	rm *.png
