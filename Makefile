.PHONY: env
env :
	conda env create -f environment.yml
	conda activate census
	python -m ipykernel install --user --name census --display-name "census"
    
.PHONY: all
all :

	jupyter-book run main.ipynb
	jupyter-book run /notebooks/EDA.ipynb
	jupyter-book run /notebooks/FeatureEngineering.ipynb
	jupyter-book run /notebooks/Modeling.ipynb

