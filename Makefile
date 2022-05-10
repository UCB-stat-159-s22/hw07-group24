.PHONY: env
env :
	conda env create -f environment.yml
	conda activate census
	python -m ipykernel install --user --name census --display-name "census"
    
.PHONY: all
all :
	jupyter execute main.ipynb
	jupyter execute notebooks/EDA.ipynb
	jupyter execute notebooks/FeatureEngineering.ipynb
	jupyter execute notebooks/Modeling.ipynb