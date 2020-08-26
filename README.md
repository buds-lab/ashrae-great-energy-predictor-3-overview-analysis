# The ASHRAE Great Energy Predictor III competition: Overview and results
This repository is for overview analysis of [a paper about the ASHRAE Great Energy Predictor Shootout III in the journal Science and Technology for the Built Environment](https://www.tandfonline.com/doi/full/10.1080/23744731.2020.1795514).

To cite this paper:

Clayton Miller, Pandarasamy Arjunan, Anjukan Kathirgamanathan, Chun Fu, Jonathan Roth, June Young Park, Chris Balbach, Krishnan Gowri, Zoltan Nagy, Anthony D. Fontanini & Jeff Haberl (2020) The ASHRAE Great Energy Predictor III competition: Overview and results, Science and Technology for the Built Environment, DOI: 10.1080/23744731.2020.1795514

## Overview of Analysis of Data *about the competition*
There are several components to the analysis in this publication in which data was collected from the Kaggle platform in various ways:

### Web Scraping 
Get meta data of Kernel notebooks, Discussion topics, and Historical competitions fromt the Kaggle website:
- [Scrape kernel notebooks, discussion topics, and historical competitions](WebScraping/Kaggle_WebScraping.ipynb)
- [Scrape discussion and comments text](WebScraping/Kaggle_WebScraping_Discussions.ipynb)
   
### Demographics of Competitors
Overview visualization(s) that gives an understanding of the people involved in the competition:
- [People/Demographics](Demographics/Map.ipynb)
   
### Discussion Board
Characterize and label discussion topics and comments from the competition:
- [Discussion Board/Text Analysis](DiscussionBoard/DiscussionAnalysis.ipynb)

### Kernel notebooks
Characterize and label notebooks and techniques topics from the competition
- [Notebook Data Cleaning](KernelNotebook/NotebookDataCleaning.ipynb) 
- [Notebook Analysis](KernelNotebook/Combined%20bar%20charts.ipynb) 

## Notebooks and Solutions Directory
In addition, there are two wiki pages that focus on a curation of the notebooks and full solutions from the competition:
- [Curation of Machine Learning Tutorials](https://github.com/buds-lab/ashrae-great-energy-predictor-3-overview-analysis/wiki/Curation-of-Machine-Learning-Tutorials)
- [Overview of the Shared Full Solutions in notebooks and discussion board](https://github.com/buds-lab/ashrae-great-energy-predictor-3-overview-analysis/wiki/Shared-solutions-in-kernel-notebooks-and-discussion-board)

