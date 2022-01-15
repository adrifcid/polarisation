This repository contains the code used in Adrián Fernández Cid's Master Thesis (University of Barcelona).

 # A study of polarisaton in bimodal social networks

## Abstract

Social polarisation is a central issue in the social sciences, and it has acquired mainstream interest in recent years. A prominent area of current research in computational social science studies the polarisation of social systems in terms of features of their graph representation. Such structural polarisation measures can capture well-grounded aspects of polarisation at a comparatively lower cost than content-based or distributional approaches, although some of them have been shown to depend on unrelated network properties like average degree or systematically give false positives on randomised networks. In this master's thesis, I explore a novel approach that implements an axiomatic polarisation measure with hierarchical clustering on bimodal networks, which are less studied in the literature. The clustering implements the well known Ward and centroid methods, as well as a new one, *poldist*, inspired by the polarisation measure used. In the validation use case, on the standard Southern Women dataset, results reasonably agree with the expected separation in two communities for the Ward and centroid methods, but not for poldist. On the other hand, the application use case, on data from the platform of the Conference on the Future of Europe, shows no significant dipoles neither in the topic-specific nor the global analysis, which (given the previous pipeline validation and the relatively low participation of the platform) points to a lack of polarisation in the data. However, further analysis on such data in terms of multipole partitions is underway and may yet reveal some structure. Current results show the proposed pipeline is a promising candidate for the study of polarisation in bimodal social networks and should be further explored.

### Supervisors

Emanuele Cozzo

Oriol Pujol Vila

## Folder structure
```bash
.
├── PFM_AdrianFernandezCid.pdf                # Project report
├── notebooks                                 # IPython notebooks of use cases
│   ├── SouthernWomen.ipynb                   # Validation use case
│   └── ConferenceOnTheFutureOfEurope.ipynb   # Application use case 
├── utils                                     # Clustering code
│   └── clustering.py  
├── data                                      # Data of use cases
│   ├── SW/                                   # Validation use case
│   └── CFE/                                  # Application use case 
├── plots                                     # Plots of use case results
│   ├── SW/                                   # Validation use case
│   └── CFE/                                  # Application use case                       
└── literature/                               # Some of the references of the project 
```
The high-level description of the project is contained, in the intended order of presentation, in PFM_AdrianFernandezCid.pdf, SouthernWomen.ipynb and ConferenceOnTheFutureOfEurope.ipynb. I recommend following that progression for revision.

**NB**: Before using the CFE data in ConferenceOnTheFutureOfEurope.ipynb you will have to unzip the corresponding file in data/CFE/.

Adrián Fernández Cid
