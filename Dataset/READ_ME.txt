Please note that for two of the WSIs, two different parts have been cropped and used as input. 
That is the reason behind having 244 thumbnails in the dataset but 242 WSIs in the manifest file.
The name of these WSIs are:
TCGA-C8-A3M8-01A-01-TSA.F27E2918-AAA7-41F0-90D3-56322773E678
TCGA-43-6770-01Z-00-DX1.466dd07f-b147-48bb-9349-fe1f5f3bcae5
========================================================================================================================================
The "k_fold_indexes.pickle" is list containing 5 lists (for each fold).
Each of the inner lists, contains two numpy arrays, the first one being the training indexes and the second one being the test indexes.
The order of the thumbnail addresses associated with the indexes can be retrieved by loading the files using the glob.glob() function.
