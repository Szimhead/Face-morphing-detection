02238-s222919-MAD
Differential Morphing Attack Detection using Deep Face Representation

Deep_Face_Representations_for_Differential_Morphing_Attack_Detection.pdf contains the reproduced method.

To reproduce, please insert necessary data in config.ini file.
Only one kind of distance metric (difference vectors or cosine distance) can be generated at a time 
(use the value using_distance_vector in config.ini to determine the desired one).

Files features*.json contain precomputed values of feature vectors for the original datasets.
In order to run the process for a different dataset, the content of these files must be erased and an empty dictionary
must be inserted ("{}")(please, note that the files cannot be deleted).

Files mds*.csv contain precomputed values of the difference vectors after applying MDS (Multidimensional Scaling) used
only for visualization. To regenerate data, their content must be erased.
Note, that for FRGC the data is clustered first, resulting in less data points (12879) than the number of difference 
vectors (30815).

The only image produced automatically is DET_produced.png. All the other images have been saved during execution of the script.