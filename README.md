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

![DET_cos](https://github.com/user-attachments/assets/220b350a-5a8b-4594-a378-e96b38380c84)
![DET_produced](https://github.com/user-attachments/assets/22e53fdd-4dd2-4171-876b-5d5d757f53ca)
![FERET_dif_vec_ubo](https://github.com/user-attachments/assets/737f2b2d-49d8-40e1-9779-695de719b7e0)
![FERET_dif_vec_opencv](https://github.com/user-attachments/assets/07abc656-c1d6-4894-a203-49b1ee265241)
![FERET_dif_vec_facemorpher](https://github.com/user-attachments/assets/ec005206-056c-4149-8fdd-204c97868d45)
![FERET_dif_vec_facefusion](https://github.com/user-attachments/assets/191207ff-efa7-44c1-a12d-63e2d8c11d2b)
![FERET_dif_vec](https://github.com/user-attachments/assets/053f7d9b-7388-4305-b427-123380167629)
![FRGC_dif_vec](https://github.com/user-attachments/assets/36b564f9-25c0-4b44-8448-2906a6a16b06)
