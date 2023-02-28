# HACRP-HAIs

Python source code and publicly available data for analyzing a history of biased hospital penalization under the Hospital Acquired Conditions Reduction Program (HACRP), a program administered by the Centers for Medicare and Medicaid Services (CMS). The associated manuscript has been submitted to a peer-reviewed journal and a link to the paper will appear once it is published. This public repository is provided to promote transparency and permit reproducibility of the associated research.


## Directories and files
All results figures, statistics, and most of the data in this repository can be exactly reproduced by running files within this repository. Below, is a breakdown of the repository's contents. The directories are numbered to indicate the order that users should follow when reproducing files within the repository and results of the associated manuscript.

<details><summary>1\_CleanCurateCompile\_CareCompare\_Data</summary>
Each python file in this directory aggregates years of archived CMS CareCompare data into a single file.

- `HACRP_Facility_Files_CombineYears.py`
- `HAI_Facility_Files_CombineYears.py`

</details>


<details><summary>2\_Preprocess\_CareCompare\_data</summary>
Each python file in this directory preprocesses time-aggregated HAI data to achieve standardized feature names and filtered-feature datasets.

- `Generate_CAUTI_data.py`
- `Generate_CLABSI_data.py`
- `Generate_MRSA_data.py`
- `Generate_CDI_data.py`

</details>

<details><summary>3\_Merge\_HAC\_with\_HAI</summary>
Jupyter notebook files in this directory merge data from HAI files with data from HACRP files. These files are also responsible for reproducing HACRP penalty assignments from scratch (a vital validation step). Each year is represented by its own file, due to the complexity of the tasks and varied changes in the HACRP program from one year to the next.

- `2015.ipynb`
- `2016.ipynb`
- `2017-Part1.ipynb`
- `2017-Part2.ipynb`
- `2018.ipynb`
- `2019.ipynb`
- `2020.ipynb`
- `2021.ipynb`
- `2022.ipynb`   
</details>

<details><summary>4\_Merge\_HAC-HAI\_with\_HCRIS</summary>
Jupyter notebook files in this directory take the merged HAC-HAI data and then merge it with data from the CMS Healthcare Cost Report Information System (HCRIS).

- <details><summary>`1_generate_filtered_PUF_df.ipynb`</summary>

     - This Jupyter notebook file checks, constructs, and/or reproduces payments from the Inpatient Prospective Payment System (IPPS) and penalties from the Hospital Acquired Conditions Reduction Program (HACRP). These data are obtained from HCRIS data sets.

- <details><summary>`2_generate_compiled_df.ipynb`</summary>

     - This Jupyter notebook produces the compiled file of HAI, HACRP, and HCRIS data that will be used in part for optimizing random sampling models, which are in-turn used to calculate the standardized infection score (SIS).

</details>


<details><summary>5\_Optimize\_random\_sampling\_models</summary>
The purpose of contents in this directory are to explain the variation in reported numbers of infections for specific types of HAIs (across years and among hospitals) as a consequence of random variation based on hospital volume. The models used here are based on a simple binomial random sampling approach (similar to a model based on coin flips).

- <details><summary>`CAUTI_opt_DataGen.py`</summary>
	- A small file (only 8 lines). The file is used to pass CAUTI-based parameters and arguments to functions in the HAI_optimize.py file.

- <details><summary>`CLABSI_opt_DataGen.py`</summary>
	- A small file (only 8 lines). The file is used to pass CLABSI-based parameters and arguments to functions in the HAI_optimize.py file.

- <details><summary>`MRSA_opt_DataGen.py`</summary>
	- A small file (only 8 lines). The file is used to pass MRSA-based parameters and arguments to functions in the HAI_optimize.py file.

- <details><summary>`CDI_opt_DataGen.py`</summary>
	- A small file (only 8 lines). The file is used to pass CDI-based parameters and arguments to functions in the HAI_optimize.py file.	  
- <details><summary>`HAI_optimize.py`</summary>

	- This python file optimizes parameters of random sampling models for particular types of HAIs (CAUTI, CLABSI, MRSA, CDI).

</details>


<details><summary>6\_Generate\_SIS\_results</summary>
The jupyter notebooks in this directory are similar to those in the Merge\_HAC\_with_HAI directory. However, rather than attempt to reproduce actual HACRP penalty assignments using numbers of observed infections, each Jupyter notebook produces HACRP penalty assignments based on the numbers of infections expected at random based on volume.

- `SIS_2015.ipynb`
- `SIS_2016.ipynb`
- `SIS_2017.ipynb`
- `SIS_2018.ipynb`
- `SIS_2019.ipynb`
- `SIS_2020.ipynb`
- `SIS_2021.ipynb`
- `SIS_2022.ipynb`

</details>


<details><summary>7\_Generate\_Final\_results</summary>
This jupyter notebook in this directory imports a single file containing merged data from all of the above directories (1 to 6). It then produces all tables and figures contained in the associated manuscript.

- `generate_final_results.ipynb`

</details>


<details><summary>data</summary>
This directory contains other directories, each containing data that are either imported or produced by the above directories (1 to 7).

- <details><summary>CareCompare\_data</summary>

   - <details><summary>CombinedFiles_HACRP</summary>
   
   		- <details><summary>`Facility.pkl`</summary>
   		A pickle file containing cleaned and curated data from the Hospital Acquired Conditions Reduction Program (HACRP) files obtained from the CMS Care Compare hospitals archive.
   
   - <details><summary>CombinedFiles_HAI</summary>
   
   		- <details><summary>`Facility.pkl`</summary> 
   		A pickle file containing cleaned and curated data on Healthcare Associated Infections (HAIs) obtained from the CMS Care Compare hospitals archive.


	
- <details><summary>Compiled\_HCRIS-HACRP-HAI-RAND</summary>
Files in this directory contain data merged from HCRIS, cost report data from RAND, and files from the CMS Care Compare archive for HAIs and the HACRP. The two files below contain the exact same data, but in different file formats.

	- `Compiled_HCRIS-HACRP-HAI-RAND.csv`
	- `Compiled_HCRIS-HACRP-HAI-RAND.pkl`
   

- <details><summary>finalized</summary>
This directory contains files that are the final product of merging data from HCRIS, RAND, and HAI and HACRP data from Care Compare, as well as data on reproduced penalty assignments, penalty assignments based on the standardized infection score (SIS), and penalty assignments based on random expectations. 

    - `final_2015.pkl`
    - `final_2016.pkl`
    - `final_2017.pkl`
    - `final_2018.pkl`
    - `final_2019.pkl`
    - `final_2020.pkl`
    - `final_2021.pkl`
    - `final_2022.pkl`
     

- <details><summary>HCRIS_data</summary>
This directory contains a file engineered from freely available SAS-based HCRIS cost report files. The file is generated by the `1_generate_filtered_PUF_df.ipynb` file. The resulting data file is then used by the `2_generate_compiled_df.ipynb` file.

     - `FilteredEngineeredPUF_p5.pkl`

- <details><summary>merged\_HAC\_HAI</summary>
Files in this directory are serialized python data files. These files contain HACRP data merged with HAI data, as well as reproduced penalty assignments and their associated data.

     - `HAI_HAC_2015.pkl`
     - `HAI_HAC_2016.pkl`
     - `HAI_HAC_2017.pkl`
     - `HAI_HAC_2018.pkl`
     - `HAI_HAC_2019.pkl`
     - `HAI_HAC_2020.pkl`
     - `HAI_HAC_2021.pkl`
     - `HAI_HAC_2022.pkl`
     - `P1_HAI_HAC_2017_holdout.pkl`
     - `P1_HAI_HAC_2017.pkl`

- <details><summary>optimized\_by\_HAI\_file\_date</summary>
This directory contains four other directories, each containing the outputs of random sample based modeling, including optimized model parameters.

     - CAUTI
     - CLABSI
     - MRSA
     - CDI

- <details><summary>preprocessed\_HAI\_data</summary>
	The directory holds curated and processed data from the CMS CareCompare Hospital archive. The contents are:
     - `CAUTI_Data.pkl`
     - `CLABSI_Data.pkl`
     - `MRSA_Data.pkl`
     - `CDI_Data.pkl`

     
- <details><summary>Rand_CostReport</summary>
This directory contains a file obtained from the RAND hospital cost report tool, which offers a single freely available file. This file is used in the current project to verify derived IPPS payment values.

     - `rand_hcris_free_2022_11_01.csv`

- <details><summary>states_codes</summary>
This directory contains a single small file. The file contains state codes used in the formation of 6-digit CMS facility numbers. These are used in verifying and engineering HCRIS data.

     - `HCRIS_STATE_CODES.csv`

</details>



<details><summary>figures</summary>
The files in this directory comprise the graphical results of the associated manuscript and its appendix.


- <details><summary>`Hists_HAC.png`</summary>
A figure showing that the distribution of random-based HAC scores is highly similar to the distribution of actual HAC scores.

- <details><summary>`Obs_v_Pred.png`</summary>
A figure showing that random expectations based on volume explain the majority of variation in reported numbers of infections across hospitals.

- <details><summary>`change_in_rank.png`</summary>
A figure showing the result of accounting for random expectations based on volume when calculating rates of HAIs. Specifically, hospital rankings in HAC scores drastically change when using the SIS to account for random expectations.

- <details><summary>`expected_penalties.png`</summary>
A figure showing the accumulation of biased HACRP penalties and inappropriate CMS savings across program years (2015 to 2022).

</details>




## Reproducing data files and results of the associated research

Instructions are provided here for exactly reproducing nearly all data files and results from scratch. The code in this project is research code, and not constructed to software development standards. The user will need a basic-to-intermediate working knowledge of python. Additionally, the instructions and file paths are MAC-based. Windows users will need to modify the source code as needed.

<details><summary>1. Download this repository or fork it on GitHub.</summary>

- The directory should be stored in a `GitHub` directory directly below the user directory. Otherwise, the user will need to change file paths in each python (.py) and jupyter notebook (.ipynb) file. 

</details>

<details><summary>2. Ensure the following software is installed:</summary>
Versions are those used in the current project. Similar versions will likely work as well. 

- python==3.8.12
- pandas==1.4.0 
- numpy==1.22.1
- scipy==1.7.3
- matplotlib==3.3.4
- matplotlib-inline==0.1.3
- jupyter-book==0.12.2
- jupyter-core==4.9.1
- ipython==8.0.1
- scikit_posthocs==0.7.0

</details>

<details><summary>3. Obtain HCRIS public use files (PUFs) for federal fiscal years (FFY 2015 - 2022) </summary>

- These files are downloaded from: `https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/Cost-Reports/Cost-Reports-by-Fiscal-Year.` 

- For each FFY, these PUFs consist of a report table, a numeric table, and an alpha-numeric table. 

- These files are too large to provision with this project's repository. 

- Store the files on this path: `~/Desktop/HCRIS/HCRIS_PUFs/`, where the tilde (~) indicates the user directory. Of course, you can store them wherever you like, so long as the path in the `1_generate_filtered_PUF_df.ipynb` file is changed to reflect the PUFs location.

</details>

<details><summary>4. Run programs following the numerical file structure.</summary>

 - <details><summary>1\_CleanCurateCompile\_CareCompare\_Data</summary>
Run these files to generate aggregated HAI and HACRP data. It doesn't matter which is run first.

	- `HACRP_Facility_Files_CombineYears.py`
	- `HAI_Facility_Files_CombineYears.py`


- <details><summary>2\_Preprocess\_CareCompare\_data</summary>
Run these files to preprocesses aggregated HAI data. Each file will take a few hours, so it's best to run each in a different terminal window. It doesn't matter which is run first.

	- `Generate_CAUTI_data.py`
	- `Generate_CLABSI_data.py`
	- `Generate_MRSA_data.py`
	- `Generate_CDI_data.py`

- <details><summary>3\_Merge\_HAC\_with\_HAI</summary>
Run each of these Jupyter notebook files. With the exception of part 1 and part 2 for 2017, it doesn't matter which notebook you run first. For 2017, run part 1 first.

	- `2015.ipynb`
	- `2016.ipynb`
	- `2017-Part1.ipynb`
	- `2017-Part2.ipynb`
	- `2018.ipynb`
	- `2019.ipynb`
	- `2020.ipynb`
	- `2021.ipynb`
	- `2022.ipynb`   

- <details><summary>4\_Merge\_HAC-HAI\_with\_HCRIS</summary>
Run these Jupyter notebook files to merge HAC/HAI data with HCRIS data. Run `1_generate_filtered_PUF_df.ipynb` first.

	- `1_generate_filtered_PUF_df.ipynb`
	- `2_generate_compiled_df.ipynb`</summary>
	
- <details><summary>5\_Optimize\_random\_sampling\_models</summary>
Run each of the `...opt_DataGen.py` files to generate optimized random expectations for each type of HAI, for each hospital in each year. Each file will take seveal hours to run, so it is recommended to run each of the 4 files in its own terminal window.

	- `CAUTI_opt_DataGen.py`
	- `CLABSI_opt_DataGen.py`
	- `MRSA_opt_DataGen.py`
	- `CDI_opt_DataGen.py`		
	- `HAI_optimize.py`

- <details><summary>6\_Generate\_SIS\_results</summary>
The jupyter notebooks in this directory are similar to those in the Merge\_HAC\_with_HAI directory. However, rather than attempt to reproduce actual HACRP penalty assignments using numbers of observed infections, each Jupyter notebook produces HACRP penalty assignments based on the numbers of infections expected at random based on volume.

	- `SIS_2015.ipynb`
	- `SIS_2016.ipynb`
	- `SIS_2017.ipynb`
	- `SIS_2018.ipynb`
	- `SIS_2019.ipynb`
	- `SIS_2020.ipynb`
	- `SIS_2021.ipynb`
	- `SIS_2022.ipynb`

- <details><summary>7\_Generate\_Final\_results</summary>
Run this jupyter notebook to reproduce all tables and figures contained in the associated manuscript.

	- `generate_final_results.ipynb`

</details>