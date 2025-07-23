#! /bin/bash
# Download the data form the paper
#Altan E, Solla SA, Miller LE, Perreault EJ (2021) Estimating the dimensionality of the manifold underlying multi-electrode neural recordings. PLoS Comput Biol 17(11): e1008591. https://doi.org/10.1371/journal.pcbi.1008591
# The dataset contains high dimensional data from linear and non-linear embedding
# We computed the FCI ID on the data contained in Fig4_data

if [ ! -d 'Manuscript_Data_and_Code' ]; then
   echo "Download of the data from
   Altan, Ege; Perreault, Sara A. Solla, Lee E. Miller, Eric J. (2020). Estimating the dimensionality of the manifold underlying multi-electrode neural recording. figshare. Software. https://doi.org/10.6084/m9.figshare.13335341.v2
   "

   wget https://figshare.com/ndownloader/files/31310512
   unzip 31310512
   rm  31310512
   mv 'Manuscript Data and Code' Manuscript_Data_and_Code
else
  echo "the dataset already exists, check the folder  'Manuscript Data and Code'"
fi

